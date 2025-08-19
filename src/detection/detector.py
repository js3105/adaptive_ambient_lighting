import cv2
import numpy as np
from picamera2 import MappedArray
from picamera2.devices import IMX500
from picamera2.devices.imx500 import postprocess_nanodet_detection
#Funktioniert
class Detection:
    def __init__(self, coords, category, conf, metadata, imx500: IMX500, picam2):
        self.category = category
        self.conf = float(conf)
        self.box = imx500.convert_inference_coords(coords, metadata, picam2)

class ObjectDetector:
    def __init__(self, imx500: IMX500, intrinsics, picam2):
        self.imx500 = imx500
        self.intrinsics = intrinsics
        self.picam2 = picam2
        self.last_detections = []
        self.TRAFFIC_LIGHT_CLASS_ID = 0
        self._gamma = 1.6
        self._clahe_clip = 2.0
        self._clahe_grid = 8
        self._s_min = 60
        self._v_min = 60
        self._use_r_dominance = True
        self._r_margin = 20

    def _labels(self):
        return [] if self.intrinsics.labels is None else self.intrinsics.labels

    def parse_detections(self, metadata):
        np_outputs = self.imx500.get_outputs(metadata, add_batch=True)
        if np_outputs is None:
            return self.last_detections
        input_w, input_h = self.imx500.get_input_size()
        bbox_normalization = getattr(self.intrinsics, "bbox_normalization", False)
        bbox_order = getattr(self.intrinsics, "bbox_order", "yx")
        threshold, iou, max_dets = 0.55, 0.65, 10

        if getattr(self.intrinsics, "postprocess", "") == "nanodet":
            boxes, scores, classes = postprocess_nanodet_detection(
                outputs=np_outputs[0], conf=threshold, iou_thres=iou, max_out_dets=max_dets
            )[0]
            from picamera2.devices.imx500.postprocess import scale_boxes
            boxes = scale_boxes(boxes, 1, 1, input_h, input_w, False, False)  # y0,x0,y1,x1
            boxes = np.array_split(boxes, 4, axis=1)
            boxes = zip(*boxes)
        else:
            boxes, scores, classes = np_outputs[0][0], np_outputs[1][0], np_outputs[2][0]
            if bbox_normalization:
                boxes = boxes / input_h
            if bbox_order == "xy":
                boxes = boxes[:, [1, 0, 3, 2]]
            boxes = np.array_split(boxes, 4, axis=1)
            boxes = zip(*boxes)

        self.last_detections = [
            Detection(box, category, score, metadata, self.imx500, self.picam2)
            for box, score, category in zip(boxes, scores, classes) if float(score) > threshold
        ]
        return self.last_detections

    def _shrink_box(self, x, y, w, h, *, fx=0.12, fy=0.18, w_max=None, h_max=None):
        sx = x + int(w * fx); ex = x + w - int(w * fx)
        sy = y + int(h * fy); ey = y + h - int(h * fy)
        if w_max is not None:
            sx = max(0, min(sx, w_max - 2)); ex = max(sx + 2, min(ex, w_max - 1))
        if h_max is not None:
            sy = max(0, min(sy, h_max - 2)); ey = max(sy + 2, min(ey, h_max - 1))
        return sx, sy, ex - sx, ey - sy

    # ---------- Farb-Erkennung (wir rechnen in RGB) ----------
    def _preprocess_rgb(self, img_rgb):
        lut = np.array([((i / 255.0) ** (1.0 / self._gamma)) * 255 for i in range(256)], dtype=np.uint8)
        img_gamma = cv2.LUT(img_rgb, lut)
        hsv = cv2.cvtColor(img_gamma, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        clahe = cv2.createCLAHE(clipLimit=self._clahe_clip, tileGridSize=(self._clahe_grid, self._clahe_grid))
        v = clahe.apply(v)
        return cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2RGB)

    def _traffic_light_masks(self, img_rgb):
        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        gate = cv2.inRange(cv2.merge([h, s, v]), (0, self._s_min, self._v_min), (179, 255, 255))
        red1, red2 = cv2.inRange(h, 0, 10), cv2.inRange(h, 160, 179)
        yellow, green = cv2.inRange(h, 15, 35), cv2.inRange(h, 45, 75)
        mask_red = cv2.bitwise_and(cv2.bitwise_or(red1, red2), gate)
        mask_yellow = cv2.bitwise_and(yellow, gate)
        mask_green = cv2.bitwise_and(green, gate)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, k, iterations=1)
        mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_OPEN, k, iterations=1)
        mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, k, iterations=1)
        if self._use_r_dominance:
            r, g, b = img_rgb[:, :, 0], img_rgb[:, :, 1], img_rgb[:, :, 2]
            r_dom = ((r.astype(np.int16) > g.astype(np.int16) + self._r_margin) &
                     (r.astype(np.int16) > b.astype(np.int16) + self._r_margin))
            mask_red = cv2.bitwise_and(mask_red, (r_dom.astype(np.uint8) * 255))
        return mask_red, mask_yellow, mask_green

    def detect_phase_by_hsv(self, roi_rgb):
        if roi_rgb is None or roi_rgb.size == 0:
            return "Unklar"
        roi_pp = self._preprocess_rgb(roi_rgb)
        m_red, m_yel, m_grn = self._traffic_light_masks(roi_pp)
        counts = [cv2.countNonZero(m_red), cv2.countNonZero(m_yel), cv2.countNonZero(m_grn)]
        if max(counts) == 0: return "Unklar"
        return ["Rot", "Gelb", "Gruen"][int(np.argmax(counts))]

    # ---------- Zeichnen & ROI ----------
    def draw_callback(self, request, stream="main"):
        if not self.last_detections:
            return
        labels = self._labels()
        with MappedArray(request, stream) as m:
            draw_surface = m.array  # Original-Buffer (XRGB8888 → 4 Kanäle)
            h_max, w_max = draw_surface.shape[:2]

            # Für Farblogik: sichere RGB-Ansicht erzeugen
            if draw_surface.ndim == 3 and draw_surface.shape[2] == 4:
                # XRGB8888 wird in NumPy i.d.R. als BGRA dargestellt → nach RGB konvertieren
                proc_rgb_full = cv2.cvtColor(draw_surface, cv2.COLOR_BGRA2RGB)
            elif draw_surface.ndim == 3 and draw_surface.shape[2] == 3:
                # Selten: RGB888
                proc_rgb_full = draw_surface[:, :, ::-1]  # falls es BGR wäre → zu RGB (sicher)
            else:
                return  # ungeeignetes Format

            for det in self.last_detections:
                try:
                    x, y, w, h = map(int, det.box)
                    x = max(0, min(x, w_max - 1)); y = max(0, min(y, h_max - 1))
                    w = min(w, w_max - x); h = min(h, h_max - y)

                    name = labels[int(det.category)] if 0 <= int(det.category) < len(labels) else f"Class {int(det.category)}"

                    if int(det.category) == self.TRAFFIC_LIGHT_CLASS_ID:
                        x, y, w, h = self._shrink_box(x, y, w, h, fx=0.18, fy=0.05, w_max=w_max, h_max=h_max)
                        # ROI aus der RGB-Prozessansicht
                        roi_rgb = proc_rgb_full[y:y+h, x:x+w]
                        if roi_rgb.size > 0:
                            phase = self.detect_phase_by_hsv(roi_rgb)
                            name = f"{name} ({phase})"

                    label = f"{name} ({det.conf:.2f})"

                    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    tx, ty = x + 5, y + 15

                    # halbtransparenter Hintergrund auf dem ORIGINAL-Buffer zeichnen (BGR(A)-Farben!)
                    overlay = draw_surface.copy()
                    cv2.rectangle(overlay, (tx, ty - th), (tx + tw, ty + baseline), (255, 255, 255), cv2.FILLED)
                    cv2.addWeighted(overlay, 0.30, draw_surface, 0.70, 0, draw_surface)

                    text_color_bgr = (0, 0, 255)
                    box_color_bgr  = (0, 255, 255) if int(det.category) == self.TRAFFIC_LIGHT_CLASS_ID else (0, 255, 0)

                    cv2.putText(draw_surface, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color_bgr, 1)
                    cv2.rectangle(draw_surface, (x, y), (x + w, y + h), box_color_bgr, 2)

                except (ValueError, IndexError) as e:
                    print(f"Error processing detection: {e}")
                    continue