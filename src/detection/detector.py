import cv2
import numpy as np
from picamera2 import MappedArray
from picamera2.devices import IMX500
from picamera2.devices.imx500 import postprocess_nanodet_detection

class Detection:
    def __init__(self, coords, category, conf, metadata, imx500: IMX500, picam2):
        self.category = category
        self.conf = float(conf)
        # liefert (x, y, w, h) im ISP-Output
        self.box = imx500.convert_inference_coords(coords, metadata, picam2)


class ObjectDetector:
    def __init__(self, imx500: IMX500, intrinsics, picam2):
        self.imx500 = imx500
        self.intrinsics = intrinsics
        self.picam2 = picam2
        self.last_detections = []
        self.TRAFFIC_LIGHT_CLASS_ID = 0

        # --- Tuning-Parameter für dunkle Räume ---
        self._gamma = 1.6
        self._clahe_clip = 2.0
        self._clahe_grid = 8
        self._s_min = 60
        self._v_min = 60
        # Optionaler Rot-Fallback (R-Kanal-Dominanz)
        self._use_r_dominance = True
        self._r_margin = 20

    def _labels(self):
        if self.intrinsics.labels is None:
            return []
        return self.intrinsics.labels

    def parse_detections(self, metadata):
        np_outputs = self.imx500.get_outputs(metadata, add_batch=True)
        if np_outputs is None:
            return self.last_detections

        input_w, input_h = self.imx500.get_input_size()
        bbox_normalization = getattr(self.intrinsics, "bbox_normalization", False)
        bbox_order = getattr(self.intrinsics, "bbox_order", "yx")
        threshold = 0.55
        iou = 0.65
        max_dets = 10

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
                boxes = boxes[:, [1, 0, 3, 2]]  # -> y0, x0, y1, x1
            boxes = np.array_split(boxes, 4, axis=1)
            boxes = zip(*boxes)

        self.last_detections = [
            Detection(box, category, score, metadata, self.imx500, self.picam2)
            for box, score, category in zip(boxes, scores, classes)
            if float(score) > threshold
        ]
        return self.last_detections

    def _shrink_box(self, x, y, w, h, *, fx=0.12, fy=0.18, w_max=None, h_max=None):
        sx = x + int(w * fx)
        ex = x + w - int(w * fx)
        sy = y + int(h * fy)
        ey = y + h - int(h * fy)

        if w_max is not None:
            sx = max(0, min(sx, w_max - 2))
            ex = max(sx + 2, min(ex, w_max - 1))
        if h_max is not None:
            sy = max(0, min(sy, h_max - 2))
            ey = max(sy + 2, min(ey, h_max - 1))

        return sx, sy, ex - sx, ey - sy

    # --- Punkt 4: robuste Farbdetektion (Gamma + CLAHE + Gate + Morph + optional R-Dominanz) ---

    def _preprocess_bgr(self, img_bgr):
        # Gamma
        lut = np.array([((i / 255.0) ** (1.0 / self._gamma)) * 255 for i in range(256)]).astype("uint8")
        img_gamma = cv2.LUT(img_bgr, lut)

        # CLAHE auf V
        hsv = cv2.cvtColor(img_gamma, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        clahe = cv2.createCLAHE(clipLimit=self._clahe_clip, tileGridSize=(self._clahe_grid, self._clahe_grid))
        v = clahe.apply(v)
        hsv = cv2.merge([h, s, v])
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def _traffic_light_masks(self, img_bgr):
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # S/V-Gate
        gate = cv2.inRange(cv2.merge([h, s, v]), (0, self._s_min, self._v_min), (179, 255, 255))

        # Hue-Ranges
        red1   = cv2.inRange(h, 0, 10)
        red2   = cv2.inRange(h, 160, 179)
        yellow = cv2.inRange(h, 15, 35)
        green  = cv2.inRange(h, 45, 90)

        mask_red    = cv2.bitwise_and(cv2.bitwise_or(red1, red2), gate)
        mask_yellow = cv2.bitwise_and(yellow, gate)
        mask_green  = cv2.bitwise_and(green, gate)

        # Morphologische Säuberung
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_red    = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, k, iterations=1)
        mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_OPEN, k, iterations=1)
        mask_green  = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, k, iterations=1)

        # Optional: R-Kanal-Dominanz als extra Sicherheit für Rot
        if self._use_r_dominance:
            b, g, r = cv2.split(img_bgr)
            r_dom = ((r.astype(np.int16) > g.astype(np.int16) + self._r_margin) &
                     (r.astype(np.int16) > b.astype(np.int16) + self._r_margin))
            r_dom = (r_dom.astype(np.uint8) * 255)
            mask_red = cv2.bitwise_and(mask_red, r_dom)

        return mask_red, mask_yellow, mask_green

    def detect_phase_by_hsv(self, roi_bgr):
        if roi_bgr is None or roi_bgr.size == 0:
            return "Unklar"

        # Vorverarbeitung stabilisiert Hue in dunkler Umgebung
        roi_pp = self._preprocess_bgr(roi_bgr)

        # Masken + Zählung
        mask_red, mask_yellow, mask_green = self._traffic_light_masks(roi_pp)
        red_pixels    = cv2.countNonZero(mask_red)
        yellow_pixels = cv2.countNonZero(mask_yellow)
        green_pixels  = cv2.countNonZero(mask_green)

        max_pixels = max(red_pixels, yellow_pixels, green_pixels)
        if max_pixels == 0:
            return "Unklar"
        if max_pixels == red_pixels:
            return "Rot"
        elif max_pixels == yellow_pixels:
            return "Gelb"
        elif max_pixels == green_pixels:
            return "Gruen"
        return "Unklar"

    # --- Ende Punkt 4 ---

    def draw_callback(self, request, stream="main"):
        if not self.last_detections:
            return
        labels = self._labels()
        with MappedArray(request, stream) as m:
            for det in self.last_detections:
                try:
                    x, y, w, h = map(int, det.box)
                    h_max, w_max = m.array.shape[:2]
                    x = max(0, min(x, w_max - 1))
                    y = max(0, min(y, h_max - 1))
                    w = min(w, w_max - x)
                    h = min(h, h_max - y)

                    name = labels[int(det.category)] if 0 <= int(det.category) < len(labels) else f"Class {int(det.category)}"

                    if int(det.category) == self.TRAFFIC_LIGHT_CLASS_ID:
                        x, y, w, h = self._shrink_box(x, y, w, h, fx=0.18, fy=0.05, w_max=w_max, h_max=h_max)
                        roi = m.array[y:y+h, x:x+w]
                        if roi.size > 0:
                            phase = self.detect_phase_by_hsv(roi)
                            name = f"{name} ({phase})"

                    label = f"{name} ({det.conf:.2f})"

                    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    tx, ty = x + 5, y + 15
                    overlay = m.array.copy()
                    cv2.rectangle(overlay, (tx, ty - th), (tx + tw, ty + baseline), (255, 255, 255), cv2.FILLED)
                    cv2.addWeighted(overlay, 0.30, m.array, 0.70, 0, m.array)

                    text_color = (0, 0, 255)
                    box_color = (0, 255, 255) if int(det.category) == self.TRAFFIC_LIGHT_CLASS_ID else (0, 255, 0)
                    cv2.putText(m.array, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
                    cv2.rectangle(m.array, (x, y), (x + w, y + h), box_color, 2)

                except (ValueError, IndexError) as e:
                    print(f"Error processing detection: {e}")
                    continue