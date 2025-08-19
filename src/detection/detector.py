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

    def _labels(self):
        """Get labels from intrinsics"""
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
    
    def _bright_mask(self, hsv, s_min=60, v_min=140):
        """Maske für helle, gesättigte Pixel (Hue egal)."""
        return cv2.inRange(hsv, (0, s_min, v_min), (179, 255, 255))

    # --- RAW-HSV mit Drittel-Logik ---
    def detect_phase_by_hsv(self, roi_bgr):
        """
        Drittel-Logik + nur helle/kräftige Pixel:
        - oben: Rot
        - mitte: Gelb
        - unten: Grün
        - Zählung nur dort, wo S>=s_min und V>=v_min.
        """
        if roi_bgr is None or roi_bgr.size == 0:
            return "Unklar"

        h, w = roi_bgr.shape[:2]
        if h < 6:
            return "Unklar"

        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        h_third = h // 3

        top    = hsv[0:h_third, :, :]           # Rot
        middle = hsv[h_third:2*h_third, :, :]   # Gelb
        bottom = hsv[2*h_third:h, :, :]         # Grün

        # nur helle/kräftige Pixel pro Segment
        top_bright    = self._bright_mask(top,    s_min=60, v_min=140)
        middle_bright = self._bright_mask(middle, s_min=60, v_min=140)
        bottom_bright = self._bright_mask(bottom, s_min=60, v_min=140)

        # Rot (zweigeteilt) im oberen Drittel, nur auf bright
        r1 = cv2.inRange(top,    (0,   70, 50), (15, 255, 255))
        r2 = cv2.inRange(top,    (160, 70, 50), (180,255, 255))
        red_mask = cv2.bitwise_or(r1, r2)
        red_pixels = cv2.countNonZero(cv2.bitwise_and(red_mask, top_bright))

        # Gelb im mittleren Drittel, nur auf bright
        y_mask = cv2.inRange(middle, (20, 70, 50), (30, 255, 255))
        yellow_pixels = cv2.countNonZero(cv2.bitwise_and(y_mask, middle_bright))

        # Grün im unteren Drittel, nur auf bright
        g_mask = cv2.inRange(bottom, (55, 70, 50), (75, 255, 255))
        green_pixels = cv2.countNonZero(cv2.bitwise_and(g_mask, bottom_bright))

        # Entscheidung
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
    # --- Ende RAW-Drittel-Logik ---

    def draw_callback(self, request, stream="main"):
        if not self.last_detections:
            return
        labels = self._labels()
        with MappedArray(request, stream) as m:
            for det in self.last_detections:
                try:
                    x, y, w, h = map(int, det.box)
                    # Ensure coordinates are within array bounds
                    h_max, w_max = m.array.shape[:2]
                    x = max(0, min(x, w_max-1))
                    y = max(0, min(y, h_max-1))
                    w = min(w, w_max-x)
                    h = min(h, h_max-y)
                    name = labels[int(det.category)] if 0 <= int(det.category) < len(labels) else f"Class {int(det.category)}"
                
                    # Für Ampeln: Phasenerkennung durchführen (Drittel-Logik)
                    if int(det.category) == self.TRAFFIC_LIGHT_CLASS_ID:
                        roi = m.array[y:y+h, x:x+w]

                        # ROI innen etwas verkleinern (z. B. 10 % an allen Seiten abschneiden)
                        shrink = 0.3
                        h_roi, w_roi = roi.shape[:2]
                        dx = int(w_roi * shrink)
                        dy = int(h_roi * shrink)
                        roi = roi[dy:h_roi-dy, dx:w_roi-dx]

                        if roi.size > 0:
                            phase = self.detect_phase_by_hsv(roi)
                            name = f"{name} ({phase})"

                    label = f"{name} ({det.conf:.2f})"

                    # Text-Hintergrund halbtransparent
                    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    tx, ty = x + 5, y + 15
                    overlay = m.array.copy()
                    cv2.rectangle(overlay, (tx, ty - th), (tx + tw, ty + baseline), (255, 255, 255), cv2.FILLED)
                    cv2.addWeighted(overlay, 0.30, m.array, 0.70, 0, m.array)

                    # Text und Rahmen zeichnen
                    text_color = (0, 0, 255)
                    box_color = (0, 255, 255) if int(det.category) == self.TRAFFIC_LIGHT_CLASS_ID else (0, 255, 0)  # 3-kanalig
                    cv2.putText(m.array, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
                    cv2.rectangle(m.array, (x, y), (x + w, y + h), box_color, 2)

                except (ValueError, IndexError) as e:
                    print(f"Error processing detection: {e}")
                    continue