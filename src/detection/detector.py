import cv2
import numpy as np
from picamera2 import MappedArray
from picamera2.devices import IMX500
from picamera2.devices.imx500 import postprocess_nanodet_detection
from datetime import datetime

# Konstanten für die Ampelphasenerkennung
MIN_ROI_H = 24
RATIO_MARGIN = 1.12

HSV_RANGES = {
    "Rot":   [((0,   70, 120), (10, 255, 255)), ((170, 70, 120), (179, 255, 255))],
    "Gelb":  [((18,  80, 140), (38, 255, 255))],
    "Gruen": [((40,  70, 120), (90, 255, 255))],
}

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

    def _is_valid_roi(self, w, h):
        if h < MIN_ROI_H or w < 5:
            return False
        # optional: Gehäuse vs. Signal – Ampel ist deutlich höher als breit
        if (h / max(1, w)) < 1.3 * RATIO_MARGIN:
            return False
        return True

    def _bright_mask(self, hsv):
        # helle, satte Pixel = potenzielle Leuchtfläche
        s, v = hsv[:, :, 1], hsv[:, :, 2]
        mask = cv2.inRange(hsv, (0, 60, 140), (179, 255, 255))
        # Rauschen entfernen
        mask = cv2.medianBlur(mask, 3)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        return mask

    def _circle_focus(self, gray, base_mask):
        """
        Versuche, die Kreisfläche der Lampe zu finden.
        Gibt eine (optionale) verfeinerte Maske zurück.
        """
        # leichte Glättung
        g = cv2.GaussianBlur(gray, (5, 5), 0)
        # Hough-Kreis: Parameter ggf. anpassen
        circles = cv2.HoughCircles(g, cv2.HOUGH_GRADIENT, dp=1.2, minDist=10,
                                   param1=80, param2=18, minRadius=4, maxRadius=60)
        if circles is None:
            return base_mask
        circles = np.uint16(np.around(circles))
        h, w = gray.shape
        circle_mask = np.zeros_like(base_mask)
        for x, y, r in circles[0, :]:
            cv2.circle(circle_mask, (x, y), r, 255, thickness=-1)
        # Schnittmenge aus hellen Pixeln und Kreis
        return cv2.bitwise_and(base_mask, circle_mask)

    def detect_phase_by_hsv(self, roi_bgr):
        """
        Robuste Farberkennung einer Ampel-ROI.
        - fokussiert helle, satte Pixel
        - optional Hough-Kreis zur Lampenlokalisierung
        - entscheidet über Hue-Histogramm in den hellen Bereichen
        """
        current_time = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')

        h, w = roi_bgr.shape[:2]
        if not self._is_valid_roi(w, h):
            return "Unklar"

        # Vorverarbeitung: leichter Median gegen Hot-Pixel
        roi = cv2.medianBlur(roi_bgr, 3)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        bright = self._bright_mask(hsv)
        if cv2.countNonZero(bright) == 0:
            return "Unklar"

        # Optional: auf Kreisflächen einschränken (falls die Sicht stabil ist)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        bright = self._circle_focus(gray, bright)

        bright_count = max(1, cv2.countNonZero(bright))

        # Prozentwerte nur auf hellen Pixeln
        def pct_for_ranges(ranges):
            total_mask = np.zeros(bright.shape, dtype=np.uint8)
            for (lo, hi) in ranges:
                m = cv2.inRange(hsv, lo, hi)
                total_mask = cv2.bitwise_or(total_mask, m)
            # nur helle Pixel zählen
            total_mask = cv2.bitwise_and(total_mask, bright)
            return cv2.countNonZero(total_mask) / bright_count

        p_red   = pct_for_ranges(HSV_RANGES["Rot"])
        p_yel   = pct_for_ranges(HSV_RANGES["Gelb"])
        p_grn   = pct_for_ranges(HSV_RANGES["Gruen"])

        # Logging (auf helle Pixel bezogen)
        h_h = hsv[:, :, 0][bright > 0]
        s_h = hsv[:, :, 1][bright > 0]
        v_h = hsv[:, :, 2][bright > 0]
        print("\n=== Bright-Pixel Color Analysis ===")
        print(f"Timestamp (UTC): {current_time}")
        print(f"ROI Size: {w}x{h}, bright_count={bright_count}")
        if h_h.size > 0:
            print(f"Hue (min/mean/max): {np.min(h_h):.0f} / {np.mean(h_h):.1f} / {np.max(h_h):.0f}")
            print(f"Sat (min/mean/max): {np.min(s_h):.0f} / {np.mean(s_h):.1f} / {np.max(s_h):.0f}")
            print(f"Val (min/mean/max): {np.min(v_h):.0f} / {np.mean(v_h):.1f} / {np.max(v_h):.0f}")
        print(f"Shares on bright pixels: Red={p_red:.2f}  Yellow={p_yel:.2f}  Green={p_grn:.2f}")

        # Entscheidung: Minimum-Anteil + Dominanz
        MIN_SHARE = 0.15     # mindestens 15% der hellen Pixel
        DOM_RATIO = 1.25     # dominante Farbe muss 25% mehr als die zweitbeste haben

        shares = {"Rot": p_red, "Gelb": p_yel, "Gruen": p_grn}
        best = max(shares, key=shares.get)
        top = shares[best]
        second = sorted(shares.values(), reverse=True)[1]

        if top < MIN_SHARE or (second > 0 and top < second * DOM_RATIO):
            phase = "Unklar"
        else:
            phase = best

        print(f"Detected phase: {phase}")
        print("========================")
        return phase