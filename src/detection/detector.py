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

    # ------------------ Helper (hell + kreisförmig) ------------------
    def _bright_mask(self, hsv, s_min=60, v_min=140):
        """Maske für helle, gesättigte Pixel."""
        return cv2.inRange(hsv, (0, s_min, v_min), (179, 255, 255))

    def _circle_masks(self, gray, min_radius, max_radius):
        """
        Finde Kreise in einem Graubild (Hough) und liefere eine Liste binärer Kreis-Masken.
        Wenn keine Kreise gefunden werden, leere Liste zurückgeben.
        """
        g = cv2.GaussianBlur(gray, (5, 5), 0)
        h, w = gray.shape[:2]
        minDist = max(8, int(min(h, w) * 0.25))
        circles = cv2.HoughCircles(g, cv2.HOUGH_GRADIENT,
                                   dp=1.2, minDist=minDist,
                                   param1=80, param2=18,
                                   minRadius=min_radius, maxRadius=max_radius)
        masks = []
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for x, y, r in circles[0, :]:
                m = np.zeros_like(gray, dtype=np.uint8)
                cv2.circle(m, (int(x), int(y)), int(r), 255, thickness=-1)
                masks.append(m)
        return masks

    def _count_color_in_segment(self, seg_bgr, color):
        """
        Zähle Pixel einer Farbe ('Rot'|'Gelb'|'Gruen') im Segment:
        - nur helle Pixel (bright mask)
        - nur innerhalb erkannter Kreise
        """
        if seg_bgr.size == 0:
            return 0

        hsv = cv2.cvtColor(seg_bgr, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(seg_bgr, cv2.COLOR_BGR2GRAY)

        # nur helle Pixel betrachten
        bright = self._bright_mask(hsv)

        # Kreisradien relativ zur Segmentgröße
        h, w = gray.shape[:2]
        r_min = max(3, int(0.08 * min(h, w)))
        r_max = max(r_min + 2, int(0.5 * min(h, w)))

        circle_ms = self._circle_masks(gray, r_min, r_max)
        if not circle_ms:
            return 0  # nur kreisförmig auswerten -> keine Kreise => keine Zählung

        # Farbmasken
        if color == "Rot":
            m1 = cv2.inRange(hsv, (0,   70, 50), (10, 255, 255))
            m2 = cv2.inRange(hsv, (170, 70, 50), (180,255, 255))
            color_mask = cv2.bitwise_or(m1, m2)
        elif color == "Gelb":
            color_mask = cv2.inRange(hsv, (18, 70, 50), (32, 255, 255))  # enger für „Gelb-Schimmer“-Resistenz
        else:  # Gruen
            color_mask = cv2.inRange(hsv, (40, 70, 50), (90, 255, 255))

        # nur helle + innerhalb Kreis
        total = 0
        for cm in circle_ms:
            inside_circle = cv2.bitwise_and(color_mask, cm)
            inside_bright = cv2.bitwise_and(inside_circle, bright)
            total += cv2.countNonZero(inside_bright)
        return total
    # -----------------------------------------------------------------

    # --- Erkennung mit Dritteln + hell + Kreis ---
    def detect_phase_by_hsv(self, roi_bgr):
        """
        - ROI in drei vertikale Drittel
        - oben nur Rot, Mitte nur Gelb, unten nur Grün
        - nur helle Pixel und nur kreisförmige Bereiche werden gezählt
        """
        if roi_bgr is None or roi_bgr.size == 0:
            return "Unklar"

        h, w = roi_bgr.shape[:2]
        if h < 6:
            return "Unklar"

        h3 = h // 3
        top_bgr    = roi_bgr[0:h3, :, :]           # Rot
        middle_bgr = roi_bgr[h3:2*h3, :, :]        # Gelb
        bottom_bgr = roi_bgr[2*h3:h, :, :]         # Grün

        red_pixels    = self._count_color_in_segment(top_bgr, "Rot")
        yellow_pixels = self._count_color_in_segment(middle_bgr, "Gelb")
        green_pixels  = self._count_color_in_segment(bottom_bgr, "Gruen")

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
    # --- Ende ---

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
                
                    # Für Ampeln: Phasenerkennung durchführen
                    if int(det.category) == self.TRAFFIC_LIGHT_CLASS_ID:
                        roi = m.array[y:y+h, x:x+w]
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
                    box_color = (0, 255, 255) if int(det.category) == self.TRAFFIC_LIGHT_CLASS_ID else (0, 255, 0)
                    cv2.putText(m.array, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
                    cv2.rectangle(m.array, (x, y), (x + w, y + h), box_color, 2)

                except (ValueError, IndexError) as e:
                    print(f"Error processing detection: {e}")
                    continue