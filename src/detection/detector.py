import cv2
import numpy as np
from picamera2 import MappedArray
from picamera2.devices import IMX500
from picamera2.devices.imx500 import postprocess_nanodet_detection
from datetime import datetime

# Konstanten für die Ampelphasenerkennung
MIN_ROI_H = 24
RATIO_MARGIN = 1.12

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
        self.user = "js3105"  # Current user's login

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

    def detect_phase_by_hsv(self, roi_bgr):
        """Erkennung der Ampelphase anhand von HSV-Farbraum-Analyse."""
        current_time = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
        
        h, w = roi_bgr.shape[:2]
        if h < MIN_ROI_H or w < 5:
            return "Unklar"

        # In HSV konvertieren
        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)

        # Print raw color data
        print("\n=== Raw Color Analysis ===")
        print(f"Timestamp (UTC): {current_time}")
        print(f"User: {self.user}")
        print(f"ROI Size: {w}x{h}")
        print("\nRaw HSV Values (min, mean, max):")
        print(f"Hue: {np.min(hsv[:,:,0]):.0f}, {np.mean(hsv[:,:,0]):.0f}, {np.max(hsv[:,:,0]):.0f}")
        print(f"Saturation: {np.min(hsv[:,:,1]):.0f}, {np.mean(hsv[:,:,1]):.0f}, {np.max(hsv[:,:,1]):.0f}")
        print(f"Value: {np.min(hsv[:,:,2]):.0f}, {np.mean(hsv[:,:,2]):.0f}, {np.max(hsv[:,:,2]):.0f}")

        # Farbmasken definieren (HSV-Werte aus YOLOv8-Beispiel)
        mask_red1 = cv2.inRange(hsv, (0, 70, 50), (10, 255, 255))
        mask_red2 = cv2.inRange(hsv, (170, 70, 50), (180, 255, 255))
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        mask_yellow = cv2.inRange(hsv, (15, 70, 50), (35, 255, 255))
        mask_green = cv2.inRange(hsv, (40, 70, 50), (90, 255, 255))

        # Pixelanzahl jeder Farbe zählen
        red_pixels = cv2.countNonZero(mask_red)
        yellow_pixels = cv2.countNonZero(mask_yellow)
        green_pixels = cv2.countNonZero(mask_green)

        # Calculate percentages
        total_pixels = w * h
        red_percentage = (red_pixels / total_pixels) * 100
        yellow_percentage = (yellow_pixels / total_pixels) * 100
        green_percentage = (green_pixels / total_pixels) * 100

        print("\nPixel Counts and Percentages:")
        print(f"Red: {red_pixels} pixels ({red_percentage:.1f}%)")
        print(f"Yellow: {yellow_pixels} pixels ({yellow_percentage:.1f}%)")
        print(f"Green: {green_pixels} pixels ({green_percentage:.1f}%)")

        # Größte Farbe bestimmen → Ampelphase
        max_pixels = max(red_pixels, yellow_pixels, green_pixels)
        
        # Add a minimum threshold to avoid false positives
        MIN_PIXELS = 50  # Adjust this threshold as needed

        if max_pixels < MIN_PIXELS:
            phase = "Unklar"
        elif max_pixels == red_pixels:
            phase = "Rot"
        elif max_pixels == yellow_pixels:
            phase = "Gelb"
        elif max_pixels == green_pixels:
            phase = "Gruen"
        else:
            phase = "Unklar"

        print(f"\nDetected phase: {phase}")
        print("========================")

        return phase

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
                    box_color = (0, 255, 255) if int(det.category) == self.TRAFFIC_LIGHT_CLASS_ID else (0, 255, 0, 0)
                    cv2.putText(m.array, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
                    cv2.rectangle(m.array, (x, y), (x + w, y + h), box_color, 2)

                except (ValueError, IndexError) as e:
                    print(f"Error processing detection: {e}")
                    continue