import cv2
import numpy as np
from picamera2 import MappedArray
from picamera2.devices import IMX500
from picamera2.devices.imx500 import postprocess_nanodet_detection

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
        # Constants for detection
        RATIO_MARGIN = 1.05
        MIN_SCORE = 6

        h, w = roi_bgr.shape[:2]
        if h < MIN_ROI_H or w < 5:
            return "Unklar"

        # In HSV konvertieren
        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)

        # Print average HSV values for debugging
        print("\n=== Raw HSV Values ===")
        print(f"Average H: {np.mean(hsv[:,:,0]):.2f}")
        print(f"Average S: {np.mean(hsv[:,:,1]):.2f}")
        print(f"Average V: {np.mean(hsv[:,:,2]):.2f}")

        # Modified HSV ranges with wider tolerances
        # Adjusted ranges for better detection
        lower_red1 = np.array([0, 50, 50])      # Reduced S and V thresholds
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 50, 50])    # Reduced S and V thresholds
        upper_red2 = np.array([180, 255, 255])  # Increased to 180 for full range

        lower_yellow = np.array([20, 50, 50])    # Adjusted yellow range and reduced thresholds
        upper_yellow = np.array([40, 255, 255])

        lower_green = np.array([40, 50, 50])     # Kept same but reduced S and V thresholds
        upper_green = np.array([90, 255, 255])

        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        mask_green = cv2.inRange(hsv, lower_green, upper_green)

        # Add Gaussian blur to reduce noise
        mask_red = cv2.GaussianBlur(mask_red, (5, 5), 0)
        mask_yellow = cv2.GaussianBlur(mask_yellow, (5, 5), 0)
        mask_green = cv2.GaussianBlur(mask_green, (5, 5), 0)

        # Calculate averages for entire masks
        red_value = np.average(mask_red)
        yellow_value = np.average(mask_yellow)
        green_value = np.average(mask_green)

        # Debug output
        print("\n=== HSV Analysis Debug ===")
        print(f"ROI Size: {w}x{h}")
        print(f"Red value: {red_value:.2f}")
        print(f"  Red range 1: H={lower_red1[0]}-{upper_red1[0]}, S={lower_red1[1]}-{upper_red1[1]}, V={lower_red1[2]}-{upper_red1[2]}")
        print(f"  Red range 2: H={lower_red2[0]}-{upper_red2[0]}, S={lower_red2[1]}-{upper_red2[1]}, V={lower_red2[2]}-{upper_red2[2]}")
        print(f"Yellow value: {yellow_value:.2f}")
        print(f"  Yellow range: H={lower_yellow[0]}-{upper_yellow[0]}, S={lower_yellow[1]}-{upper_yellow[1]}, V={lower_yellow[2]}-{upper_yellow[2]}")
        print(f"Green value: {green_value:.2f}")
        print(f"  Green range: H={lower_green[0]}-{upper_green[0]}, S={lower_green[1]}-{upper_green[1]}, V={lower_green[2]}-{upper_green[2]}")

        # Adjust scoring with lower threshold
        scores = {"Rot": red_value, "Gelb": yellow_value, "Gruen": green_value}
        winner = max(scores, key=scores.get)
        max_score = scores[winner]

        # Print decision information
        print(f"Winner: {winner} (score: {max_score:.2f})")
        print(f"Ratio margin check: {all(max_score > o * RATIO_MARGIN for o in [v for k, v in scores.items() if k != winner])}")
        print(f"Min score check: {max_score > MIN_SCORE}")
        print("========================\n")

        others = [v for k, v in scores.items() if k != winner]
        if all(max_score > o * RATIO_MARGIN for o in others) and max_score > MIN_SCORE:
            return winner
        return "Unklar"

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