import cv2
import numpy as np
from picamera2.devices.imx500 import postprocess_nanodet_detection
from picamera2.devices import IMX500  # nur für Typ-Hinweis
from config.settings import CameraSettings

class Detection:
    def __init__(self, coords, category, conf, metadata, imx500: IMX500, picam2):
        self.category = category
        self.conf = conf
        # Erwartet Inference-Koordinaten (modellabhängig) -> konvertiert zu ISP-Output
        self.box = imx500.convert_inference_coords(coords, metadata, picam2)

class ObjectDetector:
    def __init__(self, imx500: IMX500, intrinsics, picam2):
        self.imx500 = imx500
        self.intrinsics = intrinsics
        self.picam2 = picam2
        self.last_detections = []

    def _labels(self):
        labels = self.intrinsics.labels or []
        if self.intrinsics.ignore_dash_labels:
            labels = [l for l in labels if l and l != "-"]
        return labels

    def parse_detections(self, metadata):
        np_outputs = self.imx500.get_outputs(metadata, add_batch=True)
        if np_outputs is None:
            return self.last_detections

        input_w, input_h = self.imx500.get_input_size()
        bbox_normalization = self.intrinsics.bbox_normalization
        bbox_order = self.intrinsics.bbox_order
        threshold = CameraSettings.CONFIDENCE_THRESHOLD
        iou = 0.65
        max_dets = 10

        # Nanodet-Pfad wie im Beispiel
        if getattr(self.intrinsics, "postprocess", "") == "nanodet":
            boxes, scores, classes = postprocess_nanodet_detection(
                outputs=np_outputs[0], conf=threshold, iou_thres=iou, max_out_dets=max_dets
            )[0]
            from picamera2.devices.imx500.postprocess import scale_boxes
            boxes = scale_boxes(boxes, 1, 1, input_h, input_w, False, False)
        else:
            boxes, scores, classes = np_outputs[0][0], np_outputs[1][0], np_outputs[2][0]
            if bbox_normalization:
                # Beispiel normiert auf input_h
                boxes = boxes / input_h
            if bbox_order == "xy":
                # in (y0, x0, y1, x1) bringen
                boxes = boxes[:, [1, 0, 3, 2]]
            # in y0,x0,y1,x1 Spalten splitten und wieder zippen
            boxes = np.array_split(boxes, 4, axis=1)
            boxes = zip(*boxes)

        self.last_detections = [
            Detection(box, category, score, metadata, self.imx500, self.picam2)
            for box, score, category in zip(boxes, scores, classes)
            if float(score) > threshold
        ]
        return self.last_detections

    def draw_detections(self, frame):
        labels = self._labels()
        for det in self.last_detections:
            # convert_inference_coords liefert (x, y, w, h)
            x, y, w, h = map(int, det.box)
            name = labels[int(det.category)] if 0 <= int(det.category) < len(labels) else f"Class {int(det.category)}"
            label = f"{name} ({det.conf:.2f})"

            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            tx, ty = x + 5, y + 15

            overlay = frame.copy()
            cv2.rectangle(overlay, (tx, ty - th), (tx + tw, ty + baseline), (255, 255, 255), cv2.FILLED)
            cv2.addWeighted(overlay, 0.30, frame, 0.70, 0, frame)

            cv2.putText(frame, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return frame

    def process_frame(self, frame, metadata):
        self.parse_detections(metadata)
        frame = self.draw_detections(frame)
        for det in self.last_detections:
            print(f"Detected {int(det.category)} (confidence: {det.conf:.2f})")
        return frame