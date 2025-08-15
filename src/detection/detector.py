import cv2
from config.settings import CameraSettings

class ObjectDetector:
    def __init__(self, imx500, intrinsics):
        self.imx500 = imx500
        self.intrinsics = intrinsics
    
    def draw_detections(self, frame, boxes, scores, classes, confidence_threshold=CameraSettings.CONFIDENCE_THRESHOLD):
        for box, score, class_id in zip(boxes, scores, classes):
            if score > confidence_threshold:
                # Convert normalized coordinates to pixel coordinates
                h, w = frame.shape[:2]
                x1, y1, x2, y2 = box
                x1, x2 = int(x1 * w), int(x2 * w)
                y1, y2 = int(y1 * h), int(y2 * h)
                
                # Draw rectangle and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"Class {int(class_id)}: {score:.2f}"
                cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return frame
        
    def process_frame(self, frame, metadata, confidence_threshold=CameraSettings.CONFIDENCE_THRESHOLD):
        outputs = self.imx500.get_outputs(metadata, add_batch=True)
        if outputs is None:
            return frame
        
        boxes, scores, classes = outputs[0][0], outputs[1][0], outputs[2][0]
        
        # Print detections to terminal
        for score, class_id in zip(scores, classes):
            if score > confidence_threshold:
                print(f"Detected object class {int(class_id)} (confidence: {score:.2f})")
        
        # Draw detections on frame
        frame = self.draw_detections(frame, boxes, scores, classes, confidence_threshold)
        return frame