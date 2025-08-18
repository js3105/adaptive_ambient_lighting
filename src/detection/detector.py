import cv2
from config.settings import CameraSettings

class Detection:
    def __init__(self, coords, category, conf, metadata, imx500, picam2):
        """Create a Detection object with bounding box, category and confidence."""
        self.category = category
        self.conf = conf
        self.box = imx500.convert_inference_coords(coords, metadata, picam2)

class ObjectDetector:
    def __init__(self, imx500, intrinsics):
        self.imx500 = imx500
        self.intrinsics = intrinsics
        self.last_detections = []
    
    def parse_detections(self, metadata):
        """Parse output tensor into detections"""
        np_outputs = self.imx500.get_outputs(metadata, add_batch=True)
        if np_outputs is None:
            return self.last_detections
            
        boxes, scores, classes = np_outputs[0][0], np_outputs[1][0], np_outputs[2][0]
        
        # Create Detection objects for confident detections
        self.last_detections = [
            Detection(box, category, score, metadata, self.imx500, self.picam2)
            for box, score, category in zip(boxes, scores, classes)
            if score > CameraSettings.CONFIDENCE_THRESHOLD
        ]
        return self.last_detections

    def draw_detections(self, frame):
        """Draw detections with semi-transparent labels"""
        for detection in self.last_detections:
            x, y, w, h = detection.box
            label = f"Class {int(detection.category)} ({detection.conf:.2f})"

            # Calculate text size and position
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            text_x = x + 5
            text_y = y + 15

            # Create overlay for semi-transparent background
            overlay = frame.copy()
            cv2.rectangle(
                overlay,
                (text_x, text_y - text_height),
                (text_x + text_width, text_y + baseline),
                (255, 255, 255),
                cv2.FILLED
            )

            # Add semi-transparent background
            alpha = 0.30
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

            # Draw text and detection box
            cv2.putText(
                frame, label, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1
            )
            cv2.rectangle(
                frame, (x, y), (x + w, y + h),
                (0, 255, 0), thickness=2
            )

        return frame

    def process_frame(self, frame, metadata):
        # Parse new detections
        self.parse_detections(metadata)
        
        # Draw detections on frame
        frame = self.draw_detections(frame)
        
        # Print detections to terminal
        for detection in self.last_detections:
            print(f"Detected object class {int(detection.category)} (confidence: {detection.conf:.2f})")
        
        return frame