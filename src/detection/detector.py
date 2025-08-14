class ObjectDetector:
    def __init__(self, imx500, intrinsics):
        self.imx500 = imx500
        self.intrinsics = intrinsics
        
    def process_frame(self, metadata, confidence_threshold=CameraSettings.CONFIDENCE_THRESHOLD):
        outputs = self.imx500.get_outputs(metadata, add_batch=True)
        if outputs is None:
            return
        
        boxes, scores, classes = outputs[0][0], outputs[1][0], outputs[2][0]
        
        for score, class_id in zip(scores, classes):
            if score > confidence_threshold:
                print(f"Detected object class {int(class_id)} (confidence: {score:.2f})")