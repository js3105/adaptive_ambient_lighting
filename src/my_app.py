from picamera2 import Picamera2
from picamera2.devices import IMX500
from picamera2.devices.imx500 import NetworkIntrinsics
import os

def setup_camera():
    # Get directory of current script and construct path to model
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "my_model.rpk")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    
    # Initialize IMX500 with local model
    imx500 = IMX500(model_path)
    
    # Setup network intrinsics with hardcoded values
    intrinsics = imx500.network_intrinsics
    if not intrinsics:
        intrinsics = NetworkIntrinsics()
    
    # Configure settings that were previously command line arguments
    intrinsics.task = "object detection"
    intrinsics.bbox_normalization = True
    intrinsics.bbox_order = "xy"
    intrinsics.inference_rate = 30  
    intrinsics.ignore_dash_labels = True
    
    return imx500, intrinsics

def process_detections(metadata, imx500, intrinsics, confidence_threshold=0.6):
    # Get model outputs
    outputs = imx500.get_outputs(metadata, add_batch=True)
    if outputs is None:
        return
    
    # Extract detection results
    boxes, scores, classes = outputs[0][0], outputs[1][0], outputs[2][0]
    
    # Print detections that exceed confidence threshold
    for score, class_id in zip(scores, classes):
        if score > confidence_threshold:
            print(f"Detected object class {int(class_id)} (confidence: {score:.2f})")

def main():
    # Setup camera and model
    imx500, intrinsics = setup_camera()
    
    # Initialize camera with specific frame rate
    picam2 = Picamera2(imx500.camera_num)
    config = picam2.create_preview_configuration(
        controls={"FrameRate": 30},  # Hardcoded FPS
        buffer_count=4
    )
    
    print("Starting camera...")
    picam2.start(config)
    print("Camera ready! Press Ctrl+C to exit")
    
    try:
        while True:
            # Capture and process frame
            metadata = picam2.capture_metadata()
            process_detections(metadata, imx500, intrinsics, confidence_threshold=0.55)
            
    except KeyboardInterrupt:
        print("\nStopping camera...")
        picam2.stop()
        print("Camera stopped")

if __name__ == "__main__":
    main()