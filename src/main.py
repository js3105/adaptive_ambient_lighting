import sys
import os
import cv2

# Füge das src Verzeichnis zum Python-Pfad hinzu
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.camera.camera_manager import CameraManager
from src.detection.detector import ObjectDetector

def main():
    try:
        # Setup camera
        camera = CameraManager()
        camera.setup()
        print("Camera ready! Press Ctrl+C to exit")
        
        # Setup detector
        detector = ObjectDetector(camera.imx500, camera.intrinsics)
        
        # Create window for display
        cv2.namedWindow("Object Detection", cv2.WINDOW_NORMAL)
        
        # Main loop
        while True:
            # Capture frame and metadata
            frame = camera.picam2.capture_array()
            metadata = camera.picam2.capture_metadata()
            
            # Process frame and draw detections
            frame_with_detections = detector.process_frame(frame, metadata)
            
            # Show frame
            cv2.imshow("Object Detection", frame_with_detections)
            
            # Break loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
    except KeyboardInterrupt:
        print("\nStopping camera...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if camera and camera.picam2:
            camera.picam2.stop()
        cv2.destroyAllWindows()
        print("Camera stopped")

if __name__ == "__main__":
    main()