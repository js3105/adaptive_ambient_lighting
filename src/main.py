import sys
import os
import cv2
from camera.camera_manager import CameraManager
from detection.detector import ObjectDetector

def main():
    camera = None
    try:
        camera = CameraManager()
        camera.setup()
        print("Camera ready! Press 'q' to exit")

        detector = ObjectDetector(camera.imx500, camera.intrinsics, camera.picam2)

        cv2.namedWindow("Object Detection", cv2.WINDOW_NORMAL)

        while True:
            # Besser: erst Metadata, dann Frame – oder gleich Request verwenden.
            metadata = camera.picam2.capture_metadata()
            frame = camera.picam2.capture_array()

            out = detector.process_frame(frame, metadata)
            cv2.imshow("Object Detection", out)

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