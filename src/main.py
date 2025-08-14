from camera.camera_manager import CameraManager
from detection.detector import ObjectDetector

def main():
    try:
        # Setup camera
        camera = CameraManager()
        camera.setup()
        print("Camera ready! Press Ctrl+C to exit")
        
        # Setup detector
        detector = ObjectDetector(camera.imx500, camera.intrinsics)
        
        # Main loop
        while True:
            metadata = camera.picam2.capture_metadata()
            detector.process_frame(metadata)
            
    except KeyboardInterrupt:
        print("\nStopping camera...")
        camera.picam2.stop()
        print("Camera stopped")
    except Exception as e:
        print(f"Error: {e}")
        if camera and camera.picam2:
            camera.picam2.stop()

if __name__ == "__main__":
    main()