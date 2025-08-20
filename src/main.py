from camera.camera_manager import CameraManager
from detection.detector import ObjectDetector
from io.led import WS2812LedSink
import time

def main():
    camera = None
    led_sink = None
    try:
        camera = CameraManager()
        camera.setup()
        print("Preview running. Press Ctrl+C to exit.")

        # Initialize WS2812 LED on PIN18
        led_sink = WS2812LedSink(led_pin=18, led_count=1)
        
        detector = ObjectDetector(camera.imx500, camera.intrinsics, camera.picam2, led_sink=led_sink)

        # Zeichnen direkt im Preview:
        camera.picam2.pre_callback = detector.draw_callback

        # Main-Loop: nur noch Metadata holen und Detections aktualisieren
        while True:
            metadata = camera.picam2.capture_metadata()
            detector.parse_detections(metadata)
            # Kleine Pause, um CPU zu schonen (optional)
            time.sleep(0.001)

    except KeyboardInterrupt:
        print("\nStopping camera...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if camera and camera.picam2:
            camera.picam2.stop()
        if led_sink:
            led_sink.reset()
        print("Camera stopped")

if __name__ == "__main__":
    main()