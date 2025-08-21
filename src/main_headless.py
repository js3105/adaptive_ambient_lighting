from .camera.camera_manager import CameraManager
from .detection.detector import ObjectDetector
from .led.led import WS2812LedSink
import time

def main():
    camera = None
    led_sink = None
    try:
        # Kamera ohne Preview starten
        camera = CameraManager()
        camera.setup(use_preview=False)  # kein QGl/Preview

        # LEDs
        led_sink = WS2812LedSink(led_pin=18, led_count=14, ambient_color=(1, 187, 242))
        led_sink.set_ambient()

        # Detektor im Headless-Modus (kein Zeichnen)
        detector = ObjectDetector(camera.imx500, camera.intrinsics, camera.picam2,
                                  led_sink=led_sink, headless=True)

        # Headless: post_callback statt pre_callback
        camera.picam2.post_callback = detector.draw_callback

        # Loop: nur Metadaten ziehen und Detections updaten
        while True:
            metadata = camera.picam2.capture_metadata()
            detector.parse_detections(metadata)
            time.sleep(0.001)

    except KeyboardInterrupt:
        print("\nStopping camera...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        try:
            if camera and camera.picam2:
                camera.picam2.stop()
        except Exception:
            pass
        if led_sink:
            led_sink.reset()
        print("Stopped (headless)")

if __name__ == "__main__":
    main()