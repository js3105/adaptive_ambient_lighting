from typing import Tuple
import cv2
try:
    from picamera2 import Picamera2
    import libcamera
except ImportError as e:
    raise ImportError("Fehlt: python3-picamera2 (sudo apt install -y python3-picamera2)") from e

class PiCamera:
    def __init__(self, size: Tuple[int, int]=(1280, 720), hflip: bool=False, vflip: bool=False):
        self.picam = Picamera2()
        transform = libcamera.Transform(hflip=int(hflip), vflip=int(vflip))
        config = self.picam.create_video_configuration(
            main={"size": size, "format": "RGB888"},
            transform=transform,
            buffer_count=4
        )
        self.picam.configure(config)
        self.picam.start()

    def read(self):
        frame_rgb = self.picam.capture_array()
        return True, cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    def stop(self):
        try:
            self.picam.stop()
        except Exception:
            pass