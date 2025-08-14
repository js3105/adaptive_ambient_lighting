from picamera2 import Picamera2
from picamera2.devices import IMX500
from picamera2.devices.imx500 import NetworkIntrinsics
import os
from ..config.settings import CameraSettings

class CameraManager:
    def __init__(self):
        self.imx500 = None
        self.intrinsics = None
        self.picam2 = None
        
    def setup(self):
        model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "models", "my_model.rpk")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        
        self.imx500 = IMX500(model_path)
        self._setup_intrinsics()
        self._setup_camera()
        
    def _setup_intrinsics(self):
        self.intrinsics = self.imx500.network_intrinsics or NetworkIntrinsics()
        self.intrinsics.task = "object detection"
        self.intrinsics.bbox_normalization = DetectionSettings.BBOX_NORMALIZATION
        self.intrinsics.bbox_order = DetectionSettings.BBOX_ORDER
        self.intrinsics.inference_rate = DetectionSettings.INFERENCE_RATE
        self.intrinsics.ignore_dash_labels = DetectionSettings.IGNORE_DASH_LABELS
        
    def _setup_camera(self):
        self.picam2 = Picamera2(self.imx500.camera_num)
        config = self.picam2.create_preview_configuration(
            controls={"FrameRate": CameraSettings.FPS},
            buffer_count=CameraSettings.BUFFER_COUNT
        )
        self.picam2.start(config)