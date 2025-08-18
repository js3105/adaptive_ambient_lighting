from picamera2 import Picamera2
from picamera2.devices import IMX500
from picamera2.devices.imx500 import NetworkIntrinsics
import os
from config.settings import CameraSettings, DetectionSettings 

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

        # (Optional) Fortschrittsbalken fürs Firmware-Laden:
        self.imx500.show_network_fw_progress_bar()

        self._setup_camera()

        # (Optional) falls im Model gesetzt:
        if self.intrinsics.preserve_aspect_ratio:
            self.imx500.set_auto_aspect_ratio()
        
    def _setup_intrinsics(self):
        self.intrinsics = self.imx500.network_intrinsics or NetworkIntrinsics()
        self.intrinsics.task = "object detection"

        # Deine Settings übernehmen
        self.intrinsics.bbox_normalization = DetectionSettings.BBOX_NORMALIZATION
        self.intrinsics.bbox_order = DetectionSettings.BBOX_ORDER
        self.intrinsics.inference_rate = DetectionSettings.INFERENCE_RATE
        self.intrinsics.ignore_dash_labels = DetectionSettings.IGNORE_DASH_LABELS

        # Labels laden, falls nicht im RPK:
        if self.intrinsics.labels is None:
            labels_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "models", "classes.txt")
            with open(labels_path, "r") as f:
                self.intrinsics.labels = f.read().splitlines()

        # WICHTIG: Defaults anwenden
        self.intrinsics.update_with_defaults()
        
    def _setup_camera(self):
        self.picam2 = Picamera2(self.imx500.camera_num)
        # FPS an inference_rate koppeln (wie im Beispiel):
        fps = self.intrinsics.inference_rate or CameraSettings.FPS
        config = self.picam2.create_preview_configuration(
            controls={"FrameRate": fps},
            buffer_count=12  # stabiler für Realtime
        )
        self.picam2.start(config)