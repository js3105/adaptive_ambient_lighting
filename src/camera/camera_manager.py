from picamera2 import Picamera2
from picamera2.devices import IMX500
from picamera2.devices.imx500 import NetworkIntrinsics
from libcamera import controls
import time, os
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

        # optional: Fortschrittsbalken beim Laden der Netzwerk-FW
        self.imx500.show_network_fw_progress_bar()

        self._setup_camera()

        if self.intrinsics.preserve_aspect_ratio:
            self.imx500.set_auto_aspect_ratio()

    def _setup_intrinsics(self):
        self.intrinsics = self.imx500.network_intrinsics or NetworkIntrinsics()
        self.intrinsics.task = "object detection"

        self.intrinsics.bbox_normalization = DetectionSettings.BBOX_NORMALIZATION
        self.intrinsics.bbox_order = DetectionSettings.BBOX_ORDER
        self.intrinsics.inference_rate = DetectionSettings.INFERENCE_RATE
        self.intrinsics.ignore_dash_labels = DetectionSettings.IGNORE_DASH_LABELS

        if self.intrinsics.labels is None:
            labels_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "models", "classes.txt")
            with open(labels_path, "r") as f:
                self.intrinsics.labels = f.read().splitlines()

        self.intrinsics.update_with_defaults()

    def _setup_camera(self):
        self.picam2 = Picamera2(self.imx500.camera_num)
        fps = self.intrinsics.inference_rate or CameraSettings.FPS

        # 1) Preview-Konfiguration (ohne OpenCV-Show – Preview kommt vom Picamera2-Overlay)
        config = self.picam2.create_preview_configuration(
            controls={"FrameRate": fps},
            buffer_count=12
        )
        self.picam2.configure(config)

        # 2) Vorstart-Controls: Spot-Messung + kurze Belichtung bevorzugen (reduziert Überstrahlen)
        self.picam2.set_controls({
            "AeEnable": True,
            "AeMeteringMode": controls.AeMeteringModeEnum.Spot,        # Ampel als helle Teilfläche
            "AeExposureMode": controls.AeExposureModeEnum.Short,       # kurze Shutter-Zeiten
            "AeFlickerMode": controls.AeFlickerModeEnum.Auto,          # stabilisiert unter LED-Licht
        })

        # 3) Starten
        self.picam2.start(show_preview=True)

        # 4) Kurz stabilisieren lassen, dann Anti-Bloom-Preset setzen (manuell abdunkeln)
        time.sleep(0.6)
        self.apply_anti_bloom_preset(exposure_us=100000, gain=1.0, awb=False)

    # === Anti-Bloom Preset (manuelle Abdunkelung, wirkt auch fürs IMX500-Inference) ===
    def apply_anti_bloom_preset(self, *, exposure_us=100000, gain=1.0, awb=False):
        """
        Setzt die Kamera bewusst dunkler:
        - kleine ExposureTime (µs) & niedriger Gain -> weniger Bloom/Glare
        - AWB optional aus (stabilere Farben/Hue)
        Anpassbar je nach Tageslicht/Nacht:
            Tag:    exposure_us ~ 4000–9000, gain 1.0–1.5
            Abend:  exposure_us ~ 8000–14000, gain 1.2–1.8
        """
        self.picam2.set_controls({
            "AeEnable": False,                 # AE aus -> manuell
            "ExposureTime": int(exposure_us),  # µs: kleiner = dunkler
            "AnalogueGain": float(gain),
            "AwbEnable": bool(awb),
        })

    # === Optional: zusätzliches software-basiertes Abdunkeln, bevor du Frames nutzt ===
    def darken_frame_inplace(self, frame, alpha=0.9, beta=0):
        """
        Abdunkeln um ~25% per Gain (wirkt auf Anzeige/HSV, NICHT auf IMX500-Inference).
        Nutze das im draw_callback: frame[:] = convertScaleAbs(...)
        """
        return cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)