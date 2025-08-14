class CameraSettings:
    FPS = 30
    BUFFER_COUNT = 4
    CONFIDENCE_THRESHOLD = 0.55

class DetectionSettings:
    BBOX_NORMALIZATION = True
    BBOX_ORDER = "xy"
    IGNORE_DASH_LABELS = True
    INFERENCE_RATE = 30