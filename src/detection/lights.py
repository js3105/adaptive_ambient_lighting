"""
Traffic Light Detection mit YOLOv8.

Dieses Modul enthält:
- Laden des YOLOv8-Modells
- Detektion nur der Klasse "traffic light" (COCO-ID 9)
"""

from ultralytics import YOLO

# COCO-Klassen-ID für Ampeln
TRAFFIC_LIGHT_CLASS_ID = 9

class TrafficLightDetector:
    def __init__(self, model_path: str = "yolov8n.pt", conf: float = 0.25, iou: float = 0.45):
        """
        Initialisiert den YOLOv8-Detektor.

        :param model_path: Pfad zum YOLO-Modell (COCO trainiert)
        :param conf: Mindestkonfidenz für Detektionen
        :param iou: IOU-Schwelle für Non-Maximum Suppression
        """
        self.model = YOLO(model_path)
        self.conf = conf
        self.iou = iou

    def detect(self, frame):
        """
        Erkennen von Ampeln in einem Frame.

        :param frame: OpenCV-BGR-Bild
        :return: Liste von YOLO-Box-Objekten mit Ampel-Detektionen
        """
        results = self.model.predict(
            frame,
            imgsz=640,
            conf=self.conf,
            iou=self.iou,
            classes=[TRAFFIC_LIGHT_CLASS_ID],
            verbose=False
        )
        # YOLO liefert mehrere Ergebnisse; wir nehmen das erste Bild
        return list(results[0].boxes) if len(results) else []