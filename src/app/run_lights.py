"""
Minimaler Runner: Kamera → YOLO-Ampelerkennung → HSV-Phase
Ausgabe: wahlweise Live-Fenster ODER reine Konsolenausgabe.
"""

from __future__ import annotations
import cv2
from typing import List

from src.camera.picam import PiCamera
from src.detection.lights import TrafficLightDetector
from src.detection.hsv_phase import detect_phase_by_hsv
from src.utils.draw import draw_box_with_label

MODE = "window"   # "window" für Live-Fenster, "console" für Konsolenausgabe


def _extract_roi(frame, box):
    """Sicher ROI aus YOLO-Box extrahieren (x1,y1,x2,y2 werden geclippt)."""
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    H, W = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W - 1, x2), min(H - 1, y2)
    if x2 <= x1 or y2 <= y1:
        return None, (x1, y1, x2, y2)
    return frame[y1:y2, x1:x2], (x1, y1, x2, y2)


def run_windowed():
    cam = PiCamera(size=(1280, 720))
    det = TrafficLightDetector("yolov8n.pt")

    print("Starte Ampelerkennung (Fenster). Beenden mit Taste 'q'.")
    try:
        while True:
            ok, frame = cam.read()
            if not ok:
                continue

            boxes = det.detect(frame)  # Liste von YOLO-Boxen (nur traffic light)
            for b in boxes:
                roi, (x1, y1, x2, y2) = _extract_roi(frame, b)
                if roi is None:
                    continue
                phase = detect_phase_by_hsv(roi)
                draw_box_with_label(frame, x1, y1, x2, y2, f"Ampel: {phase}")

            cv2.imshow("Ampelerkennung (minimal)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cv2.destroyAllWindows()
        cam.stop()


def run_console():
    cam = PiCamera(size=(1280, 720))
    det = TrafficLightDetector("yolov8n.pt")

    print("Starte Ampelerkennung (Konsole). Stop mit Ctrl+C.")
    try:
        while True:
            ok, frame = cam.read()
            if not ok:
                continue

            boxes = det.detect(frame)
            phases: List[str] = []
            for b in boxes:
                roi, _ = _extract_roi(frame, b)
                if roi is None:
                    continue
                phases.append(detect_phase_by_hsv(roi))

            print({"detections": len(boxes), "phases": phases})
    except KeyboardInterrupt:
        pass
    finally:
        cam.stop()


if __name__ == "__main__":
    if MODE == "window":
        run_windowed()
    else:
        run_console()
    # Alternativ: kommentiere die gewünschte Variante ein/aus:
    # run_windowed()
    # run_console()