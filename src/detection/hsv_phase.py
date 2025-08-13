"""
HSV-Phasenanalyse für Ampeln (Drittelmethode).

Eingabe:  ROI als OpenCV-BGR-Array (z. B. aus der YOLO-Box ausgeschnitten)
Ausgabe:  "Rot", "Gelb", "Grün" oder "Unklar"
"""

from __future__ import annotations
import numpy as np
import cv2

# Mindesthöhe des ROI, damit die Drittelbildung Sinn ergibt
MIN_ROI_H = 24

# Wie deutlich der Gewinner höher sein muss als die anderen
RATIO_MARGIN = 1.12  # 12%

def detect_phase_by_hsv(roi_bgr: "np.ndarray") -> str:
    """
    Bestimmt die Ampelphase im gegebenen ROI per HSV-Drittelmethode.

    :param roi_bgr: Ausschnitt der Ampel (BGR, uint8)
    :return: "Rot" | "Gelb" | "Grün" | "Unklar"
    """
    if roi_bgr is None or roi_bgr.size == 0:
        return "Unklar"

    h, w = roi_bgr.shape[:2]
    if h < MIN_ROI_H or w < 5:
        return "Unklar"

    # BGR -> HSV
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)

    # Farbmasken
    lower_red1 = np.array([0,   100, 100], dtype=np.uint8)
    upper_red1 = np.array([10,  255, 255], dtype=np.uint8)
    lower_red2 = np.array([160, 100, 100], dtype=np.uint8)
    upper_red2 = np.array([179, 255, 255], dtype=np.uint8)

    lower_yellow = np.array([15, 100, 100], dtype=np.uint8)
    upper_yellow = np.array([35, 255, 255], dtype=np.uint8)

    lower_green = np.array([40, 100, 100], dtype=np.uint8)
    upper_green = np.array([90, 255, 255], dtype=np.uint8)

    mask_red = cv2.bitwise_or(
        cv2.inRange(hsv, lower_red1, upper_red1),
        cv2.inRange(hsv, lower_red2, upper_red2)
    )
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_green  = cv2.inRange(hsv, lower_green, upper_green)

    # Vertikale Drittel (klassische, hochkantige Ampel)
    t1 = h // 3
    t2 = (2 * h) // 3

    red_top    = float(np.mean(mask_red[:t1, :]))
    yellow_mid = float(np.mean(mask_yellow[t1:t2, :]))
    green_bot  = float(np.mean(mask_green[t2:, :]))

    scores = {"Rot": red_top, "Gelb": yellow_mid, "Grün": green_bot}
    winner = max(scores, key=scores.get)
    max_score = scores[winner]

    others = [v for k, v in scores.items() if k != winner]

    # Mindeststärke + eindeutiger Abstand
    if max_score > 10 and all(max_score > o * RATIO_MARGIN for o in others):
        return winner
    return "Unklar"