"""
Zeichen-Helfer für Overlays auf dem Kamerabild.
"""

import cv2
from typing import Tuple

def draw_box(img, x1: int, y1: int, x2: int, y2: int, color: Tuple[int,int,int]=(0, 255, 255), thickness: int=2):
    """Zeichnet eine Rechteck-Box."""
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

def draw_label(img, text: str, x: int, y: int, color: Tuple[int,int,int]=(0, 255, 255), scale: float=0.7, thickness: int=2):
    """Schreibt ein Label an die angegebene Position (links-oben an der Box)."""
    y = max(20, y)  # nicht aus dem Bild fallen
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, lineType=cv2.LINE_AA)

def draw_box_with_label(img, x1: int, y1: int, x2: int, y2: int, label: str,
                        color: Tuple[int,int,int]=(0, 255, 255), thickness: int=2):
    """Kombiniert Box + Label in einem Schritt."""
    draw_box(img, x1, y1, x2, y2, color=color, thickness=thickness)
    draw_label(img, label, x1, max(20, y1 - 6), color=color, scale=0.7, thickness=2)