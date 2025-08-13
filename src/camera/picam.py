"""
Pi-Kamera Wrapper (Platzhalter).

Ziel:
- Einheitliches Interface für die Kamera.
- Später auf dem Raspberry Pi mit Picamera2 implementieren.
- Jetzt noch ohne echte Funktionalität, damit andere Module bereits importieren können.

TODO (später am Pi):
- Picamera2 importieren (from picamera2 import Picamera2, import libcamera)
- In __init__ konfigurieren (size, hflip, vflip, buffer_count)
- read() → (ok, frame_bgr) zurückgeben
- stop() → Kamera stoppen/aufräumen
"""

from typing import Tuple
import numpy as np

DEFAULT_SIZE: Tuple[int, int] = (1280, 720)

class PiCamera:
    def __init__(self, size: Tuple[int, int] = DEFAULT_SIZE, hflip: bool = False, vflip: bool = False) -> None:
        self.size = size
        self.hflip = hflip
        self.vflip = vflip
        # Platzhalter: hier später Picamera2 konfigurieren

    def read(self) -> tuple[bool, "np.ndarray"]:
        """
        Liefert (ok, frame_bgr).
        Placeholder: wirf eine klare Exception, bis die Pi-Implementierung steht.
        """
        raise NotImplementedError(
            "PiCamera.read() ist noch nicht implementiert. "
            "Wird später mit Picamera2 am Raspberry Pi ergänzt."
        )

    def stop(self) -> None:
        """Aufräumen/Stop – aktuell nichts zu tun."""
        return