from typing import Optional

class LedPhaseSink:
    """
    Schnittstelle für Ampel-LED-Ausgabe.
    Ersetze 'apply_phase' später durch eine echte Implementierung (GPIO, SPI, I2C, etc.).
    """
    def __init__(self):
        self._last_phase: Optional[str] = None

    def apply_phase(self, phase: str) -> None:
        """
        phase: "Rot" | "Gelb" | "Gruen" | "Unklar"
        Default: no-op (kann durch Subklasse/DI ersetzt werden).
        """
        # Debounce: nur bei Änderung reagieren
        if phase == self._last_phase:
            return
        self._last_phase = phase
        # Placeholder: hier später echte LED-Ansteuerung einhängen
        # z.B. gpiozero, rpi_ws281x, etc.
        print(f"[LED] Phase -> {phase}")

    def reset(self):
        self._last_phase = None