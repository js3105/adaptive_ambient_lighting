from typing import Optional
import logging


class LedPhaseSink:
    def __init__(self):
        self._last_phase: Optional[str] = None

    def apply_phase(self, phase: str) -> None:
        if phase == self._last_phase:
            return
        self._last_phase = phase
        print(f"[LED] Phase -> {phase}")

    def reset(self):
        self._last_phase = None


class WS2812LedSink(LedPhaseSink):
    """
    WS2812 LED implementation for traffic light colors.
    """
    def __init__(self, led_pin=18, led_count=14, led_freq_hz=800000, led_dma=10, led_brightness=128):
        super().__init__()
        self.led_pin = led_pin
        self.led_count = led_count
        self.strip = None
        self._initialized = False

        # Traffic light phase colors
        self.phase_colors = {
            "Rot": (255, 0, 0),
            "Gelb": (255, 255, 0),
            "Gruen": (0, 255, 0),
            "Unklar": (0, 0, 0)
        }

        try:
            import rpi_ws281x
            self.strip = rpi_ws281x.PixelStrip(
                led_count, led_pin, led_freq_hz, led_dma, False, led_brightness, 0
            )
            self.strip.begin()
            self._initialized = True
            logging.info(f"WS2812 LED initialized on PIN {led_pin} with {led_count} LEDs")
        except ImportError:
            logging.warning("rpi_ws281x library not available, LED will not work")
        except Exception as e:
            logging.warning(f"Failed to initialize WS2812 LED: {e}")

    def apply_phase(self, phase: str) -> None:
        if phase == self._last_phase:
            return
        self._last_phase = phase

        if not self._initialized:
            print(f"[LED] Phase -> {phase} (WS2812 not available)")
            return

        color = self.phase_colors.get(phase, (50, 50, 50))  # fallback

        try:
            import rpi_ws281x
            r, g, b = color
            # <<< FIXED: nutze RGB statt GRB, um Rot/Grün nicht zu vertauschen >>>
            ws_color = rpi_ws281x.Color(r, g, b)

            # <<< FIXED: setze ALLE LEDs >>>
            for i in range(self.led_count):
                self.strip.setPixelColor(i, ws_color)

            self.strip.show()
            print(f"[LED] Set to {phase}: RGB{color}")

        except Exception as e:
            logging.warning(f"Error setting LED color: {e}")

    def reset(self):
        super().reset()
        if self._initialized and self.strip:
            try:
                for i in range(self.led_count):
                    self.strip.setPixelColor(i, 0)
                self.strip.show()
                logging.debug("LED strip reset to off")
            except Exception as e:
                logging.warning(f"Error resetting LED: {e}")

    def __del__(self):
        self.reset()