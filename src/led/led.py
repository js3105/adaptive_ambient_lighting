from typing import Optional
import logging

class LedPhaseSink:
    """
    Schnittstelle für Ampel-LED-Ausgabe.
    Ersetze 'apply_phase' später durch eine echte Implementierung (GPIO, SPI, I2C, etc.).
    """
    def __init__(self):
        self._last_phase: Optional[str] = None

    def GPIO(self, phase: str) -> None:
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


class WS2812LedSink(LedPhaseSink):
    """
    WS2812 LED implementation for traffic light colors on PIN18.
    """
    def __init__(self, led_pin=18, led_count=1, led_freq_hz=800000, led_dma=10, led_brightness=128):
        super().__init__()
        self.led_pin = led_pin
        self.led_count = led_count
        self.strip = None
        self._initialized = False
        
        # Color mapping for traffic light phases
        self.phase_colors = {
            "Rot": (255, 0, 0),      # Red
            "Gelb": (255, 255, 0),   # Yellow 
            "Gruen": (0, 255, 0),    # Green
            "Unklar": (0, 0, 0)      # Off/Black
        }
        
        # Try to initialize WS2812 strip
        try:
            import rpi_ws281x
            self.strip = rpi_ws281x.PixelStrip(
                led_count, led_pin, led_freq_hz, led_dma, False, led_brightness, 0
            )
            self.strip.begin()
            self._initialized = True
            logging.info(f"WS2812 LED initialized on PIN {led_pin}")
        except ImportError:
            logging.warning("rpi_ws281x library not available, LED will not work")
        except Exception as e:
            logging.warning(f"Failed to initialize WS2812 LED: {e}")
    
    def GPIO(self, phase: str) -> None:
        """
        Apply traffic light phase color to WS2812 LED.
        """
        # Debounce: nur bei Änderung reagieren
        if phase == self._last_phase:
            return
        self._last_phase = phase
        
        if not self._initialized:
            print(f"[LED] Phase -> {phase} (WS2812 not available)")
            return
            
        # Get RGB color for phase
        color = self.phase_colors.get(phase, (50, 50, 50))  # Default dim white for unknown
        
        try:
            # Convert RGB to WS2812 color format
            import rpi_ws281x
            # WS2812 typically uses GRB format: Green, Red, Blue
            r, g, b = color
            ws_color = rpi_ws281x.Color(g, r, b)  # GRB format
            
            # Set all LEDs to the same color
            for i in range(self.led_count):
                self.strip.setPixelColor(i, ws_color)
            
            # Update the LED strip
            self.strip.show()
            
            print(f"[LED] Set to {phase}: RGB{color}")
            
        except Exception as e:
            logging.warning(f"Error setting LED color: {e}")
    
    def reset(self):
        """Reset LED to off state."""
        super().reset()
        if self._initialized and self.strip:
            try:
                # Turn off all LEDs
                for i in range(self.led_count):
                    self.strip.setPixelColor(i, 0)
                self.strip.show()
                logging.debug("LED strip reset to off")
            except Exception as e:
                logging.warning(f"Error resetting LED: {e}")
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        self.reset()