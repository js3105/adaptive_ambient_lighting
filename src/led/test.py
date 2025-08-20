import time
from rpi_ws281x import PixelStrip, Color

# LED configuration:
LED_COUNT      = 14      # Anzahl der LEDs im Band
LED_PIN        = 18      # GPIO Pin (hier 18)
LED_FREQ_HZ    = 800000  # LED signal frequency in hertz (usually 800kHz)
LED_DMA        = 10      # DMA channel to use for generating signal (try 10)
LED_BRIGHTNESS = 255     # Max brightness (0-255)
LED_INVERT     = False   # True to invert the signal (False für normale Ansteuerung)
LED_CHANNEL    = 0       # Set to 1 for GPIOs 13, 19, 41, 45 or 53

def colorWipe(strip, color, wait_ms=50):
    """Füllt das Band mit einer Farbe."""
    for i in range(strip.numPixels()):
        strip.setPixelColor(i, color)
        strip.show()
        time.sleep(wait_ms/1000.0)

if __name__ == '__main__':
    # Initialisierung:
    strip = PixelStrip(LED_COUNT, LED_PIN, LED_FREQ_HZ, LED_DMA, LED_INVERT, LED_BRIGHTNESS, LED_CHANNEL)
    strip.begin()

    print('Starte LED-Demo...')
    try:
        while True:
            print('Rot...')
            colorWipe(strip, Color(255, 0, 0))  # Rot
            time.sleep(1)
            print('Grün...')
            colorWipe(strip, Color(0, 255, 0))  # Grün
            time.sleep(1)
            print('Blau...')
            colorWipe(strip, Color(0, 0, 255))  # Blau
            time.sleep(1)
    except KeyboardInterrupt:
        colorWipe(strip, Color(0,0,0), 10)     # Alles ausschalten
        print('\nLED-Band ausgeschaltet.')
