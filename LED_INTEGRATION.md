# WS2812 LED Integration

This document explains the WS2812 LED integration for displaying traffic light colors.

## Overview

The system now includes WS2812 LED support that automatically lights the LED in the same color as the detected traffic light phase:

- **Red LED** when traffic light shows "Rot" (Red)
- **Yellow LED** when traffic light shows "Gelb" (Yellow) 
- **Green LED** when traffic light shows "Gruen" (Green)
- **LED Off** when traffic light phase is "Unklar" (Unknown/Unclear)

## Hardware Setup

1. Connect WS2812 LED strip to GPIO PIN 18 on Raspberry Pi
2. Ensure proper power supply for the LED strip
3. Common wiring:
   - VCC → 5V power supply
   - GND → Ground (shared with Pi)
   - DIN → GPIO 18 (PIN 18)

## Software Components

### WS2812LedSink Class

Located in `src/io/led.py`, this class extends the existing `LedPhaseSink` and provides:

- **Hardware abstraction**: Works with or without actual WS2812 hardware
- **Color mapping**: Automatic conversion from traffic light phases to RGB colors
- **Debouncing**: Only updates LED when phase actually changes
- **Error handling**: Graceful fallback when hardware isn't available

### Configuration Options

The `WS2812LedSink` can be configured with these parameters:

```python
WS2812LedSink(
    led_pin=18,          # GPIO pin number (default: 18)
    led_count=1,         # Number of LEDs in strip (default: 1)
    led_freq_hz=800000,  # LED signal frequency (default: 800kHz)
    led_dma=10,          # DMA channel (default: 10)
    led_brightness=128   # Brightness 0-255 (default: 128)
)
```

## Usage

The LED integration is automatically handled by the main application. When traffic lights are detected and their phase is determined, the LED will automatically update to match.

### Manual Testing

You can test the LED functionality independently:

```python
from src.io.led import WS2812LedSink

# Initialize LED
led = WS2812LedSink()

# Test different phases
led.apply_phase("Rot")    # Red
led.apply_phase("Gelb")   # Yellow
led.apply_phase("Gruen")  # Green
led.apply_phase("Unklar") # Off

# Reset LED
led.reset()
```

## Integration with Traffic Light Detection

The LED integration is seamlessly connected to the traffic light detection system:

1. **Detection**: Camera detects traffic lights in the scene
2. **Analysis**: Traffic light phase is analyzed from the image ROI
3. **LED Update**: LED automatically updates to match the detected phase
4. **Debouncing**: LED only changes when phase actually changes

## Troubleshooting

### LED Not Working
- Verify WS2812 hardware is properly connected to PIN 18
- Check power supply to LED strip
- Ensure rpi_ws281x library is installed: `pip install rpi_ws281x`
- Run with sudo if permission errors occur

### Running Without Hardware
- The system gracefully falls back to console output when hardware isn't available
- Useful for development and testing on non-Pi systems

### Performance Considerations
- LED updates are debounced to prevent unnecessary SPI communication
- Minimal impact on main detection loop performance
- Cleanup handled automatically on application exit

## Dependencies

- `rpi_ws281x`: Library for controlling WS2812 LEDs on Raspberry Pi
- Hardware: WS2812/WS2812B LED strip or individual LEDs
- Raspberry Pi with GPIO access