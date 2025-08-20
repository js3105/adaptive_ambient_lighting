import cv2
import numpy as np
import logging
from picamera2 import MappedArray
from picamera2.devices import IMX500

from .postprocess import parse_detections
from .traffic_light import TrafficLightAnalyzer, TrafficLightPhase
from .common import pick_label
from .overlay import draw_label, draw_box

# Klassen-IDs (euer Datensatz)
CLASS_TL = 0  # ampel

class ObjectDetector:
    DETECTION_THRESHOLD = 0.55
    DETECTION_IOU = 0.65
    DETECTION_MAX_DETS = 10

    def __init__(self, imx500: IMX500, intrinsics, picam2, led_sink=None):
        self.imx500 = imx500
        self.intrinsics = intrinsics
        self.picam2 = picam2
        self.last_detections = []
        self.labels = [] if intrinsics.labels is None else intrinsics.labels

        # Farbanalyse
        self._tl = TrafficLightAnalyzer(
            gamma=1.6, clahe_clip=2.0, clahe_grid=8,
            s_min=60, v_min=60, use_r_dom=True, r_margin=20
        )

        # LED-Schnittstelle (optional)
        self._led_sink = led_sink  # erwartet Objekt mit .apply_phase(str)

    def set_led_sink(self, led_sink):
        self._led_sink = led_sink

    def _labels(self):
        return self.labels

    def parse_detections(self, metadata):
        self.last_detections = parse_detections(
            self.imx500, self.intrinsics, self.picam2, metadata, self.last_detections,
            threshold=self.DETECTION_THRESHOLD, iou=self.DETECTION_IOU, max_dets=self.DETECTION_MAX_DETS
        )
        return self.last_detections

    def draw_callback(self, request, stream="main"):
        if not self.last_detections:
            return

        labels = self._labels()
        with MappedArray(request, stream) as m:
            draw_surface = m.array
            h_max, w_max = draw_surface.shape[:2]

            # Farbansicht nur einmal berechnen
            if draw_surface.ndim == 3 and draw_surface.shape[2] == 4:
                proc_rgb_full = cv2.cvtColor(draw_surface, cv2.COLOR_BGRA2RGB)
            elif draw_surface.ndim == 3 and draw_surface.shape[2] == 3:
                proc_rgb_full = draw_surface[:, :, ::-1]
            else:
                return

            overlay = draw_surface.copy()

            # Wir merken uns die "dominante" Phase aus diesem Frame (erste beste Ampel)
            frame_phase_for_led = None

            for det in self.last_detections:
                try:
                    x, y, w, h = map(int, det.box)
                    x = max(0, min(x, w_max - 1)); y = max(0, min(y, h_max - 1))
                    w = min(w, w_max - x); h = min(h, h_max - y)
                    if w <= 0 or h <= 0:
                        continue

                    name = pick_label(labels, det.category)

                    # Nur für Ampeln die ROI-Analyse durchführen
                    if int(det.category) == CLASS_TL:
                        sx, sy, sw, sh = self._tl.shrink_box(x, y, w, h, fx=0.18, fy=0.05, w_max=w_max, h_max=h_max)
                        roi_rgb = proc_rgb_full[sy:sy+sh, sx:sx+sw] if (sw > 0 and sh > 0) else None
                        phase = self._tl.phase_from_roi(roi_rgb) if roi_rgb is not None else TrafficLightPhase.UNKLAR
                        name = f"{name} ({phase})"

                        # Erste gefundene Phase dieses Frames – an LED-Schnittstelle melden
                        if frame_phase_for_led is None:
                            frame_phase_for_led = phase

                    label = f"{name} ({det.conf:.2f})"
                    draw_label(overlay, label, x, y, scale=0.5, thickness=1,
                               text_bgr=(0,0,255), bg_alpha=1.0)  # BG auf overlay (und erst später blenden)

                    color = (0,255,255) if int(det.category) == CLASS_TL else (0,255,0)
                    draw_box(overlay, x, y, w, h, color, thickness=2)

                except (ValueError, IndexError) as e:
                    logging.warning(f"Error processing detection: {e}")
                    continue

            # Overlay nur einmal anwenden
            cv2.addWeighted(overlay, 0.30, draw_surface, 0.70, 0, draw_surface)

            # LED-Schnittstelle am Frame-Ende aufrufen (entprellt durch LedPhaseSink selbst)
            if self._led_sink is not None and frame_phase_for_led is not None:
                try:
                    self._led_sink.apply_phase(frame_phase_for_led)
                except Exception as e:
                    logging.warning(f"LED sink error: {e}")