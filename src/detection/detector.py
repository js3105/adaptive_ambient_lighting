# src/detection/detector.py
import cv2
import numpy as np
import logging
from picamera2 import MappedArray
from picamera2.devices import IMX500

from .postprocess import parse_detections
from .traffic_light import TrafficLightAnalyzer, TrafficLightPhase
from .common import pick_label
from .overlay import draw_label, draw_box
from .arrows import ArrowSelector, CLASS_ARROW_LEFT, CLASS_ARROW_RIGHT, CLASS_ARROW_STRAIGHT

# Klassen-IDs (euer Datensatz)
CLASS_TL = 0          # ampel
# 1: pfeil_gerade, 2: pfeil_links, 3: pfeil_rechts (siehe arrows.py)

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

        # Farb-/Phasenanalyse
        self._tl = TrafficLightAnalyzer(
            gamma=1.6, clahe_clip=2.0, clahe_grid=8,
            s_min=60, v_min=60, use_r_dom=True, r_margin=20
        )

        # LED-Schnittstelle (optional; Objekt mit .apply_phase(str))
        self._led_sink = led_sink

        # Pfeil-/Spur-Logik
        self._arrows = ArrowSelector()

                # Sticky-Auswahl (Ampel behalten, wenn Pfeil kurz weg ist)
        self._sticky = {
            "lane": None,
            "box": None,         # (x, y, w, h)
            "ttl": 0
        }
        self.STICKY_TTL_FRAMES = 12
        self.STICKY_IOU_THRESH = 0.30

    def set_led_sink(self, led_sink):
        self._led_sink = led_sink

    def _labels(self):
        return self.labels
    
    def _iou_xywh(a, b):
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        ax2, ay2 = ax + aw, ay + ah
        bx2, by2 = bx + bw, by + bh
        inter_w = max(0, min(ax2, bx2) - max(ax, bx))
        inter_h = max(0, min(ay2, by2) - max(ay, by))
        inter = inter_w * inter_h
        if inter == 0:
            return 0.0
        area_a = aw * ah
        area_b = bw * bh
        return inter / float(area_a + area_b - inter)

    def _best_iou_light(sticky_box, light_dets):
        best, best_iou = None, 0.0
        for det in light_dets:
            x, y, w, h = map(int, det.box)
            iou = _iou_xywh(sticky_box, (x, y, w, h))
            if iou > best_iou:
                best, best_iou = det, iou
        return best, best_iou

    # ---------- NN-Postprocessing ----------
    def parse_detections(self, metadata):
        self.last_detections = parse_detections(
            self.imx500, self.intrinsics, self.picam2, metadata, self.last_detections,
            threshold=self.DETECTION_THRESHOLD, iou=self.DETECTION_IOU, max_dets=self.DETECTION_MAX_DETS
        )
        return self.last_detections

    # ---------- Zeichnen & Logik ----------
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

            # === 1) Ego-Spur-ROI zeichnen
            roi_poly = self._arrows.ego_lane_roi_polygon(draw_surface.shape)
            cv2.polylines(overlay, [roi_poly], True, (0, 255, 255), 2)

            # === 2) Detections in Pfeile & Ampeln trennen
            all_arrows = []  # (det, x,y,w,h, cx,cy)
            all_lights = []  # det
            for det in self.last_detections:
                try:
                    x, y, w, h = map(int, det.box)
                    x = max(0, min(x, w_max - 1))
                    y = max(0, min(y, h_max - 1))
                    w = min(w, w_max - x)
                    h = min(h, h_max - y)
                    if w <= 0 or h <= 0:
                        continue

                    if int(det.category) == CLASS_TL:
                        all_lights.append(det)
                    elif int(det.category) in (CLASS_ARROW_LEFT, CLASS_ARROW_RIGHT, CLASS_ARROW_STRAIGHT):
                        cx, cy = self._arrows.box_center_xy(x, y, w, h)
                        all_arrows.append((det, x, y, w, h, cx, cy))

                except Exception as e:
                    logging.warning(f"Error sorting detection: {e}")
                    continue

            # === 3) Pfeil/Lane wählen (wie gehabt)
            chosen = self._arrows.choose_arrow_and_lane(all_arrows, roi_poly, w_max)

            matched_light = None
            frame_phase_for_led = None
            used_sticky = False  # Debug-Flag

            # === 4) Fall A: Wir haben einen stabilen Pfeil -> reguläres Matching
            if chosen and all_lights:
                det, x, y, w, h, cx, cy, lane_key, stable_ok = chosen
                matched_light = self._arrows.match_light_side_biased(
                    all_lights, lane_key, w_max, h_max, arrow_cx=cx, arrow_cy=cy
                )

                # Phase nur für gematchte Ampel bestimmen
                if matched_light is not None:
                    lx, ly, lw, lh = map(int, matched_light.box)
                    sx, sy, sw, sh = self._tl.shrink_box(lx, ly, lw, lh, fx=0.18, fy=0.05, w_max=w_max, h_max=h_max)
                    roi_rgb = proc_rgb_full[sy:sy+sh, sx:sx+sw] if (sw > 0 and sh > 0) else None
                    phase = self._tl.phase_from_roi(roi_rgb) if roi_rgb is not None else TrafficLightPhase.UNKLAR
                    if stable_ok:
                        frame_phase_for_led = phase

                    # --- Sticky aktualisieren (nur wenn Pfeil stabil) ---
                    if stable_ok:
                        self._sticky["lane"] = lane_key
                        self._sticky["box"]  = (lx, ly, lw, lh)
                        self._sticky["ttl"]  = self.STICKY_TTL_FRAMES

            # === 5) Fall B: Kein (stabiler) Pfeil -> Sticky versuchen
            if (matched_light is None) and (self._sticky["ttl"] > 0) and all_lights:
                prev_box = self._sticky["box"]
                if prev_box is not None:
                    cand, iou = _best_iou_light(prev_box, all_lights)
                    if cand is not None and iou >= self.STICKY_IOU_THRESH:
                        matched_light = cand
                        used_sticky = True
                        # Box updaten & TTL dekrementieren
                        lx, ly, lw, lh = map(int, matched_light.box)
                        self._sticky["box"] = (lx, ly, lw, lh)
                        self._sticky["ttl"] -= 1

                        # Phase für Anzeige/LED bestimmen (LED optional: nur wenn du willst)
                        sx, sy, sw, sh = self._tl.shrink_box(lx, ly, lw, lh, fx=0.18, fy=0.05, w_max=w_max, h_max=h_max)
                        roi_rgb = proc_rgb_full[sy:sy+sh, sx:sx+sw] if (sw > 0 and sh > 0) else None
                        frame_phase_for_led = self._tl.phase_from_roi(roi_rgb) if roi_rgb is not None else TrafficLightPhase.UNKLAR

            # === 6) Dünne Übersicht zeichnen (wie vorher)
            for det, x, y, w, h, cx, cy in all_arrows:
                name = pick_label(labels, det.category)
                draw_box(overlay, int(x), int(y), int(w), int(h), (180, 180, 180), thickness=1)
                draw_label(overlay, f"{name} ({det.conf:.2f})", int(x), int(y),
                           scale=0.5, thickness=1, text_bgr=(50, 50, 50), bg_alpha=1.0)

            for det in all_lights:
                lx, ly, lw, lh = map(int, det.box)
                draw_box(overlay, lx, ly, lw, lh, (0, 140, 255), thickness=1)
                draw_label(overlay, f"{pick_label(labels, det.category)} ({det.conf:.2f})", lx, ly,
                           scale=0.5, thickness=1, text_bgr=(0, 90, 180), bg_alpha=1.0)

            # === 7) Highlights/Badges
            if chosen:
                det, x, y, w, h, cx, cy, lane_key, stable_ok = chosen
                draw_box(overlay, int(x), int(y), int(w), int(h), (0, 255, 0), thickness=3)
                badge = f"{pick_label(labels, det.category)} | lane:{lane_key} | conf:{det.conf:.2f}"
                draw_label(overlay, badge, int(x), max(0, int(y) - 20),
                           scale=0.6, thickness=2, text_bgr=(0, 120, 0), bg_alpha=1.0)

                # Debug Stabilität
                draw_label(overlay, f"stable {self._arrows._stable_counter}/{self._arrows.STABLE_N} (lane {lane_key})",
                           10, 10, scale=0.6, thickness=2, text_bgr=(40, 200, 40), bg_alpha=1.0)

            if matched_light is not None:
                lx, ly, lw, lh = map(int, matched_light.box)
                color = (0, 255, 255) if not used_sticky else (255, 255, 0)  # sticky = gelblich markieren
                draw_box(overlay, lx, ly, lw, lh, color, thickness=3)

                label_phase = frame_phase_for_led if frame_phase_for_led is not None else "Unklar"
                sticky_tag = " (sticky)" if used_sticky else ""
                draw_label(overlay, f"ampel{sticky_tag}: {label_phase}",
                           lx, max(0, ly - 20), scale=0.6, thickness=2, text_bgr=(0, 180, 180), bg_alpha=1.0)

            # === 8) LED-Schnittstelle am Frame-Ende aufrufen
            if self._led_sink is not None and frame_phase_for_led is not None:
                try:
                    self._led_sink.apply_phase(frame_phase_for_led)
                except Exception as e:
                    logging.warning(f"LED sink error: {e}")