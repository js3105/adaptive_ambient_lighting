# src/detection/detector.py
import cv2
import numpy as np
import logging
import time
from picamera2 import MappedArray
from picamera2.devices import IMX500

from .postprocess import parse_detections
from .traffic_light import TrafficLightAnalyzer, TrafficLightPhase
from .common import pick_label
from .overlay import draw_label, draw_box
from .arrows import ArrowSelector, CLASS_ARROW_LEFT, CLASS_ARROW_RIGHT, CLASS_ARROW_STRAIGHT

# ------------------------------------------------------------
# Hilfsfunktionen für Reattach (IoU)
# ------------------------------------------------------------
def _iou_xywh(a, b):
    """IoU (Intersection-over-Union) zwischen zwei Boxen im (x,y,w,h)-Format."""
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
    """Finde die Ampel-Detection mit höchstem IoU zur Sticky-Box."""
    best, best_iou = None, 0.0
    for det in light_dets:
        try:
            x, y, w, h = map(int, det.box)
            iou = _iou_xywh(sticky_box, (x, y, w, h))
            if iou > best_iou:
                best, best_iou = det, iou
        except Exception as e:
            logging.warning(f"Sticky IoU calc error: {e}")
            continue
    return best, best_iou

# ------------------------------------------------------------
# Klassen-IDs (euer Datensatz)
# ------------------------------------------------------------
CLASS_TL = 0  # ampel
# 1: pfeil_gerade, 2: pfeil_links, 3: pfeil_rechts (siehe arrows.py)

class ObjectDetector:
    DETECTION_THRESHOLD = 0.55
    DETECTION_IOU = 0.65
    DETECTION_MAX_DETS = 10

    # Sticky „bis Grün + 2s“
    STICKY_HOLD_GREEN = 2.0   # Sekunden nach erster Grün-Sichtung
    # IoU-Reattach (wenn Detection kurz weg ist)
    STICKY_TTL_FRAMES = 820
    STICKY_IOU_THRESH = 0.05

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

        # LED-Schnittstelle (optional; Objekt mit .apply_phase(str) und .set_ambient())
        self._led_sink = led_sink

        # Pfeil-/Spur-Logik
        self._arrows = ArrowSelector()

        # Sticky-State (ereignisgetrieben: „halten bis Grün“)
        self._sticky_mode = False            # True = Ampel wird gehalten
        self._sticky_release_ts = 0.0        # >0 wenn Grün gesehen → Zeitpunkt, ab dem gelöst wird
        self._sticky_box = None              # (x,y,w,h) der gemerkten Ampel
        self._sticky_lane = None             # "left" | "straight" | "right"
        self._sticky_last_phase = None       # letzte bekannte Phase (zur Anzeige/LED)
        self._sticky_ttl_frames = 0          # Restversuche IoU-Reattach
        self._sticky_last_seen_ts = 0.0      # wann zuletzt real gesehen/reattached

        # Optionales Debug
        self._debug = False

    # ---------- Public API ----------
    def set_led_sink(self, led_sink):
        self._led_sink = led_sink

    def _labels(self):
        return self.labels

    # ---------- NN-Postprocessing ----------
    def parse_detections(self, metadata):
        try:
            self.last_detections = parse_detections(
                self.imx500, self.intrinsics, self.picam2, metadata, self.last_detections,
                threshold=self.DETECTION_THRESHOLD, iou=self.DETECTION_IOU, max_dets=self.DETECTION_MAX_DETS
            )
        except Exception as e:
            logging.error(f"parse_detections failed: {e}")
        return self.last_detections

    # ---------- Sticky-Hilfsfunktionen ----------
    def _sticky_start(self, lane_key, box_xywh, phase):
        """Sticky aktivieren/erneuern sobald ein stabiles Match existiert."""
        now = time.time()
        self._sticky_mode = True
        self._sticky_release_ts = 0.0             # noch KEIN Release geplant
        self._sticky_lane = lane_key
        self._sticky_box = tuple(map(int, box_xywh))
        self._sticky_last_phase = phase
        self._sticky_ttl_frames = self.STICKY_TTL_FRAMES
        self._sticky_last_seen_ts = now

    def _sticky_on_phase(self, phase):
        """Bei erster Grün-Sichtung Release-Timer starten/verlängern."""
        if not phase:
            return
        p = str(phase)
        now = time.time()
        if p == TrafficLightPhase.GRUEN:
            if self._sticky_release_ts <= 0.0:
                self._sticky_release_ts = now + self.STICKY_HOLD_GREEN
            else:
                self._sticky_release_ts = max(self._sticky_release_ts, now + self.STICKY_HOLD_GREEN)

    def _sticky_active(self):
        """Sticky gilt, solange wir im Sticky-Modus sind und (falls gesetzt) die Release-Zeit nicht abgelaufen ist."""
        if not self._sticky_mode:
            return False
        if self._sticky_release_ts <= 0.0:
            return True  # Noch kein Grün gesehen → unbegrenzt halten
        return time.time() < self._sticky_release_ts

    def _sticky_end_if_due(self):
        """Sticky deaktivieren, wenn der Release-Timer abgelaufen ist."""
        if self._sticky_mode and self._sticky_release_ts > 0.0 and time.time() >= self._sticky_release_ts:
            self._sticky_mode = False

    # ---------- Hauptanzeige & Logik ----------
    def draw_callback(self, request, stream="main"):
        try:
            with MappedArray(request, stream) as m:
                draw_surface = m.array
                if draw_surface.ndim != 3:
                    return
                h_max, w_max = draw_surface.shape[:2]

                # Farbansicht nur einmal berechnen
                if draw_surface.shape[2] == 4:
                    proc_rgb_full = cv2.cvtColor(draw_surface, cv2.COLOR_BGRA2RGB)
                elif draw_surface.shape[2] == 3:
                    proc_rgb_full = draw_surface[:, :, ::-1]
                else:
                    return

                overlay = draw_surface.copy()

                # === 1) Ego-Spur-ROI zeichnen
                roi_poly = self._arrows.ego_lane_roi_polygon(draw_surface.shape)
                cv2.polylines(overlay, [roi_poly], True, (0, 255, 255), 2)

                # === 2) Detections trennen
                all_arrows = []  # (det, x,y,w,h, cx,cy)
                all_lights = []  # det
                for det in self.last_detections:
                    try:
                        x, y, w, h = map(int, det.box)
                        # clamp
                        x = max(0, min(x, w_max - 1))
                        y = max(0, min(y, h_max - 1))
                        w = min(w, w_max - x)
                        h = min(h, h_max - y)
                        if w <= 0 or h <= 0:
                            continue

                        if int(det.category) == CLASS_TL:
                            all_lights.append(det)
                        elif int(det.category) in (CLASS_ARROW_LEFT, CLASS_ARROW_RIGHT, CLASS_ARROW_STRAIGHT):
                            cx = x + 0.5 * w
                            cy = y + 0.5 * h
                            all_arrows.append((det, x, y, w, h, cx, cy))
                    except Exception as e:
                        logging.warning(f"Error sorting detection: {e}")
                        continue

                # === 3) Pfeil/Lane wählen
                chosen = None
                try:
                    chosen = self._arrows.choose_arrow_and_lane(all_arrows, roi_poly, w_max)
                except Exception as e:
                    logging.warning(f"choose_arrow_and_lane failed: {e}")

                matched_light = None
                frame_phase_for_led = None
                used_sticky = False

                # === 4) Fall A: stabiler Pfeil -> Side-Bias-Matching + Sticky-Start/Phase-Handling
                try:
                    if chosen and all_lights:
                        det, x, y, w, h, cx, cy, lane_key, stable_ok = chosen
                        matched_light = self._arrows.match_light_side_biased(
                            all_lights, lane_key, w_max, h_max, arrow_cx=cx, arrow_cy=cy
                        )
                        if matched_light is not None:
                            lx, ly, lw, lh = map(int, matched_light.box)
                            sx, sy, sw, sh = self._tl.shrink_box(lx, ly, lw, lh, fx=0.18, fy=0.05, w_max=w_max, h_max=h_max)
                            if sw > 0 and sh > 0:
                                roi_rgb = proc_rgb_full[sy:sy+sh, sx:sx+sw]
                                phase = self._tl.phase_from_roi(roi_rgb)
                            else:
                                phase = TrafficLightPhase.UNKLAR

                            if stable_ok:
                                # *** ARME NUR BEI ROT ODER GELB ***
                                if phase in (TrafficLightPhase.ROT, TrafficLightPhase.GELB):
                                    frame_phase_for_led = phase
                                    self._sticky_start(lane_key, (lx, ly, lw, lh), phase)
                                # Immer: Grün überwachen, falls schon aktiv
                                self._sticky_on_phase(phase)
                            else:
                                # Noch nicht stabil: nur Grün-Release vorbereiten
                                self._sticky_on_phase(phase)
                except Exception as e:
                    logging.warning(f"matching phase failed: {e}")

                # === 5) Fall B: kein stabiler Pfeil/Match -> Sticky nutzen (falls aktiv)
                try:
                    if (matched_light is None) and self._sticky_active():
                        # 5.1 Reattach per IoU
                        if all_lights and self._sticky_box is not None and self._sticky_ttl_frames > 0:
                            cand, iou = _best_iou_light(self._sticky_box, all_lights)
                            if cand is not None and iou >= self.STICKY_IOU_THRESH:
                                matched_light = cand
                                used_sticky = True
                                lx, ly, lw, lh = map(int, matched_light.box)
                                self._sticky_box = (lx, ly, lw, lh)
                                self._sticky_ttl_frames -= 1
                                self._sticky_last_seen_ts = time.time()

                                sx, sy, sw, sh = self._tl.shrink_box(lx, ly, lw, lh, fx=0.18, fy=0.05, w_max=w_max, h_max=h_max)
                                if sw > 0 and sh > 0:
                                    roi_rgb = proc_rgb_full[sy:sy+sh, sx:sx+sw]
                                    phase = self._tl.phase_from_roi(roi_rgb)
                                else:
                                    phase = self._sticky_last_phase or TrafficLightPhase.UNKLAR

                                self._sticky_last_phase = phase
                                self._sticky_on_phase(phase)
                                frame_phase_for_led = phase

                        # 5.2 Kein Reattach → Phase aus gecachter Box
                        if matched_light is None and self._sticky_box is not None:
                            used_sticky = True
                            lx, ly, lw, lh = self._sticky_box
                            lx = max(0, min(lx, w_max - 2)); ly = max(0, min(ly, h_max - 2))
                            lw = max(2, min(lw, w_max - lx)); lh = max(2, min(lh, h_max - ly))
                            sx, sy, sw, sh = self._tl.shrink_box(lx, ly, lw, lh, fx=0.18, fy=0.05, w_max=w_max, h_max=h_max)
                            if sw > 0 and sh > 0:
                                roi_rgb = proc_rgb_full[sy:sy+sh, sx:sx+sw]
                                phase = self._tl.phase_from_roi(roi_rgb)
                                self._sticky_last_phase = phase
                                self._sticky_on_phase(phase)
                                frame_phase_for_led = phase
                except Exception as e:
                    logging.warning(f"sticky handling failed: {e}")

                # === 6) Dünne Übersicht: alle Pfeile & Ampeln
                labels = self._labels()
                for det, x, y, w, h, cx, cy in all_arrows:
                    try:
                        name = pick_label(labels, det.category)
                        draw_box(overlay, int(x), int(y), int(w), int(h), (180, 180, 180), thickness=1)
                        draw_label(overlay, f"{name} ({det.conf:.2f})", int(x), int(y),
                                   scale=0.5, thickness=1, text_bgr=(50, 50, 50), bg_alpha=1.0)
                    except Exception:
                        continue

                for det in all_lights:
                    try:
                        lx, ly, lw, lh = map(int, det.box)
                        draw_box(overlay, lx, ly, lw, lh, (0, 140, 255), thickness=1)
                        draw_label(overlay, f"{pick_label(labels, det.category)} ({det.conf:.2f})", lx, ly,
                                   scale=0.5, thickness=1, text_bgr=(0, 90, 180), bg_alpha=1.0)
                    except Exception:
                        continue

                # === 7) Highlights/Badges
                if chosen:
                    try:
                        det, x, y, w, h, cx, cy, lane_key, stable_ok = chosen
                        draw_box(overlay, int(x), int(y), int(w), int(h), (0, 255, 0), thickness=3)
                        badge = f"{pick_label(labels, det.category)} | lane:{lane_key} | conf:{det.conf:.2f}"
                        draw_label(overlay, badge, int(x), max(0, int(y) - 20),
                                   scale=0.6, thickness=2, text_bgr=(0, 120, 0), bg_alpha=1.0)
                    except Exception:
                        pass

                # Gematchte Ampel (live oder sticky)
                if matched_light is not None:
                    try:
                        lx, ly, lw, lh = map(int, matched_light.box)
                        color = (0, 255, 255) if not used_sticky else (255, 255, 0)  # sticky gelblich
                        draw_box(overlay, lx, ly, lw, lh, color, thickness=3)

                        label_phase = (frame_phase_for_led
                                       if frame_phase_for_led is not None
                                       else (self._sticky_last_phase or "Unklar"))
                        sticky_tag = " (sticky)" if used_sticky else ""
                        draw_label(overlay, f"ampel{sticky_tag}: {label_phase}",
                                   lx, max(0, ly - 20), scale=0.6, thickness=2, text_bgr=(0, 180, 180), bg_alpha=1.0)
                    except Exception:
                        pass
                else:
                    if self._sticky_active() and self._sticky_box is not None:
                        lx, ly, lw, lh = self._sticky_box
                        draw_box(overlay, lx, ly, lw, lh, (255, 255, 0), thickness=2)
                        draw_label(overlay, f"ampel (sticky only): {self._sticky_last_phase or 'Unklar'}",
                                   lx, max(0, ly - 20), scale=0.6, thickness=2, text_bgr=(0, 180, 180), bg_alpha=1.0)

                # === 8) Overlay einblenden
                cv2.addWeighted(overlay, 0.30, draw_surface, 0.70, 0, draw_surface)

                # === 9) LED steuern
                if self._led_sink is not None:
                    if frame_phase_for_led is not None:
                        # aktive Ampelphase anzeigen
                        try:
                            self._led_sink.apply_phase(frame_phase_for_led)
                        except Exception as e:
                            logging.warning(f"LED sink error: {e}")
                    else:
                        # Keine aktive Phase → Ambient, wenn nicht „sticky aktiv“
                        if not self._sticky_active():
                            try:
                                self._led_sink.set_ambient()
                            except Exception as e:
                                logging.warning(f"LED ambient error: {e}")

                # === 10) Sticky ggf. beenden
                self._sticky_end_if_due()

        except Exception as e:
            logging.error(f"draw_callback error: {e}")
            return