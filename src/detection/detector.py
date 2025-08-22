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

# ------------------------------------------------------------
# Hyperbel-Spurerkennung (weiß-Maske + robuster Fit)
# ------------------------------------------------------------
_LANE_CFG = {
    "proc_scale": 0.5,      # Downscale für Verarbeitung (Performance)
    "band_width_rel": 0.23, # Vertikalband links/rechts der Mitte (w * rel) -> Ego-Spur
    "y_min_rel": 0.45,      # nur untere Bildhälfte
    "canny_low": 60,
    "canny_high": 160,
    "huber_delta": 3.0,
    "max_iters": 15,
    "roi_expand_px": 8,     # ROI-Breite um Kurven (in Downscale-Pixeln); skaliert hoch
    "min_points_fit": 80,   # Mindestpunkte pro Seite für stabilen Fit
}

def _white_mask_hsv(rgb):
    """Weiße Fahrbahnmarkierung: niedrige Sättigung, hohe Helligkeit."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    v = clahe.apply(v)
    hsv = cv2.merge([h, s, v])
    lower = np.array([0, 0, 200], dtype=np.uint8)
    upper = np.array([179, 70, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
    return mask

def _hyperbola_x_of_y(p, y):
    a, b, c = p
    return a / (y + b + 1e-6) + c

def _jacobian(p, y):
    a, b, c = p
    denom = (y + b + 1e-6)
    J = np.stack([
        1.0 / denom,            # d/da
        -a / (denom**2),        # d/db
        np.ones_like(y)         # d/dc
    ], axis=1)
    return J

def _fit_hyperbola(points_xy, max_iters=15, huber_delta=3.0):
    """Robuster Gauss-Newton-Fit für x(y)=a/(y+b)+c. Punkte: Nx2 (x,y)."""
    pts = np.asarray(points_xy, dtype=np.float32)
    if pts.shape[0] < 5:
        return None, None
    x = pts[:,0]; y = pts[:,1]
    c0 = np.median(x); b0 = 0.0
    a0 = np.median((x - c0) * (y + b0 + 1e-3))
    p = np.array([a0, b0, c0], dtype=np.float32)
    for _ in range(max_iters):
        x_pred = _hyperbola_x_of_y(p, y)
        r = x - x_pred
        abs_r = np.abs(r)
        w = np.where(abs_r <= huber_delta, 1.0, huber_delta/(abs_r + 1e-6))
        J = _jacobian(p, y)
        W = w[:,None]
        A = J.T @ (W * J)
        bvec = J.T @ (W[:,0] * r)
        try:
            dp = np.linalg.solve(A, bvec)
        except np.linalg.LinAlgError:
            return None, None
        p = p + dp.astype(np.float32)
        if float(np.linalg.norm(dp)) < 1e-3:
            break
    final_r = x - _hyperbola_x_of_y(p, y)
    inliers = np.abs(final_r) < 3.0*huber_delta
    return p.astype(np.float32), inliers

def _collect_lane_points(rgb_full, cfg, debug=None):
    """Sammelt (x,y)-Kandidaten links/rechts der Bildmitte (Ego-Bänder)."""
    h, w = rgb_full.shape[:2]
    scale = cfg["proc_scale"]
    rgb = cv2.resize(rgb_full, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    hh, ww = rgb.shape[:2]
    mask = _white_mask_hsv(rgb)
    edges = cv2.Canny(mask, cfg["canny_low"], cfg["canny_high"])
    y_min = int(cfg["y_min_rel"] * hh)
    edges[:y_min, :] = 0
    cx = ww // 2
    band = int(cfg["band_width_rel"] * ww)
    left_band = (max(0, cx - band), cx)
    right_band = (cx, min(ww-1, cx + band))
    ys, xs = np.nonzero(edges)
    left_mask  = (xs >= left_band[0]) & (xs < left_band[1])
    right_mask = (xs >= right_band[0]) & (xs < right_band[1])
    left_pts_ds  = np.stack([xs[left_mask],  ys[left_mask]],  axis=1) if np.any(left_mask)  else np.zeros((0,2), np.int32)
    right_pts_ds = np.stack([xs[right_mask], ys[right_mask]], axis=1) if np.any(right_mask) else np.zeros((0,2), np.int32)
    left_pts  = (left_pts_ds.astype(np.float32)  / scale) if left_pts_ds.size  else np.zeros((0,2), np.float32)
    right_pts = (right_pts_ds.astype(np.float32) / scale) if right_pts_ds.size else np.zeros((0,2), np.float32)
    if debug is not None:
        debug["edges_ds"] = edges
        debug["mask_ds"] = mask
        debug["left_band"] = (int(left_band[0]/scale), int(left_band[1]/scale))
        debug["right_band"] = (int(right_band[0]/scale), int(right_band[1]/scale))
    return left_pts, right_pts

def _build_lane_roi_from_curves(p_left, p_right, shape_hw, cfg):
    """Schmale ROI als Polygon zwischen linker und rechter Hyperbel."""
    h, w = shape_hw
    y0 = int(_LANE_CFG["y_min_rel"] * h)
    y1 = h - 1
    ys = np.arange(y0, y1, 5, dtype=np.float32)
    xl = _hyperbola_x_of_y(p_left, ys)
    xr = _hyperbola_x_of_y(p_right, ys)
    xl = np.clip(xl, 0, w-1); xr = np.clip(xr, 0, w-1)
    if np.nanmean(xl) > np.nanmean(xr):
        xl, xr = xr, xl
    expand = max(2, int(cfg["roi_expand_px"] / cfg["proc_scale"]))
    xl = np.maximum(0, xl - expand)
    xr = np.minimum(w-1, xr + expand)
    left_pts  = np.stack([xl, ys], axis=1).astype(np.int32)
    right_pts = np.stack([xr, ys], axis=1).astype(np.int32)
    poly = np.vstack([left_pts, right_pts[::-1]])
    return poly

def _fit_ego_lane(rgb_full, cfg, overlay=None):
    """Liefert (roi_poly, left_params, right_params) oder (None, None, None)."""
    dbg = {}
    left_pts, right_pts = _collect_lane_points(rgb_full, cfg, debug=dbg)
    if left_pts.shape[0] < cfg["min_points_fit"] or right_pts.shape[0] < cfg["min_points_fit"]:
        return None, None, None
    pL, _ = _fit_hyperbola(left_pts, max_iters=cfg["max_iters"], huber_delta=cfg["huber_delta"])
    pR, _ = _fit_hyperbola(right_pts, max_iters=cfg["max_iters"], huber_delta=cfg["huber_delta"])
    if pL is None or pR is None:
        return None, None, None
    h, w = rgb_full.shape[:2]
    y_check = float(h - 5)
    xl = _hyperbola_x_of_y(pL, y_check)
    xr = _hyperbola_x_of_y(pR, y_check)
    if not (0 <= xl < xr < w):
        return None, None, None
    roi_poly = _build_lane_roi_from_curves(pL, pR, (h, w), cfg)
    if overlay is not None:
        def _draw_curve(params, color):
            ys = np.arange(int(0.45*h), h-1, 5, dtype=np.float32)
            xs = _hyperbola_x_of_y(params, ys)
            pts = np.stack([xs, ys], axis=1).round().astype(np.int32)
            pts = pts[(pts[:,0] >= 0) & (pts[:,0] < w)]
            if len(pts) >= 2:
                cv2.polylines(overlay, [pts], False, color, 2)
        _draw_curve(pL, (0, 255, 0))
        _draw_curve(pR, (0, 255, 0))
        cv2.polylines(overlay, [roi_poly], True, (0, 200, 255), 2)
    return roi_poly, pL, pR

# ------------------------------------------------------------
# Objekt-Detektor
# ------------------------------------------------------------
class ObjectDetector:
    DETECTION_THRESHOLD = 0.55
    DETECTION_IOU = 0.65
    DETECTION_MAX_DETS = 10

    # Sticky „bis Grün + 2s“
    STICKY_HOLD_GREEN = 2.0
    STICKY_TTL_FRAMES = 820
    STICKY_IOU_THRESH = 0.05

    # Quadranten-Logik
    QUAD_MARGIN_FRAC = 0.06   # Sicherheitsabstand von den Mittellinien
    QUAD_DEBUG = True         # Debug-Overlay aktivieren

    def __init__(self, imx500: IMX500, intrinsics, picam2, led_sink=None, headless: bool = False):
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

        # Sticky-State
        self._sticky_mode = False
        self._sticky_release_ts = 0.0
        self._sticky_lane = None
        self._sticky_box = None
        self._sticky_last_phase = None
        self._sticky_ttl_frames = 0
        self._sticky_last_seen_ts = 0.0

        # Debug / Headless
        self._debug = False
        self._headless = headless

        # ROI Cache
        self._roi_poly_cached = None
        self._roi_cached_ts = 0.0
        self._roi_cache_ttl = 0.8  # Sekunden

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
        now = time.time()
        self._sticky_mode = True
        self._sticky_release_ts = 0.0
        self._sticky_lane = lane_key
        self._sticky_box = tuple(map(int, box_xywh))
        self._sticky_last_phase = phase
        self._sticky_ttl_frames = self.STICKY_TTL_FRAMES
        self._sticky_last_seen_ts = now

    def _sticky_on_phase(self, phase):
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
        if not self._sticky_mode:
            return False
        if self._sticky_release_ts <= 0.0:
            return True
        return time.time() < self._sticky_release_ts

    def _sticky_end_if_due(self):
        if self._sticky_mode and self._sticky_release_ts > 0.0 and time.time() >= self._sticky_release_ts:
            self._sticky_mode = False

    # ---------- Quadranten-Helpers ----------
    def _draw_quadrant_debug(self, overlay, w_max, h_max, lane_key=None):
        """Zeichnet Mittellinien, Margins und hebt den relevanten Quadranten hervor."""
        if overlay is None:
            return
        x_mid = int(0.5 * w_max)
        y_mid = int(0.5 * h_max)
        x_margin = int(self.QUAD_MARGIN_FRAC * w_max)
        y_margin = int(self.QUAD_MARGIN_FRAC * h_max)

        # Mittellinien
        cv2.line(overlay, (x_mid, 0), (x_mid, h_max-1), (60, 60, 60), 1)
        cv2.line(overlay, (0, y_mid), (w_max-1, y_mid), (60, 60, 60), 1)

        # Marginlinien (gestrichelt)
        for dy in (-y_margin, +y_margin):
            y = y_mid + dy
            for x0 in range(0, w_max, 20):
                cv2.line(overlay, (x0, y), (min(x0+10, w_max-1), y), (80, 80, 80), 1)
        for dx in (-x_margin, +x_margin):
            x = x_mid + dx
            for y0 in range(0, h_max, 20):
                cv2.line(overlay, (x, y0), (x, min(y0+10, h_max-1)), (80, 80, 80), 1)

        # Quadrant-Highlight
        if lane_key:
            qmask = np.zeros(overlay.shape[:2], np.uint8)
            if lane_key == "straight":
                # oberer Quadrant (über Margin)
                y_hi = y_mid - y_margin
                cv2.rectangle(qmask, (0, 0), (w_max-1, max(0, y_hi)), 255, -1)
            elif lane_key == "left":
                x_hi = x_mid - x_margin
                cv2.rectangle(qmask, (0, 0), (max(0, x_hi), h_max-1), 255, -1)
            elif lane_key == "right":
                x_lo = x_mid + x_margin
                cv2.rectangle(qmask, (min(w_max-1, x_lo), 0), (w_max-1, h_max-1), 255, -1)
            # leicht einfärben
            color = (30, 180, 255) if lane_key == "straight" else (255, 200, 40) if lane_key == "right" else (40, 200, 255)
            tint = np.zeros_like(overlay)
            tint[:] = color
            overlay[:] = np.where(qmask[...,None]==255, (0.15*tint + 0.85*overlay).astype(overlay.dtype), overlay)

            cv2.putText(overlay, f"Quadrant: {lane_key}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240,240,240), 2, cv2.LINE_AA)

    def _match_light_by_quadrant(self, all_lights, lane_key, w_max, h_max, overlay=None):
        """
        Wählt die Ampel anhand des Quadranten:
          - 'straight' -> oberer Quadrant (cy <= y_mid - y_margin)
          - 'left'     -> linker Quadrant (cx <= x_mid - x_margin)
          - 'right'    -> rechter Quadrant (cx >= x_mid + x_margin)
        Gibt die Detection mit der höchsten Konfidenz zurück.
        """
        if not all_lights or lane_key is None:
            return None

        x_mid = 0.5 * w_max
        y_mid = 0.5 * h_max
        x_margin = self.QUAD_MARGIN_FRAC * w_max
        y_margin = self.QUAD_MARGIN_FRAC * h_max

        cands = []
        for det in all_lights:
            try:
                x, y, w, h = map(int, det.box)
                if w <= 0 or h <= 0:
                    continue
                cx = x + 0.5 * w
                cy = y + 0.5 * h

                ok = False
                if lane_key == "straight":
                    ok = (cy <= (y_mid - y_margin))
                elif lane_key == "left":
                    ok = (cx <= (x_mid - x_margin))
                elif lane_key == "right":
                    ok = (cx >= (x_mid + x_margin))

                if ok:
                    cands.append((det, float(det.conf), (int(cx), int(cy))))
                    # Debug: Kandidat markieren
                    if overlay is not None and self.QUAD_DEBUG:
                        cv2.circle(overlay, (int(cx), int(cy)), 6, (0, 200, 255), 2)
            except Exception as e:
                logging.debug(f"quadrant candidate error: {e}")

        if not cands:
            return None

        cands.sort(key=lambda t: t[1], reverse=True)
        best_det, best_conf, best_pt = cands[0]

        # Debug: Gewinner kräftig markieren
        if overlay is not None and self.QUAD_DEBUG:
            cv2.circle(overlay, best_pt, 8, (0, 255, 255), -1)
            bx, by, bw, bh = map(int, best_det.box)
            draw_box(overlay, bx, by, bw, bh, (0, 255, 255), thickness=3)
            draw_label(overlay, f"quad-best {best_conf:.2f}", bx, max(0, by-18),
                       scale=0.5, thickness=1, text_bgr=(0,140,255), bg_alpha=1.0)

        return best_det

    # ---------- Hauptanzeige & Logik ----------
    def draw_callback(self, request, stream="main"):
        try:
            with MappedArray(request, stream) as m:
                draw_surface = m.array
                if draw_surface.ndim != 3:
                    return
                h_max, w_max = draw_surface.shape[:2]

                # Farbansicht
                if draw_surface.shape[2] == 4:
                    proc_rgb_full = cv2.cvtColor(draw_surface, cv2.COLOR_BGRA2RGB)
                elif draw_surface.shape[2] == 3:
                    proc_rgb_full = draw_surface[:, :, ::-1]
                else:
                    return

                overlay = None if self._headless else draw_surface.copy()

                # === 1) Ego-Spur per Hyperbel (mit Weiß-Maske)
                roi_poly = None
                try:
                    roi_poly, pL, pR = _fit_ego_lane(proc_rgb_full, _LANE_CFG, overlay=overlay if overlay is not None else None)
                except Exception as e:
                    logging.warning(f"lane fit failed: {e}")

                # Cache/TTL
                now_ts = time.time()
                if roi_poly is not None and roi_poly.shape[0] >= 4:
                    self._roi_poly_cached = roi_poly
                    self._roi_cached_ts = now_ts
                else:
                    if self._roi_poly_cached is not None and (now_ts - self._roi_cached_ts) <= self._roi_cache_ttl:
                        roi_poly = self._roi_poly_cached
                    else:
                        roi_poly = self._arrows.ego_lane_roi_polygon(draw_surface.shape)

                # === Quadranten-Overlay (immer sichtbar in Bildschirmversion)
                if overlay is not None and self.QUAD_DEBUG:
                    self._draw_quadrant_debug(overlay, w_max, h_max, None)

                # === 2) Detections trennen
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
                            cx = x + 0.5 * w
                            cy = y + 0.5 * h
                            # Arrow‑Kandidat wird später in choose_arrow_and_lane gegen ROI geprüft
                            all_arrows.append((det, x, y, w, h, cx, cy))
                    except Exception as e:
                        logging.warning(f"Error sorting detection: {e}")
                        continue

                # === 3) Pfeil/Lane wählen (nur innerhalb der hyperbolischen Ego-ROI)
                chosen = None
                try:
                    chosen = self._arrows.choose_arrow_and_lane(all_arrows, roi_poly, w_max)
                except Exception as e:
                    logging.warning(f"choose_arrow_and_lane failed: {e}")

                matched_light = None
                frame_phase_for_led = None
                used_sticky = False

                # === 4) Stabiler Pfeil -> Quadranten-Matching + Sticky-Start/Phase
                try:
                    if chosen and all_lights:
                        det, x, y, w, h, cx, cy, lane_key, stable_ok = chosen

                        # Quadrant passend zur Fahrtrichtung highlighten
                        if overlay is not None and self.QUAD_DEBUG:
                            self._draw_quadrant_debug(overlay, w_max, h_max, lane_key)

                        matched_light = self._match_light_by_quadrant(
                            all_lights, lane_key, w_max, h_max, overlay=overlay if overlay is not None else None
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
                                if phase in (TrafficLightPhase.ROT, TrafficLightPhase.GELB):
                                    frame_phase_for_led = phase
                                    self._sticky_start(lane_key, (lx, ly, lw, lh), phase)
                                self._sticky_on_phase(phase)
                            else:
                                self._sticky_on_phase(phase)
                except Exception as e:
                    logging.warning(f"matching/phase failed: {e}")

                # === 5) Kein stabiler Pfeil/Match -> Sticky nutzen (falls aktiv)
                try:
                    if (matched_light is None) and self._sticky_active():
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

                # === 6) Zeichnen (nur wenn Overlay existiert)
                if overlay is not None:
                    labels = self._labels()
                    # Pfeile
                    for det, x, y, w, h, cx, cy in all_arrows:
                        try:
                            name = pick_label(labels, det.category)
                            draw_box(overlay, int(x), int(y), int(w), int(h), (180, 180, 180), 1)
                            draw_label(overlay, f"{name} ({det.conf:.2f})", int(x), int(y),
                                       scale=0.5, thickness=1, text_bgr=(50, 50, 50), bg_alpha=1.0)
                        except Exception:
                            continue
                    # Ampeln
                    for det in all_lights:
                        try:
                            lx, ly, lw, lh = map(int, det.box)
                            draw_box(overlay, lx, ly, lw, lh, (0, 140, 255), 1)
                            draw_label(overlay, f"{pick_label(labels, det.category)} ({det.conf:.2f})", lx, ly,
                                       scale=0.5, thickness=1, text_bgr=(0, 90, 180), bg_alpha=1.0)
                        except Exception:
                            continue
                    # Match-Highlight
                    if chosen:
                        try:
                            det, x, y, w, h, cx, cy, lane_key, stable_ok = chosen
                            draw_box(overlay, int(x), int(y), int(w), int(h), (0, 255, 0), 3)
                            badge = f"{pick_label(labels, det.category)} | lane:{lane_key} | conf:{det.conf:.2f}"
                            draw_label(overlay, badge, int(x), max(0, int(y) - 20),
                                       scale=0.6, thickness=2, text_bgr=(0, 120, 0), bg_alpha=1.0)
                        except Exception:
                            pass
                    # Gematchte Ampel
                    if matched_light is not None:
                        try:
                            lx, ly, lw, lh = map(int, matched_light.box)
                            color = (0, 255, 255) if not used_sticky else (255, 255, 0)
                            draw_box(overlay, lx, ly, lw, lh, color, 3)
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
                            draw_box(overlay, lx, ly, lw, lh, (255, 255, 0), 2)
                            draw_label(overlay, f"ampel (sticky only): {self._sticky_last_phase or 'Unklar'}",
                                       lx, max(0, ly - 20), scale=0.6, thickness=2, text_bgr=(0, 180, 180), bg_alpha=1.0)

                    # Ego-ROI leicht einfärben
                    if roi_poly is not None and roi_poly.shape[0] >= 4:
                        mask = np.zeros_like(draw_surface[:, :, :1])
                        cv2.fillPoly(mask, [roi_poly], 255)
                        draw_surface[:] = np.where(mask == 255,
                                                   (overlay * 0.35 + draw_surface * 0.65).astype(draw_surface.dtype),
                                                   draw_surface)
                    else:
                        cv2.addWeighted(overlay, 0.30, draw_surface, 0.70, 0, draw_surface)

                # === 7) LED steuern
                if self._led_sink is not None:
                    if frame_phase_for_led is not None:
                        try:
                            self._led_sink.apply_phase(frame_phase_for_led)
                        except Exception as e:
                            logging.warning(f"LED sink error: {e}")
                    else:
                        if not self._sticky_active():
                            try:
                                self._led_sink.set_ambient()
                            except Exception as e:
                                logging.warning(f"LED ambient error: {e}")

                # === 8) Sticky ggf. beenden
                self._sticky_end_if_due()

        except Exception as e:
            logging.error(f"draw_callback error: {e}")
            return