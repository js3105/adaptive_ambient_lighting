# src/detection/arrows.py
import cv2
import numpy as np

# Euer Datensatz: 1=gerade, 2=links, 3=rechts
CLASS_ARROW_STRAIGHT = 1
CLASS_ARROW_LEFT     = 2
CLASS_ARROW_RIGHT    = 3

LANE_LEFT, LANE_STRAIGHT, LANE_RIGHT = "left", "straight", "right"

class ArrowSelector:
    """
    Wählt den relevanten Pfeil innerhalb der Ego-Spur (Trapez),
    bestimmt die Spur (links/mitte/rechts) und matched die passende Ampel.

    NEU: Side-Bias-Matching:
      - Bei linkem Pfeil bevorzugt Ampeln links des Referenzpunkts
      - Bei rechtem Pfeil bevorzugt Ampeln rechts
      - Bei geradem Pfeil bevorzugt Ampeln nahe der Mitte
    """

    def __init__(self):
        # ROI-Trapez in Prozent (an Kamera/Setup anpassbar)
        self.roi_bottom_y = 0.95
        self.roi_top_y    = 0.60
        self.roi_left_bot = 0.33
        self.roi_right_bot= 0.67
        self.roi_left_top = 0.42
        self.roi_right_top= 0.58

        # Buckets über Bildbreite
        self.bucket_left_max  = 0.33
        self.bucket_right_min = 0.66

        # Entprellen
        self._stable_lane_key = None
        self._stable_counter  = 0
        self.STABLE_N         = 3

        # -------- Side-Bias Parameter (neu) --------
        # Referenz für "links/rechts von" — 'arrow', 'lane' oder 'blend'
        self.side_ref_mode = "blend"
        # Blend-Faktor für 'blend': 1.0 = nur Pfeil, 0.0 = nur Lane-Mitte
        self.side_ref_alpha = 0.6
        # Toleranz, wie weit "falsche" Seite noch akzeptiert wird (in normierten X)
        self.side_tol = 0.05
        # Vertikaler Filter: Ampeln müssen oberhalb des Pfeils liegen (Margin in Pixeln)
        self.vert_margin_px = 8

    # ---------- Geometrie ----------
    def ego_lane_roi_polygon(self, frame_shape):
        h, w = frame_shape[:2]
        bl = (int(self.roi_left_bot * w),  int(self.roi_bottom_y * h))  # bottom-left
        br = (int(self.roi_right_bot * w), int(self.roi_bottom_y * h))  # bottom-right
        tr = (int(self.roi_right_top * w), int(self.roi_top_y * h))     # top-right
        tl = (int(self.roi_left_top * w),  int(self.roi_top_y * h))     # top-left
        return np.array([bl, br, tr, tl], dtype=np.int32)

    @staticmethod
    def point_in_poly(pt, poly):
        return cv2.pointPolygonTest(poly, pt, False) >= 0

    @staticmethod
    def box_center_xy(x, y, w, h):
        return (x + 0.5 * w, y + 0.5 * h)

    # ---------- Lane/Bucket ----------
    def lane_from_category_or_pos(self, category: int, x_center_norm: float):
        if category == CLASS_ARROW_LEFT:
            return LANE_LEFT
        if category == CLASS_ARROW_RIGHT:
            return LANE_RIGHT
        if category == CLASS_ARROW_STRAIGHT:
            return LANE_STRAIGHT

        # Fallback: per x-Position bucketen
        if x_center_norm < self.bucket_left_max:
            return LANE_LEFT
        if x_center_norm > self.bucket_right_min:
            return LANE_RIGHT
        return LANE_STRAIGHT

    def bucket_mid_for_lane(self, lane_key):
        low, high = {
            LANE_LEFT:     (0.0, self.bucket_left_max),
            LANE_STRAIGHT: (self.bucket_left_max, self.bucket_right_min),
            LANE_RIGHT:    (self.bucket_right_min, 1.0)
        }[lane_key]
        return 0.5 * (low + high)

    def match_light_to_lane(self, lights, lane_key, frame_width):
        """Alte Methode: nur auf Bucket-Mitte (horizontal) matchen."""
        mid = self.bucket_mid_for_lane(lane_key)
        best, best_score = None, -1.0
        for det in lights:
            x, y, w, h = map(int, det.box)
            x_norm = (x + 0.5 * w) / float(frame_width)
            score = 1.0 - abs(x_norm - mid)
            if score > best_score:
                best_score, best = score, det
        return best

    # ---------- Side-Bias Matching (neu) ----------
    def _ref_x_norm(self, lane_mid_norm, arrow_cx_norm):
        if self.side_ref_mode == "arrow":
            return arrow_cx_norm
        if self.side_ref_mode == "lane":
            return lane_mid_norm
        # blend
        a = float(self.side_ref_alpha)
        return a * arrow_cx_norm + (1.0 - a) * lane_mid_norm

    def match_light_side_biased(self, lights, lane_key, frame_w, frame_h,
                                 arrow_cx, arrow_cy):
        """
        Bevorzuge Ampeln:
          - bei LANE_LEFT  : links vom Referenz-x
          - bei LANE_RIGHT : rechts vom Referenz-x
          - bei LANE_STRAIGHT: nahe am Referenz-x
        Filter: nur Ampeln oberhalb des Pfeils (mit vertikalem Margin).
        Fallback: wenn kein Kandidat passt, nutze match_light_to_lane.
        """
        if not lights:
            return None

        lane_mid = self.bucket_mid_for_lane(lane_key)
        arrow_cx_norm = arrow_cx / float(frame_w)
        ref_x = self._ref_x_norm(lane_mid, arrow_cx_norm)
        v_cut = arrow_cy - self.vert_margin_px

        # Kandidaten vorfiltern (oberhalb des Pfeils)
        candidates = []
        for det in lights:
            x, y, w, h = map(int, det.box)
            cx = x + 0.5 * w
            cy = y + 0.5 * h
            if cy >= v_cut:
                continue  # Ampel zu tief
            cx_norm = cx / float(frame_w)
            dx = cx_norm - ref_x  # negativ = links von Referenz, positiv = rechts

            # Seiten-Bedingung
            side_ok = True
            if lane_key == LANE_LEFT:
                side_ok = (dx <= self.side_tol)  # bevorzugt links; kleine Toleranz nach rechts
            elif lane_key == LANE_RIGHT:
                side_ok = (dx >= -self.side_tol) # bevorzugt rechts; kleine Toleranz nach links
            # LANE_STRAIGHT: kein Seitenfilter

            if not side_ok:
                continue

            # Score:
            # - Distanz-Strafe horizontal (je näher an ref_x, desto höher)
            # - Leichter Bonus, wenn "richtige" Seite (dx<0 bei links, dx>0 bei rechts)
            # - Vertikaler Bonus: je höher (kleiner cy), desto besser (früh sichtbare Ampel)
            dist_score = 1.0 - min(abs(dx) / 0.5, 1.0)           # 0..1
            side_bonus = 0.0
            if lane_key == LANE_LEFT:
                side_bonus = 0.15 * (1.0 if dx < 0 else 0.5)     # rechts neben Grenze schlechter
            elif lane_key == LANE_RIGHT:
                side_bonus = 0.15 * (1.0 if dx > 0 else 0.5)
            elif lane_key == LANE_STRAIGHT:
                side_bonus = 0.10 * (1.0 - min(abs(dx) / 0.33, 1.0))

            vert_score = 1.0 - min((arrow_cy - cy) / float(frame_h), 1.0)  # tiefer = schlechter (0..1)
            score = 0.70 * dist_score + 0.20 * side_bonus + 0.10 * vert_score

            candidates.append((score, det))

        if not candidates:
            # Fallback: alte Bucket-Mitte-Logik
            return self.match_light_to_lane(lights, lane_key, frame_w)

        candidates.sort(key=lambda t: t[0], reverse=True)
        return candidates[0][1]

    # ---------- Auswahl des relevanten Pfeils ----------
    def choose_arrow_and_lane(self, arrows_with_centers, roi_poly, frame_width):
        """
        arrows_with_centers: Liste [(det, x,y,w,h, cx,cy), ...] (cx,cy in Pixel)
        Rückgabe: tuple | None  ->  (det, x,y,w,h, cx,cy, lane_key, stable_ok)
        """
        # Nur Pfeile, deren Mittelpunkt im ROI liegt
        in_lane = []
        for det, x, y, w, h, cx, cy in arrows_with_centers:
            if self.point_in_poly((float(cx), float(cy)), roi_poly):
                in_lane.append((det, x, y, w, h, cx, cy))

        if not in_lane:
            # Reset Stabilität
            self._stable_lane_key = None
            self._stable_counter = 0
            return None

        # „vor mir“ = größtes cy (y wächst nach unten)
        in_lane.sort(key=lambda t: t[6], reverse=True)
        det, x, y, w, h, cx, cy = in_lane[0]

        lane_key = self.lane_from_category_or_pos(int(det.category), cx / float(frame_width))

        # Entprellen
        stable_ok = False
        if self._stable_lane_key == lane_key:
            self._stable_counter += 1
        else:
            self._stable_lane_key = lane_key
            self._stable_counter = 1
    
        if self._stable_counter >= self.STABLE_N:
            stable_ok = True

        return (det, x, y, w, h, cx, cy, lane_key, stable_ok)