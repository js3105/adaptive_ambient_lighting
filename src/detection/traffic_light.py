import cv2
import numpy as np

class TrafficLightPhase:
    ROT, GELB, GRUEN, UNKLAR = "Rot", "Gelb", "Gruen", "Unklar"

class TrafficLightAnalyzer:
    def __init__(self, *, gamma=1.6, clahe_clip=2.0, clahe_grid=8,
                 s_min=60, v_min=60, use_r_dom=True, r_margin=20):
        self.gamma = gamma
        self.clahe_clip = clahe_clip
        self.clahe_grid = clahe_grid
        self.s_min = s_min
        self.v_min = v_min
        self.use_r_dom = use_r_dom
        self.r_margin = r_margin

    def shrink_box(self, x, y, w, h, *, fx=0.18, fy=0.05, w_max=None, h_max=None):
        sx = x + int(w * fx); ex = x + w - int(w * fx)
        sy = y + int(h * fy); ey = y + h - int(h * fy)
        if w_max is not None:
            sx = max(0, min(sx, w_max - 2)); ex = max(sx + 2, min(ex, w_max - 1))
        if h_max is not None:
            sy = max(0, min(sy, h_max - 2)); ey = max(sy + 2, min(ey, h_max - 1))
        return sx, sy, ex - sx, ey - sy

    def _preprocess_rgb(self, img_rgb):
        lut = np.array([((i / 255.0) ** (1.0 / self.gamma)) * 255 for i in range(256)], dtype=np.uint8)
        img_gamma = cv2.LUT(img_rgb, lut)
        hsv = cv2.cvtColor(img_gamma, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        clahe = cv2.createCLAHE(clipLimit=self.clahe_clip, tileGridSize=(self.clahe_grid, self.clahe_grid))
        v = clahe.apply(v)
        return cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2RGB)

    def _masks(self, img_rgb):
        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        gate = cv2.inRange(cv2.merge([h, s, v]), (0, self.s_min, self.v_min), (179, 255, 255))
        red1, red2 = cv2.inRange(h, 0, 10), cv2.inRange(h, 160, 179)
        yellow, green = cv2.inRange(h, 15, 35), cv2.inRange(h, 45, 75)

        mask_red = cv2.bitwise_and(cv2.bitwise_or(red1, red2), gate)
        mask_yellow = cv2.bitwise_and(yellow, gate)
        mask_green = cv2.bitwise_and(green, gate)

        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, k, iterations=1)
        mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_OPEN, k, iterations=1)
        mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, k, iterations=1)

        if self.use_r_dom:
            r, g, b = img_rgb[:, :, 0], img_rgb[:, :, 1], img_rgb[:, :, 2]
            r_dom = ((r.astype(np.int16) > g.astype(np.int16) + self.r_margin) &
                     (r.astype(np.int16) > b.astype(np.int16) + self.r_margin))
            mask_red = cv2.bitwise_and(mask_red, (r_dom.astype(np.uint8) * 255))
        return mask_red, mask_yellow, mask_green

    def phase_from_roi(self, roi_rgb):
        if roi_rgb is None or roi_rgb.size == 0:
            return TrafficLightPhase.UNKLAR
        roi_pp = self._preprocess_rgb(roi_rgb)
        m_red, m_yel, m_grn = self._masks(roi_pp)
        counts = [cv2.countNonZero(m_red), cv2.countNonZero(m_yel), cv2.countNonZero(m_grn)]
        if max(counts) == 0:
            return TrafficLightPhase.UNKLAR
        return [TrafficLightPhase.ROT, TrafficLightPhase.GELB, TrafficLightPhase.GRUEN][int(np.argmax(counts))]