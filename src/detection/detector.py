import cv2
import numpy as np
from picamera2 import MappedArray
from picamera2.devices import IMX500
from picamera2.devices.imx500 import postprocess_nanodet_detection

class Detection:
    def __init__(self, coords, category, conf, metadata, imx500: IMX500, picam2):
        self.category = category
        self.conf = float(conf)
        # liefert (x, y, w, h) im ISP-Output
        self.box = imx500.convert_inference_coords(coords, metadata, picam2)


class ObjectDetector:
    def __init__(self, imx500: IMX500, intrinsics, picam2):
        self.imx500 = imx500
        self.intrinsics = intrinsics
        self.picam2 = picam2
        self.last_detections = []
        self.TRAFFIC_LIGHT_CLASS_ID = 0
        # Debug/Feintuning für HSV-Trackbars
        self.enable_hsv_debug = True   # auf True setzen, um Trackbars zu aktivieren
        self._hsv_ui_initialized = False

    def _labels(self):
        """Get labels from intrinsics"""
        if self.intrinsics.labels is None:
            return []
        return self.intrinsics.labels
    
    def parse_detections(self, metadata):
        np_outputs = self.imx500.get_outputs(metadata, add_batch=True)
        if np_outputs is None:
            return self.last_detections

        input_w, input_h = self.imx500.get_input_size()
        bbox_normalization = getattr(self.intrinsics, "bbox_normalization", False)
        bbox_order = getattr(self.intrinsics, "bbox_order", "yx")
        threshold = 0.55
        iou = 0.65
        max_dets = 10

        if getattr(self.intrinsics, "postprocess", "") == "nanodet":
            boxes, scores, classes = postprocess_nanodet_detection(
                outputs=np_outputs[0], conf=threshold, iou_thres=iou, max_out_dets=max_dets
            )[0]
            from picamera2.devices.imx500.postprocess import scale_boxes
            boxes = scale_boxes(boxes, 1, 1, input_h, input_w, False, False)  # y0,x0,y1,x1
            boxes = np.array_split(boxes, 4, axis=1)
            boxes = zip(*boxes)
        else:
            boxes, scores, classes = np_outputs[0][0], np_outputs[1][0], np_outputs[2][0]
            if bbox_normalization:
                boxes = boxes / input_h
            if bbox_order == "xy":
                boxes = boxes[:, [1, 0, 3, 2]]  # -> y0, x0, y1, x1
            boxes = np.array_split(boxes, 4, axis=1)
            boxes = zip(*boxes)

        self.last_detections = [
            Detection(box, category, score, metadata, self.imx500, self.picam2)
            for box, score, category in zip(boxes, scores, classes)
            if float(score) > threshold
        ]
        return self.last_detections

    # ---------------- HSV-Trackbar-Debug UI ----------------
    def _init_hsv_debug(self):
        """
        Erstellt ein UI-Fenster mit Trackbars für die HSV-Schwellen.
        - Rot hat zwei Hue-Bereiche (R1 und R2).
        - Für jede Farbe: Hmin/Hmax sowie Smin/Vmin (Obergrenzen S/V = 255).
        """
        cv2.namedWindow("HSV Tuner", cv2.WINDOW_NORMAL)
        # Rot Bereich 1 (z. B. 0-15)
        cv2.createTrackbar("Hmin_R1", "HSV Tuner", 0,   179, lambda v: None)
        cv2.createTrackbar("Hmax_R1", "HSV Tuner", 15,  179, lambda v: None)
        # Rot Bereich 2 (z. B. 165-179)
        cv2.createTrackbar("Hmin_R2", "HSV Tuner", 165, 179, lambda v: None)
        cv2.createTrackbar("Hmax_R2", "HSV Tuner", 179, 179, lambda v: None)
        # Gemeinsame S/V-Minima für Rot
        cv2.createTrackbar("Smin_R",   "HSV Tuner", 150, 255, lambda v: None)
        cv2.createTrackbar("Vmin_R",   "HSV Tuner", 100, 255, lambda v: None)

        # Gelb
        cv2.createTrackbar("Hmin_Y", "HSV Tuner", 20,  179, lambda v: None)
        cv2.createTrackbar("Hmax_Y", "HSV Tuner", 35,  179, lambda v: None)
        cv2.createTrackbar("Smin_Y", "HSV Tuner", 150, 255, lambda v: None)
        cv2.createTrackbar("Vmin_Y", "HSV Tuner", 100, 255, lambda v: None)

        # Grün
        cv2.createTrackbar("Hmin_G", "HSV Tuner", 65,  179, lambda v: None)
        cv2.createTrackbar("Hmax_G", "HSV Tuner", 85,  179, lambda v: None)
        cv2.createTrackbar("Smin_G", "HSV Tuner", 150, 255, lambda v: None)
        cv2.createTrackbar("Vmin_G", "HSV Tuner", 100, 255, lambda v: None)

        # Anzeige-Fenster für ROI/Masks
        cv2.namedWindow("ROI", cv2.WINDOW_NORMAL)
        cv2.namedWindow("mask_red", cv2.WINDOW_NORMAL)
        cv2.namedWindow("mask_yellow", cv2.WINDOW_NORMAL)
        cv2.namedWindow("mask_green", cv2.WINDOW_NORMAL)

        self._hsv_ui_initialized = True

    def _read_hsv_params(self):
        """
        Liest alle Trackbar-Werte und liefert fertige Schwellen-Tupel zurück.
        Rückgabe:
          red_ranges -> [((hmin1,smin,vmin),(hmax1,255,255)), ((hmin2,smin,vmin),(hmax2,255,255))]
          yellow_rng -> ((hminY,sminY,vminY),(hmaxY,255,255))
          green_rng  -> ((hminG,sminG,vminG),(hmaxG,255,255))
        """
        # Rot
        hmin_r1 = cv2.getTrackbarPos("Hmin_R1", "HSV Tuner")
        hmax_r1 = cv2.getTrackbarPos("Hmax_R1", "HSV Tuner")
        hmin_r2 = cv2.getTrackbarPos("Hmin_R2", "HSV Tuner")
        hmax_r2 = cv2.getTrackbarPos("Hmax_R2", "HSV Tuner")
        smin_r  = cv2.getTrackbarPos("Smin_R",   "HSV Tuner")
        vmin_r  = cv2.getTrackbarPos("Vmin_R",   "HSV Tuner")
        red_ranges = [((hmin_r1, smin_r, vmin_r), (hmax_r1, 255, 255)),
                      ((hmin_r2, smin_r, vmin_r), (hmax_r2, 255, 255))]

        # Gelb
        hmin_y = cv2.getTrackbarPos("Hmin_Y", "HSV Tuner")
        hmax_y = cv2.getTrackbarPos("Hmax_Y", "HSV Tuner")
        smin_y = cv2.getTrackbarPos("Smin_Y", "HSV Tuner")
        vmin_y = cv2.getTrackbarPos("Vmin_Y", "HSV Tuner")
        yellow_rng = ((hmin_y, smin_y, vmin_y), (hmax_y, 255, 255))

        # Grün
        hmin_g = cv2.getTrackbarPos("Hmin_G", "HSV Tuner")
        hmax_g = cv2.getTrackbarPos("Hmax_G", "HSV Tuner")
        smin_g = cv2.getTrackbarPos("Smin_G", "HSV Tuner")
        vmin_g = cv2.getTrackbarPos("Vmin_G", "HSV Tuner")
        green_rng = ((hmin_g, smin_g, vmin_g), (hmax_g, 255, 255))

        return red_ranges, yellow_rng, green_rng
    # -------------- Ende HSV-Trackbar-Debug UI --------------

    # --- RAW-HSV-Farblogik (minimal) ---
    def detect_phase_by_hsv(self, roi_bgr):
        """
        Minimal-Variante:
        - ROI in HSV
        - feste Masken für Rot/Gelb/Grün
        - Pixel zählen, größte Farbe gewinnt
        - Optional: HSV-Trackbar-Debug (self.enable_hsv_debug=True)
        """
        if roi_bgr is None or roi_bgr.size == 0:
            return "Unklar"

        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)

        # Trackbar-Debug initialisieren (einmalig)
        if self.enable_hsv_debug and not self._hsv_ui_initialized:
            self._init_hsv_debug()

        if self.enable_hsv_debug:
            # Werte aus UI lesen
            red_ranges, yellow_rng, green_rng = self._read_hsv_params()
            # Masken nach UI
            mask_red1 = cv2.inRange(hsv, red_ranges[0][0], red_ranges[0][1])
            mask_red2 = cv2.inRange(hsv, red_ranges[1][0], red_ranges[1][1])
            mask_red  = cv2.bitwise_or(mask_red1, mask_red2)
            mask_yellow = cv2.inRange(hsv, yellow_rng[0], yellow_rng[1])
            mask_green  = cv2.inRange(hsv, green_rng[0],  green_rng[1])

            # Anzeige
            cv2.imshow("ROI", roi_bgr)
            cv2.imshow("mask_red", mask_red)
            cv2.imshow("mask_yellow", mask_yellow)
            cv2.imshow("mask_green", mask_green)
            cv2.waitKey(1)
        else:
            # Feste Standardbereiche (wie zuvor)
            mask_red1 = cv2.inRange(hsv, (0,   150, 100), (15, 255, 255))
            mask_red2 = cv2.inRange(hsv, (165, 150, 100), (179, 255, 255))
            mask_red  = cv2.bitwise_or(mask_red1, mask_red2)
            mask_yellow = cv2.inRange(hsv, (20, 150, 100), (35, 255, 255))
            mask_green  = cv2.inRange(hsv, (65, 150, 100), (85, 255, 255))

        red_pixels    = cv2.countNonZero(mask_red)
        yellow_pixels = cv2.countNonZero(mask_yellow)
        green_pixels  = cv2.countNonZero(mask_green)

        max_pixels = max(red_pixels, yellow_pixels, green_pixels)
        if max_pixels == 0:
            return "Unklar"
        if max_pixels == red_pixels:
            return "Rot"
        elif max_pixels == yellow_pixels:
            return "Gelb"
        elif max_pixels == green_pixels:
            return "Gruen"
        return "Unklar"
    # --- Ende RAW-HSV-Farblogik ---

    def draw_callback(self, request, stream="main"):
        if not self.last_detections:
            return
        labels = self._labels()
        with MappedArray(request, stream) as m:
            for det in self.last_detections:
                try:
                    x, y, w, h = map(int, det.box)
                    # Ensure coordinates are within array bounds
                    h_max, w_max = m.array.shape[:2]
                    x = max(0, min(x, w_max-1))
                    y = max(0, min(y, h_max-1))
                    w = min(w, w_max-x)
                    h = min(h, h_max-y)
                    name = labels[int(det.category)] if 0 <= int(det.category) < len(labels) else f"Class {int(det.category)}"
                
                    # Für Ampeln: Phasenerkennung durchführen (RAW-HSV)
                    if int(det.category) == self.TRAFFIC_LIGHT_CLASS_ID:
                        roi = m.array[y:y+h, x:x+w]
                        if roi.size > 0:
                            phase = self.detect_phase_by_hsv(roi)
                            name = f"{name} ({phase})"

                    label = f"{name} ({det.conf:.2f})"

                    # Text-Hintergrund halbtransparent
                    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    tx, ty = x + 5, y + 15
                    overlay = m.array.copy()
                    cv2.rectangle(overlay, (tx, ty - th), (tx + tw, ty + baseline), (255, 255, 255), cv2.FILLED)
                    cv2.addWeighted(overlay, 0.30, m.array, 0.70, 0, m.array)

                    # Text und Rahmen zeichnen
                    text_color = (0, 0, 255)
                    box_color = (0, 255, 255) if int(det.category) == self.TRAFFIC_LIGHT_CLASS_ID else (0, 255, 0)  # 3-kanalig
                    cv2.putText(m.array, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
                    cv2.rectangle(m.array, (x, y), (x + w, y + h), box_color, 2)

                except (ValueError, IndexError) as e:
                    print(f"Error processing detection: {e}")
                    continue