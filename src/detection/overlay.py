import cv2

def draw_label(img, text, x, y, *, scale=0.5, thickness=1,
               text_bgr=(0,0,255), bg_alpha=0.30, pad=4):
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    tx, ty = x + 5, y + 15
    cv2.rectangle(img, (tx, ty - th - pad), (tx + tw + pad, ty + base), (255,255,255), cv2.FILLED)
    cv2.putText(img, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, scale, text_bgr, thickness)

def draw_box(img, x, y, w, h, color_bgr, thickness=2):
    cv2.rectangle(img, (x, y), (x + w, y + h), color_bgr, thickness)