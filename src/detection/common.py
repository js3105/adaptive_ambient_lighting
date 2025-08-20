class Detection:
    def __init__(self, coords, category, conf, metadata, imx500, picam2):
        self.category = int(category)
        self.conf = float(conf)
        self.box = imx500.convert_inference_coords(coords, metadata, picam2)

def pick_label(labels, category):
    try:
        i = int(category)
        if 0 <= i < len(labels):
            return str(labels[i])
    except Exception:
        pass
    return f"Class {int(category)}"