import numpy as np
from picamera2.devices.imx500 import postprocess_nanodet_detection
from .common import Detection

def parse_detections(imx500, intrinsics, picam2, metadata, last_detections, *,
                     threshold=0.55, iou=0.65, max_dets=10):
    np_outputs = imx500.get_outputs(metadata, add_batch=True)
    if np_outputs is None:
        return last_detections

    input_w, input_h = imx500.get_input_size()
    bbox_normalization = getattr(intrinsics, "bbox_normalization", False)
    bbox_order = getattr(intrinsics, "bbox_order", "yx")

    if getattr(intrinsics, "postprocess", "") == "nanodet":
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
            boxes = boxes[:, [1, 0, 3, 2]]
        boxes = np.array_split(boxes, 4, axis=1)
        boxes = zip(*boxes)

    return [
        Detection(box, category, score, metadata, imx500, picam2)
        for box, score, category in zip(boxes, scores, classes) if float(score) > threshold
    ]