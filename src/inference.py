"""ONNX inference with numpy and onnxruntime only (no torch or Ultralytics needed).

These functions define the preprocessing and post-processing the exported models expect.
webapp/inference.js implements the same steps in JavaScript, and tests check that the two
agree on shared fixtures.

Classify-and-locate models: resize the whole photo to 224 x 224, scale to 0..1, ImageNet
normalisation, NCHW -> softmax over 8 buildings, and one box as normalised (cx, cy, w, h).

Detector: fit the photo into 640x640, either letterboxed (shape kept, grey padding) or stretched,
scale to 0..1, NCHW -> output (1, 4 + 8, N) with box centre/size in input pixels and per-class
scores -> confidence filter -> per-class non-maximum suppression -> at most one box per building
(each building exists once on campus) -> boxes mapped back to the original image.
"""

from __future__ import annotations

import numpy as np

from src.constants import (
    CLASSIFIER_INPUT,
    DETECTOR_CONF,
    DETECTOR_IMGSZ,
    DETECTOR_IOU,
    DETECTOR_RESIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    LETTERBOX_FILL,
)

# --------------------------------------------------------------------------- classifier


def preprocess_classifier(image, size: int = CLASSIFIER_INPUT) -> np.ndarray:
    """PIL RGB image -> float32 array of shape (1, 3, size, size).

    The whole photo is resized to a square (stretched, as the dataset's photos were), so the
    model sees all of it and its box maps straight back onto the photo.
    """
    from PIL import Image

    resized = image.resize((size, size), Image.BILINEAR)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    array = (array - np.array(IMAGENET_MEAN, dtype=np.float32)) / np.array(IMAGENET_STD, dtype=np.float32)
    return array.transpose(2, 0, 1)[None].astype(np.float32)


def box_to_pixels(box_cxcywh, width: int, height: int) -> list[float]:
    """Normalised (cx, cy, w, h) -> [x0, y0, x1, y1] in photo pixels, clipped to the photo."""
    cx, cy, w, h = (float(v) for v in box_cxcywh)
    x0, x1 = (cx - w / 2) * width, (cx + w / 2) * width
    y0, y1 = (cy - h / 2) * height, (cy + h / 2) * height
    return [min(max(x0, 0), width), min(max(y0, 0), height), min(max(x1, 0), width), min(max(y1, 0), height)]


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


# --------------------------------------------------------------------------- detector


def letterbox(image, size: int = DETECTOR_IMGSZ):
    """PIL RGB image -> ((1, 3, size, size) float32 array, scale, (pad_x, pad_y))."""
    from PIL import Image

    width, height = image.size
    scale = min(size / width, size / height)
    new_w, new_h = round(width * scale), round(height * scale)
    pad_x, pad_y = (size - new_w) // 2, (size - new_h) // 2
    canvas = Image.new("RGB", (size, size), (LETTERBOX_FILL,) * 3)
    canvas.paste(image.resize((new_w, new_h), Image.BILINEAR), (pad_x, pad_y))
    array = np.asarray(canvas, dtype=np.float32) / 255.0
    return array.transpose(2, 0, 1)[None].astype(np.float32), scale, (pad_x, pad_y)


def stretch(image, size: int = DETECTOR_IMGSZ):
    """PIL RGB image -> ((1, 3, size, size) float32 array, (scale_x, scale_y), (0, 0))."""
    from PIL import Image

    width, height = image.size
    array = np.asarray(image.resize((size, size), Image.BILINEAR), dtype=np.float32) / 255.0
    return array.transpose(2, 0, 1)[None].astype(np.float32), (size / width, size / height), (0, 0)


def fit_for_detector(image, size: int = DETECTOR_IMGSZ, resize: str = DETECTOR_RESIZE):
    """Letterbox or stretch the photo for the detector. Returns (tensor, scale, pad) for decoding."""
    if resize == "stretch":
        return stretch(image, size)
    if resize == "letterbox":
        return letterbox(image, size)
    raise ValueError(f"Unknown detector resize mode {resize!r}; use 'letterbox' or 'stretch'.")


def iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """IoU of one xyxy box against an (n, 4) array of xyxy boxes."""
    x0 = np.maximum(box[0], boxes[:, 0])
    y0 = np.maximum(box[1], boxes[:, 1])
    x1 = np.minimum(box[2], boxes[:, 2])
    y1 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area = (box[2] - box[0]) * (box[3] - box[1])
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / np.maximum(area + areas - inter, 1e-9)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    """Greedy non-maximum suppression. Returns kept indices, highest score first."""
    order = np.argsort(-scores, kind="stable")
    keep = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        overlaps = iou(boxes[i], boxes[order[1:]])
        order = order[1:][overlaps <= iou_threshold]
    return keep


def decode_detections(
    output: np.ndarray,
    scale,
    pad: tuple[int, int],
    image_size: tuple[int, int],
    conf: float = DETECTOR_CONF,
    iou_threshold: float = DETECTOR_IOU,
    one_per_building: bool = True,
) -> list[dict]:
    """Raw detector output (1, 4 + classes, N) -> detections in original image pixels.

    `scale` is one number for a letterboxed photo, or (scale_x, scale_y) for a stretched one.
    Each detection is {"class_id", "confidence", "box": [x0, y0, x1, y1]}, most confident first.
    With `one_per_building`, only the most confident box for each building is kept: a second
    box with the same name is always at least partly wrong.
    """
    scale_x, scale_y = scale if isinstance(scale, (tuple, list)) else (scale, scale)
    preds = np.asarray(output)[0].T  # (N, 4 + classes)
    class_scores = preds[:, 4:]
    class_ids = class_scores.argmax(1)
    confidences = class_scores[np.arange(len(preds)), class_ids]
    mask = confidences >= conf
    if not mask.any():
        return []

    cx, cy, w, h = preds[mask, 0], preds[mask, 1], preds[mask, 2], preds[mask, 3]
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad[0]) / scale_x
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad[1]) / scale_y
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, image_size[0])
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, image_size[1])
    confidences, class_ids = confidences[mask], class_ids[mask]

    detections = []
    for c in np.unique(class_ids):
        idx = np.where(class_ids == c)[0]
        kept = nms(boxes[idx], confidences[idx], iou_threshold)
        for k in kept[:1] if one_per_building else kept:
            j = idx[k]
            detections.append(
                {
                    "class_id": int(c),
                    "confidence": float(confidences[j]),
                    "box": [float(v) for v in boxes[j]],
                }
            )
    return sorted(detections, key=lambda d: -d["confidence"])


# --------------------------------------------------------------------------- sessions


def open_session(path: str):
    import onnxruntime as ort

    return ort.InferenceSession(path, providers=["CPUExecutionProvider"])


def run_classifier(session, image) -> tuple[np.ndarray, np.ndarray]:
    """(probabilities over the buildings, box as normalised cx, cy, w, h) for one PIL image."""
    name = session.get_inputs()[0].name
    outputs = dict(
        zip(
            [o.name for o in session.get_outputs()],
            session.run(None, {name: preprocess_classifier(image)}),
            strict=True,
        )
    )
    return softmax(outputs["logits"])[0], np.asarray(outputs["box"])[0]


def run_detector(
    session,
    image,
    conf: float = DETECTOR_CONF,
    iou_threshold: float = DETECTOR_IOU,
    resize: str = DETECTOR_RESIZE,
    one_per_building: bool = True,
) -> list[dict]:
    name = session.get_inputs()[0].name
    size = session.get_inputs()[0].shape[-1]
    size = size if isinstance(size, int) else DETECTOR_IMGSZ
    tensor, scale, pad = fit_for_detector(image, size, resize)
    output = session.run(None, {name: tensor})[0]
    return decode_detections(output, scale, pad, image.size, conf, iou_threshold, one_per_building)
