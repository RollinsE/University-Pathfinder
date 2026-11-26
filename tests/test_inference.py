import numpy as np
import pytest

from src.constants import IMAGENET_MEAN, IMAGENET_STD
from src.inference import (
    decode_detections,
    fit_for_detector,
    letterbox,
    nms,
    preprocess_classifier,
    softmax,
    stretch,
)


def test_softmax():
    p = softmax(np.array([[1.0, 3.0, 2.0]]))
    assert p.sum() == pytest.approx(1.0)
    assert p.argmax() == 1


def test_classifier_preprocessing_shape_and_normalisation():  # whole photo, any aspect ratio
    from PIL import Image

    image = Image.new("RGB", (400, 300), (255, 255, 255))
    tensor = preprocess_classifier(image)
    assert tensor.shape == (1, 3, 224, 224)
    expected = (1 - IMAGENET_MEAN[0]) / IMAGENET_STD[0]
    assert tensor[0, 0].mean() == pytest.approx(expected, abs=1e-4)


def test_letterbox_geometry():
    from PIL import Image

    tensor, scale, pad = letterbox(Image.new("RGB", (1280, 640)), 640)
    assert tensor.shape == (1, 3, 640, 640)
    assert scale == 0.5
    assert pad == (0, 160)
    assert tensor[0, 0, 0, 0] == pytest.approx(114 / 255)


def test_stretch_fills_the_square_and_scales_each_axis():
    from PIL import Image

    tensor, scale, pad = stretch(Image.new("RGB", (1280, 640), (255, 255, 255)), 640)
    assert tensor.shape == (1, 3, 640, 640)
    assert scale == (0.5, 1.0) and pad == (0, 0)
    assert tensor.min() == pytest.approx(1.0)  # no grey padding anywhere


def test_fit_for_detector_rejects_unknown_modes():
    from PIL import Image

    with pytest.raises(ValueError, match="resize mode"):
        fit_for_detector(Image.new("RGB", (64, 64)), 64, "crop")


def test_nms():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], dtype=float)
    assert nms(boxes, np.array([0.9, 0.8, 0.7]), 0.45) == [0, 2]


def test_decode_maps_boxes_back():
    anchors = 3
    out = np.zeros((1, 6, anchors), dtype=np.float32)
    out[0, :, 0] = [320, 320, 100, 50, 0.1, 0.9]
    out[0, :, 1] = [100, 100, 10, 10, 0.2, 0.1]
    out[0, :, 2] = [100, 300, 40, 40, 0.6, 0.3]
    dets = decode_detections(out, 0.5, (0, 160), (1280, 640), conf=0.25, iou_threshold=0.45)
    assert [d["class_id"] for d in dets] == [1, 0]
    assert [round(v) for v in dets[0]["box"]] == [540, 270, 740, 370]


def test_decode_maps_stretched_boxes_back():
    out = np.zeros((1, 6, 1), dtype=np.float32)
    out[0, :, 0] = [320, 320, 100, 50, 0.1, 0.9]
    dets = decode_detections(out, (0.5, 1.0), (0, 0), (1280, 640))
    assert [round(v) for v in dets[0]["box"]] == [540, 295, 740, 345]


def test_decode_keeps_one_box_per_building():
    """Two far-apart boxes with the same name survive NMS; only the most confident is kept."""
    out = np.zeros((1, 6, 3), dtype=np.float32)
    out[0, :, 0] = [100, 100, 80, 80, 0.0, 0.6]
    out[0, :, 1] = [500, 500, 80, 80, 0.0, 0.9]
    out[0, :, 2] = [300, 300, 80, 80, 0.7, 0.0]
    dets = decode_detections(out, 1.0, (0, 0), (640, 640))
    assert [(d["class_id"], round(d["confidence"], 1)) for d in dets] == [(1, 0.9), (0, 0.7)]
    every = decode_detections(out, 1.0, (0, 0), (640, 640), one_per_building=False)
    assert [d["class_id"] for d in every] == [1, 0, 1]


def test_decode_with_nothing_above_threshold():
    assert decode_detections(np.zeros((1, 12, 5)), 1.0, (0, 0), (640, 640)) == []


def test_classifier_sees_the_whole_photo_and_its_box_maps_back():
    """Paint one quarter of the photo: it must land in the same quarter of the model's input."""
    from PIL import Image, ImageDraw

    from src.inference import box_to_pixels

    image = Image.new("RGB", (1600, 1200), (0, 0, 255))
    ImageDraw.Draw(image).rectangle((800, 0, 1600, 600), fill=(255, 0, 0))  # top-right quarter red
    red = preprocess_classifier(image)[0, 0]
    assert red[:100, 120:].mean() > 1.5 and red[120:, :100].mean() < 0  # top-right red, bottom-left not
    assert box_to_pixels([0.75, 0.25, 0.5, 0.5], 1600, 1200) == [800.0, 0.0, 1600.0, 600.0]
    assert box_to_pixels([0.05, 0.5, 0.4, 0.2], 100, 100) == pytest.approx([0, 40, 25, 60])
