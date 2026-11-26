"""Predictions drawn on the photo, tested with stand-in model outputs (no ONNX models needed)."""

import copy
import json
import os

from PIL import Image

import src.predict as P

DETECTION = {
    "model": "yolo11n",
    "kind": "detector",
    "detections": [
        {"building": "Poole House", "confidence": 0.87, "box": [20, 40, 110, 110]},
        {"building": "Fusion Building", "confidence": 0.41, "box": [115, 70, 150, 110]},
    ],
}
CLASSIFICATION = {
    "model": "mobilenet_v3_large",
    "kind": "classifier",
    "top3": [
        {"building": "Kimmeridge House", "probability": 0.38},
        {"building": "Poole House", "probability": 0.31},
        {"building": "Weymouth House", "probability": 0.12},
    ],
    "box": [30.0, 20.0, 120.0, 100.0],
}


def test_headlines_use_the_app_wording():
    assert P.headline(DETECTION) == "YOLO11n: this looks like Poole House (Confident, 87%)"
    assert P.headline(CLASSIFICATION) == "MobileNetV3-Large: it might be Kimmeridge House (Not sure, 38%)"
    nothing = {"model": "yolo11n", "kind": "detector", "detections": []}
    assert P.headline(nothing) == "YOLO11n: no campus building found"


def test_annotate_draws_on_a_copy():
    photo = Image.new("RGB", (160, 120), (150, 190, 230))
    for result, box in ((DETECTION, DETECTION["detections"][0]["box"]), (CLASSIFICATION, CLASSIFICATION["box"])):
        drawn = P.annotate(photo, result)
        assert drawn.width == photo.width and drawn.height > photo.height  # headline strip on top
        strip = drawn.height - photo.height
        assert drawn.getpixel((5, strip + 115)) == photo.getpixel((5, 115))  # photo below the strip
        left, mid_y = int(box[0]), int((box[1] + box[3]) / 2)
        assert drawn.getpixel((left, strip + mid_y)) != photo.getpixel((left, mid_y))  # the box's left edge
    assert photo.getpixel((20, 40)) == (150, 190, 230)  # original untouched


def test_run_predict_all_models_saves_one_image_each(run_logger, tmp_path, monkeypatch):
    LOG, _, run_dir = run_logger
    web = tmp_path / "web"
    os.makedirs(web / "models")
    for m in ("yolo11n", "mobilenet_v3_large"):  # resnet18 not exported
        (web / "models" / f"{m}.onnx").write_bytes(b"stand-in")
    Image.new("RGB", (160, 120)).save(tmp_path / "my photo.jpg")
    monkeypatch.setattr(
        P,
        "predict_one",
        lambda image, model, web_dir, **kw: copy.deepcopy(DETECTION if model == "yolo11n" else CLASSIFICATION),
    )

    results = P.run_predict(str(tmp_path / "my photo.jpg"), str(web), "all", LOG, run_dir=run_dir)
    assert [r["model"] for r in results] == ["yolo11n", "mobilenet_v3_large"]
    for m in ("yolo11n", "mobilenet_v3_large"):
        assert os.path.exists(os.path.join(run_dir, "predictions", f"my photo_{m}.jpg"))
    with open(os.path.join(run_dir, "logs", "run.log"), encoding="utf-8") as f:
        log = f.read()
    assert "not_exported=resnet18" in log
    assert '"This looks like Poole House (Confident, 87%)"' in log


def test_detector_settings_are_named_in_the_headline_and_file(run_logger, tmp_path, monkeypatch):
    LOG, _, run_dir = run_logger
    web = tmp_path / "web"
    os.makedirs(web / "models")
    (web / "models" / "yolo11n.onnx").write_bytes(b"stand-in")
    manifest = {"models": [{"id": "yolo11n", "input": {"size": 640, "resize": "letterbox"}}]}
    (web / "models" / "manifest.json").write_text(json.dumps(manifest))
    Image.new("RGB", (160, 120)).save(tmp_path / "p.jpg")
    calls = []

    def fake(image, model, web_dir, **kw):
        calls.append(kw)
        return copy.deepcopy(DETECTION)

    monkeypatch.setattr(P, "predict_one", fake)
    photo = str(tmp_path / "p.jpg")

    default = P.run_predict(photo, str(web), "yolo11n", LOG, run_dir=run_dir)[0]
    assert calls[-1] == {"conf": 0.25, "resize": "letterbox"}  # the exported setting
    assert default["annotated"].endswith("p_yolo11n.jpg")
    assert P.headline(default).startswith("YOLO11n: ")

    tried = P.run_predict(photo, str(web), "yolo11n", LOG, run_dir=run_dir, conf=0.1, resize="stretch")[0]
    assert calls[-1] == {"conf": 0.1, "resize": "stretch"}
    assert tried["annotated"].endswith("p_yolo11n_stretch_conf0.1.jpg")
    assert P.headline(tried).startswith("YOLO11n (stretch, conf 0.1): ")


def test_detector_resize_defaults_to_letterbox_without_a_manifest(tmp_path):
    assert P.detector_resize_in_manifest(str(tmp_path), "yolo11n") == "letterbox"


def test_labels_do_not_cover_each_other():
    bounds = (0, 0, 1000, 800)
    first = P.place_label((100, 200, 500, 600), (200, 40), [], bounds)
    assert first == (100, 160)  # just above the box
    taken = [(first[0], first[1], first[0] + 200, first[1] + 40)]
    second = P.place_label((120, 210, 400, 500), (200, 40), taken, bounds)
    assert second == (120, 210)  # inside the top edge instead, clear of the first
    edge = P.place_label((950, 0, 1000, 50), (200, 40), [], bounds)
    assert edge == (800, 0)  # kept inside the photo


def test_long_lines_wrap_between_words():
    lines = P._wrap("   2  Sir Michael Cobham Library 51%", 120, lambda t: 6 * len(t))
    assert all(6 * len(line) <= 120 for line in lines)
    assert " ".join(part.strip() for part in lines) == "2 Sir Michael Cobham Library 51%"
