"""The web app's JavaScript decoder must give the same detections as src/inference.py."""

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from src.inference import decode_detections

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
@pytest.mark.parametrize("resize", ["letterbox", "stretch"])
@pytest.mark.parametrize("one_per_building", [True, False])
def test_javascript_and_python_decoders_agree(tmp_path, resize, one_per_building):
    rng = np.random.default_rng(0)
    num_classes, anchors, size = 8, 400, 640
    image_w, image_h = 1600, 1200
    out = np.zeros((1, 4 + num_classes, anchors), dtype=np.float32)
    out[0, 0] = rng.uniform(0, size, anchors)
    out[0, 1] = rng.uniform(80, size - 80, anchors)
    out[0, 2] = rng.uniform(20, 300, anchors)
    out[0, 3] = rng.uniform(20, 300, anchors)
    out[0, 4:] = rng.uniform(0, 1, (num_classes, anchors)) ** 6  # mostly low scores

    if resize == "letterbox":
        scale = min(size / image_w, size / image_h)
        pad = ((size - round(image_w * scale)) // 2, (size - round(image_h * scale)) // 2)
    else:
        scale, pad = (size / image_w, size / image_h), (0, 0)
    expected = decode_detections(
        out, scale, pad, (image_w, image_h), conf=0.25, iou_threshold=0.45, one_per_building=one_per_building
    )
    assert expected, "fixture should produce some detections"

    fixture = tmp_path / "fixture.json"
    fixture.write_text(
        json.dumps(
            {
                "data": out.ravel().tolist(),
                "num_classes": num_classes,
                "anchors": anchors,
                "size": size,
                "image_w": image_w,
                "image_h": image_h,
                "conf": 0.25,
                "iou": 0.45,
                "resize": resize,
                "one_per_building": one_per_building,
            }
        )
    )
    result = subprocess.run(
        ["node", str(ROOT / "webapp" / "tests" / "decode_fixture.mjs"), str(fixture)],
        capture_output=True,
        text=True,
        check=True,
    )
    got = json.loads(result.stdout)
    assert got["geometry"]["padX"] == pad[0] and got["geometry"]["padY"] == pad[1]
    assert [d["classId"] for d in got["detections"]] == [d["class_id"] for d in expected]
    for js, py in zip(got["detections"], expected, strict=True):
        assert js["confidence"] == pytest.approx(py["confidence"], abs=1e-6)
        assert js["box"] == pytest.approx(py["box"], abs=1e-3)
