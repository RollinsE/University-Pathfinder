"""Short end-to-end training runs on the synthetic dataset (skipped without the libraries)."""

import os

import pytest

from src.constants import NUM_CLASSES


def test_classifier_trains_evaluates_and_exports(demo_dataset, run_logger, tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from src.classify import checkpoint_path, evaluate_classifier, train_classifier

    LOG, _, run_dir = run_logger
    train_classifier(
        "mobilenet_v3_large", demo_dataset, run_dir, LOG, epochs=1, batch_size=8, pretrained=False, num_workers=0
    )
    assert os.path.exists(checkpoint_path(run_dir, "mobilenet_v3_large"))
    metrics = evaluate_classifier("mobilenet_v3_large", demo_dataset, run_dir, LOG, batch_size=8, num_workers=0)
    assert metrics["images"] > 0
    assert len(metrics["confusion"]) == NUM_CLASSES
    assert 0 <= metrics["localisation"]["mean_iou"] <= 1

    pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")
    from src.export import export_classifier

    info = export_classifier("mobilenet_v3_large", run_dir, str(tmp_path), LOG, check_image=None)
    assert info["max_probability_diff"] < 1e-3 and info["max_box_diff"] < 1e-3
    assert os.path.exists(tmp_path / "mobilenet_v3_large.onnx")


def test_detector_trains_and_evaluates(demo_dataset, run_logger):
    pytest.importorskip("ultralytics")
    if os.environ.get("PATHFINDER_TEST_YOLO") != "1":
        pytest.skip("set PATHFINDER_TEST_YOLO=1 to run the detector smoke test")
    from src.detect import evaluate_detector, train_detector

    LOG, _, run_dir = run_logger
    train_detector(demo_dataset, run_dir, LOG, epochs=1, imgsz=160, batch=8, pretrained=False, workers=0, plots=False)
    metrics = evaluate_detector(demo_dataset, run_dir, LOG, imgsz=160, batch=8, plots=False)
    assert "detection" in metrics
