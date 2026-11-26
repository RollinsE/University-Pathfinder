import json
import os

import numpy as np
import pytest

from src.constants import CLASS_NAMES, NUM_CLASSES
from src.evaluate import (
    build_leaderboard,
    classification_metrics,
    results_markdown,
    run_evaluate,
    save_metrics,
    save_predictions,
)


def test_metrics_on_a_known_example():
    y_true = [0, 0, 1, 1, 2, 2]
    y_pred = [0, 1, 1, 1, 2, -1]
    m = classification_metrics(y_true, y_pred)
    assert m["accuracy"] == pytest.approx(4 / 6, abs=1e-4)
    assert m["no_prediction"] == 1
    assert m["per_class"][CLASS_NAMES[0]]["recall"] == 0.5
    assert m["per_class"][CLASS_NAMES[1]]["precision"] == pytest.approx(2 / 3, abs=1e-4)
    assert m["confusion"][2][NUM_CLASSES] == 1  # the missed image sits in the "(none)" column
    # macro F1 averages only classes present in the test labels
    f1s = [m["per_class"][CLASS_NAMES[c]]["f1"] for c in range(3)]
    assert m["macro_f1"] == pytest.approx(np.mean(f1s), abs=1e-3)


def test_top3_accuracy():
    scores = np.zeros((2, NUM_CLASSES))
    scores[0, [5, 3, 0]] = [0.5, 0.3, 0.2]  # true 0 is third
    scores[1, [5, 3, 4]] = [0.5, 0.3, 0.2]  # true 1 is missing
    m = classification_metrics([0, 1], [5, 5], scores)
    assert m["top3_accuracy"] == 0.5


def test_predictions_csv(tmp_path):
    scores = np.eye(NUM_CLASSES)[[0, 2]]
    path = save_predictions(str(tmp_path / "p.csv"), ["a.jpg", "b.jpg"], [0, 1], [0, -1], scores)
    lines = open(path).read().splitlines()
    assert lines[0].startswith("image,true,predicted")
    assert "(none)" in lines[2]


def _fake_metrics(run_dir, model, accuracy, detection=None):
    m = classification_metrics([0, 1, 2, 3], [0, 1, 2, 3 if accuracy == 1 else 0])
    m["accuracy"] = accuracy
    m["params_m"] = 3.0
    if detection:
        m["detection"] = detection
    save_metrics(run_dir, model, m)


def test_leaderboard_and_results(run_logger):
    LOG, _, run_dir = run_logger
    _fake_metrics(run_dir, "resnet18", 0.75)
    _fake_metrics(run_dir, "yolo11n", 1.0, {"map50": 0.95, "map50_95": 0.7})
    with open(os.path.join(run_dir, "metrics", "export.json"), "w") as f:
        json.dump({"models": {"yolo11n": {"onnx_mb": 10.2, "cpu_ms": 40.0}}}, f)

    board = run_evaluate(run_dir, LOG)
    assert board.iloc[0]["id"] == "yolo11n"
    assert board.iloc[0]["cpu_ms"] == 40.0
    md = open(os.path.join(run_dir, "metrics", "results.md")).read()
    assert "| YOLO11n | detector | 100.0% (4/4) | 51–100% |" in md
    summary = json.load(open(os.path.join(run_dir, "metrics", "summary.json")))
    assert summary["best_accuracy_model"] == "yolo11n"
    assert "ResNet18" in results_markdown(build_leaderboard(run_dir))


def test_evaluate_without_metrics_fails(run_logger):
    LOG, _, run_dir = run_logger
    with pytest.raises(FileNotFoundError):
        run_evaluate(run_dir, LOG)


def test_top3_gives_no_credit_when_nothing_was_detected():
    scores = np.zeros((3, NUM_CLASSES))
    m = classification_metrics([0, 1, 2], [-1, -1, -1], scores)
    assert m["top3_accuracy"] == 0.0


def test_localisation_metrics():
    from src.evaluate import box_iou_cxcywh, localisation_metrics

    truth = np.array([[0.5, 0.5, 0.4, 0.4], [0.5, 0.5, 0.4, 0.4], [0.5, 0.5, 0.4, 0.4]])
    pred = np.array([[0.5, 0.5, 0.4, 0.4], [0.6, 0.5, 0.4, 0.4], [np.nan] * 4])  # exact, shifted, none
    ious = box_iou_cxcywh(pred[:2], truth[:2])
    assert ious[0] == pytest.approx(1.0) and ious[1] == pytest.approx(0.6)  # 0.12 / 0.2
    m = localisation_metrics([0, 1, 2], [0, 2, -1], pred, truth)
    assert m["mean_iou"] == pytest.approx((1 + 0.6 + 0) / 3, abs=1e-4)
    assert m["iou50"] == pytest.approx(2 / 3, abs=1e-4)
    assert m["named_and_boxed"] == pytest.approx(1 / 3, abs=1e-4)  # second photo: wrong building
