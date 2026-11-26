import json
import os

import pytest

from src.constants import CLASS_NAMES
from src.evaluate import classification_metrics, save_metrics
from src.export import build_manifest, export_samples
from src.figures import run_figures


def test_manifest_lists_models_with_preprocessing(run_logger):
    _, run_id, run_dir = run_logger
    m = classification_metrics([0, 1], [0, 1])
    m["detection"] = {"map50": 0.9, "map50_95": 0.6}
    save_metrics(run_dir, "yolo11n", m)
    info = {"yolo11n": {"onnx_mb": 10.0, "cpu_ms": 50.0}, "mobilenet_v3_large": {"onnx_mb": 17.0, "cpu_ms": 12.0}}

    manifest = build_manifest(run_id, ["mobilenet_v3_large", "yolo11n"], info, run_dir)
    assert manifest["primary"] == "yolo11n"
    assert [c["name"] for c in manifest["classes"]] == CLASS_NAMES
    by_id = {m["id"]: m for m in manifest["models"]}
    assert by_id["mobilenet_v3_large"]["input"] == {
        "size": 224,
        "resize": "stretch",
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    }
    assert by_id["mobilenet_v3_large"]["outputs"]["box"].startswith("normalised")
    assert by_id["yolo11n"]["kind"] == "detector"
    assert by_id["yolo11n"]["metrics"]["map50"] == 0.9
    assert by_id["mobilenet_v3_large"]["metrics"] == {}


def test_samples_cover_each_building_first(demo_dataset, run_logger, tmp_path):
    LOG, _, _ = run_logger
    entries = export_samples(demo_dataset, str(tmp_path / "samples"), 8, LOG)
    assert len(entries) == 8
    assert sorted(e["building"] for e in entries) == sorted(CLASS_NAMES)
    listed = json.load(open(tmp_path / "samples" / "samples.json"))
    assert listed == entries
    assert all(os.path.exists(tmp_path / "samples" / e["file"]) for e in entries)


def _fake_run(run_dir):
    """Metrics, classifier histories and a YOLO results.csv, as a finished run leaves them."""
    for model, pred in (("resnet18", [0, 1, 2, 2]), ("yolo11n", [0, 1, 2, -1])):
        save_metrics(run_dir, model, classification_metrics([0, 1, 2, 2], pred))
    folder = os.path.join(run_dir, "artifacts", "resnet18")
    os.makedirs(folder)
    with open(os.path.join(folder, "history.csv"), "w") as f:
        f.write("epoch,lr,train_loss,train_accuracy,train_box_iou,val_loss,val_accuracy,val_box_iou\n")
        f.write(
            "1,0.001,1.2,0.5,0.3,1.0,0.6,0.4\n2,0.0005,0.8,0.7,0.6,0.7,0.8,0.7\n3,0.0002,0.6,0.9,0.7,0.72,0.8,0.75\n"
        )
    folder = os.path.join(run_dir, "artifacts", "yolo11n", "train")
    os.makedirs(folder)
    with open(os.path.join(folder, "results.csv"), "w") as f:
        f.write(
            "  epoch,  train/box_loss, train/cls_loss, train/dfl_loss, metrics/mAP50(B), metrics/mAP50-95(B),"
            " val/box_loss, val/cls_loss, val/dfl_loss\n"
        )
        f.write("1,1,1,1,0.2,0.1,1.2,1.2,1.2\n2,0.8,0.8,0.8,0.7,0.5,1,1,1\n3,0.7,0.7,0.7,0.6,0.45,1.1,1.1,1.1\n")


def test_figure_specs_cover_every_model(run_logger):
    from src.figures import build_specs

    _, _, run_dir = run_logger
    _fake_run(run_dir)
    specs = build_specs(run_dir, "light")
    assert list(specs) == ["model_comparison", "training_curves", "confusion_yolo11n", "confusion_resnet18"]

    curves = specs["training_curves"]
    names = {trace.get("name") for trace in curves["data"]}
    assert {"ResNet18", "ResNet18 box IoU", "YOLO11n mAP50", "YOLO11n mAP50-95", "YOLO11n validation loss"} <= names
    assert "YOLO11n 2 of 3" in curves["layout"]["title"]["text"]  # best mAP50-95 at epoch 2
    assert "ResNet18 2" in curves["layout"]["title"]["text"]

    comparison = specs["model_comparison"]
    assert "3/4" in " ".join(t["text"][0] for t in comparison["data"] if t.get("mode") == "text")
    assert "uncertainty of a 4-photo test set" in comparison["layout"]["title"]["text"]

    yolo = specs["confusion_yolo11n"]
    assert yolo["data"][0]["x"][-1] == "(none)"  # it missed one photo entirely
    assert "1 of 4 test photos wrong" in yolo["layout"]["title"]["text"]
    assert specs["confusion_resnet18"]["data"][0]["x"][-1] != "(none)"


def test_dark_specs_use_dark_surfaces(run_logger):
    from src.figures import THEMES, build_specs

    _, _, run_dir = run_logger
    _fake_run(run_dir)
    for spec in build_specs(run_dir, "dark").values():
        assert spec["layout"]["paper_bgcolor"] == THEMES["dark"]["surface"]


def test_figures_are_written_when_plotly_is_available(run_logger, tmp_path):
    pytest.importorskip("plotly")

    LOG, _, run_dir = run_logger
    _fake_run(run_dir)
    written = run_figures(run_dir, LOG)
    assert "training_curves.html" in written
    assert os.path.exists(written["training_curves.html"])


def test_figures_skip_cleanly_without_plotly(run_logger, monkeypatch):
    import builtins

    LOG, _, run_dir = run_logger
    _fake_run(run_dir)
    real_import = builtins.__import__

    def no_plotly(name, *args, **kwargs):
        if name.startswith("plotly"):
            raise ImportError("no plotly")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_plotly)
    assert run_figures(run_dir, LOG) == {}
    with open(os.path.join(run_dir, "logs", "run.log"), encoding="utf-8") as f:
        assert "plotly is not installed" in f.read()
