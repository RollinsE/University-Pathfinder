import json
import os
import zipfile

import pytest

import main
from src.progress import extract_zip


def test_demo_defaults_are_small_and_offline():
    args = main.parse_args(["--mode", "all", "--demo"])
    assert args.epochs == 2 and args.yolo_imgsz == 160 and args.no_pretrained
    args = main.parse_args(["--mode", "all", "--demo", "--epochs", "5"])
    assert args.epochs == 5
    args = main.parse_args(["--mode", "all"])
    assert args.epochs == 30 and args.yolo_imgsz == 640 and not args.no_pretrained


def test_unknown_model_is_rejected():
    with pytest.raises(ValueError):
        main.parse_models("resnet18,vgg16")


def test_acquire_demo_through_the_cli(tmp_path):
    run_dir = tmp_path / "runs" / "run_test"
    code = main.main(["--mode", "acquire", "--demo", "--run_dir", str(run_dir), "--data_dir", str(tmp_path / "data")])
    assert code == 0
    record = json.load(open(run_dir / "metrics" / "dataset.json"))
    assert record["source"] == "demo" and record["images"] > 0
    assert "[DONE] component=Pipeline" in open(run_dir / "logs" / "run.log").read()


def test_a_failed_stage_returns_non_zero_and_logs_it(tmp_path):
    run_dir = tmp_path / "runs" / "run_fail"
    assert main.main(["--mode", "evaluate", "--run_dir", str(run_dir)]) == 1
    assert "[FAIL] component=Pipeline" in open(run_dir / "logs" / "run.log").read()


def test_extract_zip_rejects_paths_outside_the_target(tmp_path):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("../escape.txt", "x")
    with pytest.raises(ValueError):
        extract_zip(str(archive), str(tmp_path / "out"))
    assert not os.path.exists(tmp_path / "escape.txt")
