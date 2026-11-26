"""YOLO per-epoch logging, tested with stand-ins (no Ultralytics needed)."""

import os

import numpy as np

from src.detect import _EpochLogger, _log_training_curve


def _log_text(run_dir):
    with open(os.path.join(run_dir, "logs", "run.log"), encoding="utf-8") as f:
        return f.read()


class FakeTrainer:
    epoch = 4
    epochs = 150
    tloss = np.array([1.0, 2.0, 0.5])
    metrics = {"metrics/mAP50(B)": 0.81234, "metrics/mAP50-95(B)": 0.51234}


def test_epoch_logger_writes_one_line_per_epoch(run_logger):
    LOG, _, run_dir = run_logger
    logger = _EpochLogger(LOG, "Detector:yolo11n")
    logger.train_epoch_done(FakeTrainer())
    logger(FakeTrainer())
    assert logger.logged == 1
    assert "[EVAL] component=Detector:yolo11n epoch=5/150 train_loss=3.5 val_map50=0.8123" in _log_text(run_dir)


def test_final_revalidation_is_not_logged_as_an_epoch(run_logger):
    """After training, Ultralytics fires on_fit_epoch_end again with no training epoch before it."""
    LOG, _, run_dir = run_logger
    logger = _EpochLogger(LOG, "Detector:yolo11n")
    logger.train_epoch_done(FakeTrainer())
    logger(FakeTrainer())  # epoch 5
    logger(FakeTrainer())  # the final re-validation
    assert logger.logged == 1
    assert _log_text(run_dir).count("[EVAL]") == 1


def test_epoch_logger_accepts_losses_as_a_dict(run_logger):
    """Ultralytics 8.4 gives the training losses as a dict."""
    LOG, _, run_dir = run_logger

    class DictLossTrainer(FakeTrainer):
        tloss = {"box_loss": np.array(1.0), "cls_loss": np.array(2.0), "dfl_loss": 0.5}

    logger = _EpochLogger(LOG, "Detector:yolo11n")
    logger.train_epoch_done(DictLossTrainer())
    logger(DictLossTrainer())
    assert logger.logged == 1
    assert "epoch=5/150 train_loss=3.5" in _log_text(run_dir)


def test_epoch_logger_warns_once_and_never_raises(run_logger):
    LOG, _, run_dir = run_logger
    logger = _EpochLogger(LOG, "Detector:yolo11n")

    class Broken:
        epoch = "not a number"

    for _ in range(2):
        logger.train_epoch_done(Broken())
        logger(Broken())
    assert logger.logged == 0
    assert _log_text(run_dir).count("epoch_logging_error=") == 1


def test_training_curve_backfill_from_results_csv(run_logger, tmp_path):
    LOG, _, run_dir = run_logger
    os.makedirs(tmp_path / "train")
    (tmp_path / "train" / "results.csv").write_text(
        "epoch,train/box_loss,train/cls_loss,train/dfl_loss,metrics/mAP50(B),metrics/mAP50-95(B)\n"
        "1,1.0,1.0,1.0,0.2,0.1\n"
        "2,0.5,0.5,0.5,0.6,0.4\n"
    )
    _log_training_curve(str(tmp_path), "Detector:yolo11n", LOG, backfill=True)
    text = _log_text(run_dir)
    assert "[EVAL] component=Detector:yolo11n epoch=2 train_loss=1.5 val_map50=0.6 val_map50_95=0.4" in text
    assert "best_epoch=2" in text
