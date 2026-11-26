"""Building detector: Ultralytics YOLO11n.

Unlike the classifiers, the detector finds every building in a photo and draws a box round
each, so it copes with photos that show more than one building. It is trained on the full
prepared dataset, background images included.

Two sets of test numbers are recorded:

* detection metrics on the test split: mAP50, mAP50-95, precision and recall, overall and per
  building (Ultralytics' validator);
* image-level accuracy on the same test images and labels as the classifiers, taking the
  detector's most confident box as its answer (no box counts as wrong).

Training resumes from last.pt if a run was interrupted.
"""

from __future__ import annotations

import json
import os

import numpy as np

from src.constants import CLASS_NAMES, DETECTOR_CONF, DETECTOR_IMGSZ, DETECTOR_IOU, MODEL_LABELS, NUM_CLASSES
from src.datasets import image_level_targets, write_data_yaml
from src.evaluate import box_iou_cxcywh, classification_metrics, localisation_metrics, save_metrics, save_predictions
from src.logger import component_stage, log_stage, log_step
from src.progress import progress


def detector_dir(run_dir: str, model_name: str) -> str:
    return os.path.abspath(os.path.join(run_dir, "artifacts", model_name))


def best_weights_path(run_dir: str, model_name: str) -> str:
    return os.path.join(detector_dir(run_dir, model_name), "train", "weights", "best.pt")


def _device_arg(device: str):
    """Ultralytics device: None for automatic, "0" for cuda or cuda:0, otherwise as given."""
    if device == "auto":
        return None
    if device == "cuda":
        return "0"
    if device.startswith("cuda:"):
        return device.split(":", 1)[1]
    return device


def _total_loss(losses) -> float | None:
    """Sum of the training losses, whether Ultralytics gives an array/tensor or a dict of them."""
    if losses is None:
        return None
    values = losses.values() if isinstance(losses, dict) else [losses]
    return round(sum(float(v.sum()) if hasattr(v, "sum") else float(v) for v in values), 4)


class _EpochLogger:
    """Ultralytics callback: one [EVAL] line in run.log per finished epoch.

    Ultralytics also fires on_fit_epoch_end once more after training, when it re-validates the
    best model; that call follows no training epoch, so it is skipped. Any error is reported once
    as a warning and never stops training. If no epoch was logged (a callback API change, say),
    the curve is filled in from results.csv after training.

    Register both hooks:
        model.add_callback("on_train_epoch_end", logger.train_epoch_done)
        model.add_callback("on_fit_epoch_end", logger)
    """

    def __init__(self, LOG, component: str):
        self.LOG, self.component = LOG, component
        self.logged = 0
        self.failed = False
        self.pending = False

    def train_epoch_done(self, trainer) -> None:
        self.pending = True

    def attach(self, model) -> None:
        model.add_callback("on_train_epoch_end", self.train_epoch_done)
        model.add_callback("on_fit_epoch_end", self)

    def __call__(self, trainer) -> None:
        if not self.pending:  # the final re-validation, not a training epoch
            return
        self.pending = False
        try:
            metrics = getattr(trainer, "metrics", None) or {}
            epoch = int(getattr(trainer, "epoch", -1)) + 1
            total = getattr(trainer, "epochs", None)
            fields = {
                "epoch": f"{epoch}/{total}" if total else epoch,
                "train_loss": _total_loss(getattr(trainer, "tloss", None)),
            }
            for key, name in (("metrics/mAP50(B)", "val_map50"), ("metrics/mAP50-95(B)", "val_map50_95")):
                if key in metrics:
                    fields[name] = round(float(metrics[key]), 4)
            log_stage(self.LOG, "EVAL", self.component, **fields)
            self.logged += 1
        except Exception as exc:
            if not self.failed:
                self.failed = True
                log_stage(self.LOG, "WARN", self.component, epoch_logging_error=f"{type(exc).__name__}: {exc}")


def train_detector(
    prepared_dir: str,
    run_dir: str,
    LOG,
    model_name: str = "yolo11n",
    epochs: int = 150,
    imgsz: int = DETECTOR_IMGSZ,
    batch: int = 16,
    patience: int = 30,
    pretrained: bool = True,
    workers: int = 2,
    seed: int = 42,
    device: str = "auto",
    plots: bool = True,
    overwrite: bool = False,
) -> str:
    """Train the detector. Returns the path to best.pt (reused when training already finished)."""
    from ultralytics import YOLO

    component = f"Detector:{model_name}"
    project = detector_dir(run_dir, model_name)
    weights_dir = os.path.join(project, "train", "weights")
    best, last = os.path.join(weights_dir, "best.pt"), os.path.join(weights_dir, "last.pt")
    done_marker = os.path.join(project, "train", "complete.json")

    label = MODEL_LABELS[model_name]
    if os.path.exists(done_marker) and os.path.exists(best) and not overwrite:
        log_step(LOG, component, f"{label} already trained in this run, reusing it (--overwrite retrains)")
        return best

    data_yaml = write_data_yaml(prepared_dir)
    resume = os.path.exists(last) and not os.path.exists(done_marker) and not overwrite
    if resume:
        # Ultralytics resumes with the settings the run started with and ignores new ones,
        # so log those and say so when they differ from this command's.
        saved = _checkpoint_train_args(last)
        requested = {"epochs": epochs, "patience": patience, "imgsz": imgsz, "batch": batch}
        changed = {k: (saved[k], v) for k, v in requested.items() if k in saved and saved[k] != v}
        if changed:
            log_stage(
                LOG,
                "WARN",
                component,
                note="resuming keeps the settings this run started with; use --overwrite to restart with new ones",
                **{k: f"{old} (asked for {new})" for k, (old, new) in changed.items()},
            )
        epochs = saved.get("epochs", epochs)
        patience = saved.get("patience", patience)
        imgsz = saved.get("imgsz", imgsz)
        batch = saved.get("batch", batch)
    with component_stage(
        LOG,
        component,
        f"Training {label}",
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        patience=patience,
        pretrained=pretrained,
        resume=resume,
        device=device,
    ):
        if resume:
            log_step(LOG, component, "Resuming the interrupted training run", checkpoint=last)
            model = YOLO(last)
            epoch_logger = _EpochLogger(LOG, component)
            epoch_logger.attach(model)
            log_step(LOG, component, "Handing over to Ultralytics, which prints its own progress table")
            model.train(resume=True)
        else:
            source = f"{model_name}.pt" if pretrained else f"{model_name}.yaml"
            weights = "COCO weights" if pretrained else "random weights"
            log_step(LOG, component, f"Building {label} from {weights}", source=source)
            model = YOLO(source)
            epoch_logger = _EpochLogger(LOG, component)
            epoch_logger.attach(model)
            log_step(
                LOG,
                component,
                f"Training for up to {epochs} epochs; Ultralytics prints its own progress table",
                patience=patience,
            )
            model.train(
                data=data_yaml,
                epochs=epochs,
                imgsz=imgsz,
                batch=batch,
                patience=patience,
                workers=workers,
                seed=seed,
                deterministic=True,
                device=_device_arg(device),
                project=project,
                name="train",
                exist_ok=True,
                plots=plots,
                verbose=False,
            )
        if not os.path.exists(best):
            raise FileNotFoundError(f"Training finished without writing {best}.")
        with open(done_marker, "w", encoding="utf-8") as f:
            json.dump({"epochs": epochs, "patience": patience, "imgsz": imgsz, "data": data_yaml}, f, indent=2)
        log_stage(LOG, "CHECKPOINT", component, path=best)
        log_step(LOG, component, "Reading the training curve")
        _log_training_curve(project, component, LOG, backfill=epoch_logger.logged == 0)
    return best


def _checkpoint_train_args(path: str) -> dict:
    """The training settings stored in an Ultralytics checkpoint ({} if they can't be read)."""
    try:
        import torch

        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        args = ckpt.get("train_args") or {}
        return {k: args[k] for k in ("epochs", "patience", "imgsz", "batch") if k in args}
    except Exception:  # an unreadable checkpoint is reported by Ultralytics itself
        return {}


def _log_training_curve(project: str, component: str, LOG, backfill: bool = False) -> None:
    """Log the best validation epoch from Ultralytics' results.csv.

    backfill: also write one [EVAL] line per epoch (to run.log only), for when the live
    per-epoch callback logged nothing.
    """
    import pandas as pd

    path = os.path.join(project, "train", "results.csv")
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    column = "metrics/mAP50-95(B)"
    if backfill and len(df):
        loss_columns = [c for c in ("train/box_loss", "train/cls_loss", "train/dfl_loss") if c in df.columns]
        for _, r in df.iterrows():
            log_stage(
                LOG,
                "EVAL",
                component,
                file_only=True,
                epoch=int(r["epoch"]),
                train_loss=round(float(sum(r[c] for c in loss_columns)), 4) if loss_columns else None,
                val_map50=round(float(r["metrics/mAP50(B)"]), 4) if "metrics/mAP50(B)" in df.columns else None,
                val_map50_95=round(float(r[column]), 4) if column in df.columns else None,
            )
        log_step(LOG, component, "Per-epoch lines filled in from results.csv", epochs=len(df))
    if column in df.columns and len(df):
        row = df.loc[df[column].idxmax()]
        log_stage(
            LOG,
            "RESULT",
            component,
            best_epoch=int(row["epoch"]),
            val_map50=round(float(row.get("metrics/mAP50(B)", np.nan)), 4),
            val_map50_95=round(float(row[column]), 4),
            epochs_run=len(df),
        )
    log_stage(LOG, "ARTIFACT", component, path=path)


def _per_class(values, class_index) -> dict:
    out = {name: None for name in CLASS_NAMES}
    if values is None or class_index is None:
        return out
    for value, c in zip(np.asarray(values).ravel(), np.asarray(class_index).ravel(), strict=False):
        out[CLASS_NAMES[int(c)]] = round(float(value), 4)
    return out


def top_box(result) -> np.ndarray:
    """The most confident box as normalised (cx, cy, w, h), or NaNs when nothing was found."""
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return np.full(4, np.nan)
    best = int(boxes.conf.cpu().numpy().argmax())
    x0, y0, x1, y1 = boxes.xyxyn.cpu().numpy()[best]
    return np.array([(x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0])


def image_scores(result) -> np.ndarray:
    """Per-building score for one image: the confidence of its most confident box (0 if none)."""
    scores = np.zeros(NUM_CLASSES)
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return scores
    classes = boxes.cls.cpu().numpy().astype(int)
    confs = boxes.conf.cpu().numpy()
    for c, conf in zip(classes, confs, strict=True):
        if 0 <= c < NUM_CLASSES:
            scores[c] = max(scores[c], float(conf))
    return scores


def evaluate_detector(
    prepared_dir: str,
    run_dir: str,
    LOG,
    model_name: str = "yolo11n",
    imgsz: int = DETECTOR_IMGSZ,
    batch: int = 16,
    device: str = "auto",
    plots: bool = True,
) -> dict:
    """Detection metrics and image-level accuracy on the test split."""
    from ultralytics import YOLO

    component = f"Detector:{model_name}"
    label = MODEL_LABELS[model_name]
    with component_stage(LOG, f"{component}:test", f"Testing {label} on the held-out test photos"):
        log_step(LOG, component, "Loading the best checkpoint")
        model = YOLO(best_weights_path(run_dir, model_name))
        data_yaml = write_data_yaml(prepared_dir)
        log_step(LOG, component, "Measuring detection accuracy (mAP) on the test split")

        result = model.val(
            data=data_yaml,
            split="test",
            imgsz=imgsz,
            batch=batch,
            device=_device_arg(device),
            project=detector_dir(run_dir, model_name),
            name="test",
            exist_ok=True,
            plots=plots,
            verbose=False,
        )
        box = result.box
        class_index = getattr(box, "ap_class_index", None)
        detection = {
            "map50": round(float(box.map50), 4),
            "map50_95": round(float(box.map), 4),
            "precision": round(float(box.mp), 4),
            "recall": round(float(box.mr), 4),
            "per_class_map50": _per_class(getattr(box, "ap50", None), class_index),
            "per_class_map50_95": _per_class(getattr(box, "ap", None), class_index),
        }

        targets = image_level_targets(prepared_dir, "test")
        log_step(LOG, component, "Scoring test photos one by one, as the classifiers were", images=len(targets))
        scores, boxes = [], []
        for path, _, _ in progress(targets, desc=f"{MODEL_LABELS[model_name]} test images", unit="img"):
            prediction = model.predict(
                path, imgsz=imgsz, conf=DETECTOR_CONF, iou=DETECTOR_IOU, device=_device_arg(device), verbose=False
            )[0]
            scores.append(image_scores(prediction))
            boxes.append(top_box(prediction))
        scores, boxes = np.array(scores), np.array(boxes)
        y_true = [c for _, c, _ in targets]
        y_pred = [int(s.argmax()) if s.max() > 0 else -1 for s in scores]
        true_boxes = np.array([b for _, _, b in targets], dtype=float)

        metrics = classification_metrics(y_true, y_pred, scores)
        # The detector's answer for a photo is its most confident box, compared with the largest
        # labelled box: the same single-box measure the classify-and-locate models get.
        metrics["localisation"] = localisation_metrics(y_true, y_pred, boxes, true_boxes)
        params = sum(p.numel() for p in model.model.parameters()) / 1e6
        metrics.update({"model": model_name, "kind": "detector", "detection": detection, "params_m": round(params, 2)})

        log_step(LOG, component, "Writing test metrics and predictions")
        path = save_metrics(run_dir, model_name, metrics)
        pred_path = save_predictions(
            os.path.join(run_dir, "metrics", f"{model_name}_test_predictions.csv"),
            [p for p, _, _ in targets],
            y_true,
            y_pred,
            scores,
            ious=np.nan_to_num(box_iou_cxcywh(boxes, true_boxes)),
        )
        for p in (path, pred_path):
            log_stage(LOG, "ARTIFACT", component, path=p)
        log_stage(
            LOG,
            "METRICS",
            component,
            split="test",
            map50=detection["map50"],
            map50_95=detection["map50_95"],
            precision=detection["precision"],
            recall=detection["recall"],
        )
        log_stage(
            LOG,
            "METRICS",
            component,
            split="test",
            images=metrics["images"],
            accuracy=metrics["accuracy"],
            top3_accuracy=metrics["top3_accuracy"],
            macro_f1=metrics["macro_f1"],
            no_detection=metrics["no_prediction"],
            mean_box_iou=metrics["localisation"]["mean_iou"],
            named_and_boxed_accuracy=metrics["localisation"]["named_and_boxed"],
        )
    return metrics
