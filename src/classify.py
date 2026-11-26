"""Classify-and-locate models: ResNet18 (baseline) and MobileNetV3-Large (lightweight).

Each is an ImageNet-pretrained torchvision backbone with two small heads:

* a class head: which of the 8 buildings the photo shows;
* a box head: one box round that building, as (centre x, centre y, width, height), each 0..1
  of the photo's width or height.

This is single-object localisation: one name and one box per photo. Each photo's target is the
building with the largest labelled box (see src/datasets.py). The detector (src/detect.py) is
the model that finds every building in a photo.

Both models get the same recipe, so the comparison is fair:

* input: the whole photo resized to 224 x 224. The dataset's photos were stretched to squares
  when it was made, so photos are stretched the same way for training, testing and the web app;
* loss: cross-entropy with label smoothing for the name, plus --box_weight x (L1 + generalised
  IoU) for the box;
* AdamW, cosine learning-rate decay; augmentation that moves the box with the photo (small
  shifts, scaling and rotation) plus colour jitter for day and night;
* the checkpoint kept is the epoch with the best validation accuracy (ties: higher box overlap);
  training stops early when validation loss has not improved for --patience epochs;
* the full training state is saved after every epoch, so an interrupted run resumes.
"""

from __future__ import annotations

import csv
import json
import math
import os
import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.constants import (
    CLASS_NAMES,
    CLASSIFIER_INPUT,
    IMAGENET_MEAN,
    IMAGENET_STD,
    MODEL_LABELS,
    NUM_CLASSES,
)
from src.datasets import class_counts, image_level_targets
from src.evaluate import box_iou_cxcywh, classification_metrics, localisation_metrics, save_metrics, save_predictions
from src.logger import component_stage, log_stage, log_step
from src.progress import progress

# Format tag stored in every checkpoint, checked when a checkpoint is loaded or reused.
CHECKPOINT_FORMAT = "classify_and_locate_v1"

# --------------------------------------------------------------------------- setup


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device(requested: str = "auto") -> torch.device:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"--device {requested} needs a GPU, but none is available.")
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class ClassifyAndLocate(nn.Module):
    """Backbone features -> (class logits, box as sigmoid cx, cy, w, h)."""

    def __init__(self, arch: str, num_classes: int = NUM_CLASSES, pretrained: bool = True):
        super().__init__()
        from torchvision import models

        if arch == "resnet18":
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            net = models.resnet18(weights=weights)
            features = net.fc.in_features
            net.fc = nn.Identity()
            self.body = net
        elif arch == "mobilenet_v3_large":
            weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
            net = models.mobilenet_v3_large(weights=weights)
            features = net.classifier[-1].in_features  # keep the pretrained 960 -> 1280 layer
            net.classifier[-1] = nn.Identity()
            self.body = net
        else:
            raise ValueError(f"Unknown classifier {arch!r}.")
        self.arch = arch
        self.class_head = nn.Linear(features, num_classes)
        self.box_head = nn.Sequential(nn.Linear(features, 256), nn.ReLU(inplace=True), nn.Linear(256, 4))

    def forward(self, x):
        f = self.body(x)
        return self.class_head(f), torch.sigmoid(self.box_head(f))


def build_classifier(arch: str, num_classes: int = NUM_CLASSES, pretrained: bool = True) -> nn.Module:
    return ClassifyAndLocate(arch, num_classes, pretrained)


def build_transforms(train: bool):
    """Image and box transforms (torchvision v2, which moves boxes with the image)."""
    from torchvision.transforms import v2 as T

    steps = [T.ToImage(), T.Resize((CLASSIFIER_INPUT, CLASSIFIER_INPUT), antialias=True)]
    if train:
        steps += [
            T.RandomAffine(degrees=6, translate=(0.06, 0.06), scale=(0.85, 1.1)),
            T.ColorJitter(brightness=0.4, contrast=0.3, saturation=0.3, hue=0.03),
        ]
    steps += [T.ClampBoundingBoxes(), T.ToDtype(torch.float32, scale=True), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return T.Compose(steps)


def load_rgb(path: str):
    """Open an image upright (EXIF orientation applied) as RGB."""
    from PIL import Image, ImageOps

    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def _cxcywh_to_xyxy(box, width: float, height: float) -> list[float]:
    cx, cy, w, h = box
    return [(cx - w / 2) * width, (cy - h / 2) * height, (cx + w / 2) * width, (cy + h / 2) * height]


class ImageListDataset(Dataset):
    """(image path, class id, box) targets as (image tensor, class id, box tensor).

    Boxes go in and out as normalised (cx, cy, w, h); the transform moves them with the image.
    """

    def __init__(self, targets: list[tuple[str, int, tuple]], transform):
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        from torchvision import tv_tensors

        path, label, box = self.targets[index]
        image = load_rgb(path)
        width, height = image.size
        boxes = tv_tensors.BoundingBoxes(
            [_cxcywh_to_xyxy(box, width, height)], format="XYXY", canvas_size=(height, width)
        )
        image, boxes = self.transform(image, boxes)
        x0, y0, x1, y1 = (boxes[0] / CLASSIFIER_INPUT).tolist()
        target = torch.tensor([(x0 + x1) / 2, (y0 + y1) / 2, max(x1 - x0, 1e-3), max(y1 - y0, 1e-3)])
        return image, label, target.clamp(0, 1)


def make_loader(targets, train: bool, batch_size: int, num_workers: int, seed: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        ImageListDataset(targets, build_transforms(train)),
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
        persistent_workers=num_workers > 0,
    )


def count_parameters(model: nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6


def checkpoint_path(run_dir: str, arch: str) -> str:
    return os.path.join(run_dir, "artifacts", arch, "best.pt")


def load_classifier(path: str, device: torch.device | str = "cpu") -> tuple[nn.Module, dict]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"{path} is not a classify-and-locate checkpoint. Retrain it with --overwrite.")
    model = build_classifier(checkpoint["arch"], len(checkpoint["class_names"]), pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    return model, checkpoint


def _has_checkpoint_format(path: str) -> bool:
    try:
        return torch.load(path, map_location="cpu", weights_only=False).get("format") == CHECKPOINT_FORMAT
    except Exception:
        return False


# --------------------------------------------------------------------------- training


class ClassifyAndLocateLoss(nn.Module):
    """Cross-entropy for the name + box_weight x (L1 + generalised IoU) for the box."""

    def __init__(self, label_smoothing: float = 0.1, box_weight: float = 2.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.box_weight = box_weight

    def forward(self, logits, boxes, labels, targets):
        from torchvision.ops import box_convert, generalized_box_iou_loss

        l1 = (boxes - targets).abs().mean()
        giou = generalized_box_iou_loss(
            box_convert(boxes, "cxcywh", "xyxy"), box_convert(targets, "cxcywh", "xyxy"), reduction="mean"
        )
        return self.ce(logits, labels) + self.box_weight * (l1 + giou)


def _run_epoch(model, loader, criterion, device, optimizer=None, desc: str = "") -> tuple[float, float, float]:
    """One pass over loader. Trains when an optimizer is given. Returns (loss, accuracy, mean box IoU)."""
    training = optimizer is not None
    model.train(training)
    total_loss, correct, iou_sum, seen = 0.0, 0, 0.0, 0
    with torch.set_grad_enabled(training):
        bar = progress(loader, desc=desc, unit="batch", leave=False)
        for images, labels, targets in bar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits, boxes = model(images)
            loss = criterion(logits, boxes, labels, targets)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            n = labels.size(0)
            total_loss += loss.item() * n
            correct += (logits.argmax(1) == labels).sum().item()
            iou_sum += float(box_iou_cxcywh(boxes.detach().cpu().numpy(), targets.cpu().numpy()).sum())
            seen += n
            bar.set_postfix(loss=f"{total_loss / seen:.3f}", acc=f"{correct / seen:.3f}", iou=f"{iou_sum / seen:.2f}")
    seen = max(seen, 1)
    return total_loss / seen, correct / seen, iou_sum / seen


def train_classifier(
    arch: str,
    prepared_dir: str,
    run_dir: str,
    LOG,
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    patience: int = 8,
    label_smoothing: float = 0.1,
    box_weight: float = 2.0,
    pretrained: bool = True,
    num_workers: int = 2,
    seed: int = 42,
    device: str = "auto",
    overwrite: bool = False,
) -> str:
    """Fine-tune one classify-and-locate model. Returns the checkpoint path.

    A finished model (complete.json beside best.pt) is reused. An interrupted one resumes from
    last.pt, which holds the model, optimiser, scheduler and training history after every epoch.
    Data augmentation after a resume is random again, so a resumed run is equivalent to, but not
    bit-for-bit identical with, an uninterrupted one.
    """
    component = f"Classifier:{arch}"
    ckpt_path = checkpoint_path(run_dir, arch)
    out_dir = os.path.dirname(ckpt_path)
    last_path = os.path.join(out_dir, "last.pt")
    done_marker = os.path.join(out_dir, "complete.json")
    label = MODEL_LABELS[arch]

    if overwrite:
        for path in (ckpt_path, last_path, done_marker):
            if os.path.exists(path):
                os.remove(path)
    elif os.path.exists(done_marker) and os.path.exists(ckpt_path):
        if _has_checkpoint_format(ckpt_path):
            log_step(LOG, component, f"{label} already trained in this run, reusing it (--overwrite retrains)")
            return ckpt_path
        log_stage(LOG, "WARN", component, note="saved model is not a classify-and-locate checkpoint, retraining it")
        for path in (ckpt_path, last_path, done_marker):
            if os.path.exists(path):
                os.remove(path)
    if os.path.exists(last_path) and not _has_checkpoint_format(last_path):
        os.remove(last_path)  # an interrupted checkpoint in another format cannot be resumed

    dev = pick_device(device)
    resuming = os.path.exists(last_path)
    with component_stage(
        LOG,
        component,
        f"{'Resuming' if resuming else 'Training'} {label}",
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        device=dev.type,
        pretrained=pretrained,
    ):
        set_seed(seed)
        log_step(LOG, component, "Loading train and validation images")
        train_samples = image_level_targets(prepared_dir, "train")
        valid_samples = image_level_targets(prepared_dir, "valid")
        if not train_samples or not valid_samples:
            raise ValueError(f"No labelled images found under {prepared_dir}.")
        log_stage(LOG, "INFO", component, train_images=len(train_samples), valid_images=len(valid_samples))
        per_class = class_counts([(p, c) for p, c, _ in train_samples])
        log_stage(LOG, "INFO", component, file_only=True, train_per_class=json.dumps(per_class))

        train_loader = make_loader(train_samples, True, batch_size, num_workers, seed)
        valid_loader = make_loader(valid_samples, False, batch_size, num_workers, seed)

        weights = "ImageNet weights" if pretrained else "random weights"
        log_step(LOG, component, f"Building {label} from {weights} with a building-name head and a box head")
        model = build_classifier(arch, NUM_CLASSES, pretrained=pretrained).to(dev)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = ClassifyAndLocateLoss(label_smoothing=label_smoothing, box_weight=box_weight)

        os.makedirs(out_dir, exist_ok=True)
        state = {
            "epoch": 0,
            "best_acc": -1.0,
            "best_iou": -1.0,
            "best_loss": math.inf,
            "best_epoch": 0,
            "lowest_val_loss": math.inf,
            "epochs_without_improvement": 0,
            "stopped_early": False,
            "history": [],
            "epochs_planned": epochs,
        }
        if resuming:
            saved = torch.load(last_path, map_location=dev, weights_only=False)
            if saved.get("arch") != arch:
                raise ValueError(
                    f"{last_path} belongs to {saved.get('arch')}, not {arch}. Remove it or use --overwrite."
                )
            model.load_state_dict(saved["model"])
            optimizer.load_state_dict(saved["optimizer"])
            scheduler.load_state_dict(saved["scheduler"])
            state.update(saved["state"])
            if state["epochs_planned"] != epochs:
                log_stage(
                    LOG,
                    "WARN",
                    component,
                    resumed_run_planned_epochs=state["epochs_planned"],
                    epochs_now=epochs,
                    note="the learning-rate schedule keeps the length the run started with",
                )
            log_step(
                LOG,
                component,
                f"Resuming interrupted training from epoch {state['epoch'] + 1}",
                best_epoch_so_far=state["best_epoch"],
                best_val_accuracy=round(state["best_acc"], 4),
            )

        started = time.time()
        first_epoch = state["epoch"] + 1
        if state["stopped_early"] or first_epoch > epochs:
            log_step(LOG, component, "Training had already finished before the interruption")
        else:
            log_step(
                LOG,
                component,
                f"Training epochs {first_epoch} to {epochs}",
                device=dev.type,
                patience=patience,
            )

        stop_after = os.environ.get("PATHFINDER_STOP_AFTER_EPOCH")  # test hook for the resume check
        for epoch in range(first_epoch, epochs + 1):
            if state["stopped_early"]:
                break
            current_lr = optimizer.param_groups[0]["lr"]
            train_loss, train_acc, train_iou = _run_epoch(
                model, train_loader, criterion, dev, optimizer, desc=f"{label} epoch {epoch}/{epochs}"
            )
            val_loss, val_acc, val_iou = _run_epoch(model, valid_loader, criterion, dev, desc=f"{label} validating")
            scheduler.step()

            state["history"].append(
                {
                    "epoch": epoch,
                    "lr": current_lr,
                    "train_loss": train_loss,
                    "train_accuracy": train_acc,
                    "train_box_iou": train_iou,
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                    "val_box_iou": val_iou,
                }
            )
            log_stage(
                LOG,
                "EVAL",
                component,
                epoch=f"{epoch}/{epochs}",
                train_loss=round(train_loss, 4),
                train_accuracy=round(train_acc, 4),
                val_loss=round(val_loss, 4),
                val_accuracy=round(val_acc, 4),
                val_box_iou=round(val_iou, 4),
            )

            if val_acc > state["best_acc"] or (val_acc == state["best_acc"] and val_iou > state["best_iou"]):
                state.update(best_acc=val_acc, best_iou=val_iou, best_loss=val_loss, best_epoch=epoch)
                _save_atomic(
                    {
                        "format": CHECKPOINT_FORMAT,
                        "arch": arch,
                        "state_dict": model.state_dict(),
                        "class_names": CLASS_NAMES,
                        "epoch": epoch,
                        "val_accuracy": val_acc,
                        "val_loss": val_loss,
                        "val_box_iou": val_iou,
                        "box_weight": box_weight,
                        "input_size": CLASSIFIER_INPUT,
                        "resize_mode": "stretch",
                        "mean": IMAGENET_MEAN,
                        "std": IMAGENET_STD,
                    },
                    ckpt_path,
                )
                log_stage(LOG, "CHECKPOINT", component, epoch=epoch, path=ckpt_path)
                log_step(
                    LOG,
                    component,
                    f"New best model saved at epoch {epoch}",
                    val_accuracy=round(val_acc, 4),
                    val_box_iou=round(val_iou, 4),
                )

            if val_loss < state["lowest_val_loss"] - 1e-4:
                state.update(lowest_val_loss=val_loss, epochs_without_improvement=0)
            else:
                state["epochs_without_improvement"] += 1
                if state["epochs_without_improvement"] >= patience:
                    state["stopped_early"] = True
                    log_step(LOG, component, f"Stopping early: validation loss has not improved for {patience} epochs")

            state["epoch"] = epoch
            _save_atomic(
                {
                    "format": CHECKPOINT_FORMAT,
                    "arch": arch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "state": state,
                },
                last_path,
            )
            log_stage(LOG, "CHECKPOINT", component, epoch=epoch, path=last_path, file_only=True)
            if stop_after and epoch == int(stop_after):
                raise RuntimeError(f"Stopped after epoch {epoch} on purpose (PATHFINDER_STOP_AFTER_EPOCH).")

        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Training ended without a best checkpoint at {ckpt_path}.")

        log_step(LOG, component, "Saving training history")
        history = state["history"]
        history_path = os.path.join(out_dir, "history.csv")
        with open(history_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
        log_stage(LOG, "ARTIFACT", component, path=history_path)

        with open(done_marker, "w", encoding="utf-8") as f:
            json.dump({"epochs_run": state["epoch"], "best_epoch": state["best_epoch"]}, f, indent=2)
        if os.path.exists(last_path):
            os.remove(last_path)  # training is complete; best.pt is what is kept
        log_stage(
            LOG,
            "RESULT",
            component,
            best_epoch=state["best_epoch"],
            val_accuracy=round(state["best_acc"], 4),
            val_box_iou=round(state["best_iou"], 4),
            val_loss=round(state["best_loss"], 4),
            epochs_run=state["epoch"],
            minutes_this_session=round((time.time() - started) / 60, 1),
        )
    return ckpt_path


def _save_atomic(obj, path: str) -> None:
    """torch.save to a temporary file, then rename, so a disconnect never leaves half a checkpoint."""
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


# --------------------------------------------------------------------------- evaluation


@torch.no_grad()
def predict_outputs(model, targets, batch_size: int, num_workers: int, device, desc: str):
    """Softmax probabilities and predicted boxes (normalised cx, cy, w, h) for targets, in order."""
    loader = DataLoader(
        ImageListDataset(targets, build_transforms(False)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    model.eval()
    probs, boxes = [], []
    for images, _, _ in progress(loader, desc=desc, unit="batch"):
        logits, box = model(images.to(device))
        probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        boxes.append(box.cpu().numpy())
    if not probs:
        return np.zeros((0, NUM_CLASSES)), np.zeros((0, 4))
    return np.concatenate(probs), np.concatenate(boxes)


def evaluate_classifier(
    arch: str,
    prepared_dir: str,
    run_dir: str,
    LOG,
    batch_size: int = 32,
    num_workers: int = 2,
    device: str = "auto",
) -> dict:
    """Score a trained model on the test split: the building's name and its box.

    Writes metrics JSON (including "localisation") and a predictions CSV.
    """
    component = f"Classifier:{arch}"
    dev = pick_device(device)
    label = MODEL_LABELS[arch]
    with component_stage(LOG, f"{component}:test", f"Testing {label} on the held-out test photos"):
        log_step(LOG, component, "Loading the best checkpoint")
        model, checkpoint = load_classifier(checkpoint_path(run_dir, arch), dev)
        targets = image_level_targets(prepared_dir, "test")
        log_step(LOG, component, "Scoring test photos", images=len(targets), best_epoch=checkpoint["epoch"])
        scores, boxes = predict_outputs(model, targets, batch_size, num_workers, dev, desc=f"{label} test")
        y_true = [c for _, c, _ in targets]
        y_pred = scores.argmax(1).tolist()
        true_boxes = np.array([b for _, _, b in targets], dtype=float)

        metrics = classification_metrics(y_true, y_pred, scores)
        metrics["localisation"] = localisation_metrics(y_true, y_pred, boxes, true_boxes)
        metrics.update(
            {
                "model": arch,
                "kind": "classifier",
                "best_epoch": checkpoint["epoch"],
                "val_accuracy": round(float(checkpoint["val_accuracy"]), 4),
                "params_m": round(count_parameters(model), 2),
            }
        )
        log_step(LOG, component, "Writing test metrics and predictions")
        path = save_metrics(run_dir, arch, metrics)
        pred_path = save_predictions(
            os.path.join(run_dir, "metrics", f"{arch}_test_predictions.csv"),
            [p for p, _, _ in targets],
            y_true,
            y_pred,
            scores,
            ious=box_iou_cxcywh(boxes, true_boxes),
        )
        for p in (path, pred_path):
            log_stage(LOG, "ARTIFACT", component, path=p)
        loc = metrics["localisation"]
        log_stage(
            LOG,
            "METRICS",
            component,
            split="test",
            images=metrics["images"],
            accuracy=metrics["accuracy"],
            top3_accuracy=metrics["top3_accuracy"],
            macro_f1=metrics["macro_f1"],
            mean_box_iou=loc["mean_iou"],
            named_and_boxed_accuracy=loc["named_and_boxed"],
        )
    return metrics
