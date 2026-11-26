"""Metrics shared by every model, and the leaderboard that compares them.

All three models are scored on the same test images with the same image-level label (the
building with the largest box), so their accuracy figures are directly comparable. The
detector also gets its own detection metrics (mAP, precision, recall) on the full test split.
"""

from __future__ import annotations

import csv
import json
import os

import numpy as np
import pandas as pd

from src.console import print_table
from src.constants import ALL_MODELS, CLASS_NAMES, MODEL_LABELS, MODEL_ROLES, NUM_CLASSES
from src.logger import component_stage, log_stage, log_step

NO_PREDICTION = -1


def box_iou_cxcywh(a, b) -> np.ndarray:
    """IoU of matching rows of two (n, 4) arrays of normalised (cx, cy, w, h) boxes."""
    a, b = np.asarray(a, dtype=float).reshape(-1, 4), np.asarray(b, dtype=float).reshape(-1, 4)
    ax0, ay0, ax1, ay1 = a[:, 0] - a[:, 2] / 2, a[:, 1] - a[:, 3] / 2, a[:, 0] + a[:, 2] / 2, a[:, 1] + a[:, 3] / 2
    bx0, by0, bx1, by1 = b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2, b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2
    iw = np.clip(np.minimum(ax1, bx1) - np.maximum(ax0, bx0), 0, None)
    ih = np.clip(np.minimum(ay1, by1) - np.maximum(ay0, by0), 0, None)
    inter = iw * ih
    union = a[:, 2] * a[:, 3] + b[:, 2] * b[:, 3] - inter
    return inter / np.maximum(union, 1e-9)


def localisation_metrics(y_true, y_pred, pred_boxes, true_boxes) -> dict:
    """How well one predicted box per photo matches the building's labelled box.

    mean_iou: average overlap (0 to 1) between predicted and labelled box; a photo with no box
    (a detector that found nothing) counts as 0.
    iou50: share of photos whose box overlaps the label by at least half (IoU >= 0.5).
    named_and_boxed: share of photos with the right building AND IoU >= 0.5.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    pred_boxes = np.asarray(pred_boxes, dtype=float).reshape(-1, 4)
    ious = np.nan_to_num(box_iou_cxcywh(pred_boxes, true_boxes))
    ious[np.isnan(pred_boxes).any(axis=1)] = 0.0
    hit = ious >= 0.5
    return {
        "mean_iou": round(float(ious.mean()), 4) if len(ious) else 0.0,
        "iou50": round(float(hit.mean()), 4) if len(ious) else 0.0,
        "named_and_boxed": round(float((hit & (y_true == y_pred)).mean()), 4) if len(ious) else 0.0,
    }


def wilson_interval(correct: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for an accuracy of correct/total.

    With a small test set this is the honest way to read an accuracy: 21 out of 21 is
    "probably above 85%", not "100%". Overlapping intervals mean the models are tied.
    """
    if total == 0:
        return 0.0, 1.0
    p = correct / total
    denom = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denom
    half = z * ((p * (1 - p) / total + z**2 / (4 * total**2)) ** 0.5) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def classification_metrics(y_true, y_pred, scores=None, num_classes: int = NUM_CLASSES) -> dict:
    """Accuracy, macro F1, per-class precision/recall/F1, confusion matrix and top-3 accuracy.

    y_pred may contain NO_PREDICTION (-1) for images where a detector found nothing; those count
    as wrong and appear in the confusion matrix's last column.
    scores: optional (n, num_classes) array used for top-3 accuracy.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    n = len(y_true)
    if n == 0:
        raise ValueError("No samples to score.")

    confusion = np.zeros((num_classes, num_classes + 1), dtype=int)
    for t, p in zip(y_true, y_pred, strict=True):
        confusion[t, p if p >= 0 else num_classes] += 1

    per_class = {}
    f1s = []
    for c in range(num_classes):
        tp = int(confusion[c, c])
        support = int(confusion[c].sum())
        predicted = int(confusion[:, c].sum())
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[CLASS_NAMES[c]] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
        }
        if support:
            f1s.append(f1)

    correct = int((y_true == y_pred).sum())
    low, high = wilson_interval(correct, n)
    metrics = {
        "images": n,
        "correct": correct,
        "accuracy": round(correct / n, 4),
        "accuracy_low": round(low, 4),
        "accuracy_high": round(high, 4),
        "macro_f1": round(float(np.mean(f1s)) if f1s else 0.0, 4),
        "no_prediction": int((y_pred == NO_PREDICTION).sum()),
        "per_class": per_class,
        "confusion": confusion.tolist(),
        "confusion_columns": CLASS_NAMES + ["(none)"],
    }
    if scores is not None:
        scores = np.asarray(scores, dtype=float)
        top3 = np.argsort(-scores, axis=1, kind="stable")[:, :3]
        # A building only counts as "in the top 3" if the model gave it a score: a detector that
        # finds nothing scores every building 0 and must not get credit for an arbitrary order.
        hits = [t in row and s[t] > 0 for t, row, s in zip(y_true, top3, scores, strict=True)]
        metrics["top3_accuracy"] = round(float(np.mean(hits)), 4)
    return metrics


def top_k(scores, k: int = 3) -> list[tuple[int, float]]:
    order = np.argsort(-np.asarray(scores))[:k]
    return [(int(i), float(scores[i])) for i in order]


def save_predictions(path: str, paths, y_true, y_pred, scores, ious=None) -> str:
    """One row per test image: file, true and predicted building, confidence, top 3 and box IoU."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ious = [None] * len(paths) if ious is None else list(ious)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "true", "predicted", "confidence", "correct", "top3", "box_iou"])
        for image, t, p, s, iou in zip(paths, y_true, y_pred, scores, ious, strict=True):
            confidence = float(s[p]) if p >= 0 else 0.0
            top3 = "|".join(f"{CLASS_NAMES[i]}:{v:.3f}" for i, v in top_k(s) if v > 0)
            writer.writerow(
                [
                    os.path.basename(image),
                    CLASS_NAMES[t],
                    CLASS_NAMES[p] if p >= 0 else "(none)",
                    f"{confidence:.4f}",
                    int(t == p),
                    top3,
                    "" if iou is None else f"{float(iou):.3f}",
                ]
            )
    return path


def metrics_path(run_dir: str, model: str) -> str:
    return os.path.join(run_dir, "metrics", f"{model}_test.json")


def save_metrics(run_dir: str, model: str, metrics: dict) -> str:
    path = metrics_path(run_dir, model)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return path


def load_metrics(run_dir: str, model: str) -> dict | None:
    path = metrics_path(run_dir, model)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_export_info(run_dir: str) -> dict:
    path = os.path.join(run_dir, "metrics", "export.json")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f).get("models", {})


# --------------------------------------------------------------------------- leaderboard


def build_leaderboard(run_dir: str) -> pd.DataFrame:
    exported = load_export_info(run_dir)
    rows = []
    for model in ALL_MODELS:
        m = load_metrics(run_dir, model)
        if m is None:
            continue
        info = exported.get(model, {})
        rows.append(
            {
                "model": MODEL_LABELS[model],
                "role": MODEL_ROLES[model],
                "test_accuracy": m["accuracy"],
                "accuracy_low": m.get("accuracy_low"),
                "accuracy_high": m.get("accuracy_high"),
                "correct": m.get("correct"),
                "macro_f1": m["macro_f1"],
                "top3_accuracy": m.get("top3_accuracy"),
                "box_iou": (m.get("localisation") or {}).get("mean_iou"),
                "named_and_boxed": (m.get("localisation") or {}).get("named_and_boxed"),
                "map50": (m.get("detection") or {}).get("map50"),
                "map50_95": (m.get("detection") or {}).get("map50_95"),
                "params_m": m.get("params_m"),
                "onnx_mb": info.get("onnx_mb"),
                "cpu_ms": info.get("cpu_ms"),
                "test_images": m["images"],
                "id": model,
            }
        )
    return pd.DataFrame(rows)


def _present(value) -> bool:
    return value is not None and not (isinstance(value, float) and np.isnan(value))


def results_markdown(board: pd.DataFrame) -> str:
    """The leaderboard as a Markdown table, ready to paste into the README."""

    def fmt(value, kind):
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return ""
        if kind == "pct":
            return f"{value * 100:.1f}%"
        if kind == "3":
            return f"{value:.3f}"
        if kind == "2":
            return f"{value:.2f}"
        if kind == "1":
            return f"{value:.1f}"
        return str(value)

    lines = [
        "| Model | Role | Test accuracy | 95% range | Box IoU | Named + boxed | Macro F1 | Top-3 | mAP50 | mAP50-95 "
        "| Params (M) | ONNX (MB) | CPU ms |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in board.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    r["model"],
                    r["role"],
                    fmt(r["test_accuracy"], "pct")
                    + (f" ({int(r['correct'])}/{int(r['test_images'])})" if _present(r.get("correct")) else ""),
                    f"{r['accuracy_low'] * 100:.0f}–{r['accuracy_high'] * 100:.0f}%"
                    if _present(r.get("accuracy_low"))
                    else "",
                    fmt(r.get("box_iou"), "2"),
                    fmt(r.get("named_and_boxed"), "pct"),
                    fmt(r["macro_f1"], "3"),
                    fmt(r["top3_accuracy"], "pct"),
                    fmt(r["map50"], "3"),
                    fmt(r["map50_95"], "3"),
                    fmt(r["params_m"], "1"),
                    fmt(r["onnx_mb"], "1"),
                    fmt(r["cpu_ms"], "1"),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def run_evaluate(run_dir: str, LOG) -> pd.DataFrame:
    """Collect every model's test metrics into leaderboard.csv, results.md and summary.json."""
    with component_stage(LOG, "Evaluate", "Building the results table"):
        log_step(LOG, "Evaluate", "Collecting every model's test metrics")
        board = build_leaderboard(run_dir)
        if board.empty:
            raise FileNotFoundError("No test metrics in this run yet. Run --mode classify and/or --mode detect first.")
        board = board.sort_values(["test_accuracy", "macro_f1"], ascending=False).reset_index(drop=True)

        log_step(LOG, "Evaluate", "Writing leaderboard.csv, results.md and summary.json", models=len(board))
        metrics_dir = os.path.join(run_dir, "metrics")
        csv_path = os.path.join(metrics_dir, "leaderboard.csv")
        board.to_csv(csv_path, index=False)
        md_path = os.path.join(metrics_dir, "results.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(results_markdown(board) + "\n")

        best = board.iloc[0]
        summary = {
            "best_accuracy_model": best["id"],
            "best_accuracy": float(best["test_accuracy"]),
            "models": board["id"].tolist(),
            "test_images": int(best["test_images"]),
        }
        with open(os.path.join(metrics_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        for path in (csv_path, md_path):
            log_stage(LOG, "ARTIFACT", "Evaluate", path=path)
        view = board.copy()
        view.insert(
            view.columns.get_loc("test_accuracy") + 1,
            "95%_range",
            [
                f"{lo * 100:.0f}–{hi * 100:.0f}%" if _present(lo) else ""
                for lo, hi in zip(view["accuracy_low"], view["accuracy_high"], strict=True)
            ],
        )
        view = view.drop(columns=["id", "accuracy_low", "accuracy_high", "correct"]).rename(
            columns={"test_images": "images", "named_and_boxed": "named_boxed_share"}
        )
        print_table(LOG, view, title="Test results (same images for every model)")
        log_stage(LOG, "RESULT", "Evaluate", best=best["model"], test_accuracy=float(best["test_accuracy"]))
    return board
