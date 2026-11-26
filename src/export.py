"""Export trained models to ONNX for the web app, check them, and time them on CPU.

Writes into the web app folder (default webapp/):

    models/<model>.onnx     one file per model, weights embedded
    models/manifest.json    classes, preprocessing, file sizes, test metrics, CPU timings
    samples/                a few test photos for visitors who are not on campus
    samples/samples.json    the sample list with each photo's true building

Every exported model is run once with onnxruntime. Classifier outputs are compared with
PyTorch, and the median CPU time over repeated runs is recorded for the results table.
"""

from __future__ import annotations

import inspect
import json
import os
import shutil
import statistics
import time
from datetime import datetime, timezone

import numpy as np

from src.constants import (
    BUILDINGS,
    CLASSIFIER_INPUT,
    CLASSIFIERS,
    DETECTOR_CONF,
    DETECTOR_IMGSZ,
    DETECTOR_IOU,
    DETECTOR_RESIZE,
    DETECTORS,
    IMAGENET_MEAN,
    IMAGENET_STD,
    LETTERBOX_FILL,
    MODEL_LABELS,
    MODEL_ROLES,
)
from src.datasets import image_level_samples
from src.evaluate import load_metrics
from src.logger import component_stage, log_stage, log_step
from src.progress import progress

PRIMARY_MODEL = "yolo11n"
SAMPLE_MAX_SIDE = 1024
APP_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "webapp")
APP_FILES = ("index.html", "app.js", "inference.js", "styles.css", "buildings.json")


def copy_app_files(web_dir: str) -> bool:
    """Copy the static app into web_dir when it is not the repository's webapp/ folder."""
    if os.path.abspath(web_dir) == APP_DIR:
        return False
    os.makedirs(web_dir, exist_ok=True)
    for name in APP_FILES:
        shutil.copy2(os.path.join(APP_DIR, name), os.path.join(web_dir, name))
    return True


def _onnx_export(model, dummy, path: str, output_names=("logits", "box")) -> None:
    """torch.onnx.export across PyTorch versions, always with weights inside the .onnx file."""
    import torch

    kwargs = {"input_names": ["input"], "output_names": list(output_names), "opset_version": 17}
    params = inspect.signature(torch.onnx.export).parameters
    if "dynamo" in params:
        try:
            torch.onnx.export(model, (dummy,), path, dynamo=False, **kwargs)
            return
        except Exception:
            extra = {"external_data": False} if "external_data" in params else {}
            torch.onnx.export(model, (dummy,), path, dynamo=True, **kwargs, **extra)
            return
    torch.onnx.export(model, (dummy,), path, **kwargs)


def time_session(session, feed: dict, runs: int = 20, warmup: int = 3) -> float:
    """Median milliseconds per call."""
    for _ in range(warmup):
        session.run(None, feed)
    timings = []
    for _ in range(runs):
        start = time.perf_counter()
        session.run(None, feed)
        timings.append((time.perf_counter() - start) * 1000)
    return round(statistics.median(timings), 1)


def export_classifier(arch: str, run_dir: str, models_dir: str, LOG, check_image: str | None) -> dict:
    import torch

    from src.classify import checkpoint_path, load_classifier, load_rgb
    from src.inference import open_session, preprocess_classifier, softmax

    component = f"Export:{arch}"
    with component_stage(LOG, component, f"Exporting {MODEL_LABELS[arch]} to ONNX"):
        log_step(LOG, component, "Loading the best checkpoint")
        model, _ = load_classifier(checkpoint_path(run_dir, arch), "cpu")
        path = os.path.join(models_dir, f"{arch}.onnx")
        dummy = torch.zeros(1, 3, CLASSIFIER_INPUT, CLASSIFIER_INPUT)
        log_step(LOG, component, "Converting to ONNX")
        _onnx_export(model, dummy, path)

        log_step(LOG, component, "Checking the ONNX model gives the same answers as PyTorch")
        session = open_session(path)
        tensor = preprocess_classifier(load_rgb(check_image)) if check_image else dummy.numpy()
        feed = {session.get_inputs()[0].name: tensor}
        outputs = dict(zip([o.name for o in session.get_outputs()], session.run(None, feed), strict=True))
        with torch.no_grad():
            torch_logits, torch_box = model(torch.from_numpy(tensor))
        diff = float(np.abs(softmax(outputs["logits"]) - torch.softmax(torch_logits, dim=1).numpy()).max())
        box_diff = float(np.abs(outputs["box"] - torch_box.numpy()).max())
        if diff > 1e-3 or box_diff > 1e-3:
            log_stage(
                LOG,
                "WARN",
                component,
                max_probability_diff=round(diff, 6),
                max_box_diff=round(box_diff, 6),
                note="ONNX and PyTorch disagree",
            )

        log_step(LOG, component, "Timing it on CPU")
        info = {
            "onnx_mb": round(os.path.getsize(path) / 1e6, 1),
            "cpu_ms": time_session(session, feed),
            "max_probability_diff": round(diff, 6),
            "max_box_diff": round(box_diff, 6),
        }
        log_stage(LOG, "ARTIFACT", component, path=path)
        log_stage(LOG, "METRICS", component, **info)
    return info


def export_detector(model_name: str, run_dir: str, models_dir: str, LOG, check_image: str | None) -> dict:
    from ultralytics import YOLO

    from src.classify import load_rgb
    from src.detect import best_weights_path
    from src.inference import letterbox, open_session, run_detector

    component = f"Export:{model_name}"
    with component_stage(LOG, component, f"Exporting {MODEL_LABELS[model_name]} to ONNX"):
        log_step(LOG, component, "Converting with Ultralytics")
        exported = YOLO(best_weights_path(run_dir, model_name)).export(
            format="onnx", imgsz=DETECTOR_IMGSZ, opset=17, dynamic=False, simplify=True
        )
        path = os.path.join(models_dir, f"{model_name}.onnx")
        shutil.copy2(exported, path)

        log_step(LOG, component, "Checking the ONNX model runs on a test photo")
        session = open_session(path)
        image = load_rgb(check_image) if check_image else None
        tensor = letterbox(image)[0] if image else np.zeros((1, 3, DETECTOR_IMGSZ, DETECTOR_IMGSZ), np.float32)
        feed = {session.get_inputs()[0].name: tensor}
        output_shape = list(session.run(None, feed)[0].shape)
        if image is not None:
            detections = run_detector(session, image)
            log_stage(LOG, "INFO", component, check_image=os.path.basename(check_image), detections=len(detections))

        log_step(LOG, component, "Timing it on CPU")
        info = {
            "onnx_mb": round(os.path.getsize(path) / 1e6, 1),
            "cpu_ms": time_session(session, feed, runs=10),
            "output_shape": "x".join(str(d) for d in output_shape),
        }
        log_stage(LOG, "ARTIFACT", component, path=path)
        log_stage(LOG, "METRICS", component, **info)
    return info


def export_samples(prepared_dir: str, samples_dir: str, count: int, LOG) -> list[dict]:
    """Copy up to `count` test photos (one per building first), resized for the web."""
    from PIL import Image, ImageOps

    from src.constants import CLASS_NAMES

    if os.path.isdir(samples_dir):
        for name in os.listdir(samples_dir):
            if name.startswith("sample_"):
                os.remove(os.path.join(samples_dir, name))
    os.makedirs(samples_dir, exist_ok=True)

    test = image_level_samples(prepared_dir, "test")
    chosen, seen = [], set()
    for path, label in test:  # one per building first
        if label not in seen:
            chosen.append((path, label))
            seen.add(label)
    for item in test:
        if len(chosen) >= count:
            break
        if item not in chosen:
            chosen.append(item)
    chosen = chosen[:count]

    log_step(LOG, "Export", "Copying sample photos for the web app", photos=len(chosen))
    entries = []
    for i, (path, label) in enumerate(progress(chosen, desc="Exporting sample photos", unit="img")):
        name = f"sample_{i + 1:02d}.jpg"
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            image.thumbnail((SAMPLE_MAX_SIDE, SAMPLE_MAX_SIDE))
            image.save(os.path.join(samples_dir, name), quality=85)
        entries.append({"file": name, "building": CLASS_NAMES[label]})
    with open(os.path.join(samples_dir, "samples.json"), "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)
    log_stage(LOG, "INFO", "Export", samples=len(entries), folder=samples_dir)
    return entries


def _public_metrics(metrics: dict | None) -> dict:
    if not metrics:
        return {}
    out = {"test_accuracy": metrics.get("accuracy"), "macro_f1": metrics.get("macro_f1")}
    detection = metrics.get("detection") or {}
    if detection:
        out.update({"map50": detection.get("map50"), "map50_95": detection.get("map50_95")})
    return out


def build_manifest(
    run_id: str, models: list[str], export_info: dict, run_dir: str, detector_resize: str = DETECTOR_RESIZE
) -> dict:
    entries = []
    for model in models:
        info = export_info[model]
        entry = {
            "id": model,
            "label": MODEL_LABELS[model],
            "role": MODEL_ROLES[model],
            "file": f"{model}.onnx",
            "megabytes": info["onnx_mb"],
            "cpu_ms": info["cpu_ms"],
            "metrics": _public_metrics(load_metrics(run_dir, model)),
        }
        if model in CLASSIFIERS:
            entry.update(
                {
                    "kind": "classifier",
                    "input": {
                        "size": CLASSIFIER_INPUT,
                        "resize": "stretch",
                        "mean": list(IMAGENET_MEAN),
                        "std": list(IMAGENET_STD),
                    },
                    "outputs": {"logits": "class scores", "box": "normalised cx, cy, w, h"},
                }
            )
        else:
            entry.update(
                {
                    "kind": "detector",
                    "input": {"size": DETECTOR_IMGSZ, "fill": LETTERBOX_FILL, "resize": detector_resize},
                    "conf": DETECTOR_CONF,
                    "iou": DETECTOR_IOU,
                    "one_per_building": True,
                }
            )
        entries.append(entry)
    primary = PRIMARY_MODEL if PRIMARY_MODEL in models else models[0]
    return {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "run": run_id,
        "primary": primary,
        "classes": [{"id": i, "slug": b["slug"], "name": b["name"]} for i, b in enumerate(BUILDINGS)],
        "models": entries,
    }


def run_export(
    run_dir: str,
    run_id: str,
    prepared_dir: str,
    web_dir: str,
    LOG,
    models: list[str],
    samples: int = 8,
    detector_resize: str = DETECTOR_RESIZE,
) -> dict:
    models_dir = os.path.join(web_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    test = image_level_samples(prepared_dir, "test")
    check_image = test[0][0] if test else None

    export_info = {}
    with component_stage(LOG, "Export", "Exporting models for the web app", models=",".join(models), web_dir=web_dir):
        if copy_app_files(web_dir):
            log_step(LOG, "Export", "Copying the web app's files", to=web_dir)
        for model in models:
            if model in CLASSIFIERS:
                export_info[model] = export_classifier(model, run_dir, models_dir, LOG, check_image)
            elif model in DETECTORS:
                export_info[model] = export_detector(model, run_dir, models_dir, LOG, check_image)
            else:
                raise ValueError(f"Unknown model {model!r}.")

        log_step(LOG, "Export", "Writing models/manifest.json")
        if any(m in DETECTORS for m in models):
            log_step(LOG, "Export", "Detector photo fitting for the web app", resize=detector_resize)
        manifest = build_manifest(run_id, models, export_info, run_dir, detector_resize)
        manifest_path = os.path.join(models_dir, "manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        log_stage(LOG, "ARTIFACT", "Export", path=manifest_path)

        if samples > 0:
            export_samples(prepared_dir, os.path.join(web_dir, "samples"), samples, LOG)

        record_path = os.path.join(run_dir, "metrics", "export.json")
        with open(record_path, "w", encoding="utf-8") as f:
            json.dump({"web_dir": os.path.abspath(web_dir), "models": export_info}, f, indent=2)
        log_stage(LOG, "ARTIFACT", "Export", path=record_path)
    return manifest
