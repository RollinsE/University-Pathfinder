"""Command-line entry point for the Campus Pathfinder pipeline.

    python main.py --mode all --demo                      # whole pipeline on synthetic data
    python main.py --mode all                             # real data from Roboflow
    python main.py --mode classify --run_dir runs/run_a   # one stage into an existing run folder
    python main.py --mode predict --image photo.jpg       # one photo, exported model

Modes: acquire, classify, detect, export, evaluate, predict, all. Every mode writes into one
run folder (logs, artifacts, metrics, plots); see README.md and docs/LOGGING.md.

Multi-word options can be written either way: --batch_size 32 or --batch-size 32.
"""

from __future__ import annotations

import argparse
import os
import sys

from src.console import SEPARATOR, key_value_block, raw, stage_banner
from src.constants import (
    ALL_MODELS,
    CLASSIFIERS,
    DEFAULT_SEED,
    DETECTOR_CONF,
    DETECTOR_IMGSZ,
    DETECTOR_RESIZE,
    DETECTOR_RESIZE_MODES,
    DETECTORS,
    ROBOFLOW_PROJECT,
    ROBOFLOW_VERSION,
    ROBOFLOW_WORKSPACE,
)
from src.experiment import create_run_dir, derive_run_id, ensure_run_subdirs, find_latest_run_dir
from src.logger import component_stage, log_stage, log_step, setup_logger

MODES = ["acquire", "classify", "detect", "export", "evaluate", "predict", "all"]

PIPELINE_ACTIONS = {
    "acquire": "Getting the dataset ready",
    "classify": "Training and testing the classifiers",
    "detect": "Training and testing the detector",
    "export": "Exporting models for the web app",
    "evaluate": "Comparing the models",
    "predict": "Predicting the building in one photo",
    "all": "Running the whole pipeline",
}

# Settings used with --demo unless given explicitly: small, quick and offline.
# Demo models go to a separate web app folder, so they never replace the real ones in webapp/.
DEMO_DEFAULTS = {
    "epochs": 2,
    "yolo_epochs": 3,
    "yolo_imgsz": 160,
    "batch_size": 16,
    "yolo_batch": 8,
    "samples": 4,
    "web_dir": os.path.join("runs", "demo_webapp"),
}
FULL_DEFAULTS = {
    "epochs": 30,
    "yolo_epochs": 250,
    "yolo_imgsz": DETECTOR_IMGSZ,
    "batch_size": 32,
    "yolo_batch": 16,
    "samples": 8,
    "web_dir": "webapp",
}


def parse_args(argv=None):
    p = argparse.ArgumentParser("Campus Pathfinder")

    p.add_argument("--mode", required=True, choices=MODES)
    p.add_argument("--log_level", "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    g = p.add_argument_group("run folder")
    g.add_argument("--run_base_dir", "--run-base-dir", default="runs", help="Base folder for timestamped runs.")
    g.add_argument(
        "--run_dir", "--run-dir", default=None, help="Use a specific run folder, e.g. runs/run_YYYYMMDD_HHMMSS."
    )
    g.add_argument("--resume", action="store_true", help="Continue in the latest run under --run_base_dir.")

    g = p.add_argument_group("data")
    g.add_argument("--data_dir", "--data-dir", default="data", help="Downloaded and prepared datasets.")
    g.add_argument(
        "--version", type=int, default=ROBOFLOW_VERSION, help="Roboflow dataset version (4 original, 3 augmented)."
    )
    g.add_argument(
        "--workspace", default=ROBOFLOW_WORKSPACE, help="Roboflow workspace id (change it to use a forked copy)."
    )
    g.add_argument("--project", default=ROBOFLOW_PROJECT, help="Roboflow project id within the workspace.")
    g.add_argument(
        "--source_dir",
        "--source-dir",
        default=None,
        help="Use a YOLO-format export already on disk instead of downloading.",
    )
    g.add_argument(
        "--dataset_dir",
        "--dataset-dir",
        default=None,
        help="A prepared dataset to use instead of the one this run acquired.",
    )
    g.add_argument("--demo", action="store_true", help="Synthetic data: no account or download, quick settings.")
    g.add_argument("--overwrite", action="store_true", help="Rebuild datasets and retrain models that already exist.")

    g = p.add_argument_group("models")
    g.add_argument("--models", default=",".join(ALL_MODELS), help="Comma-separated: " + ", ".join(ALL_MODELS))
    g.add_argument("--device", default="auto", help="auto (GPU if present), cuda (GPU required), cuda:0, cpu or mps.")
    g.add_argument("--seed", type=int, default=DEFAULT_SEED)
    g.add_argument("--workers", type=int, default=2, help="Data-loading worker processes.")
    g.add_argument(
        "--no_pretrained", "--no-pretrained", action="store_true", help="Train from scratch (default with --demo)."
    )

    g = p.add_argument_group("classifiers")
    g.add_argument("--epochs", type=int, default=None, help="Default 30 (2 with --demo).")
    g.add_argument("--batch_size", "--batch-size", type=int, default=None, help="Default 32 (16 with --demo).")
    g.add_argument("--lr", type=float, default=3e-4, help="AdamW learning rate. Default 3e-4.")
    g.add_argument("--weight_decay", "--weight-decay", type=float, default=1e-4, help="Default 1e-4.")
    g.add_argument("--label_smoothing", "--label-smoothing", type=float, default=0.1, help="Default 0.1.")
    g.add_argument(
        "--box_weight", "--box-weight", type=float, default=2.0, help="Weight of the box loss against the name loss."
    )
    g.add_argument(
        "--patience", type=int, default=8, help="Stop when validation loss has not improved for this many epochs."
    )

    g = p.add_argument_group("detector")
    g.add_argument("--yolo_epochs", "--yolo-epochs", type=int, default=None, help="Default 250 (3 with --demo).")
    g.add_argument("--yolo_imgsz", "--yolo-imgsz", type=int, default=None, help="Default 640 (160 with --demo).")
    g.add_argument("--yolo_batch", "--yolo-batch", type=int, default=None, help="Default 16 (8 with --demo).")
    g.add_argument(
        "--yolo_patience",
        "--yolo-patience",
        type=int,
        default=75,
        help="Stop when validation mAP has not improved for this many epochs. Default 75.",
    )

    g = p.add_argument_group("export, figures and prediction")
    g.add_argument(
        "--web_dir",
        "--web-dir",
        default=None,
        help="Web app folder that receives models/ and samples/. Default webapp (runs/demo_webapp with --demo).",
    )
    g.add_argument(
        "--samples", type=int, default=None, help="Sample photos for the web app. Default 8 (4 with --demo)."
    )
    g.add_argument("--figures_dir", "--figures-dir", default=None, help="Also copy figures here, e.g. docs/figures.")
    g.add_argument("--no_plots", "--no-plots", action="store_true", help="Skip figures and Ultralytics plots.")
    g.add_argument("--image", default=None, help="Photo for --mode predict.")
    g.add_argument(
        "--model",
        default="yolo11n",
        choices=[*ALL_MODELS, "all"],
        help="Model for --mode predict, or all to compare the three. Default yolo11n.",
    )
    g.add_argument(
        "--yolo_resize",
        "--yolo-resize",
        default=None,
        choices=DETECTOR_RESIZE_MODES,
        help=(
            "How photos are fitted into the detector's square input: letterbox (shape kept, grey edges) or "
            "stretch (squashed, as the training photos were). With --mode export it is saved for the web app "
            "(default letterbox); with --mode predict it overrides the exported setting."
        ),
    )
    g.add_argument(
        "--conf",
        type=float,
        default=None,
        help=f"Lowest detector confidence shown by --mode predict. Default {DETECTOR_CONF}.",
    )

    args = p.parse_args(argv)
    defaults = DEMO_DEFAULTS if args.demo else FULL_DEFAULTS
    for key, value in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    if args.demo:
        args.no_pretrained = True
    return args


def check_device(device: str, LOG, training: bool) -> None:
    """Fail fast when a GPU is required but missing, and say which device training will use."""
    if not training:
        return
    try:
        import torch
    except ImportError:
        return
    has_gpu = torch.cuda.is_available()
    if device.startswith("cuda") and not has_gpu:
        raise RuntimeError(
            f"--device {device} needs a GPU, but none is available. "
            "In Colab: Runtime > Change runtime type > T4 GPU > Save, then run again."
        )
    if has_gpu and (device == "auto" or device.startswith("cuda")):
        log_step(LOG, "Pipeline", f"Using GPU: {torch.cuda.get_device_name(0)}")
    elif device == "auto":
        log_stage(LOG, "WARN", "Pipeline", device="cpu", note="no GPU found, training will be slow")


def parse_models(text: str) -> list[str]:
    models = [m.strip() for m in text.split(",") if m.strip()]
    unknown = [m for m in models if m not in ALL_MODELS]
    if unknown:
        raise ValueError(f"Unknown models {unknown}. Choose from {list(ALL_MODELS)}.")
    return models


def resolve_run_dir(args):
    if args.run_dir:
        ensure_run_subdirs(args.run_dir)
        return derive_run_id(args.run_dir), args.run_dir
    if args.resume:
        latest = find_latest_run_dir(args.run_base_dir)
        if not latest:
            sys.stderr.write(f"--resume set but there are no run_ folders under {args.run_base_dir}\n")
            sys.exit(1)
        ensure_run_subdirs(latest)
        return derive_run_id(latest), latest
    return create_run_dir(base=args.run_base_dir)


def main(argv=None) -> int:
    args = parse_args(argv)
    run_id, run_dir = resolve_run_dir(args)
    LOG = setup_logger(run_id, run_dir, log_level=args.log_level)
    log_path = os.path.join(run_dir, "logs", "run.log")
    key_value_block(
        LOG,
        "Campus Pathfinder",
        {
            "mode": args.mode + (" (demo data)" if args.demo else ""),
            "run": os.path.basename(run_dir.rstrip("/")),
            "run folder": run_dir,
            "log file": log_path,
        },
    )

    try:
        with component_stage(
            LOG,
            "Pipeline",
            PIPELINE_ACTIONS[args.mode],
            mode=args.mode,
            run_id=run_id,
            demo=args.demo,
            device=args.device,
        ):
            models = parse_models(args.models)
            classifiers = [m for m in models if m in CLASSIFIERS]
            detectors = [m for m in models if m in DETECTORS]
            plots = not args.no_plots
            check_device(args.device, LOG, training=args.mode in ("classify", "detect", "all"))

            if args.mode == "predict":
                from src.predict import run_predict

                if not args.image:
                    raise ValueError("--mode predict needs --image.")
                run_predict(
                    args.image,
                    args.web_dir,
                    args.model,
                    LOG,
                    run_dir=run_dir,
                    conf=args.conf,
                    resize=args.yolo_resize,
                )
                return 0

            if args.mode in ("acquire", "all"):
                from src.acquisition import run_acquire

                stage_banner(LOG, "Stage: acquire")
                run_acquire(
                    args.data_dir,
                    run_dir,
                    LOG,
                    version=args.version,
                    workspace=args.workspace,
                    project=args.project,
                    source_dir=args.source_dir,
                    demo=args.demo,
                    seed=args.seed,
                    overwrite=args.overwrite,
                )
                log_stage(LOG, "RESULT", "Pipeline", finished="acquire", status="complete")

            needs_data = args.mode in ("classify", "detect", "export", "all")
            prepared_dir = None
            if needs_data:
                from src.acquisition import resolve_prepared_dir

                prepared_dir = resolve_prepared_dir(run_dir, args.dataset_dir)
                log_step(LOG, "Pipeline", "Using the prepared dataset", folder=prepared_dir)

            if args.mode in ("classify", "all") and classifiers:
                from src.classify import evaluate_classifier, train_classifier

                stage_banner(LOG, "Stage: classifiers")
                for arch in classifiers:
                    train_classifier(
                        arch,
                        prepared_dir,
                        run_dir,
                        LOG,
                        epochs=args.epochs,
                        batch_size=args.batch_size,
                        lr=args.lr,
                        weight_decay=args.weight_decay,
                        patience=args.patience,
                        label_smoothing=args.label_smoothing,
                        box_weight=args.box_weight,
                        pretrained=not args.no_pretrained,
                        num_workers=args.workers,
                        seed=args.seed,
                        device=args.device,
                        overwrite=args.overwrite,
                    )
                    evaluate_classifier(
                        arch, prepared_dir, run_dir, LOG, args.batch_size, args.workers, device=args.device
                    )
                log_stage(
                    LOG, "RESULT", "Pipeline", finished="classify", models=",".join(classifiers), status="complete"
                )

            if args.mode in ("detect", "all") and detectors:
                from src.detect import evaluate_detector, train_detector

                stage_banner(LOG, "Stage: detector")
                for name in detectors:
                    train_detector(
                        prepared_dir,
                        run_dir,
                        LOG,
                        model_name=name,
                        epochs=args.yolo_epochs,
                        imgsz=args.yolo_imgsz,
                        batch=args.yolo_batch,
                        patience=args.yolo_patience,
                        pretrained=not args.no_pretrained,
                        workers=args.workers,
                        seed=args.seed,
                        device=args.device,
                        plots=plots,
                        overwrite=args.overwrite,
                    )
                    evaluate_detector(
                        prepared_dir,
                        run_dir,
                        LOG,
                        model_name=name,
                        imgsz=args.yolo_imgsz,
                        batch=args.yolo_batch,
                        device=args.device,
                        plots=plots,
                    )
                log_stage(LOG, "RESULT", "Pipeline", finished="detect", models=",".join(detectors), status="complete")

            if args.mode in ("export", "all"):
                from src.export import run_export

                stage_banner(LOG, "Stage: export")
                run_export(
                    run_dir,
                    run_id,
                    prepared_dir,
                    args.web_dir,
                    LOG,
                    models=models,
                    samples=args.samples,
                    detector_resize=args.yolo_resize or DETECTOR_RESIZE,
                )
                log_stage(LOG, "RESULT", "Pipeline", finished="export", web_dir=args.web_dir, status="complete")

            if args.mode in ("evaluate", "all"):
                from src.evaluate import run_evaluate

                stage_banner(LOG, "Stage: evaluate")
                run_evaluate(run_dir, LOG)
                if plots:
                    from src.figures import run_figures

                    run_figures(run_dir, LOG, export_dir=args.figures_dir)
                log_stage(LOG, "RESULT", "Pipeline", finished="evaluate", status="complete")

        raw(
            LOG,
            "\n".join(
                [
                    "",
                    SEPARATOR,
                    "Outputs written",
                    SEPARATOR,
                    f"  run folder:  {run_dir}",
                    f"  log file:    {log_path}",
                    f"  metrics:     {os.path.join(run_dir, 'metrics')}",
                    f"  web models:  {os.path.join(args.web_dir, 'models')}" if args.mode in ("export", "all") else "",
                ]
            ).rstrip(),
        )
        return 0
    except Exception as exc:
        raw(LOG, f"\nPipeline failed: {type(exc).__name__}: {exc}\nFull details: {log_path}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
