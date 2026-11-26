# Logging

One logger per run, structured stage lines, and a context manager that times every component.
The code is in `src/logger.py` and `src/console.py`.

## One run, one folder, one log

`main.py` creates (or reuses) a run directory and calls `setup_logger(run_id, run_dir)` once.

```text
runs/run_YYYYMMDD_HHMMSS/
├── logs/run.log        every line the console shows, appended across stages
├── artifacts/          checkpoints and training histories, one folder per model
│   ├── resnet18/               best.pt, history.csv
│   ├── mobilenet_v3_large/     best.pt, history.csv
│   └── yolo11n/                train/ (Ultralytics run, weights/best.pt), test/
├── metrics/            dataset.json, <model>_test.json, <model>_test_predictions.csv,
│                       export.json, leaderboard.csv, results.md, summary.json
└── plots/              figures, light and dark PNG
```

- `--run_dir <path>` writes into a specific run. Use it to run `acquire`, `classify`, `detect`,
  `export` and `evaluate` as separate commands that share one `run.log`, or to resume.
- `--resume` continues in the latest run under `--run_base_dir`.
- With neither flag, a new timestamped run is created.

The log file is opened in append mode, so resumed and multi-stage runs keep a single history.

## Console and file

The same records go to two places, formatted differently:

- **`run.log`** keeps the full structured lines (below). Use it for grepping and for the complete
  history, including every file path written.
- **The console** shows a shorter rendering for reading as the run goes:

```text
================================================================================
STAGE: CLASSIFIERS
================================================================================
14:02:11  Classifier:resnet18: Training ResNet18 (epochs 30, batch size 32, lr 0.0003, device cuda, pretrained yes)
14:02:11  Classifier:resnet18: Loading train and validation images
14:02:12  Classifier:resnet18: train images 180, valid images 38
14:02:12  Classifier:resnet18: Building ResNet18 from ImageNet weights with a building-name head and a box head
14:02:13  Classifier:resnet18: Training epochs 1 to 30 (device cuda, patience 8)
14:02:31  Classifier:resnet18: epoch 1/30: train loss 1.6213, train accuracy 52.22%, val loss 0.9874, val accuracy 81.58%, val box iou 0.6120
14:02:31  Classifier:resnet18: New best model saved at epoch 1 (val accuracy 81.58%, val box iou 0.6120)
...
14:09:39  Classifier:resnet18: Saving training history
14:09:40  Classifier:resnet18: done in 7m 29s
```

  Every stage opens with a plain-English line saying what it is doing ("Downloading dataset from
  Roboflow", "Training ResNet18", "Exporting YOLO11n to ONNX"), and `STEP` lines narrate the work
  inside it, so the console always shows what the pipeline is busy with.
  The console drops the date, run id and project name. It formats numbers for reading, shortens paths
  to file names, and leaves `ARTIFACT` and `CHECKPOINT` lines in the file only. Stage banners, result
  tables and the end-of-run "Outputs written" list are plain text with no symbols.
- Downloads, extraction, dataset preparation, training epochs and test passes also draw green
  progress bars (`src/progress.py`). The bars are display only; nothing in them is needed from the
  log. Set `PATHFINDER_NO_PROGRESS=1` to turn them off.
- The console handler (`SafeStreamHandler`) survives a closed output stream: if a notebook or remote
  session drops the process's stdout, the run carries on and keeps writing `run.log`.
- When a run fails, the traceback appears once (from the innermost failing stage), followed by a
  `Pipeline failed: ...` block naming the log file, and `main.py` exits with status 1.

## Line format

```text
2026-10-21 14:09:52 | INFO | 20261021_130205 | campus_pathfinder | [METRICS] component=Classifier:resnet18 split=test images=45 accuracy=0.9333 top3_accuracy=1.0 macro_f1=0.931
└─ time               └─ level └─ run id        └─ project          └─ [STAGE] component=<name> key=value ...
```

Every line carries the run id, so lines from different runs can be told apart after they are
copied, grepped or merged.

## Stages

| Stage | Meaning |
|---|---|
| `START` / `DONE` / `FAIL` | Written by `component_stage`. `START` carries the stage's action in quotes ("Training ResNet18"). `DONE` and `FAIL` include `elapsed_sec`. `FAIL` is followed by the full traceback. |
| `STEP` | What the pipeline is doing now, in plain words, written by `log_step`: "Requesting download link", "Copying train split and remapping labels", "Checking the ONNX model gives the same answers as PyTorch". |
| `INFO` | Something worth knowing: resolved settings, image counts, reused outputs, early stopping. |
| `WARN` | Recoverable problem, such as a dropped label class or a building missing from a split. |
| `EVAL` | Training progress every epoch: loss and accuracy for the classifiers, loss and validation mAP for YOLO11n (from an Ultralytics callback). |
| `METRICS` | Final numbers for a dataset split, a model or an export. |
| `CHECKPOINT` | A model file was saved. |
| `ARTIFACT` | Any other file was written, with its path. |
| `RESULT` | Headline outcome of a component: best epoch, best model, a prediction. |

## Components

`Pipeline`, `Demo`, `Acquire`, `Prepare`, `Dataset`, `Classifier:<model>` and
`Classifier:<model>:test`, `Detector:<model>` and `Detector:<model>:test`, `Export`,
`Export:<model>`, `Evaluate`, `Figures`, `Predict:<model>`.

## How the code uses it

The logger is created once in `main.py` and passed down explicitly as `LOG`. Modules never create
their own logger and never call `print`; a test enforces that.

```python
from src.logger import component_stage, log_stage, log_step

def train_classifier(arch, prepared_dir, run_dir, LOG, ...):
    with component_stage(LOG, f"Classifier:{arch}", "Training ResNet18", epochs=epochs, device=device):
        log_step(LOG, f"Classifier:{arch}", "Loading train and validation images")
        ...
        log_stage(LOG, "EVAL", f"Classifier:{arch}", epoch=epoch, val_loss=0.41, val_accuracy=0.92)
        log_stage(LOG, "CHECKPOINT", f"Classifier:{arch}", epoch=epoch, path=ckpt_path)
```

Rules of thumb:

- Wrap each unit of work in `component_stage` with a plain-English action, so the console says
  what is starting and a crash always leaves a `[FAIL]` line and a traceback in `run.log`.
- Before each slow or notable step inside a stage, call `log_step` with what is about to happen.
- Use `log_stage` with keyword arguments instead of free text, so lines stay greppable.
  Keyword arguments set to `None` are left out.
- Log a path every time a file is written.
- Never log secrets. The Roboflow key is read from the environment and is not written anywhere.

## Reading a log

```bash
grep "\[METRICS\]\|\[RESULT\]" runs/run_*/logs/run.log     # the numbers
grep "\[EVAL\]" runs/run_*/logs/run.log                     # training curves
grep "\[FAIL\]" -A 30 runs/run_*/logs/run.log               # what broke
grep "elapsed_sec" runs/run_*/logs/run.log                  # where the time went
```

## Details

1. The logger name is `campus_pathfinder`.
2. Calling `setup_logger` again closes the old file handler and clears the old run-id filter, so
   re-running a setup cell in a notebook does not duplicate lines or leave files open.
3. `get_logger()` returns the run logger for code that cannot easily receive `LOG`.
4. `WARN` and `FAIL` lines are logged at WARNING and ERROR level; everything else at INFO.
5. `log_stage(..., file_only=True)` writes a line to `run.log` only, for details the console already
   shows in another form.
6. Ultralytics prints its own training table and progress bars while YOLO11n trains. A callback
   also writes one `[EVAL]` line per epoch (loss, validation mAP50 and mAP50-95) to the run log, so
   `run.log` holds the YOLO training curve as well.
