"""Generate notebooks/colab.ipynb, a thin Google Colab driver for main.py.

The notebook only provides storage, credentials and the commands; all logic lives in
main.py and src/. It is generated rather than edited by hand:

    python scripts/build_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "colab.ipynb"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": text.strip("\n").splitlines(keepends=True),
    }


CELLS = [
    md("""
# Campus Pathfinder: training on Google Colab

Trains and compares three models that recognise eight buildings on Bournemouth University's
Talbot Campus, then exports them for the web app:

| Model | Role |
|---|---|
| ResNet18 | baseline: names the building and boxes it |
| MobileNetV3-Large | lightweight: names the building and boxes it |
| YOLO11n | detector: finds and boxes every building; used by the web app |

**Before you start**

1. Add a Colab secret called `ROBOFLOW_API_KEY` (key icon in the left sidebar) holding the private
   API key from your own Roboflow workspace settings, and allow this notebook to read it.
2. Have the repository zip on your computer (e.g. `university-pathfinder.zip`). The code
   cell asks you to upload it. It lives in this Colab session only, never on Google Drive, so
   uploading a corrected zip replaces it completely.

**Saving compute units**

1. On the free CPU runtime, set `PREPARE_ONLY = True` and **Runtime > Run all**. This checks the
   setup with the smoke test and downloads the dataset, then stops. No GPU time is used.
2. Switch to **Runtime > Change runtime type > T4 GPU**, set `PREPARE_ONLY = False` and run all
   again. The dataset is reused from Drive; only training uses GPU time.

**When something goes wrong**

Any stage that fails stops the notebook and points to `run.log`. Fix the code on your computer,
zip it, re-run the code cell to upload the new zip, then re-run from the stage that failed.
Results are kept on Google Drive under `campus_pathfinder/` (`SAVE_TO_DRIVE = True`), and the run
folder name is fixed. Running again after a disconnect or a fix skips finished models, and a model
that was interrupted (any of the three) carries on from its last finished epoch.

Set `DEMO = True` to run the whole notebook on synthetic data instead, with no Roboflow key.
"""),
    md("## 1. Settings"),
    code("""
DEMO = False  # True = synthetic data, a few minutes, no Roboflow key needed
SMOKE_TEST = True  # quick end-to-end check on synthetic data before the real training
PREPARE_ONLY = False  # True = stop after the smoke test and dataset download (use a free CPU runtime)
DEVICE = "cuda"  # "cuda" = train on the GPU and stop with a clear message if there is none

# Which Roboflow dataset to train on:
#   workspace "rollins", project "bu-building-dectection-1jdtq", version 3: the photos used for the
#     published models;
#   workspace "computer-vision-6ydhl", project "bu-building-dectection", version 4: the project
#     team's dataset (version 3 there adds Roboflow augmentation).
ROBOFLOW_WORKSPACE = "rollins"
ROBOFLOW_PROJECT = "bu-building-dectection-1jdtq"
DATASET_VERSION = 3

# Where the code comes from:
#   "upload" - a zip uploaded from your computer into this session (nothing stored on Drive)
#   "drive"  - a zip already on Google Drive, at DRIVE_ZIP_PATH
#   "github" - the repository on GitHub
CODE_SOURCE = "upload"
DRIVE_ZIP_PATH = "/content/drive/MyDrive/university-pathfinder.zip"
REPO_URL = "https://github.com/RollinsE/University-Pathfinder.git"
BRANCH = "master"

# Where data, checkpoints, logs and exports go. True keeps them on Google Drive, so a disconnect
# or a code fix never costs finished training. False keeps them in this session only.
SAVE_TO_DRIVE = True

CODE_DIR = "/content/university-pathfinder"
OUTPUT_ROOT = "/content/drive/MyDrive" if SAVE_TO_DRIVE else "/content"
PROJECT_DIR = f"{OUTPUT_ROOT}/campus_pathfinder" + ("_demo" if DEMO else "")
RUN_NAME = "run_colab"  # one name per experiment; reuse a name to continue or resume that run
RUN_DIR = f"{PROJECT_DIR}/runs/{RUN_NAME}"
DATA_DIR = f"{PROJECT_DIR}/data"
WEB_DIR = f"{RUN_DIR}/webapp"  # exported models are kept per run, so runs can be compared
FIGURES_DIR = f"{RUN_DIR}/figures"
"""),
    md("## 2. Storage, code and dependencies"),
    code("""
import os

if SAVE_TO_DRIVE or CODE_SOURCE == "drive":
    from google.colab import drive

    drive.mount("/content/drive")
os.makedirs(PROJECT_DIR, exist_ok=True)
print(f"Outputs go to {PROJECT_DIR}")
"""),
    md("""
### Code

Re-run this cell whenever you have a corrected version: it replaces the code completely. Stages
already finished stay finished, because their results are kept in the output folder.
"""),
    code("""
import glob
import shutil
import tempfile
import zipfile


def install_code_from_zip(zip_path: str) -> None:
    \"\"\"Replace CODE_DIR with the repository inside zip_path, whatever its top folder is called.\"\"\"
    staging = tempfile.mkdtemp(dir="/content")
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(staging)
    roots = [
        os.path.dirname(p)
        for p in glob.glob(f"{staging}/**/main.py", recursive=True)
        if os.path.isdir(os.path.join(os.path.dirname(p), "src"))
    ]
    if not roots:
        shutil.rmtree(staging, ignore_errors=True)
        raise FileNotFoundError(f"{os.path.basename(zip_path)} has no main.py with a src/ folder. Is it the repository zip?")
    shutil.rmtree(CODE_DIR, ignore_errors=True)
    shutil.move(min(roots, key=len), CODE_DIR)
    shutil.rmtree(staging, ignore_errors=True)
    count = sum(len(files) for _, _, files in os.walk(CODE_DIR))
    print(f"Code ready: {count} files from {os.path.basename(zip_path)}")


%cd /content
if CODE_SOURCE == "upload":
    from google.colab import files

    print("Choose the repository zip on your computer.")
    uploaded = files.upload()
    zips = [name for name in uploaded if name.lower().endswith(".zip")]
    if not zips:
        raise ValueError("No .zip file was uploaded.")
    install_code_from_zip(zips[-1])
    for name in uploaded:
        os.remove(name)
elif CODE_SOURCE == "drive":
    if not os.path.exists(DRIVE_ZIP_PATH):
        raise FileNotFoundError(f"No zip at {DRIVE_ZIP_PATH}.")
    install_code_from_zip(DRIVE_ZIP_PATH)
elif CODE_SOURCE == "github":
    if os.path.isdir(f"{CODE_DIR}/.git"):
        !git -C {CODE_DIR} pull --ff-only
    else:
        !rm -rf {CODE_DIR}
        !git clone --branch {BRANCH} {REPO_URL} {CODE_DIR}
else:
    raise ValueError(f"CODE_SOURCE must be 'upload', 'drive' or 'github', not {CODE_SOURCE!r}.")
%cd {CODE_DIR}
"""),
    code("""
!pip install -q -r requirements-colab.txt
"""),
    code("""
import shlex

from IPython import get_ipython

EXIT_FILE = "/tmp/pathfinder_exit_code"


def run(args: str) -> None:
    \"\"\"Run main.py and stop the notebook if the stage fails. Arguments may span several lines.\"\"\"
    args = shlex.join(shlex.split(args))
    get_ipython().system(f"python main.py {args}; echo $? > {EXIT_FILE}")
    with open(EXIT_FILE) as f:
        code = int(f.read().strip() or 1)
    if code:
        raise RuntimeError(
            f"main.py {args.split()[1]} failed (exit code {code}). Details are in {RUN_DIR}/logs/run.log. "
            "Fix the code, re-run the Code cell to upload it, then re-run from this stage."
        )
"""),
    code("""
import torch

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
elif PREPARE_ONLY or DEMO:
    print("CPU runtime: fine for the smoke test and dataset download.")
elif DEVICE.startswith("cuda"):
    raise RuntimeError(
        "No GPU in this session, so training would run on the CPU. Runtime > Change runtime type > "
        "T4 GPU > Save, wait for it to reconnect, then run all again. If Colab cannot give you a GPU "
        "it falls back to CPU, so check the top-right corner says T4."
    )
"""),
    md("## 3. Roboflow API key"),
    code("""
if not DEMO:
    from google.colab import userdata

    try:
        os.environ["ROBOFLOW_API_KEY"] = userdata.get("ROBOFLOW_API_KEY")
        print("Roboflow API key loaded from Colab secrets.")
    except Exception as exc:
        raise RuntimeError(
            "Could not read the ROBOFLOW_API_KEY secret. Add it in the key icon on the left "
            "and allow this notebook to access it."
        ) from exc
"""),
    code("""
!python scripts/doctor.py
"""),
    md("""
## 4. Smoke test

The whole pipeline (dataset, all three models, testing, export, results and figures) on tiny
synthetic data with one epoch per model, in a throwaway folder, then a check that an interrupted
classifier resumes where it stopped. It takes a few minutes and catches
setup problems before the real training. Its numbers mean nothing. Set `SMOKE_TEST = False` to skip.
"""),
    code("""
if SMOKE_TEST:
    SMOKE_DIR = "/content/smoke"
    !rm -rf {SMOKE_DIR}
    run(
        f"--mode all --demo --run_dir {SMOKE_DIR}/runs/run_smoke --data_dir {SMOKE_DIR}/data "
        f"--web_dir {SMOKE_DIR}/webapp --epochs 1 --yolo_epochs 1"
    )
    for path in ("models/manifest.json", "models/yolo11n.onnx", "models/resnet18.onnx", "models/mobilenet_v3_large.onnx"):
        assert os.path.exists(f"{SMOKE_DIR}/webapp/{path}"), f"Smoke test did not produce {path}"

    # Resume check: stop a classifier on purpose after epoch 1, run it again, and expect it to
    # continue from epoch 2 rather than start over.
    resume_args = (
        f"--mode classify --models mobilenet_v3_large --demo --epochs 2 "
        f"--run_dir {SMOKE_DIR}/runs/run_resume --dataset_dir {SMOKE_DIR}/data/processed/demo"
    )
    os.environ["PATHFINDER_STOP_AFTER_EPOCH"] = "1"
    get_ipython().system(f"python main.py {resume_args} > /dev/null 2>&1")
    del os.environ["PATHFINDER_STOP_AFTER_EPOCH"]
    run(resume_args)
    with open(f"{SMOKE_DIR}/runs/run_resume/logs/run.log") as f:
        assert "Resuming interrupted training from epoch 2" in f.read(), "The classifier did not resume."
    print("Smoke test passed: training, resuming, testing, export and logging all work.")
"""),
    md("""
## 5. Training

One cell per stage, all writing to the same run folder and `run.log`. The training settings are
written out as command-line options in each cell: edit them there (`python main.py --help` lists
every option). Finished models are skipped when a cell is run again and interrupted ones resume;
add `--overwrite` to a command to retrain from scratch after changing its settings.
"""),
    code("""
DEMO_FLAG = "--demo" if DEMO else ""
COMMON = f"--run-dir {RUN_DIR} --data-dir {DATA_DIR} {DEMO_FLAG}"
run(f"--mode acquire {COMMON} --workspace {ROBOFLOW_WORKSPACE} --project {ROBOFLOW_PROJECT} --version {DATASET_VERSION}")
"""),
    code("""
# Training reads every image every epoch. Reading many small files from Google Drive is slow, so
# train from a copy on this session's local disk. Results and checkpoints still go to Drive.
import shutil

prepared_name = "demo" if DEMO else f"{ROBOFLOW_PROJECT}-v{DATASET_VERSION}"
DATASET_DIR = f"/content/local_data/{prepared_name}"
if not os.path.exists(f"{DATASET_DIR}/manifest.json"):
    shutil.rmtree(DATASET_DIR, ignore_errors=True)
    shutil.copytree(f"{DATA_DIR}/processed/{prepared_name}", DATASET_DIR)
print(f"Training from the local copy: {DATASET_DIR}")
"""),
    code("""
if PREPARE_ONLY:
    raise SystemExit(
        "Setup checked and dataset downloaded. Switch to a T4 GPU runtime, set PREPARE_ONLY = False "
        "and run all again to train."
    )
"""),
    md("### ResNet18 (baseline classifier)"),
    code("""
run(f\"\"\"
--mode classify --models resnet18
--run-dir {RUN_DIR} --data-dir {DATA_DIR} --dataset-dir {DATASET_DIR} {DEMO_FLAG}
--epochs 30 --patience 8
--batch-size 32 --lr 3e-4 --weight-decay 1e-4 --label-smoothing 0.1 --box-weight 2.0
--workers 2 --seed 42 --device {DEVICE}
\"\"\")
"""),
    md("### MobileNetV3-Large (lightweight classifier)"),
    code("""
run(f\"\"\"
--mode classify --models mobilenet_v3_large
--run-dir {RUN_DIR} --data-dir {DATA_DIR} --dataset-dir {DATASET_DIR} {DEMO_FLAG}
--epochs 30 --patience 8
--batch-size 32 --lr 3e-4 --weight-decay 1e-4 --label-smoothing 0.1 --box-weight 2.0
--workers 2 --seed 42 --device {DEVICE}
\"\"\")
"""),
    md(
        "### YOLO11n (detector)\n\nThe longest stage. Ultralytics prints its own per-epoch table; each epoch is also written to `run.log`."
    ),
    code("""
run(f\"\"\"
--mode detect --models yolo11n
--run-dir {RUN_DIR} --data-dir {DATA_DIR} --dataset-dir {DATASET_DIR} {DEMO_FLAG}
--yolo-epochs 250 --yolo-patience 75
--yolo-imgsz 640 --yolo-batch 16
--workers 2 --seed 42 --device {DEVICE}
\"\"\")
"""),
    md("## 6. Export and compare"),
    code("""
# The figures' PNG versions (for the README) are drawn by Kaleido, which needs a headless Chrome and
# its system libraries. The interactive figures work without them.
!for p in libnss3 libatk-bridge2.0-0t64 libcups2t64 libxcomposite1 libxdamage1 libxfixes3 libxrandr2 libgbm1 libxkbcommon0 libpango-1.0-0 libcairo2 libasound2t64; do apt-get install -y -qq "$p" > /dev/null 2>&1 || echo "could not install $p"; done
!plotly_get_chrome -y > /dev/null
"""),
    code("""
YOLO_RESIZE = "letterbox"  # or "stretch": how the web app fits photos into YOLO11n's square input
run(f"--mode export {COMMON} --dataset-dir {DATASET_DIR} --web-dir {WEB_DIR} --yolo-resize {YOLO_RESIZE}")
run(f"--mode evaluate {COMMON} --figures_dir {FIGURES_DIR}")
"""),
    code("""
import importlib

import src.notebook_viewer

importlib.reload(src.notebook_viewer)  # pick up a re-uploaded version
src.notebook_viewer.display_run(RUN_DIR)
"""),
    md("""
## 7. Try your own photos

Upload one or more photos. Each one goes through all three models, and the photo is shown with
each model's answer drawn on it: YOLO11n boxes every building it finds, and ResNet18 and
MobileNetV3-Large each box the one building they name, with their next guesses in the strip above.
The annotated copies are saved in the run's `predictions/` folder.
"""),
    code("""
from google.colab import files
from IPython.display import Image as ShowImage
from IPython.display import Markdown, display

uploaded = files.upload()
for name in uploaded:
    run(f'--mode predict --model all --image "{name}" --web-dir {WEB_DIR} --run-dir {RUN_DIR}')
    stem = os.path.splitext(os.path.basename(name))[0]
    display(Markdown(f"### {name}"))
    for model in ("yolo11n", "mobilenet_v3_large", "resnet18"):
        path = f"{RUN_DIR}/predictions/{stem}_{model}.jpg"
        if os.path.exists(path):
            display(ShowImage(path, width=720))
"""),
    md("""
### Try YOLO11n with other settings

Runs the same photos through YOLO11n four ways, without re-exporting or retraining:

- **letterbox** keeps each photo's shape and pads the edges grey (Ultralytics' default).
- **stretch** squashes the photo to a square, as Roboflow did to the training photos.
- **conf 0.1** also shows weaker detections (the app hides anything under 25%), to see what it
  nearly found.

The headline names any setting that differs from the exported web app. If one setting is clearly
better on your photos, set `YOLO_RESIZE` in the export cell and run it again so the app uses it.
"""),
    code("""
import glob

for name in uploaded:
    stem = os.path.splitext(os.path.basename(name))[0]
    for old in glob.glob(f"{RUN_DIR}/predictions/{stem}_yolo11n*.jpg"):
        os.remove(old)
    for resize in ("letterbox", "stretch"):
        for conf in (0.25, 0.1):
            run(
                f'--mode predict --model yolo11n --image "{name}" --web-dir {WEB_DIR} --run-dir {RUN_DIR} '
                f"--yolo-resize {resize} --conf {conf}"
            )
    display(Markdown(f"### {name}: YOLO11n settings"))
    for path in sorted(glob.glob(f"{RUN_DIR}/predictions/{stem}_yolo11n*.jpg")):
        display(ShowImage(path, width=720))
"""),
    md("""
## 8. Take the results home

Everything is already on your Drive under `campus_pathfinder/runs/<RUN_NAME>/`. This cell also
downloads three zips, each named after the run:

- `pathfinder_webapp_<RUN_NAME>.zip`: the web app with its models and sample photos. Copy `models/` and
  `samples/` into the repository's `webapp/` folder and commit them to publish the app.
- `pathfinder_figures_<RUN_NAME>.zip`: the figures for `docs/figures/`.
- `pathfinder_metrics_<RUN_NAME>.zip`: the results table (`results.md`), metrics and `run.log`.
"""),
    code("""
import shutil

shutil.make_archive(f"{PROJECT_DIR}/pathfinder_webapp_{RUN_NAME}", "zip", WEB_DIR)
shutil.make_archive(f"{PROJECT_DIR}/pathfinder_figures_{RUN_NAME}", "zip", FIGURES_DIR)
metrics_bundle = f"{PROJECT_DIR}/metrics_bundle"
shutil.rmtree(metrics_bundle, ignore_errors=True)
shutil.copytree(f"{RUN_DIR}/metrics", metrics_bundle)
shutil.copy(f"{RUN_DIR}/logs/run.log", metrics_bundle)
shutil.make_archive(f"{PROJECT_DIR}/pathfinder_metrics_{RUN_NAME}", "zip", metrics_bundle)

for name in ("pathfinder_webapp", "pathfinder_figures", "pathfinder_metrics"):
    files.download(f"{PROJECT_DIR}/{name}_{RUN_NAME}.zip")
"""),
]


def build() -> dict:
    return {
        "cells": CELLS,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "T4"},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(build(), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
