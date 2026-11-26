"""Shared constants: the building classes, dataset defaults and model settings.

Every part of the pipeline (dataset preparation, the three models, evaluation, export and the
web app manifest) takes its class list and class order from BUILDINGS, so a class index means
the same building everywhere.
"""

from __future__ import annotations

import re

# --------------------------------------------------------------------------- buildings

# Alphabetical by display name. The index in this list is the class id used by every model.
BUILDINGS = [
    {
        "slug": "christchurch_house",
        "name": "Christchurch House",
        "short": "Christchurch",
        "aliases": ["christchurch", "christchurch house"],
    },
    {
        "slug": "dorset_house",
        "name": "Dorset House",
        "short": "Dorset",
        "aliases": ["dorset", "dorset house"],
    },
    {
        "slug": "fusion_building",
        "name": "Fusion Building",
        "short": "Fusion",
        "aliases": ["fusion", "fusion building"],
    },
    {
        "slug": "kimmeridge_house",
        "name": "Kimmeridge House",
        "short": "Kimmeridge",
        "aliases": ["kimmeridge", "kimmeridge house"],
    },
    {
        "slug": "poole_gateway_building",
        "name": "Poole Gateway Building",
        "short": "Poole Gateway",
        "aliases": ["poole gateway", "poole gateway building", "gateway building"],
    },
    {
        "slug": "poole_house",
        "name": "Poole House",
        "short": "Poole House",
        "aliases": ["poole house"],
    },
    {
        "slug": "sir_michael_cobham_library",
        "name": "Sir Michael Cobham Library",
        "short": "Library",
        "aliases": ["sir michael cobham library", "michael cobham library", "cobham library", "library"],
    },
    {
        "slug": "weymouth_house",
        "name": "Weymouth House",
        "short": "Weymouth",
        "aliases": ["weymouth", "weymouth house"],
    },
]

CLASS_SLUGS = [b["slug"] for b in BUILDINGS]
CLASS_NAMES = [b["name"] for b in BUILDINGS]
CLASS_SHORT = [b["short"] for b in BUILDINGS]  # for chart axes
NUM_CLASSES = len(BUILDINGS)


def normalise_name(name: str) -> str:
    """Lower-case, punctuation to spaces, common misspellings fixed, leading 'the' dropped."""
    text = re.sub(r"[^a-z0-9]+", " ", str(name).lower()).strip()
    text = text.replace("micheal", "michael")
    if text.startswith("the "):
        text = text[4:]
    return text


_ALIAS_TO_INDEX = {}
for _index, _building in enumerate(BUILDINGS):
    for _alias in [_building["slug"], _building["name"], *_building["aliases"]]:
        _ALIAS_TO_INDEX[normalise_name(_alias)] = _index


def canonical_index(name: str) -> int | None:
    """Class id for any spelling of a building name (dataset label, folder name, slug), or None."""
    return _ALIAS_TO_INDEX.get(normalise_name(name))


# --------------------------------------------------------------------------- dataset

ROBOFLOW_WORKSPACE = "computer-vision-6ydhl"
ROBOFLOW_PROJECT = "bu-building-dectection"  # the project's id on Roboflow, spelling included
ROBOFLOW_VERSION = 4  # 4: original images; 3: the same images with offline augmentation
ROBOFLOW_FORMAT = "yolov8"  # YOLO txt labels plus data.yaml, read directly by Ultralytics
ROBOFLOW_API_URL = "https://api.roboflow.com"
ROBOFLOW_UNIVERSE_URL = f"https://universe.roboflow.com/{ROBOFLOW_WORKSPACE}/{ROBOFLOW_PROJECT}"

SPLITS = ("train", "valid", "test")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# --------------------------------------------------------------------------- models

CLASSIFIERS = ("resnet18", "mobilenet_v3_large")
DETECTORS = ("yolo11n",)
ALL_MODELS = CLASSIFIERS + DETECTORS

MODEL_LABELS = {
    "resnet18": "ResNet18",
    "mobilenet_v3_large": "MobileNetV3-Large",
    "yolo11n": "YOLO11n",
}

MODEL_ROLES = {
    "resnet18": "baseline classifier + box",
    "mobilenet_v3_large": "lightweight classifier + box",
    "yolo11n": "detector",
}

# Classify-and-locate input: the whole photo resized to this square. The web app reads the same
# values from models/manifest.json.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLASSIFIER_INPUT = 224

# Detector settings.
DETECTOR_IMGSZ = 640
DETECTOR_CONF = 0.25
DETECTOR_IOU = 0.45
LETTERBOX_FILL = 114
# How a photo is fitted into the detector's square input: "letterbox" keeps its shape and pads
# the edges (Ultralytics' own default); "stretch" squashes it to a square, as Roboflow did to
# the training photos.
DETECTOR_RESIZE_MODES = ("letterbox", "stretch")
DETECTOR_RESIZE = "letterbox"

DEFAULT_SEED = 42
