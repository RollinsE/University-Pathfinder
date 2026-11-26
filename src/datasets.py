"""Dataset preparation and image-level labels.

A Roboflow YOLO export is converted into a *prepared* dataset:

    <prepared_dir>/
    ├── data.yaml            Ultralytics dataset file (absolute path, canonical class names)
    ├── manifest.json        counts per split and class, dropped labels, multi-building images
    ├── train/images, train/labels
    ├── valid/images, valid/labels
    └── test/images,  test/labels

Class ids in the label files are remapped to the order in src/constants.BUILDINGS, so the
classifiers, the detector and the web app all agree on what class 3 means. Labels whose
class is not one of the buildings are dropped and counted.

The classifiers need one label per image. image_level_samples() gives each labelled image
the class of its largest box; images with no boxes (background shots) are left out of
classification but stay in the detector's dataset.
"""

from __future__ import annotations

import json
import os
import shutil
from collections import Counter

import yaml

from src.constants import CLASS_NAMES, IMAGE_EXTENSIONS, NUM_CLASSES, SPLITS, canonical_index
from src.logger import component_stage, log_stage, log_step
from src.progress import progress

# Roboflow names the validation folder "valid"; some exports use "val".
SPLIT_FOLDER_ALIASES = {"train": ("train",), "valid": ("valid", "val"), "test": ("test",)}


# --------------------------------------------------------------------------- label files


def parse_label_line(line: str):
    """One YOLO label line -> (class_id, cx, cy, w, h), all box values normalised to 0..1.

    Polygon lines (class followed by x y pairs) are reduced to their bounding box.
    Returns None for blank or malformed lines.
    """
    parts = line.split()
    if len(parts) < 5:
        return None
    try:
        class_id = int(float(parts[0]))
        values = [float(v) for v in parts[1:]]
    except ValueError:
        return None
    if len(values) == 4:
        cx, cy, w, h = values
    elif len(values) >= 6 and len(values) % 2 == 0:
        xs, ys = values[0::2], values[1::2]
        x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
        cx, cy, w, h = (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0
    else:
        return None
    if w <= 0 or h <= 0:
        return None
    return class_id, cx, cy, w, h


def read_label_file(path: str) -> list[tuple]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        rows = [parse_label_line(line) for line in f]
    return [r for r in rows if r is not None]


def largest_box_class(boxes: list[tuple]) -> int | None:
    """Class of the box with the largest area, or None for an image with no boxes."""
    if not boxes:
        return None
    return max(boxes, key=lambda b: b[3] * b[4])[0]


# --------------------------------------------------------------------------- export layout


def list_images(folder: str) -> list[str]:
    if not os.path.isdir(folder):
        return []
    return sorted(os.path.join(folder, name) for name in os.listdir(folder) if name.lower().endswith(IMAGE_EXTENSIONS))


def find_split_dir(export_dir: str, split: str) -> str | None:
    for name in SPLIT_FOLDER_ALIASES[split]:
        candidate = os.path.join(export_dir, name)
        if os.path.isdir(os.path.join(candidate, "images")):
            return candidate
    return None


def read_export_names(export_dir: str) -> list[str]:
    """Class names from an export's data.yaml (list or {id: name} mapping)."""
    yaml_path = os.path.join(export_dir, "data.yaml")
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"No data.yaml in {export_dir}. Is this a YOLO-format export?")
    with open(yaml_path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    names = data.get("names")
    if isinstance(names, dict):
        return [names[k] for k in sorted(names, key=int)]
    if isinstance(names, list):
        return names
    raise ValueError(f"{yaml_path} has no class names.")


def class_remap(export_names: list[str]) -> tuple[dict[int, int], list[str]]:
    """Map export class ids to canonical ids. Returns (mapping, names that matched nothing)."""
    mapping, unknown = {}, []
    for export_id, name in enumerate(export_names):
        index = canonical_index(name)
        if index is None:
            unknown.append(name)
        else:
            mapping[export_id] = index
    return mapping, unknown


def write_data_yaml(prepared_dir: str) -> str:
    """(Re)write data.yaml with the dataset's current absolute path.

    Called again before training, so a dataset that has moved (a different Colab session,
    a Drive folder) still resolves.
    """
    data = {
        "path": os.path.abspath(prepared_dir),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": NUM_CLASSES,
        "names": {i: name for i, name in enumerate(CLASS_NAMES)},
    }
    path = os.path.join(prepared_dir, "data.yaml")
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    return path


# --------------------------------------------------------------------------- preparation


def prepare_dataset(export_dir: str, prepared_dir: str, LOG, overwrite: bool = False) -> dict:
    """Copy an export into prepared_dir with canonical class ids. Returns the manifest."""
    manifest_path = os.path.join(prepared_dir, "manifest.json")
    if os.path.exists(manifest_path) and not overwrite:
        write_data_yaml(prepared_dir)
        manifest = load_manifest(prepared_dir)
        log_step(LOG, "Prepare", "Dataset already prepared, reusing it", images=manifest["total_images"])
        return manifest

    with component_stage(LOG, "Prepare", "Preparing dataset", source=export_dir, target=prepared_dir):
        log_step(LOG, "Prepare", "Reading class names from data.yaml")
        export_names = read_export_names(export_dir)
        mapping, unknown = class_remap(export_names)
        missing = sorted(set(range(NUM_CLASSES)) - set(mapping.values()))
        if unknown:
            log_stage(LOG, "WARN", "Prepare", dropped_classes="|".join(unknown), reason="not one of the buildings")
        if missing:
            raise ValueError(
                "The export has no labels for: " + ", ".join(CLASS_NAMES[i] for i in missing) + ". "
                "Check the class names in its data.yaml against src/constants.BUILDINGS."
            )

        if os.path.isdir(prepared_dir):
            shutil.rmtree(prepared_dir)

        manifest = {
            "source": os.path.abspath(export_dir),
            "export_names": export_names,
            "class_names": CLASS_NAMES,
            "dropped_classes": unknown,
            "splits": {},
        }
        for split in SPLITS:
            source = find_split_dir(export_dir, split)
            if source is None:
                raise FileNotFoundError(f"The export has no {split} split (looked for {SPLIT_FOLDER_ALIASES[split]}).")
            count = len(list_images(os.path.join(source, "images")))
            log_step(LOG, "Prepare", f"Copying {split} split and remapping labels", images=count)
            manifest["splits"][split] = _prepare_split(source, os.path.join(prepared_dir, split), mapping, split)

        manifest["total_images"] = sum(s["images"] for s in manifest["splits"].values())
        log_step(LOG, "Prepare", "Writing data.yaml and manifest.json")
        write_data_yaml(prepared_dir)
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        log_stage(LOG, "ARTIFACT", "Prepare", path=manifest_path)
        for split, stats in manifest["splits"].items():
            log_stage(
                LOG,
                "METRICS",
                "Prepare",
                split=split,
                images=stats["images"],
                labelled=stats["labelled_images"],
                background=stats["background_images"],
                boxes=stats["boxes"],
                multi_building=stats["multi_building_images"],
                dropped_boxes=stats["dropped_boxes"],
            )
        log_step(LOG, "Prepare", "Checking every building appears in every split")
        check_class_coverage(manifest, LOG)
    return manifest


def _prepare_split(source: str, target: str, mapping: dict[int, int], split: str) -> dict:
    image_out = os.path.join(target, "images")
    label_out = os.path.join(target, "labels")
    os.makedirs(image_out, exist_ok=True)
    os.makedirs(label_out, exist_ok=True)

    stats = {
        "images": 0,
        "labelled_images": 0,
        "background_images": 0,
        "boxes": 0,
        "dropped_boxes": 0,
        "multi_building_images": 0,
        "boxes_per_class": Counter(),
        "images_per_class": Counter(),
    }
    images = list_images(os.path.join(source, "images"))
    for image_path in progress(images, desc=f"Preparing {split}", unit="img"):
        stem = os.path.splitext(os.path.basename(image_path))[0]
        boxes = read_label_file(os.path.join(source, "labels", stem + ".txt"))
        kept = [(mapping[b[0]], *b[1:]) for b in boxes if b[0] in mapping]

        shutil.copy2(image_path, os.path.join(image_out, os.path.basename(image_path)))
        with open(os.path.join(label_out, stem + ".txt"), "w", encoding="utf-8") as f:
            for class_id, cx, cy, w, h in kept:
                f.write(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

        stats["images"] += 1
        stats["boxes"] += len(kept)
        stats["dropped_boxes"] += len(boxes) - len(kept)
        if kept:
            stats["labelled_images"] += 1
            stats["images_per_class"][CLASS_NAMES[largest_box_class(kept)]] += 1
            if len({b[0] for b in kept}) > 1:
                stats["multi_building_images"] += 1
            for b in kept:
                stats["boxes_per_class"][CLASS_NAMES[b[0]]] += 1
        else:
            stats["background_images"] += 1

    stats["boxes_per_class"] = {name: stats["boxes_per_class"].get(name, 0) for name in CLASS_NAMES}
    stats["images_per_class"] = {name: stats["images_per_class"].get(name, 0) for name in CLASS_NAMES}
    return stats


def check_class_coverage(manifest: dict, LOG) -> list[str]:
    """Warn about any building with no images in a split. Returns the warnings."""
    problems = []
    for split, stats in manifest["splits"].items():
        for name, count in stats["images_per_class"].items():
            if count == 0:
                problems.append(f"{split}:{name}")
    if problems:
        log_stage(LOG, "WARN", "Dataset", classes_without_images="|".join(problems))
    return problems


def load_manifest(prepared_dir: str) -> dict:
    with open(os.path.join(prepared_dir, "manifest.json"), encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------- image-level samples


def image_level_samples(prepared_dir: str, split: str) -> list[tuple[str, int]]:
    """(image path, class id) for every labelled image in a split, using the largest box."""
    folder = os.path.join(prepared_dir, split)
    samples = []
    for image_path in list_images(os.path.join(folder, "images")):
        stem = os.path.splitext(os.path.basename(image_path))[0]
        label = largest_box_class(read_label_file(os.path.join(folder, "labels", stem + ".txt")))
        if label is not None:
            samples.append((image_path, label))
    return samples


def image_level_targets(prepared_dir: str, split: str) -> list[tuple[str, int, tuple]]:
    """(image path, class id, box) for every labelled image, using its largest box.

    The box is the YOLO label's normalised (cx, cy, w, h): fractions of the image's width and height.
    """
    folder = os.path.join(prepared_dir, split)
    targets = []
    for image_path in list_images(os.path.join(folder, "images")):
        stem = os.path.splitext(os.path.basename(image_path))[0]
        boxes = read_label_file(os.path.join(folder, "labels", stem + ".txt"))
        if boxes:
            class_id, cx, cy, w, h = max(boxes, key=lambda b: b[3] * b[4])
            targets.append((image_path, class_id, (cx, cy, w, h)))
    return targets


def class_counts(samples: list[tuple[str, int]]) -> dict[str, int]:
    counts = Counter(label for _, label in samples)
    return {name: counts.get(i, 0) for i, name in enumerate(CLASS_NAMES)}
