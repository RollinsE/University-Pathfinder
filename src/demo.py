"""Synthetic demo data in the same layout as a Roboflow YOLO export.

Each building is drawn as a block with its own colour, roof and window pattern, on a day or
night sky, at a random position and scale. Some images contain a second, smaller building
and some contain none, so the demo exercises the same cases as the real data: multi-building
images, background images and a class that is not a building (dropped during preparation).

The demo needs no account or download and runs the whole pipeline on a CPU in minutes. Its
numbers say nothing about the real task.
"""

from __future__ import annotations

import os
import random
import shutil

import yaml

from src.constants import CLASS_NAMES
from src.logger import component_stage, log_stage, log_step
from src.progress import progress

# Roboflow-style names, plus one label that is not a building.
DEMO_EXPORT_NAMES = [name.replace(" ", "-") for name in CLASS_NAMES] + ["Bicycle"]
DEMO_SPLIT_SIZES = {"train": 10, "valid": 3, "test": 3}  # images per building per split
DEMO_BACKGROUND = {"train": 4, "valid": 2, "test": 2}
IMAGE_SIZE = (320, 240)

PALETTE = [
    (196, 120, 82),
    (214, 196, 160),
    (90, 120, 160),
    (150, 150, 150),
    (60, 140, 110),
    (180, 90, 120),
    (120, 100, 70),
    (230, 230, 230),
]


def _draw_building(draw, class_id: int, box, night: bool, rng: random.Random) -> None:
    x0, y0, x1, y1 = box
    colour = PALETTE[class_id]
    if night:
        colour = tuple(int(c * 0.45) for c in colour)
    draw.rectangle(box, fill=colour)

    roof = class_id % 3
    if roof == 1:
        draw.polygon([(x0, y0), ((x0 + x1) // 2, y0 - (y1 - y0) // 4), (x1, y0)], fill=colour)
    elif roof == 2:
        draw.rectangle((x0 + (x1 - x0) // 4, y0 - (y1 - y0) // 6, x1 - (x1 - x0) // 4, y0), fill=colour)

    cols, rows = 2 + class_id % 4, 2 + class_id // 3
    lit = (255, 230, 140)
    dark = (40, 50, 70)
    w, h = (x1 - x0) / (cols * 2 + 1), (y1 - y0) / (rows * 2 + 1)
    for r in range(rows):
        for c in range(cols):
            wx, wy = x0 + w * (2 * c + 1), y0 + h * (2 * r + 1)
            fill = lit if night and rng.random() < 0.7 else dark
            draw.rectangle((wx, wy, wx + w, wy + h), fill=fill)


def _random_box(rng: random.Random, scale: tuple[float, float]):
    width, height = IMAGE_SIZE
    bw = int(width * rng.uniform(*scale))
    bh = int(height * rng.uniform(scale[0] * 0.8, scale[1] * 0.9))
    x0 = rng.randint(0, width - bw)
    y0 = rng.randint(height // 6, height - bh)
    return x0, y0, x0 + bw, y0 + bh


def _yolo_line(class_id: int, box) -> str:
    width, height = IMAGE_SIZE
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2 / width, (y0 + y1) / 2 / height
    return f"{class_id} {cx:.6f} {cy:.6f} {(x1 - x0) / width:.6f} {(y1 - y0) / height:.6f}"


def _render(path: str, rng: random.Random, buildings: list[int], night: bool) -> list[str]:
    from PIL import Image, ImageDraw

    sky = (25, 30, 60) if night else (150, 190, 230)
    ground = (30, 45, 30) if night else (90, 140, 80)
    image = Image.new("RGB", IMAGE_SIZE, sky)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, IMAGE_SIZE[1] * 5 // 6, IMAGE_SIZE[0], IMAGE_SIZE[1]), fill=ground)

    lines = []
    for i, class_id in enumerate(buildings):
        scale = (0.45, 0.7) if i == 0 else (0.15, 0.25)
        box = _random_box(rng, scale)
        _draw_building(draw, class_id, box, night, rng)
        lines.append(_yolo_line(class_id, box))
    if rng.random() < 0.2:  # a few non-building labels, which preparation must drop
        box = _random_box(rng, (0.05, 0.1))
        draw.ellipse(box, fill=(200, 40, 40))
        lines.append(_yolo_line(len(CLASS_NAMES), box))
    image.save(path, quality=90)
    return lines


def make_demo_export(export_dir: str, LOG, seed: int = 42, overwrite: bool = False) -> str:
    """Write a synthetic Roboflow-style export to export_dir. Returns export_dir."""
    if os.path.exists(os.path.join(export_dir, "data.yaml")) and not overwrite:
        log_step(LOG, "Demo", "Demo dataset already drawn, reusing it", folder=export_dir)
        return export_dir

    with component_stage(LOG, "Demo", "Drawing synthetic demo dataset", target=export_dir, seed=seed):
        if os.path.isdir(export_dir):
            shutil.rmtree(export_dir)
        rng = random.Random(seed)
        total = 0
        for split, per_class in DEMO_SPLIT_SIZES.items():
            image_dir = os.path.join(export_dir, split, "images")
            label_dir = os.path.join(export_dir, split, "labels")
            os.makedirs(image_dir, exist_ok=True)
            os.makedirs(label_dir, exist_ok=True)

            jobs = [(c, i) for c in range(len(CLASS_NAMES)) for i in range(per_class)]
            jobs += [(None, i) for i in range(DEMO_BACKGROUND[split])]
            log_step(LOG, "Demo", f"Drawing {split} split", images=len(jobs))
            for class_id, i in progress(jobs, desc=f"Drawing {split}", unit="img"):
                night = i % 2 == 1
                if class_id is None:
                    buildings, stem = [], f"background_{i:03d}"
                else:
                    buildings = [class_id]
                    if rng.random() < 0.25:
                        buildings.append(rng.choice([c for c in range(len(CLASS_NAMES)) if c != class_id]))
                    stem = f"{CLASS_NAMES[class_id].replace(' ', '_').lower()}_{i:03d}"
                lines = _render(os.path.join(image_dir, stem + ".jpg"), rng, buildings, night)
                with open(os.path.join(label_dir, stem + ".txt"), "w", encoding="utf-8") as f:
                    f.write("\n".join(lines) + ("\n" if lines else ""))
                total += 1

        with open(os.path.join(export_dir, "data.yaml"), "w", encoding="utf-8") as f:
            yaml.safe_dump(
                {
                    "train": "../train/images",
                    "val": "../valid/images",
                    "test": "../test/images",
                    "nc": len(DEMO_EXPORT_NAMES),
                    "names": DEMO_EXPORT_NAMES,
                },
                f,
                sort_keys=False,
            )
        log_stage(LOG, "INFO", "Demo", images=total, classes=len(CLASS_NAMES))
    return export_dir
