import json
import os

import pytest
import yaml

from src.constants import CLASS_NAMES, SPLITS
from src.datasets import (
    class_remap,
    image_level_samples,
    largest_box_class,
    load_manifest,
    parse_label_line,
    prepare_dataset,
    read_label_file,
)


def test_parse_box_polygon_and_bad_lines():
    assert parse_label_line("3 0.5 0.5 0.2 0.4") == (3, 0.5, 0.5, 0.2, 0.4)
    cls, cx, cy, w, h = parse_label_line("1 0.1 0.1 0.5 0.1 0.5 0.3 0.1 0.3")
    assert cls == 1
    assert (round(cx, 3), round(cy, 3), round(w, 3), round(h, 3)) == (0.3, 0.2, 0.4, 0.2)
    assert parse_label_line("") is None
    assert parse_label_line("x 0.5 0.5 0.2 0.2") is None
    assert parse_label_line("2 0.5 0.5 0 0.2") is None


def test_largest_box_wins():
    boxes = [(1, 0.5, 0.5, 0.1, 0.1), (4, 0.5, 0.5, 0.6, 0.5), (2, 0.2, 0.2, 0.3, 0.3)]
    assert largest_box_class(boxes) == 4
    assert largest_box_class([]) is None


def test_class_remap_reorders_and_drops_unknown():
    names = list(reversed([n.replace(" ", "-") for n in CLASS_NAMES])) + ["buildings"]
    mapping, unknown = class_remap(names)
    assert unknown == ["buildings"]
    assert mapping[0] == len(CLASS_NAMES) - 1
    assert len(mapping) == len(CLASS_NAMES)


def _write_export(root, names, labels_by_split):
    for split, items in labels_by_split.items():
        os.makedirs(root / split / "images", exist_ok=True)
        os.makedirs(root / split / "labels", exist_ok=True)
        for stem, lines in items.items():
            (root / split / "images" / f"{stem}.jpg").write_bytes(b"not really a jpeg")
            if lines is not None:
                (root / split / "labels" / f"{stem}.txt").write_text("\n".join(lines) + "\n")
    (root / "data.yaml").write_text(yaml.safe_dump({"names": names}))


def test_prepare_remaps_ids_and_counts(tmp_path, run_logger):
    LOG, _, _ = run_logger
    export_names = ["buildings"] + list(reversed(CLASS_NAMES))  # unknown first, buildings reversed
    every_class = {f"img_{i}": [f"{i + 1} 0.5 0.5 0.4 0.4"] for i in range(len(CLASS_NAMES))}
    labels = {split: dict(every_class) for split in SPLITS}
    labels["train"]["mixed"] = ["0 0.5 0.5 0.9 0.9", "1 0.5 0.5 0.6 0.6", "2 0.2 0.2 0.1 0.1"]
    labels["train"]["background"] = None
    export = tmp_path / "export"
    _write_export(export, export_names, labels)

    manifest = prepare_dataset(str(export), str(tmp_path / "prepared"), LOG)
    train = manifest["splits"]["train"]
    assert manifest["dropped_classes"] == ["buildings"]
    assert train["background_images"] == 1
    assert train["dropped_boxes"] == 1
    assert train["multi_building_images"] == 1

    # export id 1 is the last building alphabetically -> canonical id 7
    assert read_label_file(str(tmp_path / "prepared" / "train" / "labels" / "img_0.txt"))[0][0] == 7
    # in "mixed" the unknown full-frame box is dropped, so the largest kept box (export id 1) wins
    samples = dict(image_level_samples(str(tmp_path / "prepared"), "train"))
    mixed = [p for p in samples if p.endswith("mixed.jpg")][0]
    assert samples[mixed] == 7

    data = yaml.safe_load((tmp_path / "prepared" / "data.yaml").read_text())
    assert data["names"][0] == CLASS_NAMES[0]
    assert os.path.isabs(data["path"])
    assert load_manifest(str(tmp_path / "prepared"))["total_images"] == manifest["total_images"]


def test_prepare_refuses_an_export_missing_a_building(tmp_path, run_logger):
    LOG, _, _ = run_logger
    names = [n.replace(" ", "-") for n in CLASS_NAMES[:-1]]
    _write_export(tmp_path / "export", names, {s: {"a": ["0 0.5 0.5 0.2 0.2"]} for s in SPLITS})
    with pytest.raises(ValueError):
        prepare_dataset(str(tmp_path / "export"), str(tmp_path / "prepared"), LOG)


def test_demo_dataset_has_every_building_in_every_split(demo_dataset):
    manifest = load_manifest(demo_dataset)
    for split in SPLITS:
        counts = manifest["splits"][split]["images_per_class"]
        assert all(counts[name] > 0 for name in CLASS_NAMES), (split, counts)
    assert manifest["dropped_classes"] == ["Bicycle"]
    record = json.loads(json.dumps(manifest))
    assert record["splits"]["test"]["background_images"] > 0


def test_image_level_targets_give_the_largest_box(demo_dataset):
    from src.datasets import image_level_samples, image_level_targets

    targets = image_level_targets(demo_dataset, "test")
    samples = image_level_samples(demo_dataset, "test")
    assert [(p, c) for p, c, _ in targets] == samples
    for _, _, (cx, cy, w, h) in targets:
        assert 0 < w <= 1 and 0 < h <= 1 and 0 <= cx <= 1 and 0 <= cy <= 1
