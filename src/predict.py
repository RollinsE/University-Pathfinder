"""Predict the building in a photo with the exported ONNX models, and draw the answer on it.

Uses the same files and the same pre- and post-processing as the web app:

    python main.py --mode predict --image photo.jpg                      # the detector
    python main.py --mode predict --image photo.jpg --model all          # all three models
    python main.py --mode predict --image photo.jpg --model mobilenet_v3_large
    python main.py --mode predict --image photo.jpg --yolo_resize stretch --conf 0.1   # try settings

For each model, the photo is saved with the result drawn on it to
<run>/predictions/<photo>_<model>.jpg, with a headline giving the answer in words:

- YOLO11n (detector): a numbered box round every building it found (at most one per building),
  with name and confidence. --yolo_resize and --conf try other settings without re-exporting;
  the saved photo's name then says which, e.g. <photo>_yolo11n_stretch_conf0.1.jpg.
- ResNet18 and MobileNetV3-Large (classify-and-locate): one box round the building they name,
  with its name and confidence; their next two guesses are listed in the headline strip.

The notebook displays these images.
"""

from __future__ import annotations

import json
import os

from src.constants import ALL_MODELS, CLASS_NAMES, CLASSIFIERS, DETECTOR_CONF, DETECTOR_RESIZE, MODEL_LABELS
from src.evaluate import top_k
from src.logger import component_stage, log_stage, log_step

UNSURE_BELOW = 0.5
MAX_SIDE = 1280
PRIMARY = (25, 164, 99)  # the detector's most confident box (same green as the web app)
SECONDARY = (42, 120, 214)  # its other boxes
CLASSIFIER_COLOUR = {"resnet18": (235, 104, 52), "mobilenet_v3_large": (20, 140, 98)}  # as in the figures


def confidence_word(p: float) -> str:
    """The same words the web app uses."""
    if p >= 0.75:
        return "Confident"
    if p >= UNSURE_BELOW:
        return "Fairly sure"
    return "Not sure"


def headline(result: dict) -> str:
    label = MODEL_LABELS[result["model"]]
    if result.get("settings"):
        label = f"{label} ({result['settings']})"
    if result["kind"] == "detector":
        if not result["detections"]:
            return f"{label}: no campus building found"
        best = result["detections"][0]
        answer, p = best["building"], best["confidence"]
    else:
        best = result["top3"][0]
        answer, p = best["building"], best["probability"]
    lead = "it might be" if p < UNSURE_BELOW else "this looks like"
    return f"{label}: {lead} {answer} ({confidence_word(p)}, {p:.0%})"


def _font(size: int):
    from PIL import ImageFont

    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow older than 10.1
        return ImageFont.load_default()


def _banner_lines(result: dict) -> list[str]:
    """Headline first, then one line per alternative answer."""
    lines = [headline(result)]
    if result["kind"] == "classifier":
        lines += [f"   also possible: {i['building']} {i['probability']:.0%}" for i in result["top3"][1:]]
    else:
        lines += [
            f"   {n}  {d['building']} {d['confidence']:.0%}" for n, d in enumerate(result["detections"][1:], start=2)
        ]
    return lines


def _wrap(text: str, width: float, measure) -> list[str]:
    """Split text into lines no wider than `width`, breaking between words."""
    indent = text[: len(text) - len(text.lstrip())]
    lines, current = [], ""
    for word in text.split():
        trial = f"{current} {word}" if current else f"{indent}{word}"
        if current and measure(trial) > width:
            lines.append(current)
            current = f"{indent}   {word}"
        else:
            current = trial
    return [*lines, current] if current else lines


def _overlaps(a, b) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def place_label(box, size, taken, bounds):
    """Top-left corner for a box's label that stays inside `bounds` and clear of `taken` labels.

    Tries just above the box, then inside its top edge, then inside and just below its bottom
    edge, and falls back to the first spot when every one collides.
    """
    x0, y0, x1, y1 = box
    w, h = size
    left, top, right, bottom = bounds
    x = min(max(x0, left), max(left, right - w))
    spots = [y0 - h, y0, y1 - h, y1]
    spots = [min(max(y, top), bottom - h) for y in spots]
    for y in spots:
        rect = (x, y, x + w, y + h)
        if not any(_overlaps(rect, t) for t in taken):
            return x, y
    return x, spots[0]


def annotate(image, result: dict):
    """Return a copy of the photo with the result drawn on it.

    The headline and the other answers sit in a strip above the photo, so they never cover a
    box. Box labels are placed so they don't hide one another.
    """
    from PIL import Image, ImageDraw

    photo = image.copy()
    photo.thumbnail((MAX_SIDE, MAX_SIDE))
    scale = photo.width / image.width
    side = max(photo.size)
    line = max(3, side // 250)
    font = _font(max(14, side // 40))

    probe = ImageDraw.Draw(photo)
    text_width = photo.width - 4 * line
    lines = []
    for i, text in enumerate(_banner_lines(result)):
        lines += [(part, i == 0) for part in _wrap(text, text_width, lambda t: probe.textlength(t, font=font))]
    line_h = probe.textbbox((0, 0), "Hg", font=font)[3] + 2 * line
    banner = len(lines) * line_h + 2 * line
    canvas = Image.new("RGB", (photo.width, photo.height + banner), (20, 20, 20))
    canvas.paste(photo, (0, banner))
    draw = ImageDraw.Draw(canvas)
    for row, (text, is_headline) in enumerate(lines):
        colour = (255, 255, 255) if is_headline else (200, 200, 200)
        draw.text((2 * line, line + row * line_h), text, font=font, fill=colour)

    if result["kind"] == "detector":
        primary_ink, secondary_ink = (8, 24, 15), (255, 255, 255)
        boxes = [
            (det["box"], f"{n}  {det['building']}  {det['confidence']:.0%}", PRIMARY if n == 1 else SECONDARY)
            for n, det in enumerate(result["detections"], start=1)
        ]
        inks = [primary_ink] + [secondary_ink] * (len(boxes) - 1)
    elif result.get("box"):
        best = result["top3"][0]
        colour = CLASSIFIER_COLOUR.get(result["model"], (200, 200, 200))
        boxes = [(result["box"], f"{best['building']}  {best['probability']:.0%}", colour)]
        inks = [(255, 255, 255)]
    else:
        boxes, inks = [], []

    scaled = [tuple(v * scale + (banner if k % 2 else 0) for k, v in enumerate(box)) for box, _, _ in boxes]
    # Outlines first and labels after, so no outline crosses a label. The most confident box
    # chooses its label spot first.
    for rect, (_, _, colour) in reversed(list(zip(scaled, boxes, strict=True))):
        draw.rectangle(rect, outline=colour, width=line)
    taken = []
    bounds = (0, banner, canvas.width, canvas.height)
    for rect, (_, text, colour), ink in zip(scaled, boxes, inks, strict=True):
        x0, y0, x1, y1 = draw.textbbox((0, 0), text, font=font)
        size = (x1 - x0 + 2 * line, y1 + 2 * line)
        x, y = place_label(rect, size, taken, bounds)
        taken.append((x, y, x + size[0], y + size[1]))
        draw.rectangle(taken[-1], fill=colour)
        draw.text((x + line, y + line), text, font=font, fill=ink)
    return canvas


def detector_resize_in_manifest(web_dir: str, model: str) -> str:
    """The resize mode the web app uses for this detector (letterbox when the manifest doesn't say)."""
    path = os.path.join(web_dir, "models", "manifest.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for entry in json.load(f).get("models", []):
                if entry.get("id") == model:
                    return entry.get("input", {}).get("resize", "letterbox")
    return DETECTOR_RESIZE


def predict_one(image, model: str, web_dir: str, conf: float = DETECTOR_CONF, resize: str = DETECTOR_RESIZE) -> dict:
    from src.inference import box_to_pixels, open_session, run_classifier, run_detector

    path = os.path.join(web_dir, "models", f"{model}.onnx")
    session = open_session(path)
    if model in CLASSIFIERS:
        probs, box = run_classifier(session, image)
        ranked = [{"building": CLASS_NAMES[i], "probability": round(p, 4)} for i, p in top_k(probs)]
        pixels = [round(v, 1) for v in box_to_pixels(box, *image.size)]
        return {"model": model, "kind": "classifier", "top3": ranked, "box": pixels}
    detections = run_detector(session, image, conf=conf, resize=resize)
    return {
        "model": model,
        "kind": "detector",
        "detections": [
            {
                "building": CLASS_NAMES[d["class_id"]],
                "confidence": round(d["confidence"], 4),
                "box": [round(v, 1) for v in d["box"]],
            }
            for d in detections
        ],
    }


def run_predict(
    image_path: str,
    web_dir: str,
    model: str,
    LOG,
    run_dir: str | None = None,
    conf: float | None = None,
    resize: str | None = None,
) -> list[dict]:
    """Predict with one model (or "all"), log the answers and save annotated copies of the photo.

    `conf` and `resize` change the detector's settings for this prediction only; left as None,
    it uses the default threshold and the resize mode the web app was exported with.
    """
    from PIL import Image, ImageOps

    order = ["yolo11n", *[m for m in ALL_MODELS if m != "yolo11n"]]  # the app's model first
    models = [m for m in order if model == "all" or m == model]
    available = [m for m in models if os.path.exists(os.path.join(web_dir, "models", f"{m}.onnx"))]
    missing = [m for m in models if m not in available]
    if not available:
        raise FileNotFoundError(f"No exported models in {os.path.join(web_dir, 'models')}. Run --mode export first.")

    name = os.path.basename(image_path)
    stem = os.path.splitext(name)[0]
    out_dir = os.path.join(run_dir, "predictions") if run_dir else None
    results = []
    with component_stage(LOG, "Predict", f"Finding the building in {name}", models=",".join(available)):
        if missing:
            log_stage(LOG, "WARN", "Predict", not_exported=",".join(missing))
        log_step(LOG, "Predict", "Loading the photo")
        with Image.open(image_path) as raw:
            image = ImageOps.exif_transpose(raw).convert("RGB")

        for m in available:
            component = f"Predict:{m}"
            log_step(LOG, component, f"Running {MODEL_LABELS[m]}")
            if m in CLASSIFIERS:
                result = predict_one(image, m, web_dir)
                suffix = ""
            else:
                exported = detector_resize_in_manifest(web_dir, m)
                used_resize = resize or exported
                used_conf = DETECTOR_CONF if conf is None else conf
                result = predict_one(image, m, web_dir, conf=used_conf, resize=used_resize)
                changed = []  # settings that differ from the web app's, named in the headline and file
                if used_resize != exported:
                    changed.append(used_resize)
                if used_conf != DETECTOR_CONF:
                    changed.append(f"conf {used_conf:g}")
                result["settings"] = ", ".join(changed)
                suffix = "".join("_" + c.replace(" ", "") for c in changed)
                log_stage(LOG, "INFO", component, resize=used_resize, conf=used_conf)
            answer = headline(result).split(": ", 1)[1]
            log_step(LOG, component, answer[0].upper() + answer[1:])
            log_stage(LOG, "RESULT", component, file_only=True, answer=answer)
            if result["kind"] == "classifier":
                for rank, item in enumerate(result["top3"], 1):
                    log_stage(LOG, "INFO", component, file_only=True, rank=rank, **item)
                log_stage(LOG, "INFO", component, building=result["top3"][0]["building"], box=result["box"])
            for i, det in enumerate(result.get("detections", []), 1):
                log_stage(LOG, "INFO", component, box_number=i, **det)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
                path = os.path.join(out_dir, f"{stem}_{m}{suffix}.jpg")
                annotate(image, result).save(path, quality=90)
                result["annotated"] = path
                log_stage(LOG, "ARTIFACT", component, path=path)
            log_stage(LOG, "INFO", component, file_only=True, result=json.dumps(result))
            results.append(result)
    return results
