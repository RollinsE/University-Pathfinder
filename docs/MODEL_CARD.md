# Model card

## Purpose

Name the building in a photo taken on Bournemouth University's Talbot Campus, as a first step
towards camera-based wayfinding. The web app uses the detector; the two classifiers are there to
compare against.

## Models

| Model | Role | Output | Starting weights |
|---|---|---|---|
| ResNet18 | baseline classify-and-locate | one building per photo (probabilities for all eight) and one box round it | ImageNet (torchvision `IMAGENET1K_V1`) |
| MobileNetV3-Large | lightweight classify-and-locate | as above | ImageNet (torchvision `IMAGENET1K_V2`) |
| YOLO11n | detector, used by the web app | a box, building and confidence for every building found, or none | COCO (Ultralytics `yolo11n.pt`) |

## Training

**Classify-and-locate models** (`src/classify.py`), same recipe for both:

- the pretrained backbone feeds two small heads: an 8-way building head and a box head that
  outputs one box as (centre x, centre y, width, height), each 0 to 1 of the photo;
- input: the whole photo resized to 224 x 224, stretched as the dataset's photos were, so the
  model sees all of it and its box maps straight back onto the photo;
- target: the building with the largest labelled box, and that box;
- loss: cross-entropy with label smoothing 0.1 for the name, plus 2 x (L1 + generalised IoU) for
  the box (`--box-weight`);
- AdamW, learning rate 3e-4, weight decay 1e-4, cosine decay over 30 epochs, batch 32;
- augmentation that moves the box with the photo (small shifts, scaling, rotation up to 6 degrees)
  plus colour jitter for day and night lighting;
- the checkpoint kept is the epoch with the best validation accuracy (ties: higher box IoU);
  training stops early when validation loss has not improved for 8 epochs;
- fixed seed (42); the full training state is saved after every epoch, so an interrupted run
  resumes from its last finished epoch.

**Detector** (`src/detect.py`): Ultralytics YOLO11n, 640 px, up to 250 epochs with patience 75,
batch 16, Ultralytics' default augmentation, fixed seed. Validation mAP on 45 photos varies a lot
between epochs, so a long patience keeps one bad epoch from ending training early. Training resumes from `last.pt` after an
interruption.

## Evaluation

Every model is scored on the held-out **test** split, which is not used for training or for
choosing checkpoints.

- **Box accuracy, all models.** Each model's one box per photo (for the detector, its most
  confident box) is compared with the largest labelled box: mean IoU (overlap, 0 to 1), and
  "named and boxed", the share of photos with the right building and IoU of at least 0.5.
- **Image-level accuracy, all models.** Each test photo's true label is the building with the
  largest box. Classifiers answer with their top class; the detector answers with its most
  confident box, and a photo where it finds nothing counts as wrong. Also reported: macro F1,
  top-3 accuracy, per-building precision and recall, and a confusion matrix.
- **Detection metrics, detector only.** mAP50, mAP50-95, precision and recall on the test split
  (Ultralytics validator), overall and per building.
- **Speed and size.** Each exported ONNX model's file size and median CPU time per photo with
  onnxruntime.

Results for a run are in `runs/<run>/metrics/results.md` and in the README.

## Intended use

- Suggesting which campus building is in a photo, as an aid to finding your way.
- A worked comparison of a baseline CNN, a mobile-sized CNN and a small detector on a small
  custom dataset.

## Not intended for

- Any building other than the eight listed, or any other campus.
- Safety-critical navigation, accessibility routing, or anything where a wrong answer causes harm.
- Identifying or tracking people.

## Limitations

- **One box per photo.** ResNet18 and MobileNetV3-Large box only the building they name; in a
  photo of two buildings they box one. YOLO11n boxes each.
- **Closed set.** The classify-and-locate models always name one of the eight buildings, even for
  a photo of something else. The detector can say it found nothing, which is one reason the app uses it.
- **Small test set.** A few dozen test photos: one photo is worth several points of accuracy, so
  treat small differences between models as ties.
- **Similar buildings.** Buildings with similar materials and window patterns are the likeliest to
  be confused; check the confusion matrices.
- **Viewpoint and season.** Accuracy drops for viewpoints, seasons and lighting the training photos
  did not cover.
- **Confidence is not probability.** Scores rank answers; they are not calibrated chances of being
  right. The web app shows them in words ("Confident", "Fairly sure", "Not sure") and gives a
  retake tip below 50%.

## Licences

The classifiers use torchvision's ImageNet weights. YOLO11 is developed by Ultralytics under
AGPL-3.0; models trained with it, and software that distributes them, may be covered by AGPL-3.0
unless an Ultralytics Enterprise licence applies. See NOTICE.md.
