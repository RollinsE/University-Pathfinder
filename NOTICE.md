# Notice

## Project team

Campus Pathfinder started as a group coursework project at Bournemouth University. The project
team photographed and annotated the eight buildings, published the dataset on Roboflow, and
compared ResNet classifiers with a YOLOv5 detector.

## Development

Developed by Rollins Edeh into a standalone, reproducible project:

- a tested Python package with a command-line pipeline, structured run logging and a Colab
  notebook;
- one class list shared by every model, and all models scored on the same held-out test photos;
- ResNet18 as the baseline, MobileNetV3-Large as a lightweight classifier and YOLO11n as the
  detector;
- ONNX export and an in-browser web app that names the buildings in a photo.

## Dataset

[BU Building Dectection](https://universe.roboflow.com/computer-vision-6ydhl/bu-building-dectection)
on Roboflow Universe, created by the project team. See docs/DATA_CARD.md.

## Third-party software

- [Ultralytics](https://github.com/ultralytics/ultralytics) (AGPL-3.0) for YOLO11 training and export.
  Models trained with it, and software that distributes them, may fall under AGPL-3.0 unless an
  Ultralytics Enterprise licence applies.
- [PyTorch and torchvision](https://pytorch.org/) (BSD-style) for the classifiers and ImageNet weights.
- [ONNX Runtime Web](https://github.com/microsoft/onnxruntime) (MIT) to run the models in the browser.
