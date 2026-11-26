# Web app

A static, phone-first page that names the building in a photo. The model runs in the browser
with [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/), so there is no server to
run or pay for, nothing to wake up, and photos never leave the device.

## What the visitor sees

1. **Take photo** (opens the rear camera on a phone) or **Upload**.
2. **Sample photos**, for visitors who are not on campus.
3. A green loading bar the first time a model downloads (about 10 MB for the detector).
4. **The result:**
   - the photo with every building found boxed and numbered;
   - the answer card: the most confident building, a confidence word (Confident, Fairly sure,
     Not sure) and the percentage;
   - below 50% the card says "It might be" and suggests a closer, straight-on photo; with no
     building found it says so and suggests stepping back instead of guessing;
   - "Also in this photo", numbered to match the boxes;
   - a building card with an **Open in Maps** link.
5. **Compare the three models** (closed by default): the same photo through ResNet18,
   MobileNetV3-Large and YOLO11n, with each model's top answers and its time on this device. The
   classifiers download only when this panel is opened.
6. Footer: the privacy note, the eight buildings it knows, and how it works with credits and the
   test results from `models/manifest.json`.

Light and dark themes follow the device setting. Layout is a single column, 560 px wide at most.

## Files

```text
webapp/
├── index.html, styles.css, app.js   the page
├── inference.js                     pre- and post-processing (mirrors src/inference.py)
├── buildings.json                   map location and optional description per building
├── models/                          written by --mode export
│   ├── manifest.json                classes, preprocessing, sizes, test metrics
│   ├── yolo11n.onnx                 the detector (primary)
│   ├── mobilenet_v3_large.onnx      comparison only
│   └── resnet18.onnx                comparison only
├── samples/                         written by --mode export (test photos + samples.json)
└── tests/                           Node tests: node --test "webapp/tests/*.test.mjs"
```

The page reads everything about the models from `manifest.json`: which one is primary, input
sizes, normalisation, thresholds and class names. Re-exporting updates the app with no code
changes.

## Building descriptions

Each building's card shows a description when `buildings.json` has one. They start empty: add a
line or two per building (what is inside it, for example) and the cards pick them up. The Maps
link drops a pin on the building's `lat` and `lng` when they are set, and otherwise searches Google
Maps for `mapsQuery`. Google Maps doesn't know every building by name, so coordinates are more
reliable: right-click the building in Google Maps and click the coordinates at the top of the menu
to copy them.

## Run it locally

```bash
python main.py --mode export --run_dir runs/<run>   # writes webapp/models and webapp/samples
make serve                                          # http://localhost:8000
```

The page must be served over HTTP (not opened as a file) so it can fetch the models. Phone
cameras need HTTPS, which GitHub Pages provides; on a laptop, Upload works over plain HTTP.

## Publish it

`.github/workflows/pages.yml` publishes `webapp/` to GitHub Pages on every push that changes it.

1. Run the export and commit `webapp/models/` and `webapp/samples/` (`.gitignore` allows these
   `.onnx` files; ResNet18 is the largest at roughly 45 MB, under GitHub's 100 MB limit).
2. In the repository settings, set **Pages > Source** to **GitHub Actions**.
3. Push. The app appears at `https://<user>.github.io/<repo>/`.

To keep the repository small, export only the detector and MobileNet
(`--models mobilenet_v3_large,yolo11n`); the comparison panel shows whichever models the
manifest lists.

## Details

- ONNX Runtime Web 1.20.1 is loaded from jsDelivr and runs single-threaded on WebAssembly
  (static hosting has no cross-origin isolation, which threads need).
- Photos are scaled to at most 1600 px on the long side before anything else, and EXIF rotation
  is applied.
- Classify-and-locate input: the whole photo resized to 224 x 224, ImageNet normalisation; outputs
  `logits` and `box` (normalised cx, cy, w, h). Detector input: the photo fitted into 640 x 640,
  either letterboxed with grey (114) padding or stretched, as `input.resize` in the manifest says
  (set with `--yolo_resize` at export; letterbox by default). Boxes are kept above 0.25 confidence,
  overlapping boxes for the same building are merged by non-maximum suppression at IoU 0.45, and
  then only the most confident box for each building is shown (`one_per_building`), since each
  building exists once on campus. `tests/test_webapp_parity.py` checks that the JavaScript and
  Python decoders agree in every combination.
