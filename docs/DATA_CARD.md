# Data card

## Source

[BU Building Dectection](https://universe.roboflow.com/computer-vision-6ydhl/bu-building-dectection)
on Roboflow Universe (workspace `computer-vision-6ydhl`, project `bu-building-dectection`; the
spelling is the project's id). It was created by the project team.

- **Content:** smartphone photos of eight buildings on Bournemouth University's Talbot Campus,
  taken by day and at night, from varied angles and distances, some partly hidden by trees,
  people or other buildings. About 32 photos per building, plus background photos (sky,
  trees, paths) with no buildings in them, which teach the detector what is not a building.
- **Labels:** one box round each visible building, drawn by the team.
- **Versions:** version 4 holds the original photos and is the default. Version 3 holds the same
  photos with offline augmentation added to the training split by Roboflow. Choose one with
  `--version`.
- **Splits:** the train, valid and test splits come from the Roboflow export and are used as they
  are, so every model is trained, tuned and tested on the same photos.

The exact counts for a download are written to `data/processed/<project>-v<version>/manifest.json`
and logged in `run.log` by the `Prepare` stage.

## The eight buildings

Christchurch House, Dorset House, Fusion Building, Kimmeridge House, Poole Gateway Building,
Poole House, Sir Michael Cobham Library and Weymouth House. The class order (alphabetical) is
defined once in `src/constants.py` and used by every model and by the web app.

## Preparation

`src/datasets.py` turns the export into a prepared dataset:

- class ids are remapped to the canonical order by name, whatever order the export uses;
- labels that are not one of the eight buildings are dropped and counted;
- a building with no photos in a split is reported as a warning;
- `data.yaml` is rewritten with the dataset's absolute path for Ultralytics.

The classifiers need one label per photo. Each photo's label is the building with the largest
box. Background photos have no label and are left out of classifier training and of every
model's accuracy figure; the detector still trains on them, and they count in its mAP.

## Getting the data

The dataset is public. Downloading it needs a free Roboflow account's private API key in the
`ROBOFLOW_API_KEY` environment variable (or a `.env` file, or a Colab secret). The key is never
logged or saved. `python main.py --mode acquire` downloads, extracts and prepares it with
progress bars; `--source_dir` uses an export already on disk.

```text
data/raw/<project>-v<version>.zip     downloaded archive
data/raw/<project>-v<version>/        extracted export
data/processed/<project>-v<version>/  prepared dataset
```

`data/` is ignored by Git.

## Licence and reuse

Check the licence shown on the Roboflow Universe page before redistributing the photos. The web
app ships a handful of test photos (`webapp/samples/`) so visitors who are not on campus can try
it; remove them, or export with `--samples 0`, if the licence does not allow that.

## Known limitations

- **Small.** About 32 photos per building. One photo moves a building's test accuracy by several
  points, so small differences between models are noise.
- **Near-duplicates.** Several photos of a building were taken from similar spots. If similar
  shots fall in different splits, test accuracy overstates how well a model handles a new
  viewpoint.
- **One season, one period.** Photos come from one collection period. Scaffolding, new signage,
  seasonal foliage or new buildings can all change what a building looks like.
- **People.** Campus photos can include passers-by. They are not labelled or used, but they are
  present in the images.

## Demo data

`--demo` replaces the dataset with synthetic drawings of eight coloured blocks
(`src/demo.py`) in the same layout, including multi-building photos, background photos and a
non-building label. It exists to test the pipeline without an account; its numbers mean nothing
about the real task.
