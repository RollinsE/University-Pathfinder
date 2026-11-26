// Decode a detector-output fixture written by tests/test_webapp_parity.py and print the result
// as JSON, so the Python test can compare it with src/inference.py.
//
//   node webapp/tests/decode_fixture.mjs fixture.json
import { readFileSync } from "node:fs";

import { decodeDetections, detectorGeometry } from "../inference.js";

const fixture = JSON.parse(readFileSync(process.argv[2], "utf8"));
const geometry = detectorGeometry(fixture.image_w, fixture.image_h, fixture.size, fixture.resize);
const detections = decodeDetections(
  Float32Array.from(fixture.data),
  fixture.num_classes,
  fixture.anchors,
  geometry,
  fixture.image_w,
  fixture.image_h,
  fixture.conf,
  fixture.iou,
  fixture.one_per_building,
);
process.stdout.write(JSON.stringify({ geometry, detections }));
