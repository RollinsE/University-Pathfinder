// Run with: node --test webapp/tests/
import assert from "node:assert/strict";
import { test } from "node:test";

import {
  boxToPixels,
  confidenceWord,
  decodeDetections,
  letterboxGeometry,
  nms,
  placeLabel,
  rgbaToNormalisedTensor,
  softmax,
  stretchGeometry,
  topK,
} from "../inference.js";

test("softmax sums to one and keeps order", () => {
  const p = softmax([1, 3, 2]);
  assert.ok(Math.abs(p.reduce((a, b) => a + b, 0) - 1) < 1e-9);
  assert.deepEqual(topK(p, 3), [1, 2, 0]);
});

test("box maps back to photo pixels and is clipped", () => {
  assert.deepEqual(boxToPixels([0.5, 0.5, 0.5, 0.5], 400, 200), [100, 50, 300, 150]);
  assert.deepEqual(boxToPixels([0.1, 0.5, 0.4, 0.2], 100, 100).map(Math.round), [0, 40, 30, 60]);
});

test("letterbox pads the short side evenly", () => {
  const g = letterboxGeometry(1280, 640, 640);
  assert.equal(g.scale, 0.5);
  assert.equal(g.w, 640);
  assert.equal(g.h, 320);
  assert.equal(g.padX, 0);
  assert.equal(g.padY, 160);
});

test("normalised tensor is channel-first", () => {
  const rgba = new Uint8ClampedArray([255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 255, 255]);
  const t = rgbaToNormalisedTensor(rgba, 2, [0, 0, 0], [1, 1, 1]);
  assert.deepEqual(Array.from(t), [1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1]);
});

test("nms drops the overlapping lower score", () => {
  const boxes = [
    [0, 0, 10, 10],
    [1, 1, 11, 11],
    [50, 50, 60, 60],
  ];
  assert.deepEqual(nms(boxes, [0.9, 0.8, 0.7], 0.45), [0, 2]);
});

test("decode maps boxes back to the original image", () => {
  // Two classes, three anchors, layout (1, 4 + 2, 3).
  const anchors = 3;
  const data = new Float32Array(6 * anchors);
  const set = (row, a, v) => (data[row * anchors + a] = v);
  // anchor 0: class 1, centred box in letterboxed space
  set(0, 0, 320); set(1, 0, 320); set(2, 0, 100); set(3, 0, 50); set(4, 0, 0.1); set(5, 0, 0.9);
  // anchor 1: below the confidence threshold
  set(0, 1, 100); set(1, 1, 100); set(2, 1, 10); set(3, 1, 10); set(4, 1, 0.2); set(5, 1, 0.1);
  // anchor 2: class 0
  set(0, 2, 100); set(1, 2, 300); set(2, 2, 40); set(3, 2, 40); set(4, 2, 0.6); set(5, 2, 0.3);
  const geometry = letterboxGeometry(1280, 640, 640);
  const dets = decodeDetections(data, 2, anchors, geometry, 1280, 640, 0.25, 0.45);
  assert.equal(dets.length, 2);
  assert.equal(dets[0].classId, 1);
  assert.deepEqual(dets[0].box.map((v) => Math.round(v)), [540, 270, 740, 370]);
  assert.equal(dets[1].classId, 0);
});

test("stretch fills the square and decodes each axis separately", () => {
  const g = stretchGeometry(1280, 640, 640);
  assert.deepEqual([g.scaleX, g.scaleY, g.padX, g.padY], [0.5, 1, 0, 0]);
  const data = new Float32Array([320, 320, 100, 50, 0.1, 0.9]);
  const dets = decodeDetections(data, 2, 1, g, 1280, 640, 0.25, 0.45);
  assert.deepEqual(dets[0].box.map((v) => Math.round(v)), [540, 295, 740, 345]);
});

test("one box per building unless asked for every box", () => {
  const anchors = 3;
  const data = new Float32Array(6 * anchors);
  const set = (row, a, v) => (data[row * anchors + a] = v);
  [[100, 0, 0.6], [500, 0, 0.9], [300, 0.7, 0]].forEach(([c, s0, s1], a) => {
    set(0, a, c); set(1, a, c); set(2, a, 80); set(3, a, 80); set(4, a, s0); set(5, a, s1);
  });
  const g = letterboxGeometry(640, 640, 640);
  const one = decodeDetections(data, 2, anchors, g, 640, 640, 0.25, 0.45);
  assert.deepEqual(one.map((d) => d.classId), [1, 0]);
  assert.ok(Math.abs(one[0].confidence - 0.9) < 1e-6);
  const every = decodeDetections(data, 2, anchors, g, 640, 640, 0.25, 0.45, false);
  assert.deepEqual(every.map((d) => d.classId), [1, 0, 1]);
});

test("confidence words", () => {
  assert.equal(confidenceWord(0.9), "Confident");
  assert.equal(confidenceWord(0.6), "Fairly sure");
  assert.equal(confidenceWord(0.3), "Not sure");
});

test("labels stay inside the photo and clear of each other", () => {
  const first = placeLabel([100, 200, 500, 600], 200, 40, [], 1000, 800);
  assert.deepEqual(first, [100, 160]);
  const second = placeLabel([120, 210, 400, 500], 200, 40, [[100, 160, 300, 200]], 1000, 800);
  assert.deepEqual(second, [120, 210]);
  assert.deepEqual(placeLabel([950, 0, 1000, 50], 200, 40, [], 1000, 800), [800, 0]);
});
