// Pre- and post-processing for the exported models.
//
// Mirrors src/inference.py step for step, and tests check the two agree on shared fixtures.
// Functions here work on plain arrays and typed arrays so they run in the browser and in
// Node; drawing images into canvases happens in app.js.

/** Softmax over a 1-D array of logits. */
export function softmax(logits) {
  const max = Math.max(...logits);
  const exps = Array.from(logits, (v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((v) => v / sum);
}

/** Normalised (cx, cy, w, h) box -> [x0, y0, x1, y1] in photo pixels, clipped to the photo. */
export function boxToPixels(box, width, height) {
  const [cx, cy, w, h] = box;
  const clip = (v, max) => Math.min(Math.max(v, 0), max);
  return [
    clip((cx - w / 2) * width, width),
    clip((cy - h / 2) * height, height),
    clip((cx + w / 2) * width, width),
    clip((cy + h / 2) * height, height),
  ];
}

/** Indices of the k largest values, largest first. */
export function topK(values, k = 3) {
  return Array.from(values, (v, i) => [v, i])
    .sort((a, b) => b[0] - a[0] || a[1] - b[1])
    .slice(0, k)
    .map(([, i]) => i);
}

/** RGBA pixels (size x size) -> Float32Array NCHW, ImageNet-normalised. */
export function rgbaToNormalisedTensor(rgba, size, mean, std) {
  const plane = size * size;
  const out = new Float32Array(3 * plane);
  for (let i = 0; i < plane; i++) {
    for (let c = 0; c < 3; c++) {
      out[c * plane + i] = (rgba[i * 4 + c] / 255 - mean[c]) / std[c];
    }
  }
  return out;
}

/** Detector letterbox geometry for an image of width x height into a size x size square. */
export function letterboxGeometry(width, height, size) {
  const scale = Math.min(size / width, size / height);
  const w = Math.round(width * scale);
  const h = Math.round(height * scale);
  return { scale, scaleX: scale, scaleY: scale, w, h, padX: Math.floor((size - w) / 2), padY: Math.floor((size - h) / 2) };
}

/** Detector stretch geometry: the whole photo squashed to a size x size square, no padding. */
export function stretchGeometry(width, height, size) {
  return { scaleX: size / width, scaleY: size / height, w: size, h: size, padX: 0, padY: 0 };
}

/** Geometry for the manifest's resize mode ("letterbox" when not given). */
export function detectorGeometry(width, height, size, resize = "letterbox") {
  if (resize === "stretch") return stretchGeometry(width, height, size);
  if (resize === "letterbox") return letterboxGeometry(width, height, size);
  throw new Error(`Unknown detector resize mode ${resize}`);
}

/** RGBA pixels (size x size) -> Float32Array NCHW scaled to 0..1. */
export function rgbaToUnitTensor(rgba, size) {
  const plane = size * size;
  const out = new Float32Array(3 * plane);
  for (let i = 0; i < plane; i++) {
    out[i] = rgba[i * 4] / 255;
    out[plane + i] = rgba[i * 4 + 1] / 255;
    out[2 * plane + i] = rgba[i * 4 + 2] / 255;
  }
  return out;
}

function iou(a, b) {
  const x0 = Math.max(a[0], b[0]);
  const y0 = Math.max(a[1], b[1]);
  const x1 = Math.min(a[2], b[2]);
  const y1 = Math.min(a[3], b[3]);
  const inter = Math.max(0, x1 - x0) * Math.max(0, y1 - y0);
  const union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
  return inter / Math.max(union, 1e-9);
}

/** Greedy non-maximum suppression. Returns kept indices, highest score first. */
export function nms(boxes, scores, iouThreshold) {
  const order = scores
    .map((s, i) => [s, i])
    .sort((a, b) => b[0] - a[0] || a[1] - b[1])
    .map(([, i]) => i);
  const keep = [];
  const removed = new Set();
  for (let n = 0; n < order.length; n++) {
    const i = order[n];
    if (removed.has(i)) continue;
    keep.push(i);
    for (let m = n + 1; m < order.length; m++) {
      const j = order[m];
      if (!removed.has(j) && iou(boxes[i], boxes[j]) > iouThreshold) removed.add(j);
    }
  }
  return keep;
}

/**
 * Raw detector output -> detections in original image pixels, most confident first.
 *
 * data: Float32Array of shape (1, 4 + numClasses, anchors), row-major.
 * Each detection is { classId, confidence, box: [x0, y0, x1, y1] }.
 * With onePerBuilding, only the most confident box for each building is kept.
 */
export function decodeDetections(
  data,
  numClasses,
  anchors,
  geometry,
  imageW,
  imageH,
  conf,
  iouThreshold,
  onePerBuilding = true,
) {
  const scaleX = geometry.scaleX ?? geometry.scale;
  const scaleY = geometry.scaleY ?? geometry.scale;
  const candidates = [];
  for (let a = 0; a < anchors; a++) {
    let best = -1;
    let bestScore = -Infinity;
    for (let c = 0; c < numClasses; c++) {
      const s = data[(4 + c) * anchors + a];
      if (s > bestScore) {
        bestScore = s;
        best = c;
      }
    }
    if (bestScore < conf) continue;
    const cx = data[a];
    const cy = data[anchors + a];
    const w = data[2 * anchors + a];
    const h = data[3 * anchors + a];
    const clampX = (v) => Math.min(Math.max(v, 0), imageW);
    const clampY = (v) => Math.min(Math.max(v, 0), imageH);
    candidates.push({
      classId: best,
      confidence: bestScore,
      box: [
        clampX((cx - w / 2 - geometry.padX) / scaleX),
        clampY((cy - h / 2 - geometry.padY) / scaleY),
        clampX((cx + w / 2 - geometry.padX) / scaleX),
        clampY((cy + h / 2 - geometry.padY) / scaleY),
      ],
    });
  }

  const detections = [];
  const classes = [...new Set(candidates.map((d) => d.classId))].sort((a, b) => a - b);
  for (const c of classes) {
    const group = candidates.filter((d) => d.classId === c);
    const kept = nms(
      group.map((d) => d.box),
      group.map((d) => d.confidence),
      iouThreshold,
    );
    for (const k of onePerBuilding ? kept.slice(0, 1) : kept) detections.push(group[k]);
  }
  return detections.sort((a, b) => b.confidence - a.confidence);
}

/** Plain-words confidence for the answer card. */
export function confidenceWord(p) {
  if (p >= 0.75) return "Confident";
  if (p >= 0.5) return "Fairly sure";
  return "Not sure";
}

/**
 * Top-left corner for a box's label that stays inside the canvas and clear of labels already
 * placed. Tries just above the box, then inside its top edge, then inside and just below its
 * bottom edge, and falls back to the first spot when every one collides.
 * box: [x0, y0, x1, y1]; taken: array of [x0, y0, x1, y1].
 */
export function placeLabel(box, width, height, taken, canvasW, canvasH) {
  const [x0, y0, , y1] = box;
  const x = Math.min(Math.max(x0, 0), Math.max(0, canvasW - width));
  const clampY = (y) => Math.min(Math.max(y, 0), canvasH - height);
  const spots = [y0 - height, y0, y1 - height, y1].map(clampY);
  const overlaps = (a, b) => a[0] < b[2] && b[0] < a[2] && a[1] < b[3] && b[1] < a[3];
  for (const y of spots) {
    const rect = [x, y, x + width, y + height];
    if (!taken.some((t) => overlaps(rect, t))) return [x, y];
  }
  return [x, spots[0]];
}
