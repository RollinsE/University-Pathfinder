// Campus Pathfinder web app.
//
// Loads models/manifest.json (written by `python main.py --mode export`), runs the primary
// model (the detector) on a photo in the browser with ONNX Runtime Web, and shows the
// answer. The other models load only when the comparison panel is opened.

import {
  boxToPixels,
  confidenceWord,
  decodeDetections,
  detectorGeometry,
  placeLabel,
  rgbaToNormalisedTensor,
  rgbaToUnitTensor,
  softmax,
  topK,
} from "./inference.js";

const ORT_VERSION = "1.20.1";
const MAX_PHOTO_SIDE = 1600;
const UNSURE_BELOW = 0.5;

const state = {
  manifest: null,
  buildings: {},
  samples: [],
  sessions: new Map(),
  photo: null, // canvas holding the current photo at working resolution
  sampleBuilding: null, // true building when the photo is a sample
};

const $ = (id) => document.getElementById(id);

// --------------------------------------------------------------------------- setup

if (window.ort) {
  ort.env.wasm.wasmPaths = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
  ort.env.wasm.numThreads = 1; // static hosting has no cross-origin isolation, so no threads
}

async function fetchJSON(url) {
  const response = await fetch(url, { cache: "no-cache" });
  if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
  return response.json();
}

function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (key === "class") node.className = value;
    else if (key === "text") node.textContent = value;
    else node.setAttribute(key, value);
  }
  for (const child of children) if (child) node.append(child);
  return node;
}

function className(id) {
  return state.manifest.classes[id].name;
}

function modelEntry(id) {
  return state.manifest.models.find((m) => m.id === id);
}

function percent(p) {
  return `${Math.round(p * 100)}%`;
}

// --------------------------------------------------------------------------- status

function showNotice(message) {
  const notice = $("notice");
  notice.textContent = message;
  notice.hidden = !message;
}

function showLoader(label, fraction = null) {
  $("loader").hidden = false;
  $("loader-label").textContent = label;
  const fill = $("loader-fill");
  if (fraction === null) {
    fill.classList.add("indeterminate");
    fill.style.width = "";
  } else {
    fill.classList.remove("indeterminate");
    fill.style.width = `${Math.round(fraction * 100)}%`;
  }
}

function hideLoader() {
  $("loader").hidden = true;
}

// --------------------------------------------------------------------------- models

async function fetchWithProgress(url, onProgress) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
  const total = Number(response.headers.get("content-length")) || 0;
  if (!response.body || !total) {
    onProgress(null);
    return new Uint8Array(await response.arrayBuffer());
  }
  const reader = response.body.getReader();
  const chunks = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    onProgress(received / total, received, total);
  }
  const bytes = new Uint8Array(received);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.length;
  }
  return bytes;
}

async function getSession(entry) {
  if (state.sessions.has(entry.id)) return state.sessions.get(entry.id);
  const mb = (n) => (n / 1e6).toFixed(1);
  const bytes = await fetchWithProgress(`models/${entry.file}`, (fraction, received, total) => {
    const detail = fraction === null ? "" : ` (${mb(received)} of ${mb(total)} MB)`;
    showLoader(`Loading ${entry.label}${detail}`, fraction);
  });
  showLoader(`Starting ${entry.label}`);
  const session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });
  state.sessions.set(entry.id, session);
  hideLoader();
  return session;
}

function classifierInput(photo, input) {
  // The whole photo, resized to a square: the model sees all of it and its box maps straight back.
  const { size, mean, std } = input;
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(photo, 0, 0, size, size);
  const rgba = ctx.getImageData(0, 0, size, size).data;
  return new ort.Tensor("float32", rgbaToNormalisedTensor(rgba, size, mean, std), [1, 3, size, size]);
}

function detectorInput(photo, input) {
  // Letterboxed (shape kept, grey edges) or stretched to a square, as the manifest says.
  const { size, fill, resize = "letterbox" } = input;
  const geometry = detectorGeometry(photo.width, photo.height, size, resize);
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.fillStyle = `rgb(${fill}, ${fill}, ${fill})`;
  ctx.fillRect(0, 0, size, size);
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(photo, geometry.padX, geometry.padY, geometry.w, geometry.h);
  const rgba = ctx.getImageData(0, 0, size, size).data;
  return { tensor: new ort.Tensor("float32", rgbaToUnitTensor(rgba, size), [1, 3, size, size]), geometry };
}

async function runModel(entry, photo) {
  const session = await getSession(entry);
  const started = performance.now();
  let outcome;
  if (entry.kind === "classifier") {
    const feeds = { [session.inputNames[0]]: classifierInput(photo, entry.input) };
    const outputs = await session.run(feeds);
    const box = outputs.box ? boxToPixels(Array.from(outputs.box.data), photo.width, photo.height) : null;
    outcome = { kind: "classifier", probs: softmax((outputs.logits || outputs[session.outputNames[0]]).data), box };
  } else {
    const { tensor, geometry } = detectorInput(photo, entry.input);
    const output = (await session.run({ [session.inputNames[0]]: tensor }))[session.outputNames[0]];
    const [, rows, anchors] = output.dims;
    const detections = decodeDetections(
      output.data,
      rows - 4,
      anchors,
      geometry,
      photo.width,
      photo.height,
      entry.conf,
      entry.iou,
      entry.one_per_building ?? true,
    );
    outcome = { kind: "detector", detections };
  }
  outcome.ms = Math.round(performance.now() - started);
  return outcome;
}

// --------------------------------------------------------------------------- photo

function toWorkingCanvas(source, width, height) {
  const scale = Math.min(1, MAX_PHOTO_SIDE / Math.max(width, height));
  const canvas = document.createElement("canvas");
  canvas.width = Math.round(width * scale);
  canvas.height = Math.round(height * scale);
  canvas.getContext("2d").drawImage(source, 0, 0, canvas.width, canvas.height);
  return canvas;
}

function drawPhoto(detections = []) {
  const photo = state.photo;
  const canvas = $("photo");
  canvas.width = photo.width;
  canvas.height = photo.height;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(photo, 0, 0);

  const line = Math.max(3, Math.round(Math.max(photo.width, photo.height) / 250));
  const font = Math.max(14, Math.round(Math.max(photo.width, photo.height) / 40));
  const colourOf = (i) => (i === 0 ? cssVar("--box") : cssVar("--box-secondary"));
  // Outlines first and labels after, so no outline crosses a label.
  ctx.lineWidth = line;
  detections.forEach((d, i) => {
    const [x0, y0, x1, y1] = d.box;
    ctx.strokeStyle = colourOf(i);
    ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
  });
  // The most confident box chooses its label spot first.
  ctx.font = `600 ${font}px system-ui, sans-serif`;
  ctx.textBaseline = "middle";
  const taken = [];
  detections.forEach((d, i) => {
    const label = `${i + 1}  ${className(d.classId)}`;
    const width = ctx.measureText(label).width + font;
    const height = font * 1.5;
    const [x, y] = placeLabel(d.box, width, height, taken, canvas.width, canvas.height);
    taken.push([x, y, x + width, y + height]);
    ctx.fillStyle = colourOf(i);
    ctx.fillRect(x, y, width, height);
    ctx.fillStyle = i === 0 ? "#08180f" : "#ffffff";
    ctx.fillText(label, x + font / 2, y + height / 2);
  });
  const names = detections.map((d) => className(d.classId)).join(", ");
  canvas.setAttribute("aria-label", names ? `Your photo. Buildings found: ${names}.` : "Your photo.");
}

// --------------------------------------------------------------------------- answer

function renderAnswer(outcome) {
  const answer = $("answer");
  const others = $("others");
  answer.replaceChildren();
  others.replaceChildren();
  others.hidden = true;

  let topClass = null;
  let topScore = 0;
  let ranked = [];

  if (outcome.kind === "detector") {
    drawPhoto(outcome.detections);
    if (outcome.detections.length) {
      topClass = outcome.detections[0].classId;
      topScore = outcome.detections[0].confidence;
      ranked = outcome.detections.slice(1).map((d, i) => ({ badge: i + 2, classId: d.classId, score: d.confidence }));
    }
  } else {
    drawPhoto([]);
    const order = topK(outcome.probs, 3);
    topClass = order[0];
    topScore = outcome.probs[topClass];
    ranked = order.slice(1).map((c) => ({ badge: null, classId: c, score: outcome.probs[c] }));
  }

  if (topClass === null) {
    answer.append(
      el("div", { class: "answer-name", text: "No campus building found" }),
      el("p", {
        class: "answer-tip",
        text: "Try getting more of the building in frame, or step back so the whole front is visible.",
      }),
    );
    $("building-card").hidden = true;
  } else {
    const unsure = topScore < UNSURE_BELOW;
    const word = confidenceWord(topScore);
    const several = outcome.kind === "detector" && outcome.detections.length > 1;
    answer.append(
      el(
        "div",
        { class: "answer-kicker" },
        several ? el("span", { class: "badge primary", text: "1", "aria-hidden": "true" }) : null,
        el("span", { text: unsure ? "It might be" : "This looks like" }),
      ),
      el("div", { class: "answer-name", text: className(topClass) }),
      el(
        "div",
        { class: "answer-confidence" },
        el("span", { class: `answer-word${unsure ? " unsure" : ""}`, text: word }),
        el("span", { class: "answer-percent", text: `${percent(topScore)} confidence` }),
      ),
    );
    if (unsure) {
      answer.append(
        el("p", { class: "answer-tip", text: "Not sure about this one. A closer, straight-on photo usually helps." }),
      );
    }
    renderBuildingCard(topClass);
  }

  if (state.sampleBuilding) {
    answer.append(el("p", { class: "answer-tip", text: `Sample photo of ${state.sampleBuilding}.` }));
  }

  if (ranked.length) {
    const heading = outcome.kind === "detector" ? "Also in this photo" : "Other possibilities";
    others.append(el("li", { class: "others-heading", text: heading }));
    for (const item of ranked) {
      others.append(
        el(
          "li",
          {},
          item.badge ? el("span", { class: "badge", text: String(item.badge), "aria-hidden": "true" }) : null,
          el("span", { text: className(item.classId) }),
          el("span", { class: "muted", text: percent(item.score) }),
        ),
      );
    }
    others.hidden = false;
  }
}

function renderBuildingCard(classId) {
  const card = $("building-card");
  const cls = state.manifest.classes[classId];
  const info = state.buildings[cls.slug] || {};
  // A pin on the building's coordinates when buildings.json has them; Google Maps doesn't
  // recognise every building by name.
  const query =
    info.lat != null && info.lng != null
      ? `${info.lat},${info.lng}`
      : info.mapsQuery || `${cls.name}, Bournemouth University, Talbot Campus`;
  const link = el("a", {
    href: `https://www.google.com/maps/search/?api=1&query=${encodeURIComponent(query)}`,
    target: "_blank",
    rel: "noopener",
    text: "Open in Maps",
  });
  const parts = [el("h3", { text: cls.name })];
  if (info.description) parts.push(el("p", { text: info.description }));
  card.replaceChildren(...parts, link);
  card.hidden = false;
}

// --------------------------------------------------------------------------- compare

function compareCard(entry) {
  return el(
    "article",
    { class: "model-card", id: `card-${entry.id}` },
    el(
      "div",
      { class: "model-head" },
      el("div", {}, el("div", { class: "model-name", text: entry.label }), el("div", { class: "model-role", text: entry.role })),
      el("div", { class: "model-time", text: "waiting" }),
    ),
  );
}

function fillCompareCard(entry, outcome) {
  const card = $(`card-${entry.id}`);
  card.querySelector(".model-time").textContent = `${outcome.ms} ms on this device`;
  card.querySelectorAll(".prob, .model-role.result").forEach((n) => n.remove());

  if (outcome.kind === "classifier") {
    const order = topK(outcome.probs, 3);
    const block = el("div", { class: "prob" });
    order.forEach((c, i) => {
      const p = outcome.probs[c];
      const bar = el("div", { class: `prob-bar${i === 0 ? " top" : ""}` }, el("div"));
      bar.firstChild.style.width = percent(p);
      block.append(el("span", { text: className(c) }), el("span", { class: "prob-value", text: percent(p) }), bar);
    });
    card.append(block);
  } else if (outcome.detections.length) {
    const block = el("div", { class: "prob" });
    outcome.detections.forEach((d, i) => {
      const bar = el("div", { class: `prob-bar${i === 0 ? " top" : ""}` }, el("div"));
      bar.firstChild.style.width = percent(d.confidence);
      block.append(
        el("span", { text: `${i + 1}. ${className(d.classId)}` }),
        el("span", { class: "prob-value", text: percent(d.confidence) }),
        bar,
      );
    });
    card.append(block);
  } else {
    card.append(el("div", { class: "prob" }, el("span", { text: "No building found" })));
  }
}

async function runCompare() {
  if (!state.photo || !$("compare").open) return;
  const grid = $("compare-grid");
  grid.replaceChildren(...state.manifest.models.map(compareCard));
  for (const entry of state.manifest.models) {
    try {
      fillCompareCard(entry, await runModel(entry, state.photo));
    } catch (error) {
      $(`card-${entry.id}`).querySelector(".model-time").textContent = "could not run";
      console.error(error);
    }
  }
  hideLoader();
}

// --------------------------------------------------------------------------- flow

async function analyse() {
  showNotice("");
  $("result").hidden = false;
  drawPhoto([]);
  $("answer").replaceChildren(el("div", { class: "answer-kicker", text: "Looking…" }));
  $("others").hidden = true;
  $("building-card").hidden = true;
  try {
    const primary = modelEntry(state.manifest.primary) || state.manifest.models[0];
    renderAnswer(await runModel(primary, state.photo));
    $("result").scrollIntoView({ behavior: "smooth", block: "start" });
    await runCompare();
  } catch (error) {
    hideLoader();
    console.error(error);
    showNotice("Something went wrong running the model. Reload the page and try again.");
  }
}

async function handleFile(file) {
  if (!file || !state.manifest) return;
  state.sampleBuilding = null;
  document.querySelectorAll(".sample").forEach((b) => b.setAttribute("aria-pressed", "false"));
  try {
    const bitmap = await createImageBitmap(file, { imageOrientation: "from-image" });
    state.photo = toWorkingCanvas(bitmap, bitmap.width, bitmap.height);
    bitmap.close?.();
  } catch {
    showNotice("Couldn't read that image. Try a JPEG or PNG photo.");
    return;
  }
  await analyse();
}

async function useSample(entry, button) {
  document.querySelectorAll(".sample").forEach((b) => b.setAttribute("aria-pressed", String(b === button)));
  const image = new Image();
  image.src = `samples/${entry.file}`;
  await image.decode();
  state.photo = toWorkingCanvas(image, image.naturalWidth, image.naturalHeight);
  state.sampleBuilding = entry.building;
  await analyse();
}

// --------------------------------------------------------------------------- static parts

function renderKnownList() {
  const list = $("known-list");
  list.replaceChildren(...state.manifest.classes.map((c) => el("li", { text: c.name })));
}

function renderResultsTable() {
  const rows = state.manifest.models.filter((m) => m.metrics && m.metrics.test_accuracy != null);
  if (!rows.length) return;
  const table = el("table", { class: "results" });
  table.append(
    el(
      "tr",
      {},
      el("th", { text: "Model" }),
      el("th", { class: "num", text: "Test accuracy" }),
      el("th", { class: "num", text: "Download" }),
    ),
  );
  for (const m of rows) {
    table.append(
      el(
        "tr",
        {},
        el("td", { text: m.label }),
        el("td", { class: "num", text: `${(m.metrics.test_accuracy * 100).toFixed(1)}%` }),
        el("td", { class: "num", text: `${m.megabytes} MB` }),
      ),
    );
  }
  $("results-table").replaceChildren(table);
}

async function loadSamples() {
  try {
    state.samples = await fetchJSON("samples/samples.json");
  } catch {
    return;
  }
  if (!state.samples.length) return;
  const row = $("sample-row");
  state.samples.forEach((entry, i) => {
    const button = el(
      "button",
      { class: "sample", type: "button", "aria-pressed": "false", "aria-label": `Sample photo ${i + 1}` },
      el("img", { src: `samples/${entry.file}`, alt: "", loading: "lazy" }),
    );
    button.addEventListener("click", () => useSample(entry, button));
    row.append(button);
  });
  $("samples").hidden = false;
}

async function init() {
  for (const id of ["camera-input", "upload-input"]) {
    $(id).addEventListener("change", (event) => {
      const file = event.target.files[0];
      event.target.value = "";
      handleFile(file);
    });
  }
  $("compare").addEventListener("toggle", runCompare);

  if (!window.ort) {
    showNotice("Couldn't load ONNX Runtime Web. Check your connection and reload.");
    return;
  }
  try {
    state.manifest = await fetchJSON("models/manifest.json");
  } catch {
    showNotice("No models found yet. Run `python main.py --mode export` to add them to webapp/models/.");
    return;
  }
  state.buildings = await fetchJSON("buildings.json").catch(() => ({}));
  renderKnownList();
  renderResultsTable();
  await loadSamples();
}

init();
