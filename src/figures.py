"""Results figures with Plotly: one consistent style, built from a finished run folder.

Each figure is first built as a plain dict (a Plotly figure spec: {"data": [...],
"layout": {...}}), so its content and wording can be tested without Plotly installed.
Plotly is only needed to write the files:

- <run>/plots/<name>.html         interactive, hover for exact values (light theme); the notebook
                                  shows these
- <export_dir>/<name>_light.png   and <name>_dark.png, for the README (GitHub shows the one
                                  matching the reader's theme)

Figures
-------
model_comparison  test accuracy per model with its 95% range, so ties are visible
speed_size        CPU time per photo and download size of each exported model
training_curves   validation scores and losses by epoch: both classifiers and YOLO11n
confusion_<model> true building against the model's answer on the test photos

Style
-----
- Each model keeps one colour in every figure: YOLO11n (the app's model) blue, ResNet18 orange,
  MobileNetV3-Large aqua. Legends or direct labels always name the series.
- Thin lines (2px), ringed markers, hairline grid, no chart border; one y-axis per panel.
- Titles state the finding and are computed from the numbers, so a new run rewrites them.
"""

from __future__ import annotations

import csv
import json
import os
import textwrap

from src.constants import CLASS_SHORT, CLASSIFIERS, MODEL_LABELS
from src.evaluate import build_leaderboard, load_metrics
from src.logger import component_stage, log_stage, log_step

COMPONENT = "Figures"
FONT = 'Inter, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif'
WIDTH = 900
LEFT = 24

THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "ink": "#0b0b0b",
        "ink2": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "blue": "#2a78d6",
        "orange": "#eb6834",
        "aqua": "#1baf7a",
        "neutral": "#a3a19a",
        "sequential": ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#104281"],
        "cell_ink_dark": "#ffffff",
    },
    "dark": {
        "surface": "#1a1a19",
        "ink": "#f0efec",
        "ink2": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "blue": "#3987e5",
        "orange": "#d95926",
        "aqua": "#199e70",
        "neutral": "#6f6e69",
        "sequential": ["#1a1a19", "#184f95", "#1c5cab", "#2a78d6", "#5598e7", "#9ec5f4"],
        "cell_ink_dark": "#0b0b0b",
    },
}

MODEL_COLOUR = {"yolo11n": "blue", "resnet18": "orange", "mobilenet_v3_large": "aqua"}
FIGURE_ORDER = ("model_comparison", "speed_size", "training_curves")


# --------------------------------------------------------------------------- shared layout


def _wrap(text: str, size: int, width: int = WIDTH) -> tuple[str, int]:
    """Plotly does not wrap titles; break the text into lines that fit the figure width."""
    chars = max(30, int((width - 120) / (size * 0.52)))
    lines = textwrap.wrap(text, chars) or [""]
    return "<br>".join(lines), len(lines)


def _axis(t: dict, title: str | None = None, **extra) -> dict:
    axis = {
        "showgrid": True,
        "gridcolor": t["grid"],
        "gridwidth": 1,
        "zeroline": False,
        "showline": True,
        "linecolor": t["axis"],
        "linewidth": 1,
        "ticks": "",
        "automargin": True,
        "tickfont": {"color": t["muted"], "size": 12},
    }
    if title:
        axis["title"] = {"text": title, "font": {"color": t["ink2"], "size": 12}, "standoff": 10}
    axis.update(extra)
    return axis


def _layout(t: dict, title: str, subtitle: str, height: int, legend: bool = False, **extra) -> dict:
    title_html, title_lines = _wrap(title, 19)
    sub_html, sub_lines = _wrap(subtitle, 13)
    top = 28 + 26 * title_lines + 19 * sub_lines + (34 if legend else 14)
    layout = {
        "template": "none",
        "width": WIDTH,
        "height": height + top,
        "paper_bgcolor": t["surface"],
        "plot_bgcolor": t["surface"],
        "font": {"family": FONT, "size": 13, "color": t["ink2"]},
        "title": {
            "text": f"<b>{title_html}</b>",
            "font": {"size": 19, "color": t["ink"]},
            "subtitle": {"text": sub_html, "font": {"size": 13, "color": t["ink2"]}},
            "x": 0,
            "xref": "container",
            "xanchor": "left",
            "y": 1,
            "yref": "container",
            "yanchor": "top",
            "pad": {"t": 22, "l": LEFT},
        },
        "margin": {"t": top, "l": 70, "r": 40, "b": 64},
        "showlegend": legend,
        "hoverlabel": {"bgcolor": t["surface"], "bordercolor": t["axis"], "font": {"color": t["ink"], "family": FONT}},
    }
    if legend:
        layout["legend"] = {
            "orientation": "h",
            "x": 0,
            "xref": "paper",
            "xanchor": "left",
            "y": 1.0,
            "yanchor": "bottom",
            "font": {"color": t["ink2"], "size": 12},
            "bgcolor": "rgba(0,0,0,0)",
        }
    layout.update(extra)
    return layout


def _ring(t: dict, color: str, size: int = 10) -> dict:
    return {"color": color, "size": size, "line": {"color": t["surface"], "width": 2}}


def _colour(t: dict, model: str) -> str:
    return t[MODEL_COLOUR.get(model, "neutral")]


# --------------------------------------------------------------------------- figure specs


def model_comparison(rows: list[dict], mode: str = "light") -> dict:
    """Test accuracy per model as a dot with its 95% range; overlapping ranges are ties.

    rows: [{"id", "accuracy", "low", "high", "correct", "images"}], any order.
    """
    t = THEMES[mode]
    rows = sorted(rows, key=lambda r: r["accuracy"])
    data = []
    for r in rows:
        acc, lo, hi = r["accuracy"] * 100, r["low"] * 100, r["high"] * 100
        data.append(
            {
                "type": "scatter",
                "mode": "markers",
                "name": MODEL_LABELS[r["id"]],
                "x": [acc],
                "y": [MODEL_LABELS[r["id"]]],
                "error_x": {
                    "type": "data",
                    "symmetric": False,
                    "array": [hi - acc],
                    "arrayminus": [acc - lo],
                    "color": _colour(t, r["id"]),
                    "thickness": 2,
                    "width": 7,
                },
                "marker": _ring(t, _colour(t, r["id"]), 13),
                "hovertemplate": (
                    f"<b>{MODEL_LABELS[r['id']]}</b><br>{r['correct']} of {r['images']} test photos correct"
                    f"<br>accuracy {acc:.1f}%<br>95% range {lo:.0f}–{hi:.0f}%<extra></extra>"
                ),
            }
        )
        data.append(
            {
                "type": "scatter",
                "mode": "text",
                "x": [hi + 1.5],
                "y": [MODEL_LABELS[r["id"]]],
                "text": [f"<b>{r['correct']}/{r['images']}</b> correct ({acc:.0f}%), range {lo:.0f}–{hi:.0f}%"],
                "textposition": "middle right",
                "cliponaxis": False,
                "textfont": {"color": t["ink"], "size": 12},
                "hoverinfo": "skip",
                "showlegend": False,
            }
        )

    best = max(rows, key=lambda r: (r["accuracy"], r["low"]))
    others = [r for r in rows if r is not best]
    tied = [r for r in others if r["high"] >= best["low"]]
    n = best["images"]
    if others and len(tied) == len(others):
        title = f"All {len(rows)} models are within the uncertainty of a {n}-photo test set"
    elif tied:
        names = ", ".join(MODEL_LABELS[r["id"]] for r in tied)
        title = f"{MODEL_LABELS[best['id']]} is most accurate, statistically tied with {names}"
    else:
        title = f"{MODEL_LABELS[best['id']]} names the most test photos correctly"
    subtitle = (
        f"Share of the {n} held-out test photos each model named correctly. Lines show the 95% range: "
        "with this few photos, overlapping ranges mean the models are effectively tied."
    )
    low = min(r["low"] for r in rows) * 100
    layout = _layout(
        t,
        title,
        subtitle,
        height=70 + 64 * len(rows),
        xaxis=_axis(t, "Test accuracy (%)", range=[max(0, low - 6), 150], tickvals=[0, 20, 40, 60, 70, 80, 90, 100]),
        yaxis=_axis(
            t, None, showgrid=False, showline=False, ticklabelstandoff=10, tickfont={"color": t["ink"], "size": 13}
        ),
    )
    layout["margin"]["l"] = 150
    return {"data": data, "layout": layout}


def speed_size(rows: list[dict], mode: str = "light") -> dict:
    """CPU time per photo for each exported model, labelled with its download size.

    rows: [{"id", "cpu_ms", "onnx_mb"}].
    """
    t = THEMES[mode]
    rows = sorted(rows, key=lambda r: -r["cpu_ms"])
    data = [
        {
            "type": "bar",
            "orientation": "h",
            "x": [r["cpu_ms"] for r in rows],
            "y": [MODEL_LABELS[r["id"]] for r in rows],
            "marker": {"color": [_colour(t, r["id"]) for r in rows], "cornerradius": 4},
            "width": 0.55,
            "text": [f"  {r['cpu_ms']:.0f} ms per photo, {r['onnx_mb']:.1f} MB download" for r in rows],
            "textposition": "outside",
            "cliponaxis": False,
            "textfont": {"color": t["ink"], "size": 12},
            "hovertemplate": "<b>%{y}</b><br>%{x:.1f} ms per photo<extra></extra>",
        }
    ]
    fastest = min(rows, key=lambda r: r["cpu_ms"])
    smallest = min(rows, key=lambda r: r["onnx_mb"])
    if fastest is smallest:
        title = f"{MODEL_LABELS[fastest['id']]} is both the fastest and the smallest download"
    else:
        title = f"{MODEL_LABELS[fastest['id']]} is fastest; {MODEL_LABELS[smallest['id']]} is the smallest download"
    subtitle = (
        "Median time to process one photo on a CPU with ONNX Runtime (lower is better), and the size of "
        "each model file the web app downloads. Phones will be slower; the order usually holds."
    )
    top = max(r["cpu_ms"] for r in rows)
    layout = _layout(
        t,
        title,
        subtitle,
        height=70 + 64 * len(rows),
        xaxis=_axis(t, "Milliseconds per photo", range=[0, top * 1.9], rangemode="tozero"),
        yaxis=_axis(
            t, None, showgrid=False, showline=False, ticklabelstandoff=10, tickfont={"color": t["ink"], "size": 13}
        ),
    )
    layout["margin"]["l"] = 150
    return {"data": data, "layout": layout}


def training_curves(classifiers: dict, yolo: dict | None, mode: str = "light") -> dict:
    """Four panels: classifier validation accuracy and loss, YOLO11n validation mAP and loss.

    classifiers: {arch: {"epoch": [...], "val_accuracy": [...], "val_loss": [...], "best_epoch": int}}
    yolo: {"epoch", "map50", "map50_95", "val_loss", "train_loss", "best_epoch"} or None
    """
    t = THEMES[mode]
    gap = 0.1
    domains = {"x": [0, 0.5 - gap / 2], "x2": [0.5 + gap / 2, 1], "y_top": [0.58, 1], "y_bottom": [0, 0.42]}
    axes = {
        # name: (x axis, y axis, x domain, y domain)
        "acc": ("x", "y", domains["x"], domains["y_top"]),
        "map": ("x2", "y2", domains["x2"], domains["y_top"]),
        "closs": ("x3", "y3", domains["x"], domains["y_bottom"]),
        "yloss": ("x4", "y4", domains["x2"], domains["y_bottom"]),
    }
    data = []

    def line(panel, x, y, name, colour, group, show_legend, dash="solid", fmt=".3f", suffix=""):
        xa, ya, _, _ = axes[panel]
        data.append(
            {
                "type": "scatter",
                "mode": "lines",
                "x": x,
                "y": y,
                "xaxis": xa,
                "yaxis": ya,
                "name": name,
                "legendgroup": group,
                "showlegend": show_legend,
                "line": {"color": colour, "width": 2, "dash": dash},
                "hovertemplate": f"<b>{name}</b><br>epoch %{{x}}<br>%{{y:{fmt}}}{suffix}<extra></extra>",
            }
        )

    def best_marker(panel, x, y, colour, name):
        xa, ya, _, _ = axes[panel]
        data.append(
            {
                "type": "scatter",
                "mode": "markers",
                "x": [x],
                "y": [y],
                "xaxis": xa,
                "yaxis": ya,
                "showlegend": False,
                "marker": _ring(t, colour, 11),
                "hovertemplate": f"<b>{name}</b><br>kept epoch %{{x}}<extra></extra>",
            }
        )

    for arch, h in classifiers.items():
        colour, label = _colour(t, arch), MODEL_LABELS[arch]
        acc = [v * 100 for v in h["val_accuracy"]]
        line("acc", h["epoch"], acc, label, colour, arch, True, fmt=".1f", suffix="%")
        if h.get("val_box_iou"):
            iou = [v * 100 for v in h["val_box_iou"]]
            line(
                "acc",
                h["epoch"],
                iou,
                f"{label} box IoU",
                colour,
                f"{arch}_iou",
                True,
                dash="dot",
                fmt=".1f",
                suffix="%",
            )
        line("closs", h["epoch"], h["val_loss"], label, colour, arch, False)
        if h.get("best_epoch") in h["epoch"]:
            i = h["epoch"].index(h["best_epoch"])
            best_marker("acc", h["best_epoch"], acc[i], colour, label)
            best_marker("closs", h["best_epoch"], h["val_loss"][i], colour, label)

    if yolo:
        colour = _colour(t, "yolo11n")
        line("map", yolo["epoch"], yolo["map50"], "YOLO11n mAP50", colour, "yolo_map50", True)
        line("map", yolo["epoch"], yolo["map50_95"], "YOLO11n mAP50-95", colour, "yolo_map", True, dash="dot")
        line("yloss", yolo["epoch"], yolo["val_loss"], "YOLO11n validation loss", colour, "yolo_val", True)
        if yolo.get("train_loss"):
            line("yloss", yolo["epoch"], yolo["train_loss"], "YOLO11n training loss", t["neutral"], "yolo_tr", True)
        if yolo.get("best_epoch") in yolo["epoch"]:
            i = yolo["epoch"].index(yolo["best_epoch"])
            best_marker("map", yolo["best_epoch"], yolo["map50_95"][i], colour, "YOLO11n")

    panel_titles = {
        "acc": "Classifiers: accuracy (solid) and box IoU (dotted), %",
        "map": "YOLO11n: validation mAP",
        "closs": "Classifiers: validation loss",
        "yloss": "YOLO11n: loss (box + class + DFL)",
    }
    layout_axes, annotations = {}, []
    for panel, (xa, ya, xd, yd) in axes.items():
        x_key = "xaxis" if xa == "x" else f"xaxis{xa[1:]}"
        y_key = "yaxis" if ya == "y" else f"yaxis{ya[1:]}"
        layout_axes[x_key] = _axis(
            t, "Epoch" if panel in ("closs", "yloss") else None, domain=xd, anchor=ya, rangemode="tozero"
        )
        extra = {"range": [0, 104], "ticksuffix": ""} if panel == "acc" else {}
        if panel == "map":
            extra = {"range": [0, 1.04], "tickformat": ".1f"}
        layout_axes[y_key] = _axis(t, None, domain=yd, anchor=xa, **extra)
        annotations.append(
            {
                "text": f"<b>{panel_titles[panel]}</b>",
                "x": xd[0],
                "y": yd[1] + 0.06,
                "xref": "paper",
                "yref": "paper",
                "xanchor": "left",
                "yanchor": "bottom",
                "showarrow": False,
                "font": {"size": 13, "color": t["ink"]},
            }
        )
        has_data = any(d["xaxis"] == xa for d in data)
        if not has_data:
            annotations.append(
                {
                    "text": "not trained in this run",
                    "x": (xd[0] + xd[1]) / 2,
                    "y": (yd[0] + yd[1]) / 2,
                    "xref": "paper",
                    "yref": "paper",
                    "showarrow": False,
                    "font": {"size": 12, "color": t["muted"]},
                }
            )

    kept = [f"{MODEL_LABELS[a]} {h['best_epoch']}" for a, h in classifiers.items() if h.get("best_epoch")]
    if yolo and yolo.get("best_epoch"):
        kept.append(f"YOLO11n {yolo['best_epoch']} of {len(yolo['epoch'])}")
    title = "Epochs kept: " + ", ".join(kept) if kept else "Training curves"
    subtitle = (
        "Scores on the validation photos after each epoch. Rings mark the epoch each model kept "
        "(classifiers: best validation accuracy; YOLO11n: best mAP50-95). Classifier losses include "
        "label smoothing and the box loss, so they level off above 0. Box IoU is how much each "
        "classifier's box overlaps the labelled one."
    )
    layout = _layout(t, title, subtitle, height=640, annotations=annotations, **layout_axes)
    layout["showlegend"] = True
    layout["legend"] = {
        "orientation": "h",
        "x": 0,
        "xanchor": "left",
        "y": -0.12,
        "yanchor": "top",
        "font": {"color": t["ink2"], "size": 12},
        "bgcolor": "rgba(0,0,0,0)",
    }
    layout["margin"]["t"] += 40  # room for the top panels' titles
    layout["margin"]["b"] = 110  # room for the legend
    layout["height"] += 40 + 46
    return {"data": data, "layout": layout}


def confusion(metrics: dict, model: str, mode: str = "light") -> dict:
    """Heatmap of true building (rows) against the model's answer (columns), with counts."""
    t = THEMES[mode]
    matrix = [list(map(int, row)) for row in metrics["confusion"]]
    columns = list(CLASS_SHORT) + ["(none)"]
    if not any(row[-1] for row in matrix):  # no "(none)" answers: drop that column
        matrix = [row[:-1] for row in matrix]
        columns = columns[:-1]
    rows = list(CLASS_SHORT)
    share = [[c / max(sum(row), 1) for c in row] for row in matrix]
    steps = t["sequential"]
    colorscale = [[i / (len(steps) - 1), c] for i, c in enumerate(steps)]

    annotations = []
    for r, row in enumerate(matrix):
        for c, count in enumerate(row):
            if count:
                dark_cell = share[r][c] > 0.55
                annotations.append(
                    {
                        "x": columns[c],
                        "y": rows[r],
                        "text": str(count),
                        "showarrow": False,
                        "font": {"size": 13, "color": t["cell_ink_dark"] if dark_cell else t["ink"]},
                    }
                )

    data = [
        {
            "type": "heatmap",
            "z": share,
            "x": columns,
            "y": rows,
            "zmin": 0,
            "zmax": 1,
            "colorscale": colorscale,
            "showscale": False,
            "xgap": 2,
            "ygap": 2,
            "customdata": matrix,
            "hovertemplate": (
                "true <b>%{y}</b>, answered <b>%{x}</b><br>%{customdata} photos (%{z:.0%} of this building)"
                "<extra></extra>"
            ),
        }
    ]

    n = sum(sum(row) for row in matrix)
    correct = sum(matrix[i][i] for i in range(len(rows)))
    wrong = n - correct
    label = MODEL_LABELS[model]
    mistakes = sorted(
        ((matrix[r][c], rows[r], columns[c]) for r in range(len(rows)) for c in range(len(columns)) if r != c),
        key=lambda m: -m[0],
    )
    if wrong == 0:
        title = f"{label} named all {n} test photos correctly"
    elif wrong == 1:
        _, true, answer = mistakes[0]
        title = f"{label} got 1 of {n} test photos wrong: {true} named as {answer}"
    elif mistakes[0][0] == 1:
        title = f"{label} got {wrong} of {n} test photos wrong, each a different mix-up"
    else:
        count, true, answer = mistakes[0]
        title = f"{label} got {wrong} of {n} test photos wrong; most often {true} named as {answer} ({count} photos)"
    subtitle = (
        "Rows are the building in the photo, columns are the model's answer, and the diagonal is correct. "
        "Numbers are photos; shading is the share of each row."
    )
    layout = _layout(
        t,
        title,
        subtitle,
        height=520,
        annotations=annotations,
        xaxis=_axis(t, "Model's answer", showgrid=False, showline=False, tickangle=-30, side="bottom"),
        yaxis=_axis(t, "Building in the photo", showgrid=False, showline=False, autorange="reversed"),
    )
    layout["margin"]["l"] = 120
    layout["width"] = 760
    return {"data": data, "layout": layout}


# --------------------------------------------------------------------------- inputs from a run folder


def _read_csv(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [{k.strip(): v for k, v in row.items()} for row in csv.DictReader(f)]


def load_classifier_history(run_dir: str, arch: str) -> dict | None:
    path = os.path.join(run_dir, "artifacts", arch, "history.csv")
    if not os.path.exists(path):
        return None
    rows = _read_csv(path)
    if not rows:
        return None
    h = {
        "epoch": [int(float(r["epoch"])) for r in rows],
        "val_accuracy": [float(r["val_accuracy"]) for r in rows],
        "val_loss": [float(r["val_loss"]) for r in rows],
    }
    if all(r.get("val_box_iou") not in (None, "") for r in rows):
        h["val_box_iou"] = [float(r["val_box_iou"]) for r in rows]
    marker = os.path.join(run_dir, "artifacts", arch, "complete.json")
    if os.path.exists(marker):
        with open(marker, encoding="utf-8") as f:
            h["best_epoch"] = json.load(f).get("best_epoch")
    else:
        h["best_epoch"] = h["epoch"][max(range(len(rows)), key=lambda i: (h["val_accuracy"][i], -h["val_loss"][i]))]
    return h


def load_yolo_history(run_dir: str, model: str = "yolo11n") -> dict | None:
    path = os.path.join(run_dir, "artifacts", model, "train", "results.csv")
    if not os.path.exists(path):
        return None
    rows = _read_csv(path)
    if not rows or "metrics/mAP50-95(B)" not in rows[0]:
        return None

    def total(row, prefix):
        keys = [f"{prefix}/box_loss", f"{prefix}/cls_loss", f"{prefix}/dfl_loss"]
        return sum(float(row[k]) for k in keys if k in row and row[k] not in ("", None))

    y = {
        "epoch": [int(float(r["epoch"])) for r in rows],
        "map50": [float(r["metrics/mAP50(B)"]) for r in rows],
        "map50_95": [float(r["metrics/mAP50-95(B)"]) for r in rows],
        "val_loss": [total(r, "val") for r in rows],
        "train_loss": [total(r, "train") for r in rows],
    }
    best = max(range(len(rows)), key=lambda i: y["map50_95"][i])
    y["best_epoch"] = y["epoch"][best]
    return y


def _missing(value) -> bool:
    return value is None or (isinstance(value, float) and value != value)  # None or NaN


def build_specs(run_dir: str, mode: str = "light") -> dict:
    """Every figure the run folder allows, as {name: spec}, in display order."""
    specs = {}
    board = build_leaderboard(run_dir)
    if not board.empty:
        rows = []
        for _, r in board.iterrows():
            if _missing(r.get("accuracy_low")) or _missing(r.get("correct")):
                continue
            rows.append(
                {
                    "id": r["id"],
                    "accuracy": float(r["test_accuracy"]),
                    "low": float(r["accuracy_low"]),
                    "high": float(r["accuracy_high"]),
                    "correct": int(r["correct"]),
                    "images": int(r["test_images"]),
                }
            )
        if rows:
            specs["model_comparison"] = model_comparison(rows, mode)
        timed = [
            {"id": r["id"], "cpu_ms": float(r["cpu_ms"]), "onnx_mb": float(r["onnx_mb"])}
            for _, r in board.iterrows()
            if not _missing(r.get("cpu_ms")) and not _missing(r.get("onnx_mb"))
        ]
        if timed:
            specs["speed_size"] = speed_size(timed, mode)

    classifiers = {a: h for a in CLASSIFIERS if (h := load_classifier_history(run_dir, a))}
    yolo = load_yolo_history(run_dir)
    if classifiers or yolo:
        specs["training_curves"] = training_curves(classifiers, yolo, mode)

    for model in ("yolo11n", *CLASSIFIERS):
        metrics = load_metrics(run_dir, model)
        if metrics and metrics.get("confusion"):
            specs[f"confusion_{model}"] = confusion(metrics, model, mode)
    return specs


# --------------------------------------------------------------------------- writing


def write_html(spec: dict, path: str) -> None:
    import plotly.graph_objects as go

    name = os.path.splitext(os.path.basename(path))[0]
    go.Figure(spec).write_html(
        path,
        include_plotlyjs="cdn",
        full_html=True,
        config={
            "displaylogo": False,
            "responsive": True,
            # The camera button saves a sharp PNG even where Kaleido cannot run.
            "toImageButtonOptions": {"format": "png", "filename": name, "scale": 2},
        },
    )


def run_figures(run_dir: str, LOG, export_dir: str | None = None) -> dict:
    """Interactive HTML in <run>/plots/ and, with export_dir, light and dark PNGs.

    Returns {file name: path}. Without Plotly the figures are skipped with a warning; without a
    working Kaleido the PNGs are skipped with a warning and the HTML is still written.
    """
    written = {}
    with component_stage(LOG, COMPONENT, "Drawing figures", export_dir=export_dir):
        light = build_specs(run_dir, "light")
        if not light:
            log_stage(LOG, "WARN", COMPONENT, skipped="nothing to plot yet")
            return written
        try:
            import plotly.graph_objects as go
        except ImportError:
            log_stage(LOG, "WARN", COMPONENT, skipped="plotly is not installed (pip install plotly)")
            return written

        plots_dir = os.path.join(run_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        for name in os.listdir(plots_dir):  # start from an empty folder, so only this set of figures remains
            if name.endswith((".png", ".html")):
                os.remove(os.path.join(plots_dir, name))
        for name, spec in light.items():
            log_step(LOG, COMPONENT, f"Drawing {name.replace('_', ' ')}")
            path = os.path.join(plots_dir, f"{name}.html")
            write_html(spec, path)
            log_stage(LOG, "ARTIFACT", COMPONENT, figure=name, path=path)
            written[f"{name}.html"] = path

        if export_dir:
            os.makedirs(export_dir, exist_ok=True)
            log_step(LOG, COMPONENT, "Writing light and dark PNGs for the README", to=export_dir)
            dark = build_specs(run_dir, "dark")
            try:
                for mode, specs in (("light", light), ("dark", dark)):
                    for name, spec in specs.items():
                        path = os.path.join(export_dir, f"{name}_{mode}.png")
                        go.Figure(spec).write_image(path, scale=2)
                        log_stage(LOG, "ARTIFACT", COMPONENT, figure=name, mode=mode, path=path)
                        written[f"{name}_{mode}.png"] = path
            except Exception as exc:  # usually Kaleido's Chrome missing, or its system libraries
                reason = " ".join(f"{type(exc).__name__}: {exc}".split())
                log_stage(
                    LOG,
                    "WARN",
                    COMPONENT,
                    png="skipped",
                    reason=reason[:600],
                    fix="install Chrome for Kaleido with: plotly_get_chrome -y",
                )
        log_stage(LOG, "RESULT", COMPONENT, figures=len(light), files=len(written))
    return written
