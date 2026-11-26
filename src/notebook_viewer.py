"""Notebook helper for viewing a finished run.

    from src.notebook_viewer import display_run
    display_run(RUN_DIR)

Shows the results table, the dataset summary and the figures the run saved, without
re-running anything.
"""

from __future__ import annotations

import json
import os


def display_run(run_dir: str, show_plots: bool = True) -> None:
    import pandas as pd
    from IPython.display import Markdown, display

    metrics = os.path.join(run_dir, "metrics")
    shown = False

    dataset = os.path.join(metrics, "dataset.json")
    if os.path.exists(dataset):
        with open(dataset, encoding="utf-8") as f:
            d = json.load(f)
        splits = ", ".join(f"{k} {v}" for k, v in d.get("splits", {}).items())
        display(Markdown(f"**Dataset:** {d.get('source')} ({d.get('images')} images: {splits})"))

    results = os.path.join(metrics, "results.md")
    leaderboard = os.path.join(metrics, "leaderboard.csv")
    if os.path.exists(results):
        display(Markdown("## Test results"))
        with open(results, encoding="utf-8") as f:
            display(Markdown(f.read()))
        shown = True
    elif os.path.exists(leaderboard):
        display(Markdown("## Test results"))
        display(pd.read_csv(leaderboard))
        shown = True

    plots = os.path.join(run_dir, "plots")
    if show_plots and os.path.isdir(plots):
        from IPython.display import HTML

        order = ["model_comparison", "speed_size", "training_curves"]
        names = [n for n in os.listdir(plots) if n.endswith(".html")]
        names.sort(key=lambda n: (order.index(n[:-5]) if n[:-5] in order else len(order), n))
        for name in names:  # interactive Plotly figures from src/figures.py
            with open(os.path.join(plots, name), encoding="utf-8") as f:
                display(HTML(f.read()))
            shown = True

    if not shown:
        display(Markdown(f"Nothing to show yet in `{run_dir}`. Run the training and evaluate stages first."))
