import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_core_entrypoints_exist():
    assert (ROOT / "main.py").exists()
    for module in (
        "logger",
        "console",
        "experiment",
        "progress",
        "constants",
        "acquisition",
        "datasets",
        "demo",
        "classify",
        "detect",
        "evaluate",
        "inference",
        "export",
        "figures",
        "predict",
        "notebook_viewer",
    ):
        assert (ROOT / "src" / f"{module}.py").exists(), module


def test_documentation_files_exist():
    for name in ("README.md", "LICENSE", "NOTICE.md"):
        assert (ROOT / name).exists(), name
    for name in ("LOGGING.md", "MODEL_CARD.md", "DATA_CARD.md", "WEB_APP.md"):
        assert (ROOT / "docs" / name).exists(), name


def test_web_app_files_exist():
    for name in ("index.html", "app.js", "inference.js", "styles.css", "buildings.json"):
        assert (ROOT / "webapp" / name).exists(), name


def test_pipeline_code_logs_through_the_run_logger():
    """No bare print() in src/: everything must reach run.log."""
    offenders = []
    for path in sorted((ROOT / "src").glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "print":
                offenders.append(f"{path.name}:{node.lineno}")
    assert not offenders, offenders


def test_buildings_json_covers_every_class():
    from src.constants import CLASS_SLUGS

    buildings = json.loads((ROOT / "webapp" / "buildings.json").read_text(encoding="utf-8"))
    assert sorted(buildings) == sorted(CLASS_SLUGS)


def test_notebook_is_in_sync_with_builder():
    sys.path.insert(0, str(ROOT / "scripts"))
    import build_notebook

    on_disk = json.loads((ROOT / "notebooks" / "colab.ipynb").read_text(encoding="utf-8"))
    assert on_disk == build_notebook.build()
