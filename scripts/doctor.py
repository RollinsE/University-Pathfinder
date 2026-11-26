"""Repository and environment health check for the Campus Pathfinder pipeline.

    python scripts/doctor.py

Exits non-zero only when a repository file or a core library is missing. Training,
export and web-app readiness are reported but do not fail the check.
"""

from __future__ import annotations

import importlib
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CORE = ["numpy", "pandas", "PIL", "yaml", "requests"]
TRAINING = ["torch", "torchvision", "ultralytics"]
EXPORT = ["onnx", "onnxruntime"]
OPTIONAL = ["tqdm", "plotly", "kaleido", "onnxscript"]


def exists(path: str) -> bool:
    return (ROOT / path).exists()


def version_of(module: str) -> str | None:
    try:
        return getattr(importlib.import_module(module), "__version__", "installed")
    except Exception:
        return None


def main() -> int:
    checks = {
        "main.py": exists("main.py"),
        "src package": exists("src/__init__.py"),
        "logger": exists("src/logger.py"),
        "requirements": exists("requirements.txt"),
        "README": exists("README.md"),
        "gitignore": exists(".gitignore"),
        "web app": exists("webapp/index.html"),
    }

    print("Campus Pathfinder doctor")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Project root: {ROOT}")
    for label, ok in checks.items():
        print(f"{label}: {'OK' if ok else 'MISSING'}")

    for module in CORE:
        version = version_of(module)
        checks[f"import {module}"] = version is not None
        print(f"{module}: {version or 'MISSING'}")

    training_ready = True
    for module in TRAINING:
        version = version_of(module)
        training_ready = training_ready and version is not None
        print(f"{module}: {version or 'not installed (needed for training)'}")

    export_ready = True
    for module in EXPORT:
        version = version_of(module)
        export_ready = export_ready and version is not None
        print(f"{module}: {version or 'not installed (needed for export and --mode predict)'}")

    for module in OPTIONAL:
        print(f"{module}: {version_of(module) or 'not installed (optional)'}")

    gpu = "no"
    if version_of("torch"):
        import torch

        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            gpu = "Apple MPS"
    print(f"GPU: {gpu}")

    from src.acquisition import get_api_key

    print(f"Roboflow API key: {'set' if get_api_key() else 'not set (needed to download the real dataset)'}")
    print(f"Node.js (web app tests): {'present' if shutil.which('node') else 'not found'}")

    manifest = ROOT / "webapp" / "models" / "manifest.json"
    if manifest.exists():
        models = json.loads(manifest.read_text(encoding="utf-8")).get("models", [])
        missing = [m["file"] for m in models if not (ROOT / "webapp" / "models" / m["file"]).exists()]
        state = f"{len(models)} models" + (f", missing files: {', '.join(missing)}" if missing else "")
    else:
        state = "no models exported yet"
    print(f"Web app models: {state}")

    print(f"Ready for demo: {'yes' if all(checks.values()) and training_ready else 'no'}")
    print(f"Ready for training: {'yes' if training_ready else 'no'}{' (CPU only: slow)' if gpu == 'no' else ''}")
    print(f"Ready for export: {'yes' if export_ready else 'no'}")

    missing = [label for label, ok in checks.items() if not ok]
    if missing:
        print(f"Missing: {', '.join(missing)}")
        print("Doctor: FAIL")
        return 1
    print("Doctor: OK")
    return 0


if __name__ == "__main__":
    os.chdir(ROOT)
    sys.exit(main())
