import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_gitignore_blocks_data_weights_and_secrets():
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    for pattern in ("data/", "runs/", "*.pt", "*.pth", "*.onnx", "*.zip", ".env", "!.env.example"):
        assert pattern in gitignore, pattern


def test_web_models_and_samples_can_be_committed():
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "!webapp/models/*.onnx" in gitignore


def test_no_api_keys_in_the_repository():
    """Catch a literal key such as api_key="AbC123..." in any tracked text file."""
    pattern = re.compile(r"""api_key\s*[=:]\s*["'][A-Za-z0-9]{16,}["']""", re.IGNORECASE)
    suspicious = []
    for path in ROOT.rglob("*"):
        skip_dirs = {".git", "data", "runs", "node_modules", ".venv"}
        if not path.is_file() or skip_dirs & set(path.parts) or path.suffix in {".png", ".jpg", ".onnx", ".pt"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if pattern.search(text):
            suspicious.append(str(path.relative_to(ROOT)))
    assert not suspicious, suspicious
