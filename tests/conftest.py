import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("PATHFINDER_NO_PROGRESS", "1")  # keep test output free of progress bars


@pytest.fixture
def run_logger(tmp_path):
    """A fresh run folder and logger: returns (LOG, run_id, run_dir)."""
    from src.experiment import create_run_dir
    from src.logger import setup_logger

    run_id, run_dir = create_run_dir(str(tmp_path / "runs"))
    return setup_logger(run_id, run_dir), run_id, run_dir


@pytest.fixture(scope="session")
def demo_dataset(tmp_path_factory):
    """A prepared synthetic dataset shared by the tests: returns its folder."""
    from src.acquisition import run_acquire
    from src.experiment import create_run_dir
    from src.logger import setup_logger

    base = tmp_path_factory.mktemp("demo")
    run_id, run_dir = create_run_dir(str(base / "runs"))
    LOG = setup_logger(run_id, run_dir)
    record = run_acquire(str(base / "data"), run_dir, LOG, demo=True, seed=7)
    return record["prepared_dir"]
