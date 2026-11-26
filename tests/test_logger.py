import os

import pytest

from src.console import ConsoleFormatter, format_duration, format_value
from src.logger import component_stage, log_stage, log_step


def _read_log(run_dir):
    with open(os.path.join(run_dir, "logs", "run.log"), encoding="utf-8") as f:
        return f.read()


def test_structured_lines_reach_run_log(run_logger):
    LOG, run_id, run_dir = run_logger
    log_stage(LOG, "METRICS", "Classifier:resnet18", split="test", accuracy=0.9412, skipped=None)
    text = _read_log(run_dir)
    expected = "[METRICS] component=Classifier:resnet18 split=test accuracy=0.9412"
    assert f"| {run_id} | campus_pathfinder | {expected}" in text
    assert "skipped" not in text


def test_failure_writes_one_traceback(run_logger):
    LOG, _, run_dir = run_logger
    with pytest.raises(RuntimeError):
        with component_stage(LOG, "Outer"):
            with component_stage(LOG, "Inner"):
                raise RuntimeError("boom")
    text = _read_log(run_dir)
    assert "[FAIL] component=Inner" in text
    assert "[FAIL] component=Outer" in text
    assert text.count("Traceback (most recent call last)") == 1


def test_console_formatting():
    assert format_value("accuracy", 0.9412) == "94.12%"
    assert format_value("macro_f1", 0.9412) == "0.9412"
    assert format_value("images", 1280) == "1,280"
    assert format_duration(252) == "4m 12s"


def _console_lines(LOG):
    """Render this logger's records the way the console would."""
    import logging

    captured = []

    class Capture(logging.Handler):
        def emit(self, record):
            captured.append(ConsoleFormatter()._render(record))

    handler = Capture()
    LOG.addHandler(handler)
    return captured, handler


def test_steps_and_actions_say_what_is_happening(run_logger):
    LOG, _, run_dir = run_logger
    lines, handler = _console_lines(LOG)
    try:
        with component_stage(LOG, "Acquire", "Downloading dataset from Roboflow", version=4):
            log_step(LOG, "Acquire", "Extracting archive", size_mb=12.5)
        with component_stage(LOG, "Evaluate"):
            pass
    finally:
        LOG.removeHandler(handler)

    assert lines[0] == "Acquire: Downloading dataset from Roboflow (version 4)"
    assert lines[1] == "Acquire: Extracting archive (size mb 12.5)"
    assert lines[3] == "Evaluate: started"
    text = _read_log(run_dir)
    assert '[START] component=Acquire "Downloading dataset from Roboflow" version=4' in text
    assert '[STEP] component=Acquire "Extracting archive" size_mb=12.5' in text
