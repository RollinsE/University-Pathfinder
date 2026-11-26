"""Run logger for the Campus Pathfinder pipeline.

One console + file logger per run, structured ``[STAGE] component=... key=value``
lines, and a context manager that times each pipeline stage.

run.log keeps the full structured lines. The console renders the same records
in a shorter, readable form (src/console.py).

Usage
-----
    from src.logger import setup_logger, log_stage, log_step, component_stage

    run_id, run_dir = create_run_dir(base="runs")
    LOG = setup_logger(run_id, run_dir)

    with component_stage(LOG, "Classifier:resnet18", "Training ResNet18", epochs=30, device="cuda"):
        log_step(LOG, "Classifier:resnet18", "Loading train and validation images")
        train(...)
        log_stage(LOG, "CHECKPOINT", "Classifier:resnet18", path=checkpoint_path)

    log_stage(LOG, "METRICS", "Classifier:resnet18", split="test", accuracy=0.9412, macro_f1=0.9380)
"""

import logging
import os
import time
from contextlib import contextmanager

from src.console import make_console_handler

LOGGER_NAME = "campus_pathfinder"


def setup_logger(run_id: str, run_dir: str, log_level: str = "INFO"):
    """
    Console + file logger with consistent formatting.

    Writes to <run_dir>/logs/run.log in append mode, so separate commands and
    resumed runs continue the same file.
    """
    logs_dir = os.path.join(run_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    log_path = os.path.join(logs_dir, "run.log")

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    for handler in list(logger.handlers):
        handler.close()
    logger.handlers.clear()
    logger.filters.clear()
    logger.propagate = False

    fmt = logging.Formatter(
        fmt=f"%(asctime)s | %(levelname)s | %(run_id)s | {LOGGER_NAME} | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    class RunIdFilter(logging.Filter):
        def filter(self, record):
            record.run_id = run_id
            return True

    logger.addFilter(RunIdFilter())

    logger.addHandler(make_console_handler(logger.level))

    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    file_only = {"cp_file_only": True}
    logger.info("=" * 70, extra=file_only)
    logger.info(f"Logger initialized. log_path={log_path}", extra=file_only)
    logger.info("=" * 70, extra=file_only)
    return logger


def get_logger():
    """Return the run logger from any module without passing it around.

    Call setup_logger() once at the entry point (CLI / notebook) first.
    """
    return logging.getLogger(LOGGER_NAME)


def log_stage(LOG, stage: str, component: str, file_only: bool = False, **kwargs):
    """
    Standard structured log line.

    stage: START | STEP | INFO | WARN | DONE | FAIL | EVAL | CHECKPOINT | ARTIFACT | METRICS | RESULT
    component: Pipeline | Acquire | Prepare | Dataset | Demo | Classifier:<model> | Detector:<model>
               | Evaluate | Figures | Export | Predict
    file_only: write to run.log but keep it off the console (it is already shown another way).
    """
    _emit(LOG, stage, component, None, kwargs, file_only)


def log_step(LOG, component: str, message: str, **kwargs):
    """
    Say what the pipeline is doing now, in plain words.

    Console: ``Acquire: Downloading dataset from Roboflow (version 4)``
    run.log: ``[STEP] component=Acquire "Downloading dataset from Roboflow" version=4``
    """
    _emit(LOG, "STEP", component, message, kwargs, file_only=False)


def _emit(LOG, stage: str, component: str, message: str | None, kwargs: dict, file_only: bool):
    fields = {k: v for k, v in kwargs.items() if v is not None}
    extras = " ".join(f"{k}={v}" for k, v in fields.items())
    msg = f"[{stage}] component={component}"
    if message:
        msg += f' "{message}"'
    if extras:
        msg += f" {extras}"
    level = logging.WARNING if stage == "WARN" else logging.ERROR if stage == "FAIL" else logging.INFO
    LOG.log(
        level,
        msg,
        extra={
            "cp_stage": stage,
            "cp_component": component,
            "cp_message": message,
            "cp_fields": fields,
            "cp_file_only": file_only,
        },
    )


@contextmanager
def component_stage(LOG, component: str, action: str | None = None, **kwargs):
    """
    Context manager that logs:
      [START] component=... "<action>" key=value ...
      [DONE]  component=... elapsed_sec=...
    and on exception:
      [FAIL]  component=... elapsed_sec=...
      plus stack trace.

    action: what the stage is doing, in plain words ("Downloading dataset from Roboflow").
    The console shows it in place of "started".

    The traceback is written once, by the innermost stage that saw the error;
    outer stages only add their own [FAIL] line.
    """
    _emit(LOG, "START", component, action, kwargs, file_only=False)
    t0 = time.time()
    try:
        yield
    except Exception as exc:
        log_stage(LOG, "FAIL", component, elapsed_sec=round(time.time() - t0, 1))
        if not getattr(exc, "_cp_logged", False):
            LOG.exception(f"Exception in component={component}")
            try:
                exc._cp_logged = True
            except Exception:
                pass
        raise
    else:
        log_stage(LOG, "DONE", component, elapsed_sec=round(time.time() - t0, 1))
