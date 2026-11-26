"""Dataset acquisition: Roboflow download, a local export, or synthetic demo data.

The Roboflow project is public on Roboflow Universe, so any account's API key can download
it. The key is read from the ROBOFLOW_API_KEY environment variable (or a .env file) and is
never logged or written to disk.

    data/
    ├── raw/<project>-v<version>.zip       downloaded archive
    ├── raw/<project>-v<version>/          extracted Roboflow export
    └── processed/<project>-v<version>/    prepared dataset (see src/datasets.py)
"""

from __future__ import annotations

import json
import os
import time

from src.constants import (
    ROBOFLOW_API_URL,
    ROBOFLOW_FORMAT,
    ROBOFLOW_PROJECT,
    ROBOFLOW_VERSION,
    ROBOFLOW_WORKSPACE,
)
from src.datasets import prepare_dataset
from src.logger import component_stage, log_stage, log_step
from src.progress import download_file, extract_zip

API_KEY_ENV = "ROBOFLOW_API_KEY"


def load_env_file(path: str = ".env") -> None:
    """Set variables from a simple KEY=value .env file, without overriding the environment."""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def get_api_key() -> str | None:
    load_env_file()
    key = os.environ.get(API_KEY_ENV, "").strip()
    return key or None


def fetch_export_link(
    workspace: str,
    project: str,
    version: int,
    fmt: str,
    api_key: str,
    LOG,
    max_wait_sec: int = 300,
    poll_sec: int = 5,
) -> str:
    """Ask the Roboflow API for a download link, waiting while the export is generated."""
    import requests

    url = f"{ROBOFLOW_API_URL}/{workspace}/{project}/{version}/{fmt}"
    deadline = time.time() + max_wait_sec
    waited = False
    while True:
        response = requests.get(url, params={"api_key": api_key}, timeout=60)
        if response.status_code in (401, 403):
            raise PermissionError(
                f"Roboflow refused the request (HTTP {response.status_code}). Check that {API_KEY_ENV} "
                "holds a valid key for your own Roboflow account."
            )
        if response.status_code == 404:
            raise FileNotFoundError(f"Roboflow has no dataset at {workspace}/{project}/{version} ({fmt}).")
        response.raise_for_status()
        payload = response.json()
        link = (payload.get("export") or {}).get("link")
        if link:
            return link
        if time.time() > deadline:
            raise TimeoutError(f"Roboflow did not finish preparing the {fmt} export within {max_wait_sec}s.")
        if not waited:
            log_step(LOG, "Acquire", "Waiting for Roboflow to generate the export")
            waited = True
        time.sleep(poll_sec)


def download_roboflow(
    raw_dir: str,
    LOG,
    workspace: str = ROBOFLOW_WORKSPACE,
    project: str = ROBOFLOW_PROJECT,
    version: int = ROBOFLOW_VERSION,
    fmt: str = ROBOFLOW_FORMAT,
    overwrite: bool = False,
) -> str:
    """Download and extract a Roboflow export. Returns the extracted folder."""
    export_dir = os.path.join(raw_dir, f"{project}-v{version}")
    if os.path.exists(os.path.join(export_dir, "data.yaml")) and not overwrite:
        log_step(LOG, "Acquire", "Dataset already downloaded, reusing it", folder=export_dir)
        return export_dir

    api_key = get_api_key()
    if not api_key:
        raise RuntimeError(
            f"{API_KEY_ENV} is not set. Create a free Roboflow account, copy the private API key from "
            "your workspace settings and set it as an environment variable, a .env entry or a Colab secret."
        )

    with component_stage(
        LOG,
        "Acquire",
        "Downloading dataset from Roboflow",
        workspace=workspace,
        project=project,
        version=version,
        format=fmt,
    ):
        log_step(LOG, "Acquire", "Requesting download link")
        link = fetch_export_link(workspace, project, version, fmt, api_key, LOG)
        zip_path = os.path.join(raw_dir, f"{project}-v{version}.zip")
        log_step(LOG, "Acquire", "Downloading archive")
        download_file(link, zip_path, desc=f"Downloading {project} v{version}")
        size_mb = round(os.path.getsize(zip_path) / 1e6, 1)
        log_stage(LOG, "ARTIFACT", "Acquire", path=zip_path, size_mb=size_mb)
        log_step(LOG, "Acquire", "Extracting archive", size_mb=size_mb)
        members = extract_zip(zip_path, export_dir, desc="Extracting dataset")
        log_stage(LOG, "INFO", "Acquire", extracted_files=len(members), folder=export_dir)
    return export_dir


def run_acquire(
    data_dir: str,
    run_dir: str,
    LOG,
    version: int = ROBOFLOW_VERSION,
    source_dir: str | None = None,
    demo: bool = False,
    workspace: str = ROBOFLOW_WORKSPACE,
    project: str = ROBOFLOW_PROJECT,
    seed: int = 42,
    overwrite: bool = False,
) -> dict:
    """Get an export (demo, local folder or Roboflow) and prepare it. Records the result in the run."""
    raw_dir = os.path.join(data_dir, "raw")
    processed_dir = os.path.join(data_dir, "processed")
    source_label = "synthetic demo data" if demo else (f"local export {source_dir}" if source_dir else "Roboflow")
    remote = not (demo or source_dir)
    log_step(
        LOG,
        "Acquire",
        "Acquiring dataset",
        source=source_label,
        workspace=workspace if remote else None,
        project=project if remote else None,
        version=version if remote else None,
    )

    if demo:
        from src.demo import make_demo_export

        export_dir = make_demo_export(os.path.join(raw_dir, "demo-export"), LOG, seed=seed, overwrite=overwrite)
        prepared_dir = os.path.join(processed_dir, "demo")
        source = "demo"
    elif source_dir:
        log_step(LOG, "Acquire", "Using a dataset export already on disk", folder=source_dir)
        export_dir = source_dir
        prepared_dir = os.path.join(processed_dir, os.path.basename(os.path.normpath(source_dir)))
        source = "local"
    else:
        export_dir = download_roboflow(
            raw_dir, LOG, workspace=workspace, project=project, version=version, overwrite=overwrite
        )
        prepared_dir = os.path.join(processed_dir, f"{project}-v{version}")
        source = "roboflow"

    manifest = prepare_dataset(export_dir, prepared_dir, LOG, overwrite=overwrite)
    record = {
        "source": source,
        "workspace": workspace if source == "roboflow" else None,
        "project": project if source == "roboflow" else None,
        "version": None if demo else version,
        "export_dir": os.path.abspath(export_dir),
        "prepared_dir": os.path.abspath(prepared_dir),
        "images": manifest["total_images"],
        "splits": {k: v["images"] for k, v in manifest["splits"].items()},
    }
    path = dataset_record_path(run_dir)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    log_stage(LOG, "ARTIFACT", "Acquire", path=path)
    log_stage(LOG, "RESULT", "Acquire", images=record["images"], **record["splits"])
    return record


def dataset_record_path(run_dir: str) -> str:
    return os.path.join(run_dir, "metrics", "dataset.json")


def resolve_prepared_dir(run_dir: str, dataset_dir: str | None = None) -> str:
    """The prepared dataset for this run: --dataset_dir if given, else the one acquire recorded."""
    if dataset_dir:
        return dataset_dir
    path = dataset_record_path(run_dir)
    if not os.path.exists(path):
        raise FileNotFoundError(
            "No dataset recorded for this run. Run --mode acquire first (same --run_dir), "
            "or pass --dataset_dir pointing at a prepared dataset."
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)["prepared_dir"]
