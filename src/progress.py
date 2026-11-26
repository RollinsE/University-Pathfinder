"""Green progress bars for downloads, extraction, file copies and training loops.

Uses tqdm.auto, so the same call draws a text bar in a terminal and a widget bar in
Jupyter or Colab. If tqdm is not installed the helpers fall back to plain iteration,
so nothing in the pipeline depends on the bars being drawn.

    from src.progress import progress, download_file, extract_zip

    for path in progress(paths, desc="Copying images", unit="img"):
        ...
"""

from __future__ import annotations

import os
import zipfile

BAR_COLOUR = "green"

try:
    from tqdm.auto import tqdm as _tqdm
except ImportError:  # pragma: no cover - exercised only where tqdm is missing
    _tqdm = None


class _NullBar:
    """Stand-in with the parts of the tqdm interface the pipeline uses."""

    def __init__(self, iterable=None, total=None, **_):
        self.iterable = iterable
        self.total = total
        self.n = 0

    def __iter__(self):
        return iter(self.iterable if self.iterable is not None else [])

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def update(self, n=1):
        self.n += n

    def set_postfix(self, *args, **kwargs):
        pass

    def set_description(self, *args, **kwargs):
        pass

    def close(self):
        pass


def progress(iterable=None, total=None, desc: str = "", unit: str = "it", leave: bool = True, **kwargs):
    """A green tqdm bar (or a silent stand-in when tqdm is unavailable or bars are disabled)."""
    if _tqdm is None or os.environ.get("PATHFINDER_NO_PROGRESS") == "1":
        return _NullBar(iterable=iterable, total=total)
    return _tqdm(
        iterable,
        total=total,
        desc=desc,
        unit=unit,
        leave=leave,
        colour=BAR_COLOUR,
        dynamic_ncols=True,
        **kwargs,
    )


def download_file(url: str, dest: str, desc: str = "Downloading", timeout: int = 60, session=None) -> str:
    """Stream a URL to dest with a byte-count progress bar. Writes to a .part file first."""
    import requests

    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
    tmp = dest + ".part"
    http = session or requests
    with http.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0)) or None
        bar = progress(total=total, desc=desc, unit="B", unit_scale=True, unit_divisor=1024)
        with open(tmp, "wb") as f, bar:
            for chunk in response.iter_content(chunk_size=1 << 16):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
    os.replace(tmp, dest)
    return dest


def extract_zip(path: str, dest: str, desc: str = "Extracting") -> list[str]:
    """Extract a zip archive member by member with a progress bar. Returns the member names."""
    os.makedirs(dest, exist_ok=True)
    root = os.path.abspath(dest)
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        for member in progress(members, desc=desc, unit="file"):
            target = os.path.abspath(os.path.join(dest, member.filename))
            if not target.startswith(root + os.sep) and target != root:
                raise ValueError(f"Unsafe path in archive: {member.filename}")
            archive.extract(member, dest)
        return [m.filename for m in members]
