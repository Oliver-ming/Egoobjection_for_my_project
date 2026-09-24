"""Paths, constants and small I/O helpers shared by every script in ``tools/``.

All default paths are absolute and derived from this file's location, so the scripts work
from any current working directory.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Sequence, Union

SOLUTION_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SOLUTION_ROOT.parent
RESOURCE_DIR = PROJECT_ROOT / "resource"
DATA_ROOT = RESOURCE_DIR / "data"
IMAGE_ROOT = DATA_ROOT / "images"
SPLIT_JSON = {split: DATA_ROOT / f"{split}.json" for split in ("train", "val", "test")}

GENERATED_DATA_DIR = SOLUTION_ROOT / "data"
DOCS_DIR = SOLUTION_ROOT / "docs"
WORK_DIRS = SOLUTION_ROOT / "work_dirs"

CLASS_NAMES = ("pillow", "box", "book", "bottle", "chair", "mug", "door", "shelf", "plate", "sofa")
CATEGORY_IDS = tuple(range(len(CLASS_NAMES)))

PathLike = Union[str, os.PathLike]


def load_json(path: PathLike) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(obj: Any, path: PathLike, indent: int | None = None) -> Path:
    """Write ``obj`` as JSON atomically (temp file + rename) so readers never see a partial file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    separators = None if indent is not None else (",", ":")
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=indent, separators=separators)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return path


def load_predictions(source: Union[PathLike, Sequence[dict]]) -> list[dict]:
    """Return a COCO-results list (``{"image_id", "category_id", "bbox": [x, y, w, h], "score"}``) from a path or
    an in-memory list (returned as-is, not copied)."""
    preds = load_json(source) if isinstance(source, (str, os.PathLike)) else source
    if not isinstance(preds, list):
        raise TypeError(f"COCO results must be a JSON list of detections, got {type(preds).__name__}")
    return preds


def clip_xywh(bbox: Sequence[float], width: float, height: float) -> tuple[float, float, float, float]:
    """Clip an ``[x, y, w, h]`` box to the image; the result may have w or h <= 0."""
    x, y, w, h = bbox
    x1, x2 = min(max(x, 0.0), width), min(max(x + w, 0.0), width)
    y1, y2 = min(max(y, 0.0), height), min(max(y + h, 0.0), height)
    return x1, y1, x2 - x1, y2 - y1
