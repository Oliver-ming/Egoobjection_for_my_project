"""Shared plumbing for the Ultralytics pipeline (train.py / predict_coco.py).

Importing this module (before any other ultralytics import) does three things:
  * forces YOLO_AUTOINSTALL=False,
  * points Ultralytics' WEIGHTS_DIR (AMP-check yolo26n.pt download) at solution/weights instead of <cwd>/weights,
  * redirects the label `*.cache` files and `cache='disk'` `*.npy` files, which Ultralytics writes next to the
    labels/images, from resource/data and solution/data/yolo into work_dirs/ultra/_cache
    (resource/ is read-only; train vs trainval would otherwise thrash the same labels/train.cache).
The trainer subclasses below are importable by reference, so DDP worker processes (which unpickle the trainer class)
import this module and get the same patches.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ["YOLO_AUTOINSTALL"] = "False"

SOLUTION = Path(__file__).resolve().parents[1]
DATA_ROOT = Path("/home/zhangyuming/Egoobjection/resource/data").resolve()
YOLO_DATA = (SOLUTION / "data" / "yolo").resolve()  # clipped labels; images are symlinks into DATA_ROOT
WEIGHTS = SOLUTION / "weights"
WORK = SOLUTION / "work_dirs" / "ultra"
CACHE = WORK / "_cache"
DATA_YAMLS = Path(__file__).resolve().parent / "data"

import ultralytics.utils  # noqa: E402
from ultralytics import RTDETR, YOLO  # noqa: E402
from ultralytics.data.base import BaseDataset  # noqa: E402
from ultralytics.data.dataset import YOLODataset  # noqa: E402
from ultralytics.models.rtdetr.train import RTDETRTrainer  # noqa: E402
from ultralytics.models.yolo.detect import DetectionTrainer  # noqa: E402
from ultralytics.nn.tasks import RTDETRDetectionModel  # noqa: E402

ultralytics.utils.WEIGHTS_DIR = WEIGHTS


def _needs_cache_redirect(p: str | Path) -> bool:
    """Redirect caches out of resource/ and out of solution/data/yolo (train vs trainval share labels/train)."""
    r = Path(p).resolve()
    return r.is_relative_to(DATA_ROOT) or r.is_relative_to(YOLO_DATA)


_orig_load_or_scan_cache = YOLODataset._load_or_scan_cache


def _load_or_scan_cache(self, cache_path, cache_hash):
    cache_path = Path(cache_path)
    if _needs_cache_redirect(cache_path):
        CACHE.mkdir(parents=True, exist_ok=True)
        # the hash in the name keeps train-only and train+val caches (same labels/train parent) from thrashing
        cache_path = CACHE / f"{cache_path.parent.name}_{cache_path.stem}_{cache_hash[:12]}.cache"
    return _orig_load_or_scan_cache(self, cache_path, cache_hash)


_orig_check_cache_disk = BaseDataset.check_cache_disk


def _check_cache_disk(self, *args, **kwargs):
    npy = []
    for f in self.im_files:
        f = Path(f)
        # resolve() follows image symlinks into resource/; also catch YOLO_DATA paths
        if _needs_cache_redirect(f):
            d = CACHE / "npy" / f.parent.name
            d.mkdir(parents=True, exist_ok=True)
            npy.append(d / f"{f.stem}.npy")
        else:
            npy.append(f.with_suffix(".npy"))
    self.npy_files = npy
    return _orig_check_cache_disk(self, *args, **kwargs)


YOLODataset._load_or_scan_cache = _load_or_scan_cache
BaseDataset.check_cache_disk = _check_cache_disk


class EgoDetectionTrainer(DetectionTrainer):
    """DetectionTrainer whose module applies the cache/weights patches above (also inside DDP workers)."""


class EgoRTDETRTrainer(RTDETRTrainer):
    """RTDETRTrainer whose module applies the cache/weights patches above (also inside DDP workers)."""


def resolve_weights(name: str | Path) -> str:
    """Return a local path for `name`: existing path > solution/weights/<name> > download into solution/weights/."""
    p = Path(name)
    if p.suffix in {".yaml", ".yml"}:  # model config (from scratch), resolved by Ultralytics itself
        return str(name)
    if p.exists():
        return str(p.resolve())
    local = WEIGHTS / p.name
    if not local.exists():
        from ultralytics.utils.downloads import attempt_download_asset

        WEIGHTS.mkdir(parents=True, exist_ok=True)
        attempt_download_asset(str(local))  # downloads the GitHub release asset named local.name to this path
    if not local.exists():
        raise FileNotFoundError(f"could not find or download weights '{name}'")
    return str(local)


def load_model(name: str | Path):
    """Load a checkpoint/config with the right wrapper class: RTDETR for RT-DETR models, YOLO otherwise."""
    path = resolve_weights(name)
    if Path(path).suffix in {".yaml", ".yml"}:
        return (RTDETR if "rtdetr" in Path(path).name.lower() else YOLO)(path)
    model = YOLO(path)
    if isinstance(model.model, RTDETRDetectionModel):
        model = RTDETR(path)
    return model


def is_rtdetr(model) -> bool:
    return isinstance(model, RTDETR) or isinstance(getattr(model, "model", None), RTDETRDetectionModel)


def require_cuda_visible_devices():
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        sys.exit("Refusing to run: set CUDA_VISIBLE_DEVICES explicitly to your assigned GPU(s) (see CONVENTIONS.md).")
