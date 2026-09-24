#!/usr/bin/env python3
"""COCO bbox evaluation that reproduces ``resource/eval.py`` exactly, plus sanity warnings and a metrics json.

The metric path is the official one: ``COCO(gt).loadRes(preds)`` -> ``COCOeval(iouType="bbox")`` ->
``evaluate / accumulate / summarize``; per-class AP is computed like ``eval.py`` (IoU 0.50:0.95, area=all,
maxDets=100). Differences, all outside the metric itself:

* predictions whose ``image_id`` is not in the GT are dropped with a warning (``eval.py`` would crash). This
  makes it possible to score full-val predictions against ``val_mini.json``;
* an empty prediction list yields all-zero metrics instead of a crash;
* warnings for images with more than ``--max-dets`` detections (pycocotools applies maxDets per image *and
  category*, so extra detections still count locally but may be cut by a per-image cap elsewhere) and for
  category ids outside the GT set (silently ignored by COCOeval).

CLI::

    python tools/coco_eval.py --gt resource/data/val.json --pred work_dirs/exp/val_preds.json \
        --out work_dirs/exp/val_metrics.json --per-class

Python::

    from tools.coco_eval import evaluate
    metrics = evaluate("resource/data/val.json", preds_list)   # -> {"AP": ..., "AP50": ..., "per_class": {...}}
"""
from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Sequence, Union

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.common import SPLIT_JSON, PathLike, dump_json, load_predictions

STAT_NAMES = ("AP", "AP50", "AP75", "APs", "APm", "APl", "AR1", "AR10", "AR100", "ARs", "ARm", "ARl")
COCO_MAX_DETS = 100


def _quiet(enabled: bool):
    """Context manager swallowing pycocotools' stdout chatter when ``enabled``."""
    return contextlib.redirect_stdout(io.StringIO()) if enabled else contextlib.nullcontext()


def load_gt(gt: Union[PathLike, COCO], verbose: bool = True) -> COCO:
    if isinstance(gt, COCO):
        return gt
    with _quiet(not verbose):
        return COCO(str(gt))


def sanity_check(coco_gt: COCO, preds: Sequence[dict], max_dets: int = COCO_MAX_DETS) -> tuple[list[dict], list[str]]:
    """Return ``(predictions on GT images, warnings)``; nothing else is modified."""
    warnings = []
    gt_img_ids, gt_cat_ids = set(coco_gt.getImgIds()), set(coco_gt.getCatIds())

    unknown_imgs = Counter(p["image_id"] for p in preds if p["image_id"] not in gt_img_ids)
    if unknown_imgs:
        warnings.append(f"dropped {sum(unknown_imgs.values())} predictions on {len(unknown_imgs)} image_ids "
                        f"not in GT (e.g. {next(iter(unknown_imgs))})")
        preds = [p for p in preds if p["image_id"] in gt_img_ids]

    unknown_cats = sorted({p["category_id"] for p in preds} - gt_cat_ids)
    if unknown_cats:
        warnings.append(f"prediction category_id(s) not in GT categories (ignored by COCOeval): {unknown_cats}")

    per_image = Counter(p["image_id"] for p in preds)
    crowded = {img: n for img, n in per_image.items() if n > max_dets}
    if crowded:
        warnings.append(f"{len(crowded)} images have > {max_dets} detections (max {max(crowded.values())}); "
                        f"pycocotools caps per image AND category, a per-image cap would score lower")
    return list(preds), warnings


def _empty_results(coco_gt: COCO) -> COCO:
    coco_dt = COCO()
    coco_dt.dataset = {"images": coco_gt.dataset["images"], "categories": coco_gt.dataset["categories"],
                       "annotations": []}
    coco_dt.createIndex()
    return coco_dt


def per_class_ap(coco_eval: COCOeval, coco_gt: COCO) -> dict[str, dict[str, float]]:
    """AP (IoU 0.50:0.95) and AP50 per category at area=all, maxDets=100 -- same slicing as ``eval.py``."""
    precision = coco_eval.eval["precision"]  # [T, R, K, A, M]
    area_all, max_det_100 = 0, 2
    result = {}
    for k, cat_id in enumerate(map(int, coco_eval.params.catIds)):  # numpy ints after COCOeval.evaluate()
        p = precision[:, :, k, area_all, max_det_100]
        p50 = precision[0, :, k, area_all, max_det_100]
        name = coco_gt.cats[cat_id]["name"]
        result[name] = {"id": cat_id,
                        "AP": float(p[p > -1].mean()) if (p > -1).any() else float("nan"),
                        "AP50": float(p50[p50 > -1].mean()) if (p50 > -1).any() else float("nan")}
    return result


def evaluate(gt: Union[PathLike, COCO], preds: Union[PathLike, Sequence[dict]], verbose: bool = True,
             print_per_class: bool = False, max_dets: int = COCO_MAX_DETS) -> dict:
    """Evaluate COCO-results ``preds`` (path or list) against ``gt`` (path or ``COCO``) for bbox.

    Returns ``{"AP", "AP50", ..., "ARl", "per_class": {name: {"id", "AP", "AP50"}}, "num_images",
    "num_predictions", "warnings"}``. The input list is not mutated.
    """
    coco_gt = load_gt(gt, verbose)
    preds, warnings = sanity_check(coco_gt, load_predictions(preds), max_dets)
    for message in warnings:
        print(f"[Warn] {message}", file=sys.stderr)

    with _quiet(not verbose):
        # loadRes adds keys (id, area, segmentation, ...) to each dict, so hand it copies.
        coco_dt = coco_gt.loadRes([dict(p) for p in preds]) if preds else _empty_results(coco_gt)
        coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

    metrics: dict = {name: float(v) for name, v in zip(STAT_NAMES, coco_eval.stats)}
    metrics["per_class"] = per_class_ap(coco_eval, coco_gt)
    metrics.update(num_images=len(coco_gt.getImgIds()), num_predictions=len(preds), warnings=warnings)
    if print_per_class:
        print("\nPer-category AP (IoU=0.50:0.95, area=all, maxDets=100):")
        rows = sorted(metrics["per_class"].items(), key=lambda kv: (np.isnan(kv[1]["AP"]), -kv[1]["AP"]))
        for name, row in rows:
            print(f"{name:>20s}: {row['AP']:0.4f}   (AP50 {row['AP50']:0.4f})")
    return metrics


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gt", type=Path, default=SPLIT_JSON["val"], help="GT COCO json (default: val.json)")
    parser.add_argument("--pred", type=Path, required=True, help="COCO results json (list of detections)")
    parser.add_argument("--out", type=Path, default=None, help="write metrics json here")
    parser.add_argument("--per-class", action="store_true", help="print per-class AP / AP50")
    parser.add_argument("--max-dets", type=int, default=COCO_MAX_DETS, help="warn above this many dets per image")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    metrics = evaluate(args.gt, args.pred, print_per_class=args.per_class, max_dets=args.max_dets)
    metrics.update(gt=os.fspath(args.gt), pred=os.fspath(args.pred))
    print("\n" + "  ".join(f"{name}={metrics[name]:.4f}" for name in STAT_NAMES))
    if args.out:
        print(f"wrote {dump_json(metrics, args.out, indent=2)}")


if __name__ == "__main__":
    main()
