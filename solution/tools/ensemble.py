#!/usr/bin/env python3
"""Fuse N COCO-results files into one: Weighted Boxes Fusion, or NMS / soft-NMS for comparison.

WBF follows Solovyev et al., "Weighted boxes fusion" (2021) and the reference ``ensemble_boxes`` implementation:
per image and class, boxes of all models are sorted by ``score * model_weight``; each box joins the fused box it
overlaps most if IoU > ``iou_thr`` (else starts a new cluster); fused coordinates are the score-weighted mean of the
cluster, and the fused score is rescaled by the number of contributing boxes / models according to ``conf_type``:

* ``avg``                    -- mean(score*w) * min(M, n) / sum(W)   (clusters seen by few models are down-weighted)
* ``max``                    -- max(score*w) / max(W)
* ``box_and_model_avg``      -- sum(score*w) / sum(w of boxes) * sum(W of models present) / sum(W)
* ``absent_model_aware_avg`` -- sum(score*w) / (sum(w of boxes) + sum(W of absent models))

where ``M`` = number of models, ``n`` = boxes in the cluster, ``W`` = model weights. Coordinates stay in absolute
pixels (IoU is invariant to the per-axis normalization the reference uses), so no image sizes are needed.
``nms`` / ``soft-nms`` concatenate all models (scores scaled by ``w / max(W)``) and run class-wise greedy NMS /
Gaussian soft-NMS. Every method keeps the top ``max_per_image`` detections per image.

CLI::

    python tools/ensemble.py --preds a.json b.json c.json --weights 2 1 1 --out fused.json \
        [--method wbf] [--iou-thr 0.6] [--conf-type avg] [--gt resource/data/val.json]

Python (e.g. test-time-augmentation fusion of one image; boxes are absolute ``x1 y1 x2 y2``)::

    from tools.ensemble import FusionConfig, fuse, ensemble_predictions
    boxes, scores, labels = fuse([boxes_a, boxes_b], [scores_a, scores_b], [labels_a, labels_b])
    fused_coco = ensemble_predictions([preds_a, preds_b], weights=[1, 1], config=FusionConfig(iou_thr=0.55))
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.common import dump_json, load_predictions

METHODS = ("wbf", "nms", "soft-nms")
CONF_TYPES = ("avg", "max", "box_and_model_avg", "absent_model_aware_avg")
SEQUENTIAL_BELOW_IMAGES = 64  # process pools are not worth their start-up cost for fewer images


@dataclass(frozen=True)
class FusionConfig:
    method: str = "wbf"
    iou_thr: float = 0.6
    skip_box_thr: float = 0.001  # drop input boxes scoring below this; also the soft-NMS pruning threshold
    conf_type: str = "avg"  # WBF only
    max_per_image: int = 100  # <= 0 disables the cap
    sigma: float = 0.5  # soft-NMS Gaussian sigma

    def __post_init__(self) -> None:
        if self.method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}, got {self.method!r}")
        if self.conf_type not in CONF_TYPES:
            raise ValueError(f"conf_type must be one of {CONF_TYPES}, got {self.conf_type!r}")
        if not 0.0 <= self.iou_thr <= 1.0 or self.sigma <= 0:
            raise ValueError("iou_thr must be in [0, 1] and sigma > 0")


# ----------------------------------------------------------------------------------------------------------------
# per-class kernels (boxes: (n, 4) absolute x1 y1 x2 y2 with positive area)
# ----------------------------------------------------------------------------------------------------------------
def _iou_one_to_many(box: np.ndarray, boxes: np.ndarray, areas: np.ndarray) -> np.ndarray:
    inter_w = np.minimum(boxes[:, 2], box[2]) - np.maximum(boxes[:, 0], box[0])
    inter_h = np.minimum(boxes[:, 3], box[3]) - np.maximum(boxes[:, 1], box[1])
    inter = np.clip(inter_w, 0, None) * np.clip(inter_h, 0, None)
    return inter / ((box[2] - box[0]) * (box[3] - box[1]) + areas - inter)


def _areas(boxes: np.ndarray) -> np.ndarray:
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def _wbf_single_class(boxes: np.ndarray, wscores: np.ndarray, models: np.ndarray, weights: np.ndarray,
                      iou_thr: float, conf_type: str) -> tuple[np.ndarray, np.ndarray]:
    """WBF for one class; ``wscores`` are ``score * weight[model]``. Returns fused boxes and scores."""
    n = len(boxes)
    fused = np.empty((n, 4))
    fused_areas = np.empty(n)
    coord_sum = np.empty((n, 4))
    conf_sum, conf_max, weight_sum = np.empty(n), np.empty(n), np.empty(n)
    count = np.empty(n, dtype=np.int64)
    present = np.zeros((n, len(weights)), dtype=bool)
    k = 0
    for i in np.argsort(-wscores, kind="stable"):
        box, s, m = boxes[i], wscores[i], models[i]
        j = -1
        if k:
            ious = _iou_one_to_many(box, fused[:k], fused_areas[:k])
            best = int(np.argmax(ious))
            if ious[best] > iou_thr:
                j = best
        if j < 0:
            j, k = k, k + 1
            fused[j], coord_sum[j] = box, s * box
            conf_sum[j] = conf_max[j] = s
            weight_sum[j], count[j] = weights[m], 1
        else:
            coord_sum[j] += s * box
            conf_sum[j] += s
            conf_max[j] = max(conf_max[j], s)
            weight_sum[j] += weights[m]
            count[j] += 1
            if conf_sum[j] > 0:
                fused[j] = coord_sum[j] / conf_sum[j]
        present[j, m] = True
        fused_areas[j] = (fused[j, 2] - fused[j, 0]) * (fused[j, 3] - fused[j, 1])

    conf_sum, conf_max, weight_sum, count, present = conf_sum[:k], conf_max[:k], weight_sum[:k], count[:k], present[:k]
    total_weight = weights.sum()
    if conf_type == "avg":
        scores = conf_sum / count * np.minimum(len(weights), count) / total_weight
    elif conf_type == "max":
        scores = conf_max / weights.max()
    elif conf_type == "box_and_model_avg":
        scores = conf_sum / weight_sum * (present @ weights) / total_weight
    else:  # absent_model_aware_avg
        scores = conf_sum / (weight_sum + (~present) @ weights)
    return fused[:k], scores


def _nms_single_class(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> tuple[np.ndarray, np.ndarray]:
    areas = _areas(boxes)
    order = np.argsort(-scores, kind="stable")
    keep = []
    while order.size:
        i, rest = order[0], order[1:]
        keep.append(i)
        order = rest[_iou_one_to_many(boxes[i], boxes[rest], areas[rest]) <= iou_thr]
    return boxes[keep], scores[keep]


def _soft_nms_single_class(boxes: np.ndarray, scores: np.ndarray, sigma: float,
                           min_score: float) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian soft-NMS (Bodla et al. 2017): decay overlapping scores by ``exp(-iou^2 / sigma)``."""
    areas, scores = _areas(boxes), scores.astype(np.float64, copy=True)
    remaining = np.arange(len(boxes))
    keep, kept_scores = [], []
    while remaining.size:
        top = int(np.argmax(scores[remaining]))
        i = remaining[top]
        keep.append(i)
        kept_scores.append(scores[i])
        remaining = np.delete(remaining, top)
        if not remaining.size:
            break
        ious = _iou_one_to_many(boxes[i], boxes[remaining], areas[remaining])
        scores[remaining] *= np.exp(-(ious ** 2) / sigma)
        remaining = remaining[scores[remaining] >= min_score]
    return boxes[keep], np.asarray(kept_scores)


# ----------------------------------------------------------------------------------------------------------------
# per-image fusion
# ----------------------------------------------------------------------------------------------------------------
def _fuse_flat(boxes: np.ndarray, scores: np.ndarray, labels: np.ndarray, models: np.ndarray,
               weights: np.ndarray, cfg: FusionConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fuse one image given flat arrays with a per-box model index. Output sorted by score, capped."""
    boxes = np.concatenate([np.minimum(boxes[:, :2], boxes[:, 2:]), np.maximum(boxes[:, :2], boxes[:, 2:])], axis=1)
    valid = (scores >= cfg.skip_box_thr) & (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    boxes, scores, labels, models = boxes[valid], scores[valid], labels[valid], models[valid]

    out_boxes, out_scores, out_labels = [], [], []
    for label in np.unique(labels):
        sel = labels == label
        if cfg.method == "wbf":
            b, s = _wbf_single_class(boxes[sel], scores[sel] * weights[models[sel]], models[sel], weights,
                                     cfg.iou_thr, cfg.conf_type)
        else:
            s = scores[sel] * weights[models[sel]] / weights.max()
            if cfg.method == "nms":
                b, s = _nms_single_class(boxes[sel], s, cfg.iou_thr)
            else:
                b, s = _soft_nms_single_class(boxes[sel], s, cfg.sigma, cfg.skip_box_thr)
        out_boxes.append(b)
        out_scores.append(s)
        out_labels.append(np.full(len(s), label, dtype=np.int64))
    if not out_boxes:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=np.int64)
    boxes, scores, labels = np.concatenate(out_boxes), np.concatenate(out_scores), np.concatenate(out_labels)
    order = np.argsort(-scores, kind="stable")
    if cfg.max_per_image > 0:
        order = order[:cfg.max_per_image]
    return boxes[order], scores[order], labels[order]


def _check_weights(weights: Optional[Sequence[float]], n_models: int) -> np.ndarray:
    weights = np.ones(n_models) if weights is None else np.asarray(weights, dtype=np.float64)
    if weights.shape != (n_models,) or not (weights > 0).all():
        raise ValueError(f"need {n_models} positive weights, got {weights.tolist()}")
    return weights


def fuse(boxes_list: Sequence[np.ndarray], scores_list: Sequence[np.ndarray], labels_list: Sequence[np.ndarray],
         weights: Optional[Sequence[float]] = None, config: FusionConfig = FusionConfig()
         ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fuse the detections of several models (or TTA views) on ONE image.

    ``boxes_list[m]`` is an ``(n_m, 4)`` array of absolute ``x1 y1 x2 y2``; scores/labels are ``(n_m,)``.
    Same contract as ``ensemble_boxes.weighted_boxes_fusion`` (without coordinate normalization).
    Returns ``(boxes (k, 4), scores (k,), labels (k,))`` sorted by descending score.
    """
    weights = _check_weights(weights, len(boxes_list))
    boxes = np.concatenate([np.asarray(b, dtype=np.float64).reshape(-1, 4) for b in boxes_list])
    scores = np.concatenate([np.asarray(s, dtype=np.float64).reshape(-1) for s in scores_list])
    labels = np.concatenate([np.asarray(lb, dtype=np.int64).reshape(-1) for lb in labels_list])
    models = np.concatenate([np.full(len(np.asarray(s).reshape(-1)), m) for m, s in enumerate(scores_list)])
    if not len(boxes) == len(scores) == len(labels):
        raise ValueError("boxes, scores and labels must have matching lengths per model")
    return _fuse_flat(boxes, scores, labels, models.astype(np.int64), weights, config)


def weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=None, iou_thr: float = 0.6,
                          skip_box_thr: float = 0.001, conf_type: str = "avg"):
    """Drop-in style wrapper mirroring ``ensemble_boxes.weighted_boxes_fusion`` (absolute coordinates, no cap)."""
    config = FusionConfig("wbf", iou_thr, skip_box_thr, conf_type, max_per_image=0)
    return fuse(boxes_list, scores_list, labels_list, weights, config)


# ----------------------------------------------------------------------------------------------------------------
# COCO-results level
# ----------------------------------------------------------------------------------------------------------------
def _to_arrays(preds: Sequence[dict]) -> tuple[np.ndarray, np.ndarray]:
    """COCO results -> (image_ids int64 (n,), float64 (n, 6) = [label, score, x1, y1, x2, y2])."""
    image_ids = np.fromiter((p["image_id"] for p in preds), dtype=np.int64, count=len(preds))
    rows = np.array([(p["category_id"], p["score"], *p["bbox"]) for p in preds], dtype=np.float64).reshape(-1, 6)
    rows[:, 4:] += rows[:, 2:4]
    return image_ids, rows


def _fuse_chunk(task: tuple[list[tuple[int, np.ndarray]], np.ndarray, FusionConfig]) -> list[tuple]:
    groups, weights, cfg = task
    results = []
    for image_id, g in groups:  # g columns: model, label, score, x1, y1, x2, y2
        boxes, scores, labels = _fuse_flat(g[:, 3:], g[:, 2], g[:, 1].astype(np.int64), g[:, 0].astype(np.int64),
                                           weights, cfg)
        results.append((image_id, boxes, scores, labels))
    return results


def ensemble_predictions(pred_lists: Sequence[Sequence[dict]], weights: Optional[Sequence[float]] = None,
                         config: FusionConfig = FusionConfig(), workers: Optional[int] = None) -> list[dict]:
    """Fuse several COCO-results lists image by image (in parallel); returns a COCO-results list."""
    weights = _check_weights(weights, len(pred_lists))
    ids_rows = [_to_arrays(preds) for preds in pred_lists]
    image_ids = np.concatenate([ids for ids, _ in ids_rows])
    table = np.concatenate([np.column_stack([np.full(len(rows), m, dtype=np.float64), rows])
                            for m, (_, rows) in enumerate(ids_rows)])
    order = np.argsort(image_ids, kind="stable")
    image_ids, table = image_ids[order], table[order]
    unique_ids, starts = np.unique(image_ids, return_index=True)
    groups = [(int(i), g) for i, g in zip(unique_ids, np.split(table, starts[1:]))] if len(table) else []

    workers = workers if workers is not None else min(32, os.cpu_count() or 1)
    if workers <= 1 or len(groups) < SEQUENTIAL_BELOW_IMAGES:
        fused = _fuse_chunk((groups, weights, config))
    else:
        n_chunks = min(len(groups), workers * 4)
        tasks = [(groups[c::n_chunks], weights, config) for c in range(n_chunks)]
        with mp.Pool(workers) as pool:
            fused = [r for chunk in pool.imap_unordered(_fuse_chunk, tasks) for r in chunk]
        fused.sort(key=lambda r: r[0])

    return [{"image_id": image_id, "category_id": int(label),
             "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)], "score": float(score)}
            for image_id, boxes, scores, labels in fused
            for (x1, y1, x2, y2), score, label in zip(boxes, scores, labels)]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preds", type=Path, nargs="+", required=True, help="COCO results json files to fuse")
    parser.add_argument("--weights", type=float, nargs="+", default=None, help="one positive weight per file")
    parser.add_argument("--out", type=Path, required=True, help="output COCO results json")
    parser.add_argument("--method", choices=METHODS, default="wbf")
    parser.add_argument("--iou-thr", type=float, default=0.6, help="cluster (WBF) / suppression (NMS) IoU")
    parser.add_argument("--skip-box-thr", type=float, default=0.001, help="ignore input boxes below this score")
    parser.add_argument("--conf-type", choices=CONF_TYPES, default="avg", help="WBF fused-score rule")
    parser.add_argument("--max-per-image", type=int, default=100, help="keep top-k per image (<= 0: all)")
    parser.add_argument("--sigma", type=float, default=0.5, help="soft-NMS Gaussian sigma")
    parser.add_argument("--workers", type=int, default=None, help="processes (default min(32, cpus))")
    parser.add_argument("--gt", type=Path, default=None, help="optional GT json: print AP of inputs and output")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = FusionConfig(args.method, args.iou_thr, args.skip_box_thr, args.conf_type, args.max_per_image,
                          args.sigma)
    pred_lists = [load_predictions(p) for p in args.preds]
    start = time.perf_counter()
    fused = ensemble_predictions(pred_lists, args.weights, config, args.workers)
    elapsed = time.perf_counter() - start
    dump_json(fused, args.out)
    print(f"{config.method}: fused {sum(map(len, pred_lists))} boxes from {len(pred_lists)} files into "
          f"{len(fused)} boxes on {len({p['image_id'] for p in fused})} images in {elapsed:.1f}s -> {args.out}")

    if args.gt is not None:
        from tools.coco_eval import evaluate, load_gt

        coco_gt = load_gt(args.gt, verbose=False)
        for path, preds in [*zip(args.preds, pred_lists), (args.out, fused)]:
            m = evaluate(coco_gt, preds, verbose=False)
            print(f"  AP={m['AP']:.4f} AP50={m['AP50']:.4f} AP75={m['AP75']:.4f}  {path}")


if __name__ == "__main__":
    main()
