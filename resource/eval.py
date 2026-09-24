#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate COCO-style detections using pycocotools.

Usage:
  python evaluate_coco.py --gt path/to/gt.json --pred path/to/pred.json
"""

import argparse
import json
from collections import defaultdict

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def eval_one_ioutype(coco_gt: COCO, coco_dt_json: str, iou_type: str, per_class: bool = False):
    """Evaluate one IoU type: 'bbox' | 'segm' | 'keypoints'."""
    # Load results (predictions must use the *same* category_id & image_id space as GT)
    coco_dt = coco_gt.loadRes(coco_dt_json)

    coco_eval = COCOeval(coco_gt, coco_dt, iouType=iou_type)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()   # prints the standard COCO summary

    if per_class and iou_type in {"bbox", "segm"}:
        per_category_ap(coco_eval, coco_gt)
    elif per_class and iou_type == "keypoints":
        print("[Info] Per-class AP for keypoints is not implemented in this helper.")


def per_category_ap(coco_eval: COCOeval, coco_gt: COCO):
    """
    Print per-category AP at IoU=0.50:0.95 (COCO main metric).
    Adapted from common practice: averages over IoU thresholds, area=all, maxDets=100.
    """
    precisions = coco_eval.eval["precision"]  # [TxRxKxAxM]
    if precisions is None:
        print("[Warn] precisions is None, skip per-class AP.")
        return

    # dimensions
    T, R, K, A, M = precisions.shape
    # area index: 0 = all; maxDets index: 2 -> 100 (following COCO convention)
    area_ind = 0
    maxdet_ind = 2

    cat_ids = coco_gt.getCatIds()
    cats = coco_gt.loadCats(cat_ids)
    catid_to_name = {c["id"]: c["name"] for c in cats}

    print("\nPer-category AP (IoU=0.50:0.95, area=all, maxDets=100):")
    lines = []
    for k, cat_id in enumerate(cat_ids):
        # precision for category k, all IoU & recall, area all, maxDets=100
        p = precisions[:, :, k, area_ind, maxdet_ind]
        p = p[p > -1]  # filter out invalid entries -1
        ap = p.mean() if p.size else float("nan")
        lines.append((catid_to_name.get(cat_id, str(cat_id)), ap))

    # Sort by AP desc
    lines.sort(key=lambda x: (0.0 if (x[1] != x[1]) else -x[1]))  # NaN last
    for name, ap in lines:
        print(f"{name:>20s}: {ap:0.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", required=True, help="Path to GT COCO json (with images/annotations/categories).")
    parser.add_argument("--pred", required=True, help="Path to predictions COCO json (list of detections).")
    parser.add_argument("--types", nargs="+", default=["bbox"], choices=["bbox", "segm", "keypoints"],
                        help="IoU types to evaluate.")
    parser.add_argument("--per-class", action="store_true", help="Print per-class AP (bbox/segm).")
    args = parser.parse_args()

    # Load GT once
    coco_gt = COCO(args.gt)

    # Quick sanity: if prediction categories don’t match GT, warn
    try:
        with open(args.pred, "r") as f:
            preds = json.load(f)
        if isinstance(preds, list) and preds and "category_id" in preds[0]:
            gt_cat_ids = set(coco_gt.getCatIds())
            dt_cat_ids = {int(det.get("category_id", -1)) for det in preds}
            if not dt_cat_ids.issubset(gt_cat_ids):
                missing = dt_cat_ids - gt_cat_ids
                if missing:
                    print(f"[Warn] Some prediction category_id(s) not in GT categories: {sorted(missing)}")
    except Exception:
        pass

    for iou_type in args.types:
        print("\n" + "=" * 80)
        print(f"Evaluating IoU type: {iou_type}")
        eval_one_ioutype(coco_gt, args.pred, iou_type, per_class=args.per_class)


if __name__ == "__main__":
    main()
