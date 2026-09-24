#!/usr/bin/env python
"""Predict a full split with an Ultralytics model and write a flat COCO results json (original-image xywh).

  CUDA_VISIBLE_DEVICES=0 python ultra/predict_coco.py --weights work_dirs/ultra/<run>/weights/best.pt \
      --split val --eval [--imgsz 1024] [--augment] [--half] [--nms false]

The image list comes from resource/data/<split>.json, so every official image is predicted exactly once.
YOLO26: `--nms default|true` = one-to-many head + NMS (the Ultralytics 8.4.160 default), `--nms false` = NMS-free
one-to-one head. Either way the output is the top `--max-det` boxes per image by score.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ego_ultra as E  # noqa: E402  (must precede other ultralytics imports)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from ultralytics.models.yolo.detect import DetectionPredictor  # noqa: E402
from ultralytics.utils import nms, ops  # noqa: E402


class MultiLabelPredictor(DetectionPredictor):
    """DetectionPredictor with multi-label NMS (a box may be kept for several classes), as in mAP-style validation."""

    def postprocess(self, preds, img, orig_imgs, **kwargs):
        preds = nms.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            self.args.classes,
            self.args.agnostic_nms,
            multi_label=True,
            max_det=self.args.max_det,
            end2end=getattr(self.model, "end2end", False),
        )
        if not isinstance(orig_imgs, list):
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)[..., ::-1]
        return self.construct_results(preds, img, orig_imgs)


def load_split(split: str, gt: str | None):
    gt = Path(gt) if gt else E.DATA_ROOT / f"{split}.json"
    d = json.loads(gt.read_text())
    images = []
    for im in d["images"]:
        path = E.DATA_ROOT / "images" / im["file_name"]
        stem_id = int(path.stem)
        if stem_id != im["id"]:
            raise ValueError(f"file stem {path.stem} != image_id {im['id']}")
        if not path.exists():
            raise FileNotFoundError(path)
        images.append(dict(id=im["id"], path=str(path), w=im["width"], h=im["height"]))
    return gt, images, bool(d.get("annotations"))


def coco_eval(gt_path: Path, dets: list, img_ids: list[int] | None = None) -> dict:
    """pycocotools bbox eval, same calls as resource/eval.py (optionally restricted to img_ids)."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    coco_gt = COCO(str(gt_path))
    if not dets:
        print("[eval] no detections -> AP = 0")
        return dict(AP=0.0, AP50=0.0, AP75=0.0)
    coco_dt = coco_gt.loadRes(dets)
    ev = COCOeval(coco_gt, coco_dt, iouType="bbox")
    if img_ids is not None:
        ev.params.imgIds = img_ids
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    names = ["AP", "AP50", "AP75", "APs", "APm", "APl", "AR1", "AR10", "AR100", "ARs", "ARm", "ARl"]
    out = {k: round(float(v), 5) for k, v in zip(names, ev.stats)}
    prec = ev.eval["precision"]  # [T, R, K, A, M]
    per_class = {}
    for k, cid in enumerate(ev.params.catIds):
        p = prec[:, :, k, 0, 2]
        p = p[p > -1]
        per_class[coco_gt.cats[cid]["name"]] = round(float(p.mean()), 4) if p.size else float("nan")
    out["per_class"] = per_class
    print("[eval] per-class AP:", json.dumps(per_class))
    print(f"[eval] AP={out['AP']:.4f}  AP50={out['AP50']:.4f}  AP75={out['AP75']:.4f}")
    return out


def batches_by_shape(images: list[dict], bs: int):
    """Chunks of indices with identical (h, w) so the predictor can use minimal (rect) letterboxing."""
    order = sorted(range(len(images)), key=lambda i: (images[i]["h"], images[i]["w"], images[i]["id"]))
    chunk = []
    for i in order:
        if chunk and ((images[chunk[0]]["h"], images[chunk[0]]["w"]) != (images[i]["h"], images[i]["w"]) or len(chunk) == bs):
            yield chunk
            chunk = []
        chunk.append(i)
    if chunk:
        yield chunk


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--split", choices=["val", "test", "train"], default="val")
    ap.add_argument("--gt", default=None, help="COCO json listing the images (default resource/data/<split>.json)")
    ap.add_argument("--imgsz", type=int, default=None, help="default: imgsz the checkpoint was trained at")
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.7, help="NMS IoU (ignored by NMS-free heads)")
    ap.add_argument("--max-det", type=int, default=100)
    ap.add_argument("--augment", action="store_true", help="Ultralytics TTA (flip + 3 scales; YOLO NMS heads only)")
    ap.add_argument("--half", action="store_true", help="FP16 inference")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--nms", choices=["default", "true", "false"], default="default", help="YOLO26 head selection")
    ap.add_argument("--multi-label", action="store_true", help="multi-label NMS (NMS models only)")
    ap.add_argument("--no-rect", action="store_true", help="square letterbox instead of minimal padding")
    ap.add_argument("--device", default="0")
    ap.add_argument("--workers", type=int, default=8, help="image decode threads")
    ap.add_argument("--limit", type=int, default=0, help="only the first N images of the split (debug)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--eval", action="store_true")
    args = ap.parse_args()
    E.require_cuda_visible_devices()

    gt_path, images, has_ann = load_split(args.split, args.gt)
    if args.limit:
        images = images[: args.limit]
    model = E.load_model(args.weights)
    rtdetr = E.is_rtdetr(model)
    imgsz = args.imgsz or int((getattr(model, "ckpt", None) or {}).get("train_args", {}).get("imgsz", 1024))
    nms_arg = {"default": None, "true": True, "false": False}[args.nms]
    if args.augment and rtdetr:
        print("[warn] RT-DETR ignores augment=True in Ultralytics (no TTA path); running single-scale.")
        args.augment = False
    if args.augment and nms_arg is False:
        print("[warn] YOLO26 end-to-end head does not support TTA; running single-scale.")
        args.augment = False
    if args.multi_label and rtdetr:
        print("[warn] --multi-label has no effect for RT-DETR (its top-k is already over queries x classes).")

    weights = Path(args.weights)
    run = weights.parent.parent.name if weights.parent.name == "weights" else weights.stem
    tag = f"{run}_{args.split}_{imgsz}" + ("_tta" if args.augment else "") + (f"_nms{args.nms}" if args.nms != "default" else "")
    tag += ("_ml" if args.multi_label else "") + ("_fp16" if args.half else "")
    out = Path(args.out) if args.out else E.WORK / "preds" / f"{tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    pred_kw = dict(
        imgsz=imgsz, conf=args.conf, iou=args.iou, max_det=args.max_det, augment=args.augment, device=args.device,
        verbose=False, rect=not args.no_rect, nms=nms_arg, quantize=16 if args.half else None,
    )
    predictor = MultiLabelPredictor if (args.multi_label and not rtdetr) else None
    print(f"[predict] {type(model).__name__} {weights} split={args.split} n={len(images)} imgsz={imgsz} -> {out}")

    dets, n_with, shape_mismatch = [], 0, 0
    speed = np.zeros(3)
    chunks = list(batches_by_shape(images, args.batch))
    pool = ThreadPoolExecutor(args.workers)
    inflight = deque()
    t0 = time.time()
    ci = 0
    while ci < len(chunks) or inflight:
        while ci < len(chunks) and len(inflight) < 3:  # bounded prefetch: at most 3 batches of decoded images
            idx = chunks[ci]
            inflight.append((idx, [pool.submit(cv2.imread, images[i]["path"]) for i in idx]))
            ci += 1
        idx, futs = inflight.popleft()
        ims = [f.result() for f in futs]
        results = model.predict(ims, batch=len(ims), predictor=predictor, **pred_kw)
        for i, im, r in zip(idx, ims, results):
            meta = images[i]
            h, w = im.shape[:2]
            if (h, w) != (meta["h"], meta["w"]):
                shape_mismatch += 1
            b = r.boxes
            if b is None or len(b) == 0:
                continue
            xyxy = b.xyxy.float().cpu().numpy()
            xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clip(0, w)
            xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clip(0, h)
            conf = b.conf.float().cpu().numpy()
            cls = b.cls.cpu().numpy().astype(int)
            keep = np.argsort(-conf, kind="stable")[: args.max_det]
            n_with += 1
            for k in keep:
                x1, y1, x2, y2 = xyxy[k]
                dets.append(
                    dict(
                        image_id=int(meta["id"]),
                        category_id=int(cls[k]),
                        bbox=[round(float(x1), 2), round(float(y1), 2), round(float(x2 - x1), 2), round(float(y2 - y1), 2)],
                        score=round(float(conf[k]), 5),
                    )
                )
        if results:
            speed += np.array([results[0].speed[k] for k in ("preprocess", "inference", "postprocess")]) * len(results)
    torch.cuda.synchronize()
    wall = time.time() - t0
    pool.shutdown()

    out.write_text(json.dumps(dets))
    n = len(images)
    ids = {d["image_id"] for d in dets}
    assert ids <= {im["id"] for im in images}, "predicted image_id outside the split"
    summary = dict(
        weights=str(weights), model=type(model).__name__, split=args.split, n_images=n, imgsz=imgsz,
        augment=args.augment, half=args.half, nms=args.nms, multi_label=args.multi_label, rect=not args.no_rect,
        n_dets=len(dets), images_with_dets=n_with, dets_per_img=round(len(dets) / max(n, 1), 1),
        shape_mismatch=shape_mismatch, wall_s=round(wall, 1), ms_per_img_wall=round(1000 * wall / max(n, 1), 1),
        ms_per_img_pre_inf_post=[round(x, 2) for x in speed / max(n, 1)],
    )
    print("[predict] " + json.dumps(summary))
    if args.eval:
        if not has_ann:
            print(f"[eval] {gt_path} has no annotations, skipping eval")
        else:
            summary["metrics"] = coco_eval(gt_path, dets, [im["id"] for im in images] if args.limit else None)
    out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
