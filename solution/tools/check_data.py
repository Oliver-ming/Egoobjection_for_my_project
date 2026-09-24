#!/usr/bin/env python3
"""Data sanity report for the competition data, written as markdown (default ``solution/docs/data_check.md``).

Checks:

a. **Image files / EXIF** -- for every train/val/test image: EXIF orientation tag and raw stored size read with
   PIL (no transpose), plus the shape ``cv2.imread`` actually returns (OpenCV/mmcv/Ultralytics auto-rotate by
   EXIF). Any disagreement with json ``width``/``height`` is reported.
b. **YOLO labels vs COCO** -- every ``labels/<split>/<stem>.txt`` is compared with the COCO boxes converted to
   normalized ``cx cy w h`` after clipping to the image (box count, classes, coordinates within ``--tol``), and
   the treatment of boxes that extend outside the image (clipped or not) is determined.
c. **Box statistics** -- degenerate / out-of-image boxes, per-class counts, COCO small/medium/large split (by the
   ``area`` field, which is what COCOeval uses), aspect ratios, boxes per image, image-size histogram.

Example::

    python tools/check_data.py                    # full check, ~1 min on many cores
    python tools/check_data.py --no-decode        # header-only EXIF check (faster)
"""
from __future__ import annotations

import argparse
import datetime
import multiprocessing as mp
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.common import DATA_ROOT, DOCS_DIR, load_json

SPLITS = ("train", "val", "test")
LABELED_SPLITS = ("train", "val")
EXIF_ORIENTATION_TAG = 0x0112
TRANSPOSING_ORIENTATIONS = {5, 6, 7, 8}  # these swap width and height when applied
LABEL_PRECISION_TOL = 2e-6  # labels are written with 6 decimals
# ultralytics.data.utils.verify_image_label (8.4.x) drops the whole image as "corrupt" outside this range.
ULTRALYTICS_COORD_RANGE = (-0.01, 1.01)
COCO_AREA_BUCKETS = (("small", 0.0, 32.0 ** 2), ("medium", 32.0 ** 2, 96.0 ** 2), ("large", 96.0 ** 2, np.inf))


# ----------------------------------------------------------------------------------------------------------------
# a. image files / EXIF
# ----------------------------------------------------------------------------------------------------------------
def _init_worker() -> None:
    import cv2

    cv2.setNumThreads(1)


def probe_image(task: tuple[int, str, bool]) -> dict:
    """Read raw size + EXIF orientation with PIL and, if requested, the shape returned by ``cv2.imread``."""
    import cv2
    from PIL import Image

    image_id, path, decode = task
    result = {"id": image_id, "raw_wh": None, "orientation": None, "cv2_wh": None, "error": None}
    try:
        with Image.open(path) as im:
            result["raw_wh"] = im.size
            result["orientation"] = im.getexif().get(EXIF_ORIENTATION_TAG)
        if decode:
            array = cv2.imread(path, cv2.IMREAD_COLOR)
            if array is None:
                result["error"] = "cv2.imread returned None"
            else:
                result["cv2_wh"] = (array.shape[1], array.shape[0])
    except Exception as exc:  # corrupted / unreadable file: report instead of aborting the whole check
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def check_images(datasets: dict[str, dict], image_root: Path, decode: bool, workers: int) -> dict[str, dict]:
    tasks = [(im["id"], str(image_root / im["file_name"]), decode) for ds in datasets.values() for im in ds["images"]]
    with mp.Pool(workers, initializer=_init_worker) as pool:
        probes = {p["id"]: p for p in pool.imap_unordered(probe_image, tasks, chunksize=32)}

    report = {}
    for split, ds in datasets.items():
        r = {"images": len(ds["images"]), "orientation": Counter(), "oriented": [], "raw_mismatch": [],
             "cv2_mismatch": [], "errors": []}
        for im in ds["images"]:
            probe, json_wh = probes[im["id"]], (im["width"], im["height"])
            if probe["error"]:
                r["errors"].append((im["id"], probe["error"]))
            if probe["raw_wh"] is None:
                continue
            orientation = probe["orientation"] or 1
            r["orientation"][orientation] += 1
            raw_wh = tuple(probe["raw_wh"])
            rotated_wh = raw_wh[::-1] if orientation in TRANSPOSING_ORIENTATIONS else raw_wh
            if orientation != 1:
                r["oriented"].append({"id": im["id"], "orientation": orientation, "json_wh": json_wh,
                                      "raw_wh": raw_wh, "cv2_wh": probe["cv2_wh"],
                                      "json_is_raw": json_wh == raw_wh, "json_is_rotated": json_wh == rotated_wh})
            if raw_wh != json_wh:
                r["raw_mismatch"].append((im["id"], raw_wh, json_wh))
            if probe["cv2_wh"] is not None and tuple(probe["cv2_wh"]) != json_wh:
                r["cv2_mismatch"].append((im["id"], tuple(probe["cv2_wh"]), json_wh))
        r["decoded"] = decode
        report[split] = r
    return report


# ----------------------------------------------------------------------------------------------------------------
# b. YOLO labels vs COCO
# ----------------------------------------------------------------------------------------------------------------
def coco_to_yolo(bbox: list[float], width: float, height: float, clip: bool) -> np.ndarray:
    x1, y1, w, h = bbox
    x2, y2 = x1 + w, y1 + h
    if clip:
        x1, x2 = np.clip([x1, x2], 0, width)
        y1, y2 = np.clip([y1, y2], 0, height)
    return np.array([(x1 + x2) / 2 / width, (y1 + y2) / 2 / height, (x2 - x1) / width, (y2 - y1) / height])


def read_yolo_label(path: Path) -> np.ndarray:
    """Return an ``(n, 5)`` array ``[cls, cx, cy, w, h]``; raises on malformed lines."""
    rows = [line.split() for line in path.read_text().splitlines() if line.strip()]
    if any(len(row) != 5 for row in rows):
        raise ValueError(f"{path}: expected 5 columns per line")
    return np.array(rows, dtype=np.float64).reshape(-1, 5)


def greedy_match(gt: np.ndarray, labels: np.ndarray) -> list[tuple[int, int, float]]:
    """Match rows of two ``[cls, cx, cy, w, h]`` arrays: same class, smallest L-inf coordinate distance first."""
    if len(gt) == 0 or len(labels) == 0:
        return []
    dist = np.abs(gt[:, None, 1:] - labels[None, :, 1:]).max(axis=2)
    dist[gt[:, None, 0] != labels[None, :, 0]] = np.inf
    matches = []
    for flat in np.argsort(dist, axis=None, kind="stable"):
        i, j = np.unravel_index(flat, dist.shape)
        if not np.isfinite(dist[i, j]):
            break
        if all(i != mi and j != mj for mi, mj, _ in matches):
            matches.append((int(i), int(j), float(dist[i, j])))
    return matches


LABEL_COUNT_KEYS = ("images", "missing_label_file", "extra_label_files", "label_boxes", "matched_boxes",
                    "count_mismatch", "class_mismatch", "coord_mismatch", "same_order_as_json",
                    "label_values_out_of_range", "label_zero_size", "ultralytics_rejects")


def check_yolo_labels(ds: dict, label_dir: Path, tol: float) -> dict:
    """Per-split label comparison: the ``LABEL_COUNT_KEYS`` counts plus ``max_abs_err`` (over matched boxes) and
    ``boundary`` (how boxes crossing the image border are encoded: clipped / unclipped / ...)."""
    anns_by_image = defaultdict(list)
    for ann in ds["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)

    r = Counter(dict.fromkeys(LABEL_COUNT_KEYS, 0))
    boundary = Counter()
    max_err = 0.0
    label_stems = {p.stem for p in label_dir.glob("*.txt")}
    json_stems = {Path(im["file_name"]).stem for im in ds["images"]}
    r["extra_label_files"] = len(label_stems - json_stems)
    for im in ds["images"]:
        width, height = im["width"], im["height"]
        path = label_dir / f"{Path(im['file_name']).stem}.txt"
        if not path.exists():
            r["missing_label_file"] += 1
            continue
        labels = read_yolo_label(path)
        anns = anns_by_image[im["id"]]
        gt = np.array([[a["category_id"], *coco_to_yolo(a["bbox"], width, height, clip=True)] for a in anns])
        gt = gt.reshape(-1, 5)
        r["images"] += 1
        r["label_boxes"] += len(labels)
        r["label_values_out_of_range"] += int(((labels[:, 1:] < 0) | (labels[:, 1:] > 1)).sum())
        r["label_zero_size"] += int(((labels[:, 3] <= 0) | (labels[:, 4] <= 0)).sum())
        lo, hi = ULTRALYTICS_COORD_RANGE
        r["ultralytics_rejects"] += int(len(labels) > 0 and (labels[:, 1:].min() < lo or labels[:, 1:].max() > hi))
        if len(labels) != len(gt):
            r["count_mismatch"] += 1
        if sorted(labels[:, 0].tolist()) != sorted(gt[:, 0].tolist()):
            r["class_mismatch"] += 1
        if len(labels) == len(gt) and np.array_equal(labels[:, 0], gt[:, 0]) and \
                np.abs(labels[:, 1:] - gt[:, 1:]).max(initial=0.0) <= tol:
            r["same_order_as_json"] += 1

        matches = greedy_match(gt, labels)
        if any(err > tol for _, _, err in matches):
            r["coord_mismatch"] += 1
        r["matched_boxes"] += len(matches)
        max_err = max([max_err] + [err for _, _, err in matches])
        for i, j, _ in matches:
            x, y, w, h = anns[i]["bbox"]
            if x >= 0 and y >= 0 and x + w <= width and y + h <= height:
                continue
            raw = coco_to_yolo(anns[i]["bbox"], width, height, clip=False)
            err_clip = np.abs(labels[j, 1:] - gt[i, 1:]).max()
            err_raw = np.abs(labels[j, 1:] - raw).max()
            clip_ok, raw_ok = err_clip <= LABEL_PRECISION_TOL, err_raw <= LABEL_PRECISION_TOL
            key = {(True, False): "clipped", (False, True): "unclipped", (True, True): "indistinguishable",
                   (False, False): "neither"}[(bool(clip_ok), bool(raw_ok))]
            boundary[key] += 1
    return {**r, "max_abs_err": max_err, "boundary": boundary}


# ----------------------------------------------------------------------------------------------------------------
# c. box statistics
# ----------------------------------------------------------------------------------------------------------------
def box_stats(ds: dict, class_names: dict[int, str]) -> dict:
    images = {im["id"]: im for im in ds["images"]}
    anns = ds["annotations"]
    s = {"image_sizes": Counter((im["width"], im["height"]) for im in ds["images"])}
    if not anns:
        return s
    xywh = np.array([a["bbox"] for a in anns], dtype=np.float64)
    area = np.array([a["area"] for a in anns], dtype=np.float64)
    cls = np.array([a["category_id"] for a in anns])
    img_w = np.array([images[a["image_id"]]["width"] for a in anns], dtype=np.float64)
    img_h = np.array([images[a["image_id"]]["height"] for a in anns], dtype=np.float64)
    x1, y1, w, h = xywh.T
    x2, y2 = x1 + w, y1 + h

    s["num_boxes"] = len(anns)
    s["unknown_category"] = int((~np.isin(cls, list(class_names))).sum())
    s["unknown_image"] = sum(a["image_id"] not in images for a in anns)
    s["area_neq_wh"] = int((np.abs(area - w * h) > 1e-3 * np.maximum(1.0, area)).sum())
    s["iscrowd"] = sum(bool(a.get("iscrowd", 0)) for a in anns)
    s["degenerate"] = [anns[i] for i in np.flatnonzero((w <= 1) | (h <= 1))]
    s["duplicates"] = len(anns) - len({(a["image_id"], a["category_id"], tuple(a["bbox"])) for a in anns})

    overshoot = {"left": -x1, "top": -y1, "right": x2 - img_w, "bottom": y2 - img_h}
    s["outside"] = {side: (int((v > 0).sum()), float(v.max(initial=0.0))) for side, v in overshoot.items()}
    worst = np.max(np.stack(list(overshoot.values())), axis=0)
    s["outside_any"] = int((worst > 0).sum())
    s["outside_over_1px"] = int((worst > 1).sum())
    s["fully_outside"] = [anns[i] for i in np.flatnonzero((x1 >= img_w) | (y1 >= img_h) | (x2 <= 0) | (y2 <= 0))]

    buckets = np.select([(area >= lo) & (area <= hi) for _, lo, hi in COCO_AREA_BUCKETS], range(3), default=-1)
    aspect = w / np.maximum(h, 1e-6)
    s["per_class"] = {
        c: {"count": int((cls == c).sum()),
            "buckets": [int(((cls == c) & (buckets == b)).sum()) for b in range(3)],
            "median_aspect": float(np.median(aspect[cls == c])) if (cls == c).any() else float("nan"),
            "median_rel_size": float(np.median(np.sqrt(area / (img_w * img_h))[cls == c])) if (cls == c).any()
            else float("nan")}
        for c in class_names
    }
    s["buckets"] = [int((buckets == b).sum()) for b in range(3)]
    s["aspect_q"] = np.percentile(aspect, [1, 5, 25, 50, 75, 95, 99])
    s["rel_size_q"] = np.percentile(np.sqrt(area / (img_w * img_h)), [1, 5, 25, 50, 75, 95, 99])
    per_image = np.array(list(Counter(a["image_id"] for a in anns).values()))
    s["boxes_per_image"] = (float(per_image.mean()), int(np.median(per_image)), int(per_image.max()))
    s["images_without_boxes"] = len(images) - len(per_image)
    return s


# ----------------------------------------------------------------------------------------------------------------
# markdown rendering
# ----------------------------------------------------------------------------------------------------------------
def _table(header: list[str], rows: list[list]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in rows]
    return lines


def _pct(n: int, total: int) -> str:
    return f"{n} ({100.0 * n / total:.1f}%)" if total else str(n)


def yolo_labels_safe(labels: dict[str, dict], images: dict[str, dict]) -> tuple[bool, list[str]]:
    reasons = []
    for split, r in labels.items():
        for key in ("missing_label_file", "count_mismatch", "class_mismatch", "coord_mismatch",
                    "label_values_out_of_range", "label_zero_size"):
            if r[key]:
                reasons.append(f"{split}: {key}={r[key]}")
    for split, r in images.items():
        if split in labels and r["oriented"]:
            reasons.append(f"{split}: {len(r['oriented'])} images are re-oriented by EXIF-aware loaders")
        if split in labels and (r["raw_mismatch"] or r["cv2_mismatch"]):
            reasons.append(f"{split}: image size disagrees with json")
    return not reasons, reasons


def render_report(images: dict, labels: dict, stats: dict, class_names: dict[int, str], tol: float,
                  label_root: Path) -> str:
    safe, reasons = yolo_labels_safe(labels, images)
    n_oriented = sum(len(r["oriented"]) for r in images.values())
    n_raw_mismatch = sum(len(r["raw_mismatch"]) for r in images.values())
    n_cv2_mismatch = sum(len(r["cv2_mismatch"]) for r in images.values())
    decoded = all(r["decoded"] for r in images.values())
    boundary = sum((r["boundary"] for r in labels.values()), Counter())

    out = ["# Data check", "",
           f"Generated by `tools/check_data.py` on {datetime.date.today().isoformat()}.", "", "## Verdict", ""]
    if safe:
        out.append(f"- **YOLO labels (`{label_root}`) are SAFE to use as-is**: every train/val label file "
                   f"matches the COCO boxes clipped to the image (same count and classes; max |error| "
                   f"{max(r['max_abs_err'] for r in labels.values()):.1e} normalized, tolerance {tol:g}).")
    else:
        notes = []
        if not any(r["count_mismatch"] or r["class_mismatch"] for r in labels.values()) and \
                boundary["unclipped"] and not boundary["clipped"] and not boundary["neither"]:
            notes.append("Box counts and classes match COCO exactly, but boxes crossing the image border are NOT "
                         "clipped (they equal the raw COCO boxes normalized), so coordinates fall outside [0, 1].")
        if any(r["ultralytics_rejects"] for r in labels.values()):
            rejects = ", ".join(f"{s} {r['ultralytics_rejects']}" for s, r in labels.items())
            notes.append(f"Ultralytics would silently drop whole images as corrupt ({rejects} images); regenerate "
                         "clipped labels with `tools/prepare_data.py --yolo-dir ...`.")
        out.append(f"- **YOLO labels (`{label_root}`) are NOT safe to use as-is.** " + " ".join(notes)
                   + " Failed checks: " + "; ".join(reasons) + ".")
    out.append(f"- **EXIF orientation**: {n_oriented} of {sum(r['images'] for r in images.values())} images carry "
               f"an orientation tag != 1; {n_raw_mismatch} images have a raw size different from json; "
               + (f"{n_cv2_mismatch} images decode with `cv2.imread` to a shape different from json."
                  if decoded else "cv2 decode check skipped (--no-decode).")
               + (" EXIF auto-rotation is therefore a non-issue for this data." if n_oriented == 0 and
                  n_raw_mismatch == 0 and n_cv2_mismatch == 0 else " See section 1 before training."))
    out.append(f"- Boxes extending outside the image (train+val): "
               f"{sum(s['outside_any'] for s in stats.values() if 'outside_any' in s)}; the YOLO labels store them "
               f"clipped={boundary['clipped']}, unclipped={boundary['unclipped']}, "
               f"indistinguishable (<{LABEL_PRECISION_TOL:g})={boundary['indistinguishable']}, "
               f"neither={boundary['neither']}. The COCO json boxes are unclipped too; clip them for training "
               f"(mmdet's CocoDataset already skips boxes with no in-image area), but score against the official "
               f"GT as-is.")
    degenerate = [a for s in stats.values() for a in s.get("degenerate", [])]
    out += [f"- Degenerate GT boxes (w or h <= 1 px): {len(degenerate)}.", ""]

    out += ["## 1. Image files and EXIF orientation", ""]
    out += _table(["split", "images", "orientation != 1", "raw size != json", "cv2.imread shape != json",
                   "unreadable", "orientation histogram"],
                  [[s, r["images"], len(r["oriented"]), len(r["raw_mismatch"]),
                    len(r["cv2_mismatch"]) if r["decoded"] else "n/a", len(r["errors"]),
                    ", ".join(f"{k}: {v}" for k, v in sorted(r["orientation"].items()))]
                   for s, r in images.items()])
    oriented = [o for r in images.values() for o in r["oriented"]]
    if oriented:
        n_raw = sum(o["json_is_raw"] for o in oriented)
        n_rot = sum(o["json_is_rotated"] and not o["json_is_raw"] for o in oriented)
        out += ["", f"Of the {len(oriented)} tagged images, json (width, height) equals the raw size for {n_raw} and "
                    f"the EXIF-rotated size (only) for {n_rot}. First examples:", ""]
        out += _table(["id", "orientation", "json wh", "raw wh", "cv2 wh"],
                      [[o["id"], o["orientation"], o["json_wh"], o["raw_wh"], o["cv2_wh"]] for o in oriented[:10]])
    for r in images.values():
        for kind in ("raw_mismatch", "cv2_mismatch", "errors"):
            for item in r[kind][:10]:
                out.append(f"- {kind}: {item}")
    out.append("")

    out += ["## 2. YOLO labels vs COCO (boxes clipped to the image, then normalized cx cy w h)", ""]
    out += _table(["split", "label files", "missing", "extra", "boxes (label / matched)", "count mismatch",
                   "class mismatch", f"coord mismatch (>{tol:g})", "max abs err", "same line order as json"],
                  [[s, r["images"], r["missing_label_file"], r["extra_label_files"],
                    f"{r['label_boxes']} / {r['matched_boxes']}", r["count_mismatch"], r["class_mismatch"],
                    r["coord_mismatch"], f"{r['max_abs_err']:.1e}", _pct(r["same_order_as_json"], r["images"])]
                   for s, r in labels.items()])
    out += ["", f"Label values outside [0, 1]: {sum(r['label_values_out_of_range'] for r in labels.values())}; "
                f"label boxes with w or h <= 0: {sum(r['label_zero_size'] for r in labels.values())}. "
                f"Images Ultralytics would drop as corrupt (a coordinate outside {list(ULTRALYTICS_COORD_RANGE)}): "
                + ", ".join(f"{s} {r['ultralytics_rejects']}" for s, r in labels.items()) + ".", ""]

    out += ["## 3. Box statistics", ""]
    for split in LABELED_SPLITS:
        s = stats[split]
        out.append(f"- **{split}**: {s['num_boxes']} boxes in {sum(s['image_sizes'].values())} images "
                   f"(boxes/image mean {s['boxes_per_image'][0]:.2f}, median {s['boxes_per_image'][1]}, "
                   f"max {s['boxes_per_image'][2]}; images without boxes {s['images_without_boxes']}); "
                   f"iscrowd {s['iscrowd']}; unknown category {s['unknown_category']}; unknown image "
                   f"{s['unknown_image']}; area != w*h {s['area_neq_wh']}; exact duplicate boxes {s['duplicates']}.")
        out.append(f"  - outside image: any side {_pct(s['outside_any'], s['num_boxes'])}, by more than 1 px "
                   f"{s['outside_over_1px']}, fully outside {len(s['fully_outside'])}; per side (count, max px): "
                   + ", ".join(f"{k} ({n}, {m:.2f})" for k, (n, m) in s["outside"].items()) + ".")
        small, medium, large = s["buckets"]
        out.append(f"  - COCO area buckets: small {_pct(small, s['num_boxes'])}, medium {_pct(medium, s['num_boxes'])}"
                   f", large {_pct(large, s['num_boxes'])}.")
        out.append("  - aspect ratio w/h percentiles [1, 5, 25, 50, 75, 95, 99]: "
                   + ", ".join(f"{v:.2f}" for v in s["aspect_q"]) + ".")
        out.append("  - relative size sqrt(area / image area) percentiles [1, 5, 25, 50, 75, 95, 99]: "
                   + ", ".join(f"{v:.3f}" for v in s["rel_size_q"]) + ".")
    for a in degenerate:
        out.append(f"- degenerate box: ann id {a['id']}, image {a['image_id']}, class {class_names[a['category_id']]}"
                   f", bbox {a['bbox']}, area {a['area']}.")
    fully_outside = Counter((split, a["category_id"], tuple(a["bbox"]))
                            for split in LABELED_SPLITS for a in stats[split]["fully_outside"])
    for (split, cat, bbox), n in sorted(fully_outside.items()):
        out.append(f"- box fully outside its image ({split}): class {class_names[cat]}, bbox {list(bbox)}, "
                   f"in {n} images. Unmatchable by in-image predictions (caps recall of val slightly).")
    out += ["", "Per class (train / val counts; train small / medium / large; train medians):", ""]
    out += _table(["id", "class", "train", "val", "train S / M / L", "median w/h", "median rel. size"],
                  [[c, name, stats["train"]["per_class"][c]["count"], stats["val"]["per_class"][c]["count"],
                    " / ".join(map(str, stats["train"]["per_class"][c]["buckets"])),
                    f"{stats['train']['per_class'][c]['median_aspect']:.2f}",
                    f"{stats['train']['per_class'][c]['median_rel_size']:.3f}"]
                   for c, name in class_names.items()])
    out += ["", "Image sizes (width x height):", ""]
    all_sizes = sorted(set().union(*(s["image_sizes"] for s in stats.values())),
                       key=lambda wh: -sum(s["image_sizes"][wh] for s in stats.values()))
    out += _table(["size"] + list(stats), [[f"{w}x{h}"] + [stats[sp]["image_sizes"][(w, h)] for sp in stats]
                                            for w, h in all_sizes])
    return "\n".join(out) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT,
                        help="directory with {train,val,test}.json, images/ and labels/")
    parser.add_argument("--label-root", type=Path, default=None,
                        help="YOLO label root with {train,val}/ (default: <data-root>/labels)")
    parser.add_argument("--out", type=Path, default=DOCS_DIR / "data_check.md", help="markdown report path")
    parser.add_argument("--tol", type=float, default=1e-3, help="normalized coordinate tolerance for YOLO labels")
    parser.add_argument("--workers", type=int, default=min(64, os.cpu_count() or 1), help="image probe processes")
    parser.add_argument("--no-decode", action="store_true", help="skip decoding every image with cv2.imread")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    datasets = {split: load_json(args.data_root / f"{split}.json") for split in SPLITS}
    class_names = {c["id"]: c["name"] for c in datasets["train"]["categories"]}

    images = check_images(datasets, args.data_root / "images", decode=not args.no_decode, workers=args.workers)
    label_root = args.label_root or args.data_root / "labels"
    labels = {s: check_yolo_labels(datasets[s], label_root / s, args.tol) for s in LABELED_SPLITS}
    stats = {split: box_stats(ds, class_names) for split, ds in datasets.items()}

    report = render_report(images, labels, stats, class_names, args.tol, label_root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")
    print(report)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
