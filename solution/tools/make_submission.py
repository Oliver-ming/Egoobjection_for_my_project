#!/usr/bin/env python3
"""Build a Kaggle submission CSV from COCO-results test predictions, then validate it.

Build steps: drop predictions on unknown images / invalid categories / non-finite values, clip boxes to the image,
round (bbox 2 decimals, score 5 decimals by default), drop boxes with w or h <= 0 after rounding, keep the top
``--max-per-image`` by score, write a temporary filtered json and convert it with the OFFICIAL
``convert_to_submission()`` imported from ``resource/convert_to_submission.py`` (so the official formatting is used).

Validation (also available standalone with ``--validate-only``): columns exactly ``id,predictions``; one row per
test image (ids == test.json ids, no duplicates); every cell is a JSON list whose entries have
``image_id == row id``, ``category_id`` int in 0..9, ``bbox`` of 4 finite numbers with w, h > 0, ``score`` in [0, 1],
and at most ``--max-per-image`` entries. Exit code 1 on any validation error.

Usage::

    python tools/make_submission.py --pred work_dirs/exp/test_preds.json --out work_dirs/exp/submission.csv
    python tools/make_submission.py --validate-only work_dirs/exp/submission.csv
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.common import CATEGORY_IDS, CLASS_NAMES, RESOURCE_DIR, SPLIT_JSON, dump_json, load_json, load_predictions

MAX_REPORTED_ERRORS = 20
BOUNDS_TOL = 0.01  # px; bbox rounding may push a clipped edge this far


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def official_convert_to_submission():
    """Import ``convert_to_submission`` from the official script (its directory is added to ``sys.path``).

    Bytecode writing is disabled for the import so no ``__pycache__`` appears inside ``resource/``.
    """
    if str(RESOURCE_DIR) not in sys.path:
        sys.path.insert(0, str(RESOURCE_DIR))
    dont_write_bytecode, sys.dont_write_bytecode = sys.dont_write_bytecode, True
    try:
        from convert_to_submission import convert_to_submission
    finally:
        sys.dont_write_bytecode = dont_write_bytecode
    return convert_to_submission


def filter_predictions(preds: list[dict], image_sizes: dict[int, tuple[float, float]], max_per_image: int = 100,
                       bbox_decimals: int = 2, score_decimals: int = 5) -> tuple[list[dict], Counter]:
    """Clean predictions for submission; returns ``(kept predictions, counts of what was dropped / changed)``."""
    stats = Counter(input=len(preds))
    per_image: dict[int, list[tuple[float, dict]]] = defaultdict(list)
    for p in preds:
        image_id, category_id, bbox, score = p.get("image_id"), p.get("category_id"), p.get("bbox"), p.get("score")
        if image_id not in image_sizes:
            stats["dropped_unknown_image"] += 1
            continue
        if not _is_int(category_id) or category_id not in CATEGORY_IDS:
            stats["dropped_bad_category"] += 1
            continue
        if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4 and all(map(_is_number, bbox))
                and _is_number(score)):
            stats["dropped_non_finite"] += 1
            continue
        width, height = image_sizes[image_id]
        x, y, w, h = bbox
        corners = [min(max(v, 0.0), limit) for v, limit in ((x, width), (y, height), (x + w, width), (y + h, height))]
        stats["clipped"] += corners != [x, y, x + w, y + h]
        x1, y1, x2, y2 = (round(v, bbox_decimals) for v in corners)
        w, h = round(x2 - x1, bbox_decimals), round(y2 - y1, bbox_decimals)
        if w <= 0 or h <= 0:
            stats["dropped_empty_box"] += 1
            continue
        per_image[image_id].append((score, {"image_id": image_id, "category_id": category_id,
                                            "bbox": [x1, y1, w, h], "score": round(score, score_decimals)}))

    kept = []
    for image_id, items in per_image.items():
        items.sort(key=lambda item: -item[0])
        if max_per_image > 0 and len(items) > max_per_image:
            stats["dropped_over_max_per_image"] += len(items) - max_per_image
            items = items[:max_per_image]
        kept.extend(p for _, p in items)
    stats["kept"] = len(kept)
    return kept, stats


def build_submission(pred_path: Path, out_csv: Path, test_json: Path, max_per_image: int = 100,
                     bbox_decimals: int = 2, score_decimals: int = 5, keep_filtered: Path | None = None) -> Counter:
    test = load_json(test_json)
    image_sizes = {im["id"]: (im["width"], im["height"]) for im in test["images"]}
    kept, stats = filter_predictions(load_predictions(pred_path), image_sizes, max_per_image, bbox_decimals,
                                     score_decimals)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if keep_filtered is None:
        fd, tmp = tempfile.mkstemp(dir=out_csv.parent, prefix=f".{out_csv.stem}.", suffix=".filtered.json")
        os.close(fd)
    filtered_path = dump_json(kept, keep_filtered if keep_filtered is not None else tmp)
    try:
        official_convert_to_submission()(str(filtered_path), str(out_csv), image_ids=None, test_json=str(test_json))
    finally:
        if keep_filtered is None:
            filtered_path.unlink(missing_ok=True)
    return stats


@dataclass
class ValidationReport:
    errors: Counter = field(default_factory=Counter)  # error kind -> occurrences
    examples: list[str] = field(default_factory=list)
    warnings: Counter = field(default_factory=Counter)
    rows: int = 0
    predictions: int = 0
    empty_rows: int = 0
    max_per_row: int = 0
    per_class: Counter = field(default_factory=Counter)
    score_range: tuple[float, float] = (math.inf, -math.inf)

    @property
    def ok(self) -> bool:
        return not self.errors

    def error(self, kind: str, detail: str) -> None:
        self.errors[kind] += 1
        if len(self.examples) < MAX_REPORTED_ERRORS:
            self.examples.append(f"{kind}: {detail}")

    def summary(self) -> str:
        lines = [f"rows={self.rows} predictions={self.predictions} empty_rows={self.empty_rows} "
                 f"max_per_row={self.max_per_row} score_range=[{self.score_range[0]:.5f}, {self.score_range[1]:.5f}]"
                 if self.predictions else f"rows={self.rows} predictions=0",
                 "per class: " + ", ".join(f"{CLASS_NAMES[c]}={self.per_class[c]}" for c in CATEGORY_IDS)]
        lines += [f"WARNING {kind}: {n}" for kind, n in self.warnings.items()]
        if self.ok:
            lines.append("VALIDATION PASSED")
        else:
            lines.append("VALIDATION FAILED: " + ", ".join(f"{kind}={n}" for kind, n in self.errors.items()))
            lines += [f"  - {example}" for example in self.examples]
        return "\n".join(lines)


def _validate_entry(entry, row_id: int, image_size: tuple[float, float], report: ValidationReport) -> None:
    where = f"row id {row_id}"
    if not isinstance(entry, dict) or not {"image_id", "category_id", "bbox", "score"} <= entry.keys():
        report.error("bad_entry", f"{where}: {str(entry)[:80]}")
        return
    if not _is_int(entry["image_id"]) or entry["image_id"] != row_id:
        report.error("image_id_mismatch", f"{where}: image_id {entry['image_id']!r}")
    category_id = entry["category_id"]
    if not _is_int(category_id) or category_id not in CATEGORY_IDS:
        report.error("bad_category_id", f"{where}: {category_id!r}")
    else:
        report.per_class[category_id] += 1
    bbox = entry["bbox"]
    if not (isinstance(bbox, list) and len(bbox) == 4 and all(map(_is_number, bbox))):
        report.error("bad_bbox", f"{where}: {bbox!r}")
    elif bbox[2] <= 0 or bbox[3] <= 0:
        report.error("non_positive_wh", f"{where}: {bbox!r}")
    else:
        x, y, w, h = bbox
        width, height = image_size
        if x < -BOUNDS_TOL or y < -BOUNDS_TOL or x + w > width + BOUNDS_TOL or y + h > height + BOUNDS_TOL:
            report.warnings["bbox_outside_image"] += 1
    score = entry["score"]
    if not _is_number(score) or not 0.0 <= score <= 1.0:
        report.error("bad_score", f"{where}: {score!r}")
    else:
        report.score_range = (min(report.score_range[0], score), max(report.score_range[1], score))


def validate_submission(csv_path: Path, test_json: Path, max_per_image: int = 100) -> ValidationReport:
    report = ValidationReport()
    test = load_json(test_json)
    image_sizes = {im["id"]: (im["width"], im["height"]) for im in test["images"]}
    try:
        df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    except Exception as exc:
        report.error("unreadable_csv", f"{type(exc).__name__}: {exc}")
        return report
    if list(df.columns) != ["id", "predictions"]:
        report.error("bad_columns", f"{list(df.columns)}")
        return report
    report.rows = len(df)
    if report.rows != len(image_sizes):
        report.error("row_count", f"{report.rows} rows, expected {len(image_sizes)}")

    seen_ids = Counter()
    for raw_id, cell in zip(df["id"], df["predictions"]):
        if not raw_id.strip().isdigit():
            report.error("bad_id", repr(raw_id))
            continue
        row_id = int(raw_id)
        seen_ids[row_id] += 1
        if row_id not in image_sizes:
            report.error("unknown_id", str(row_id))
            continue
        try:
            entries = json.loads(cell)
        except json.JSONDecodeError as exc:
            report.error("bad_json", f"row id {row_id}: {exc}")
            continue
        if not isinstance(entries, list):
            report.error("not_a_list", f"row id {row_id}: {type(entries).__name__}")
            continue
        if max_per_image > 0 and len(entries) > max_per_image:
            report.error("too_many_predictions", f"row id {row_id}: {len(entries)} > {max_per_image}")
        report.predictions += len(entries)
        report.empty_rows += not entries
        report.max_per_row = max(report.max_per_row, len(entries))
        for entry in entries:
            _validate_entry(entry, row_id, image_sizes[row_id], report)

    duplicates = [i for i, n in seen_ids.items() if n > 1]
    if duplicates:
        report.error("duplicate_id", f"{len(duplicates)} ids, e.g. {duplicates[0]}")
    missing = set(image_sizes) - set(seen_ids)
    if missing:
        report.error("missing_id", f"{len(missing)} test ids have no row, e.g. {next(iter(missing))}")
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--pred", type=Path, help="COCO results json on the test images")
    mode.add_argument("--validate-only", type=Path, metavar="CSV", help="only validate an existing submission CSV")
    parser.add_argument("--out", type=Path, help="submission CSV to write (required with --pred)")
    parser.add_argument("--test", type=Path, default=SPLIT_JSON["test"], help="test.json with the image list")
    parser.add_argument("--max-per-image", type=int, default=100, help="top-k per image (<= 0: no cap)")
    parser.add_argument("--bbox-decimals", type=int, default=2)
    parser.add_argument("--score-decimals", type=int, default=5)
    parser.add_argument("--keep-filtered", type=Path, default=None,
                        help="keep the filtered COCO json passed to the official converter at this path")
    args = parser.parse_args(argv)
    if args.pred is not None and args.out is None:
        parser.error("--out is required with --pred")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    csv_path = args.validate_only
    if args.pred is not None:
        stats = build_submission(args.pred, args.out, args.test, args.max_per_image, args.bbox_decimals,
                                 args.score_decimals, args.keep_filtered)
        print("\nfiltering: " + ", ".join(f"{k}={v}" for k, v in sorted(stats.items())))
        csv_path = args.out
    report = validate_submission(csv_path, args.test, args.max_per_image)
    print(f"\nvalidation of {csv_path}:\n{report.summary()}")
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
