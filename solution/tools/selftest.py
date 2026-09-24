#!/usr/bin/env python3
"""End-to-end tests of the shared tooling (CPU only). Artifacts go to ``work_dirs/_tooling_tests/``.

Tests:

* ``eval_gt``        -- GT-as-prediction on val gives AP == 1.
* ``eval_official``  -- on a jittered prediction file, ``tools/coco_eval.py`` reproduces ``resource/eval.py``: identical
                        printed summary / per-class lines (subprocess) and bit-identical ``COCOeval.stats`` (in-process).
* ``wbf_reference``  -- WBF output equals a literal transcription of the reference ``ensemble_boxes`` algorithm
                        for every ``conf_type`` on random clustered inputs with random weights.
* ``ens_self``       -- a file fused with itself (wbf, weights 1 1) keeps its AP.
* ``ens_noisy``      -- 3 noisy copies of val GT: WBF AP >= mean single-file AP (NMS / soft-NMS reported too).
* ``ens_timing``     -- 2000 images x 5 models x 100 boxes through the CLI, must finish well under a minute.
* ``sub_roundtrip``  -- synthetic test predictions (incl. images without predictions) -> CSV validates.
* ``sub_corrupt``    -- the validator rejects corrupted CSVs (and ``--validate-only`` exits 1).

Usage::

    CUDA_VISIBLE_DEVICES="" python tools/selftest.py                 # all tests
    CUDA_VISIBLE_DEVICES="" python tools/selftest.py --only ens_noisy ens_timing
"""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import os
import re
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pycocotools.coco import COCO

from tools.coco_eval import STAT_NAMES, evaluate, load_gt
from tools.common import RESOURCE_DIR, SOLUTION_ROOT, SPLIT_JSON, WORK_DIRS, dump_json, load_json
from tools.ensemble import CONF_TYPES, FusionConfig, ensemble_predictions, weighted_boxes_fusion
from tools.make_submission import validate_submission

TOOLS = Path(__file__).resolve().parent
ENV = dict(os.environ, CUDA_VISIBLE_DEVICES="")


def run(cmd: list, check: bool = True) -> subprocess.CompletedProcess:
    proc = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, env=ENV, cwd=SOLUTION_ROOT)
    if check and proc.returncode != 0:
        raise AssertionError(f"command failed ({proc.returncode}): {' '.join(map(str, cmd))}\n{proc.stderr[-2000:]}")
    return proc


# ----------------------------------------------------------------------------------------------------------------
# synthetic data
# ----------------------------------------------------------------------------------------------------------------
def gt_as_predictions(gt: dict) -> list[dict]:
    return [{"image_id": a["image_id"], "category_id": a["category_id"], "bbox": list(a["bbox"]), "score": 1.0}
            for a in gt["annotations"]]


def noisy_predictions(gt: dict, seed: int, jitter: float = 0.06, drop: float = 0.1, duplicate: float = 0.1,
                      flip: float = 0.03, false_per_image: float = 1.5) -> list[dict]:
    """Perturbed GT: jittered boxes, random scores, dropped / duplicated / mislabeled boxes, random false boxes."""
    rng = np.random.default_rng(seed)
    sizes = {im["id"]: (im["width"], im["height"]) for im in gt["images"]}
    preds = []

    def jittered(bbox, sigma):
        x, y, w, h = bbox
        dx, dy, dw, dh = rng.normal(0.0, sigma, 4) * [w, h, w, h]
        return [x + dx, y + dy, max(w + dw, 1.0), max(h + dh, 1.0)]

    for ann in gt["annotations"]:
        if rng.random() < drop:
            continue
        category = ann["category_id"] if rng.random() > flip else int(rng.integers(10))
        score = float(rng.uniform(0.3, 1.0))
        preds.append({"image_id": ann["image_id"], "category_id": category,
                      "bbox": jittered(ann["bbox"], jitter), "score": score})
        if rng.random() < duplicate:
            preds.append({"image_id": ann["image_id"], "category_id": category,
                          "bbox": jittered(ann["bbox"], 2 * jitter), "score": score * float(rng.uniform(0.3, 0.9))})
    for image_id, (width, height) in sizes.items():
        for _ in range(rng.poisson(false_per_image)):
            w, h = rng.uniform(0.05, 0.5) * width, rng.uniform(0.05, 0.5) * height
            preds.append({"image_id": image_id, "category_id": int(rng.integers(10)),
                          "bbox": [float(rng.uniform(0, width - w)), float(rng.uniform(0, height - h)), w, h],
                          "score": float(rng.uniform(0.0, 0.7))})
    return preds


def synthetic_model_outputs(n_images: int, n_models: int, boxes_per_model: int, seed: int) -> list[list[dict]]:
    """Per image 30 shared objects; each model emits jittered copies of them plus near-duplicates / clutter."""
    rng = np.random.default_rng(seed)
    outputs = [[] for _ in range(n_models)]
    for image_id in range(1, n_images + 1):
        n_obj = 30
        wh = rng.uniform(20, 600, (n_obj, 2))
        xy = rng.uniform(0, 1, (n_obj, 2)) * (np.array([1920, 1080]) - wh)
        cls = rng.integers(0, 10, n_obj)
        for m in range(n_models):
            src = np.concatenate([np.arange(n_obj), rng.integers(0, n_obj, boxes_per_model - n_obj)])
            noise = rng.normal(0, 0.05, (boxes_per_model, 4)) * np.tile(wh[src], 2)
            noise[n_obj:] *= 4
            boxes = np.concatenate([xy[src], wh[src]], axis=1) + noise
            boxes[:, 2:] = np.maximum(boxes[:, 2:], 2.0)
            scores = rng.uniform(0.001, 1.0, boxes_per_model)
            outputs[m].extend({"image_id": image_id, "category_id": int(c), "bbox": b.tolist(), "score": float(s)}
                              for b, s, c in zip(boxes, scores, cls[src]))
    return outputs


# ----------------------------------------------------------------------------------------------------------------
# reference WBF (literal transcription of ensemble_boxes.weighted_boxes_fusion, float64) -- test oracle only
# ----------------------------------------------------------------------------------------------------------------
def _ref_weighted_box(boxes: list, conf_type: str) -> np.ndarray:
    box = np.zeros(8)
    conf, conf_list, w = 0.0, [], 0.0
    for b in boxes:
        box[4:] += b[1] * b[4:]
        conf += b[1]
        conf_list.append(b[1])
        w += b[2]
    box[0] = boxes[0][0]
    box[1] = np.max(conf_list) if conf_type == "max" else conf / len(boxes)
    box[2], box[3] = w, -1
    box[4:] /= conf
    return box


def _ref_find_match(weighted: np.ndarray, new_box: np.ndarray, iou_thr: float) -> int:
    if weighted.shape[0] == 0:
        return -1
    b = weighted[:, 4:]
    xa, ya = np.maximum(b[:, 0], new_box[4]), np.maximum(b[:, 1], new_box[5])
    xb, yb = np.minimum(b[:, 2], new_box[6]), np.minimum(b[:, 3], new_box[7])
    inter = np.maximum(xb - xa, 0) * np.maximum(yb - ya, 0)
    union = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) + (new_box[6] - new_box[4]) * (new_box[7] - new_box[5]) - inter
    ious = inter / union
    ious[weighted[:, 0] != new_box[0]] = -1
    best = int(np.argmax(ious))
    return -1 if ious[best] <= iou_thr else best


def reference_wbf(boxes_list, scores_list, labels_list, weights, iou_thr, skip_box_thr, conf_type):
    weights = np.asarray(weights, dtype=np.float64)
    filtered = defaultdict(list)
    for t, (boxes, scores, labels) in enumerate(zip(boxes_list, scores_list, labels_list)):
        for box, score, label in zip(boxes, scores, labels):
            if score >= skip_box_thr:
                filtered[int(label)].append([int(label), score * weights[t], weights[t], t, *box])
    overall = []
    for rows in filtered.values():
        rows = np.array(rows)
        rows = rows[rows[:, 1].argsort()[::-1]]
        clusters, weighted = [], np.empty((0, 8))
        for row in rows:
            index = _ref_find_match(weighted, row, iou_thr)
            if index != -1:
                clusters[index].append(row)
                weighted[index] = _ref_weighted_box(clusters[index], conf_type)
            else:
                clusters.append([row.copy()])
                weighted = np.vstack((weighted, row.copy()))
        for i, cluster in enumerate(clusters):
            cluster = np.array(cluster)
            if conf_type == "box_and_model_avg":
                weighted[i, 1] = weighted[i, 1] * len(cluster) / weighted[i, 2]
                _, idx = np.unique(cluster[:, 3], return_index=True)
                weighted[i, 1] = weighted[i, 1] * cluster[idx, 2].sum() / weights.sum()
            elif conf_type == "absent_model_aware_avg":
                mask = np.ones(len(weights), dtype=bool)
                mask[np.unique(cluster[:, 3]).astype(int)] = False
                weighted[i, 1] = weighted[i, 1] * len(cluster) / (weighted[i, 2] + weights[mask].sum())
            elif conf_type == "max":
                weighted[i, 1] = weighted[i, 1] / weights.max()
            else:
                weighted[i, 1] = weighted[i, 1] * min(len(weights), len(cluster)) / weights.sum()
        overall.append(weighted)
    if not overall:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0)
    overall = np.concatenate(overall)
    overall = overall[overall[:, 1].argsort()[::-1]]
    return overall[:, 4:], overall[:, 1], overall[:, 0]


# ----------------------------------------------------------------------------------------------------------------
# tests (each returns a one-line result, raises AssertionError on failure)
# ----------------------------------------------------------------------------------------------------------------
def test_eval_gt(ctx: dict) -> str:
    metrics = evaluate(ctx["coco_val"], gt_as_predictions(ctx["val"]), verbose=False)
    assert metrics["AP"] > 0.9999 and metrics["AR100"] > 0.9999, metrics
    return f"GT-as-pred on val: AP={metrics['AP']:.6f} AP50={metrics['AP50']:.6f} AR100={metrics['AR100']:.6f}"


def _summary_lines(stdout: str) -> list[str]:
    return [line.strip() for line in stdout.splitlines() if line.strip().startswith("Average ")]


def _per_class_values(stdout: str) -> dict[str, str]:
    return dict(re.findall(r"^\s*(\w+): (\d\.\d{4})", stdout, flags=re.M))


def test_eval_official(ctx: dict) -> str:
    pred_path = ctx["jitter_path"]
    official = run([sys.executable, RESOURCE_DIR / "eval.py", "--gt", SPLIT_JSON["val"], "--pred", pred_path,
                    "--per-class"]).stdout
    ours = run([sys.executable, TOOLS / "coco_eval.py", "--gt", SPLIT_JSON["val"], "--pred", pred_path,
                "--per-class", "--out", ctx["out"] / "jitter_metrics.json"]).stdout
    assert len(_summary_lines(official)) == 12 and _summary_lines(official) == _summary_lines(ours), \
        "printed COCO summaries differ"
    per_class = _per_class_values(official)
    assert len(per_class) == 10 and per_class == _per_class_values(ours), "printed per-class AP differ"

    spec = importlib.util.spec_from_file_location("official_eval", RESOURCE_DIR / "eval.py")
    module = importlib.util.module_from_spec(spec)
    dont_write_bytecode, sys.dont_write_bytecode = sys.dont_write_bytecode, True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = dont_write_bytecode
    captured = []

    class CapturingCOCOeval(module.COCOeval):
        def summarize(self):
            super().summarize()
            captured.append(self)

    module.COCOeval = CapturingCOCOeval
    with contextlib.redirect_stdout(io.StringIO()):
        module.eval_one_ioutype(COCO(str(SPLIT_JSON["val"])), str(pred_path), "bbox")
    metrics = evaluate(ctx["coco_val"], str(pred_path), verbose=False)
    ours_stats = np.array([metrics[name] for name in STAT_NAMES])
    assert np.array_equal(captured[0].stats, ours_stats), (captured[0].stats, ours_stats)
    return (f"jittered val preds: 12 summary lines + {len(per_class)} per-class lines identical, stats bit-identical "
            f"(AP={metrics['AP']:.4f} AP50={metrics['AP50']:.4f})")


def test_wbf_reference(ctx: dict) -> str:
    rng = np.random.default_rng(123)
    cases = 0
    for trial in range(60):
        n_models, n_obj = int(rng.integers(1, 5)), int(rng.integers(1, 12))
        centers, sizes = rng.uniform(0.2, 0.8, (n_obj, 2)), rng.uniform(0.05, 0.3, (n_obj, 2))
        classes = rng.integers(0, 3, n_obj)
        boxes_list, scores_list, labels_list = [], [], []
        for _ in range(n_models):
            src = rng.integers(0, n_obj, int(rng.integers(0, 3 * n_obj)))
            c = centers[src] + rng.normal(0, 0.03, (len(src), 2))
            s = sizes[src] * rng.uniform(0.8, 1.2, (len(src), 2))
            boxes_list.append(np.concatenate([c - s / 2, c + s / 2], axis=1))
            scores_list.append(rng.uniform(0.0, 1.0, len(src)))
            labels_list.append(classes[src])
        if sum(map(len, scores_list)) == 0:
            continue
        weights = rng.uniform(0.5, 3.0, n_models)
        iou_thr = float(rng.choice([0.4, 0.55, 0.7]))
        for conf_type in CONF_TYPES:
            ref = reference_wbf(boxes_list, scores_list, labels_list, weights, iou_thr, 0.05, conf_type)
            ours = weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights, iou_thr, 0.05, conf_type)
            assert len(ref[1]) == len(ours[1]), (trial, conf_type, len(ref[1]), len(ours[1]))
            for a, b in zip(ref, ours):
                assert np.allclose(a, b, rtol=1e-9, atol=1e-12), (trial, conf_type)
            cases += 1
    return f"{cases} random cases (4 conf_types, 1-4 weighted models) identical to the reference algorithm"


def test_ens_self(ctx: dict) -> str:
    preds = ctx["jitter"]
    base = evaluate(ctx["coco_val"], preds, verbose=False)["AP"]
    fused_path = ctx["out"] / "jitter_self_wbf.json"
    run([sys.executable, TOOLS / "ensemble.py", "--preds", ctx["jitter_path"], ctx["jitter_path"],
         "--weights", 1, 1, "--method", "wbf", "--out", fused_path])
    fused = evaluate(ctx["coco_val"], str(fused_path), verbose=False)["AP"]
    assert abs(fused - base) < 5e-3, (base, fused)
    return f"AP single={base:.4f} self-WBF={fused:.4f} (delta {fused - base:+.4f})"


def test_ens_noisy(ctx: dict) -> str:
    paths = []
    for seed in (1, 2, 3):
        path = ctx["out"] / f"noisy_{seed}.json"
        dump_json(noisy_predictions(ctx["val"], seed=seed), path)
        paths.append(path)
    singles = [evaluate(ctx["coco_val"], str(p), verbose=False)["AP"] for p in paths]
    results = {}
    for method in ("wbf", "nms", "soft-nms"):
        out = ctx["out"] / f"noisy_fused_{method}.json"
        run([sys.executable, TOOLS / "ensemble.py", "--preds", *paths, "--method", method, "--out", out])
        results[method] = evaluate(ctx["coco_val"], str(out), verbose=False)["AP"]
    for conf_type in CONF_TYPES[1:]:
        fused = ensemble_predictions([load_json(p) for p in paths], config=FusionConfig(conf_type=conf_type))
        results[f"wbf/{conf_type}"] = evaluate(ctx["coco_val"], fused, verbose=False)["AP"]
    assert results["wbf"] >= np.mean(singles), (singles, results)
    return (f"single APs {', '.join(f'{a:.4f}' for a in singles)} (mean {np.mean(singles):.4f}); fused: "
            + ", ".join(f"{k}={v:.4f}" for k, v in results.items()))


def test_ens_timing(ctx: dict) -> str:
    n_images, n_models, per_model = 2000, 5, 100
    timing_dir = ctx["out"] / "timing"
    paths = []
    for m, preds in enumerate(synthetic_model_outputs(n_images, n_models, per_model, seed=7)):
        paths.append(dump_json(preds, timing_dir / f"model_{m}.json"))
    report = {}
    for method in ("wbf", "soft-nms"):
        start = time.perf_counter()
        proc = run([sys.executable, TOOLS / "ensemble.py", "--preds", *paths, "--method", method,
                    "--out", timing_dir / f"fused_{method}.json"])
        report[method] = (time.perf_counter() - start, proc.stdout.strip().splitlines()[-1])
    fused = load_json(timing_dir / "fused_wbf.json")
    per_image = pd.Series([p["image_id"] for p in fused]).value_counts()
    assert per_image.max() <= 100 and len(per_image) == n_images
    assert all(t < 60 for t, _ in report.values()), report
    return (f"{n_images} images x {n_models} models x {per_model} boxes ({n_images * n_models * per_model} boxes): "
            + "; ".join(f"{m} CLI wall {t:.1f}s [{line.split(' in ')[-1].split(' ->')[0]} fusion]"
                        for m, (t, line) in report.items()))


def _synthetic_test_predictions(test: dict, seed: int) -> tuple[list[dict], set[int]]:
    """Random test predictions (some off-image, some images with > 100, a few invalid); also the ids left empty."""
    rng = np.random.default_rng(seed)
    images = test["images"]
    no_pred = {images[i]["id"] for i in rng.choice(len(images), 200, replace=False)}
    preds = []
    for im in images:
        if im["id"] in no_pred:
            continue
        width, height = im["width"], im["height"]
        for _ in range(int(rng.choice([rng.integers(1, 30), rng.integers(100, 160)], p=[0.9, 0.1]))):
            w, h = rng.uniform(1, 0.6 * width), rng.uniform(1, 0.6 * height)
            x, y = rng.uniform(-0.1 * width, width), rng.uniform(-0.1 * height, height)
            preds.append({"image_id": im["id"], "category_id": int(rng.integers(10)), "bbox": [x, y, w, h],
                          "score": float(rng.uniform(0, 1))})
    first = images[0]
    preds += [{"image_id": 123, "category_id": 0, "bbox": [0, 0, 10, 10], "score": 0.5},
              {"image_id": first["id"], "category_id": 12, "bbox": [0, 0, 10, 10], "score": 0.5},
              {"image_id": first["id"], "category_id": 1, "bbox": [first["width"] + 5, 0, 10, 10], "score": 0.9}]
    return preds, no_pred


def test_sub_roundtrip(ctx: dict) -> str:
    preds, no_pred = _synthetic_test_predictions(ctx["test"], seed=5)
    pred_path = dump_json(preds, ctx["out"] / "test_preds_synthetic.json")
    csv_path, filtered_path = ctx["out"] / "submission_synthetic.csv", ctx["out"] / "test_preds_filtered.json"
    proc = run([sys.executable, TOOLS / "make_submission.py", "--pred", pred_path, "--out", csv_path,
                "--keep-filtered", filtered_path])
    assert "VALIDATION PASSED" in proc.stdout, proc.stdout[-2000:]
    report = validate_submission(csv_path, SPLIT_JSON["test"])
    assert report.ok and report.rows == 2000 and report.max_per_row == 100 and not report.warnings, report

    expected = defaultdict(list)
    for p in load_json(filtered_path):
        expected[p["image_id"]].append(p)
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    rows = {int(i): json.loads(cell) for i, cell in zip(df["id"], df["predictions"])}
    assert all(rows[i] == expected.get(i, []) for i in rows), "CSV content differs from the filtered predictions"
    assert all(rows[i] == [] for i in no_pred) and report.empty_rows == 2000 - len(expected)
    assert [int(i) for i in df["id"]] == [im["id"] for im in ctx["test"]["images"]], "row order != test.json"
    filtering = next(line for line in proc.stdout.splitlines() if line.startswith("filtering:"))
    return (f"{report.rows} rows (test.json order), {report.predictions} preds, {report.empty_rows} empty rows "
            f"(incl. all {len(no_pred)} images without input preds); CSV == filtered json; {filtering}")


def test_sub_corrupt(ctx: dict) -> str:
    good = ctx["out"] / "submission_synthetic.csv"
    if not good.exists():
        test_sub_roundtrip(ctx)
    df = pd.read_csv(good, dtype=str, keep_default_na=False)
    row = int(np.flatnonzero(df["predictions"] != "[]")[0])

    def with_entry(mutate):
        entries = json.loads(df.at[row, "predictions"])
        mutate(entries[0])
        bad = df.copy()
        bad.at[row, "predictions"] = json.dumps(entries)
        return bad

    too_many = df.copy()
    too_many.at[row, "predictions"] = json.dumps(json.loads(df.at[row, "predictions"])[:1] * 101)
    bad_json = df.copy()
    bad_json.at[row, "predictions"] = "[{'image_id': 1"
    variants = {
        "row_count": df.iloc[:-1],
        "duplicate_id": pd.concat([df, df.iloc[:1]]),
        "bad_columns": df.rename(columns={"id": "ID"}),
        "bad_json": bad_json,
        "too_many_predictions": too_many,
        "image_id_mismatch": with_entry(lambda e: e.update(image_id=e["image_id"] + 1)),
        "bad_category_id": with_entry(lambda e: e.update(category_id=10)),
        "bad_score": with_entry(lambda e: e.update(score=1.5)),
        "non_positive_wh": with_entry(lambda e: e["bbox"].__setitem__(2, 0.0)),
        "bad_bbox": with_entry(lambda e: e["bbox"].__setitem__(0, float("nan"))),
    }
    caught = []
    for kind, bad in variants.items():
        path = ctx["out"] / "corrupt" / f"{kind}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        bad.to_csv(path, index=False)
        report = validate_submission(path, SPLIT_JSON["test"])
        assert not report.ok and kind in report.errors, (kind, report.summary())
        caught.append(kind)
    assert run([sys.executable, TOOLS / "make_submission.py", "--validate-only",
                ctx["out"] / "corrupt" / "bad_score.csv"], check=False).returncode == 1
    assert run([sys.executable, TOOLS / "make_submission.py", "--validate-only", good]).returncode == 0
    return f"all {len(caught)} corruptions rejected ({', '.join(caught)}); --validate-only exit codes 1 / 0"


TESTS = {"eval_gt": test_eval_gt, "eval_official": test_eval_official, "wbf_reference": test_wbf_reference,
         "ens_self": test_ens_self, "ens_noisy": test_ens_noisy, "ens_timing": test_ens_timing,
         "sub_roundtrip": test_sub_roundtrip, "sub_corrupt": test_sub_corrupt}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="+", choices=list(TESTS), default=list(TESTS))
    parser.add_argument("--out-dir", type=Path, default=WORK_DIRS / "_tooling_tests")
    args = parser.parse_args(argv)

    val = load_json(SPLIT_JSON["val"])
    jitter = noisy_predictions(val, seed=0)
    ctx = {"out": args.out_dir, "val": val, "coco_val": load_gt(SPLIT_JSON["val"], verbose=False),
           "test": load_json(SPLIT_JSON["test"]), "jitter": jitter,
           "jitter_path": dump_json(jitter, args.out_dir / "val_jitter.json")}

    results = {}
    for name in args.only:
        start = time.perf_counter()
        try:
            detail, ok = TESTS[name](ctx), True
        except Exception as exc:  # report every test, keep going
            detail, ok = f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=3)}", False
        results[name] = {"ok": ok, "seconds": round(time.perf_counter() - start, 1), "detail": detail}
        print(f"[{'PASS' if ok else 'FAIL'}] {name} ({results[name]['seconds']}s): {detail}", flush=True)
    dump_json(results, args.out_dir / "selftest_results.json", indent=2)
    return 0 if all(r["ok"] for r in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
