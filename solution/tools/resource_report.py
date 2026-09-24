#!/usr/bin/env python3
"""Summarise GPU / wall-time / progress for live or finished training runs.

Supports:
* MMDetection work dirs (``work_dirs/<name>/`` with mmengine ``<timestamp>/vis_data/scalars.json``)
* Ultralytics run dirs (``work_dirs/ultra/<name>/`` with ``results.csv``, ``args.yaml``, train logs)

Partial runs are fine: reports progress so far.

Usage::

    CUDA_VISIBLE_DEVICES="" env/bin/python tools/resource_report.py \\
      --mmdet codino_swinl_ego:3 --ultra y26x_1024_train:1 \\
      --out docs/resource_usage.json --md docs/resource_usage.md
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.common import SOLUTION_ROOT, WORK_DIRS, dump_json  # noqa: E402

LOGS = SOLUTION_ROOT / "logs"
TS_RE = re.compile(r"^(\d{4}/\d{2}/\d{2}\s+\d{2}:\d{2}:\d{2})")
ULTRA_EPOCH_RE = re.compile(
    r"^\s*(\d+)/(\d+)\s+([\d.]+G)\s+"
)  # Epoch GPU_mem ... from results is preferred
ULTRA_PROGRESS_RE = re.compile(
    r"^\s*(\d+)/(\d+)\s+([\d.]+)G\b"
)


def _parse_ts(line: str) -> Optional[datetime]:
    m = TS_RE.match(line.strip())
    if not m:
        return None
    return datetime.strptime(m.group(1), "%Y/%m/%d %H:%M:%S")


def _fmt_hours(seconds: Optional[float]) -> Optional[str]:
    if seconds is None:
        return None
    h = seconds / 3600.0
    if h < 1:
        return f"{seconds / 60.0:.1f} min"
    return f"{h:.2f} h"


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _find_mmdet_scalars(work_dir: Path) -> tuple[Optional[Path], Optional[Path], Optional[Path]]:
    """Return (scalars.json, run_log, timestamp_dir). Prefer the newest timestamp subdir."""
    candidates = sorted(
        [p for p in work_dir.iterdir() if p.is_dir() and (p / "vis_data" / "scalars.json").is_file()],
        key=lambda p: p.name,
    )
    if not candidates:
        # sometimes scalars live directly under work_dir/vis_data
        direct = work_dir / "vis_data" / "scalars.json"
        if direct.is_file():
            logs = list(work_dir.glob("*.log"))
            return direct, (logs[-1] if logs else None), work_dir
        return None, None, None
    ts = candidates[-1]
    scalars = ts / "vis_data" / "scalars.json"
    log = ts / f"{ts.name}.log"
    if not log.is_file():
        logs = list(ts.glob("*.log"))
        log = logs[-1] if logs else None
    return scalars, log, ts


def _mmdet_wall_from_log(log_path: Optional[Path], scalars_path: Path) -> tuple[Optional[float], Optional[str], Optional[str]]:
    """Wall seconds from first/last mmengine timestamps; fallback to scalars mtime - log mtime."""
    start = end = None
    if log_path and log_path.is_file():
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                ts = _parse_ts(line)
                if ts is not None:
                    start = ts
                    break
            # last timestamp: read from end
        with open(log_path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 65536))
            tail = f.read().decode("utf-8", errors="replace")
        for line in reversed(tail.splitlines()):
            ts = _parse_ts(line)
            if ts is not None:
                end = ts
                break
    if start is not None and end is not None and end >= start:
        return (end - start).total_seconds(), start.isoformat(sep=" "), end.isoformat(sep=" ")
    # fallback: file mtimes
    try:
        t0 = log_path.stat().st_mtime if log_path and log_path.is_file() else scalars_path.stat().st_mtime
        t1 = scalars_path.stat().st_mtime
        if t1 >= t0:
            return t1 - t0, datetime.fromtimestamp(t0).isoformat(sep=" "), datetime.fromtimestamp(t1).isoformat(sep=" ")
    except OSError:
        pass
    return None, None, None


def parse_mmdet_run(name: str, n_gpus: int, work_dir: Path) -> dict[str, Any]:
    scalars_path, log_path, ts_dir = _find_mmdet_scalars(work_dir)
    out: dict[str, Any] = {
        "name": name,
        "framework": "mmdet",
        "work_dir": str(work_dir),
        "n_gpus": n_gpus,
        "partial": True,
        "status": "missing_scalars",
    }
    if scalars_path is None:
        return out
    rows = _read_jsonl(scalars_path)
    train_rows = [r for r in rows if "iter" in r and "coco/bbox_mAP" not in r and "time" in r]
    # val metrics may appear as separate scalar dumps with coco/bbox_mAP
    val_rows = [r for r in rows if "coco/bbox_mAP" in r or "bbox_mAP" in r]

    times = [float(r["time"]) for r in train_rows if r.get("time") is not None]
    memories = [float(r["memory"]) for r in train_rows if r.get("memory") is not None]
    data_times = [float(r["data_time"]) for r in train_rows if r.get("data_time") is not None]

    last = train_rows[-1] if train_rows else None
    epoch = int(last["epoch"]) if last and "epoch" in last else None
    it = int(last["iter"]) if last and "iter" in last else None
    step = int(last["step"]) if last and "step" in last else it

    # mean s/iter: prefer rows after warmup (iter > 200) when available
    steady = [float(r["time"]) for r in train_rows if int(r.get("iter", 0)) >= 200 and r.get("time") is not None]
    s_per_iter = (sum(steady) / len(steady)) if steady else ((sum(times) / len(times)) if times else None)

    wall_s, t_start, t_end = _mmdet_wall_from_log(log_path, scalars_path)
    # Alternative estimate: step * mean_time (rough; LoggerHook interval is usually 50)
    # Prefer real wall clock from log timestamps.

    best_map = None
    best_epoch = None
    for r in val_rows:
        m = r.get("coco/bbox_mAP", r.get("bbox_mAP"))
        if m is None:
            continue
        m = float(m)
        ep = r.get("epoch", r.get("step"))
        if best_map is None or m > best_map:
            best_map = m
            best_epoch = ep

    # Also scan the run log for "coco/bbox_mAP" lines if no val scalars yet
    if best_map is None and log_path and log_path.is_file():
        map_re = re.compile(r"coco/bbox_mAP:\s*([0-9.]+)")
        ep_re = re.compile(r"Epoch\(val\)\s*\[(\d+)\]")
        cur_ep = None
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                em = ep_re.search(line)
                if em:
                    cur_ep = int(em.group(1))
                mm = map_re.search(line)
                if mm:
                    m = float(mm.group(1))
                    if best_map is None or m > best_map:
                        best_map = m
                        best_epoch = cur_ep

    gpu_hours = (wall_s / 3600.0 * n_gpus) if wall_s is not None else None
    out.update({
        "status": "running_or_partial",
        "timestamp_dir": str(ts_dir) if ts_dir else None,
        "scalars": str(scalars_path),
        "log": str(log_path) if log_path else None,
        "epochs_done": epoch,  # current epoch index (1-based in mmengine logs)
        "last_iter": it,
        "last_step": step,
        "n_log_points": len(train_rows),
        "s_per_iter": round(s_per_iter, 4) if s_per_iter is not None else None,
        "mean_data_time_s": round(sum(data_times) / len(data_times), 4) if data_times else None,
        "peak_memory_MiB_per_gpu": max(memories) if memories else None,
        "wall_seconds": round(wall_s, 1) if wall_s is not None else None,
        "wall_time": _fmt_hours(wall_s),
        "start": t_start,
        "end_or_last": t_end,
        "gpu_hours": round(gpu_hours, 3) if gpu_hours is not None else None,
        "best_val_mAP": best_map,
        "best_val_epoch": best_epoch,
    })
    return out


def _find_ultra_log(name: str, run_dir: Path) -> Optional[Path]:
    candidates = sorted(LOGS.glob(f"{name}_*.log"), key=lambda p: p.stat().st_mtime)
    if candidates:
        return candidates[-1]
    # also check run_dir
    local = sorted(run_dir.glob("*.log"), key=lambda p: p.stat().st_mtime)
    return local[-1] if local else None


def _ultra_wall(log_path: Optional[Path], run_dir: Path) -> tuple[Optional[float], Optional[str], Optional[str]]:
    start = end = None
    if log_path and log_path.is_file():
        # Ultralytics logs rarely have mmengine-style timestamps; use file ctime/mtime
        # and also parse "Starting training" / progress if present.
        st = log_path.stat()
        # Prefer birth/ctime of log as start; mtime as last update
        t0 = st.st_ctime
        t1 = st.st_mtime
        # results.csv mtime may be more recent once epochs finish
        results = run_dir / "results.csv"
        if results.is_file():
            t1 = max(t1, results.stat().st_mtime)
        args = run_dir / "args.yaml"
        if args.is_file():
            t0 = min(t0, args.stat().st_ctime)
        return t1 - t0, datetime.fromtimestamp(t0).isoformat(sep=" "), datetime.fromtimestamp(t1).isoformat(sep=" ")
    args = run_dir / "args.yaml"
    results = run_dir / "results.csv"
    if args.is_file():
        t0 = args.stat().st_ctime
        t1 = results.stat().st_mtime if results.is_file() else args.stat().st_mtime
        return t1 - t0, datetime.fromtimestamp(t0).isoformat(sep=" "), datetime.fromtimestamp(t1).isoformat(sep=" ")
    return None, None, None


def _parse_ultra_log_progress(log_path: Path) -> dict[str, Any]:
    """Extract epoch progress, peak GPU_mem, and rough s/iter from the train log.

    Ultralytics tees a single carriage-return-updated progress line, so we split on
    ``\\r`` as well as ``\\n`` and strip ANSI before matching.
    """
    info: dict[str, Any] = {}
    peak_g = None
    last_epoch = last_total = None
    # "1/80      20.8G     0.8457 ..." (epoch row) — not anchored to line start (CR fragments)
    row_re = re.compile(r"(?:^|[\r\n])\s*(\d+)/(\d+)\s+([\d.]+)G\b")
    # also match bare "1/80   20.8G" anywhere after ANSI strip
    row_re2 = re.compile(r"\b(\d+)/(\d+)\s+([\d.]+)G\b")
    # Progress bar fragments: "100/1250 3.6it/s"
    it_re = re.compile(r"(\d+)/(\d+)\s+([\d.]+)it/s")
    it_s_samples = []
    raw = log_path.read_text(encoding="utf-8", errors="replace")
    plain = re.sub(r"\x1b\[[0-9;]*m", "", raw)
    for frag in re.split(r"[\r\n]+", plain):
        frag = frag.strip()
        if not frag:
            continue
        m = row_re2.search(frag)
        if m:
            # Heuristic: epoch/max_epochs when max <= 1000 and looks like training header
            a, b, g = int(m.group(1)), int(m.group(2)), float(m.group(3))
            # epoch rows have max_epochs typically 40..300; batch progress has max ~1250
            if b <= 500 and g >= 1.0:  # GPU mem in GiB
                last_epoch, last_total = a, b
                peak_g = g if peak_g is None else max(peak_g, g)
        for im in it_re.finditer(frag):
            rate = float(im.group(3))
            # batch progress: denominator usually >> 100
            if int(im.group(2)) > 100 and rate > 0:
                it_s_samples.append(1.0 / rate)
    # Fall back: scan whole text with row_re2 for any GPU_mem
    if peak_g is None:
        for m in row_re2.finditer(plain):
            g = float(m.group(3))
            if g >= 1.0:
                peak_g = g if peak_g is None else max(peak_g, g)
                a, b = int(m.group(1)), int(m.group(2))
                if b <= 500:
                    last_epoch, last_total = a, b
    if last_epoch is not None:
        info["epochs_done"] = last_epoch  # current epoch in progress
        info["max_epochs"] = last_total
    if peak_g is not None:
        info["peak_memory_GiB"] = peak_g
        info["peak_memory_MiB_per_gpu"] = round(peak_g * 1024, 1)
    if it_s_samples:
        sample = it_s_samples[-50:]
        info["s_per_iter"] = round(sum(sample) / len(sample), 4)
    return info


def parse_ultra_run(name: str, n_gpus: int, run_dir: Path) -> dict[str, Any]:
    out: dict[str, Any] = {
        "name": name,
        "framework": "ultralytics",
        "work_dir": str(run_dir),
        "n_gpus": n_gpus,
        "partial": True,
        "status": "missing_run_dir",
    }
    if not run_dir.is_dir():
        return out

    args_path = run_dir / "args.yaml"
    results_path = run_dir / "results.csv"
    log_path = _find_ultra_log(name, run_dir)
    out["log"] = str(log_path) if log_path else None
    out["args"] = str(args_path) if args_path.is_file() else None

    max_epochs = None
    if args_path.is_file():
        # minimal parse without requiring pyyaml API quirks
        text = args_path.read_text(encoding="utf-8", errors="replace")
        m = re.search(r"^epochs:\s*(\d+)", text, re.M)
        if m:
            max_epochs = int(m.group(1))
        m = re.search(r"^batch:\s*(\d+)", text, re.M)
        if m:
            out["batch"] = int(m.group(1))
        m = re.search(r"^imgsz:\s*(\d+)", text, re.M)
        if m:
            out["imgsz"] = int(m.group(1))

    best_map = best_epoch = None
    epochs_completed = 0
    if results_path.is_file() and results_path.stat().st_size > 0:
        with open(results_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                epochs_completed += 1
                # Ultralytics columns vary; prefer metrics/mAP50-95(B)
                key = None
                for k in row:
                    if "mAP50-95" in k or k.strip() == "metrics/mAP50-95(B)":
                        key = k
                        break
                if key and row[key] not in (None, ""):
                    try:
                        m = float(row[key])
                    except ValueError:
                        continue
                    ep = int(float(row.get("epoch", epochs_completed)))
                    if best_map is None or m > best_map:
                        best_map = m
                        best_epoch = ep
        out["results_csv"] = str(results_path)
        out["epochs_completed"] = epochs_completed

    log_info = _parse_ultra_log_progress(log_path) if log_path else {}
    wall_s, t_start, t_end = _ultra_wall(log_path, run_dir)
    gpu_hours = (wall_s / 3600.0 * n_gpus) if wall_s is not None else None

    out.update({
        "status": "running_or_partial",
        "max_epochs": max_epochs or log_info.get("max_epochs"),
        "epochs_done": log_info.get("epochs_done") or epochs_completed,
        "s_per_iter": log_info.get("s_per_iter"),
        "peak_memory_MiB_per_gpu": log_info.get("peak_memory_MiB_per_gpu"),
        "peak_memory_GiB": log_info.get("peak_memory_GiB"),
        "wall_seconds": round(wall_s, 1) if wall_s is not None else None,
        "wall_time": _fmt_hours(wall_s),
        "start": t_start,
        "end_or_last": t_end,
        "gpu_hours": round(gpu_hours, 3) if gpu_hours is not None else None,
        "best_val_mAP": best_map,
        "best_val_epoch": best_epoch,
    })
    return out


def resolve_work_dir(kind: str, name: str) -> Path:
    if kind == "mmdet":
        return WORK_DIRS / name
    return WORK_DIRS / "ultra" / name


def parse_spec(spec: str) -> tuple[str, int]:
    if ":" not in spec:
        raise argparse.ArgumentTypeError(f"expected name:gpus, got {spec!r}")
    name, gpus = spec.rsplit(":", 1)
    return name, int(gpus)


def to_markdown(runs: list[dict[str, Any]]) -> str:
    lines = [
        "# Resource usage (live / partial runs OK)",
        "",
        "Generated by `tools/resource_report.py`. Regenerated at the end of training for the final statement.",
        "",
        "| run | GPUs | wall so far | GPU-hours | epochs | s/iter | peak mem / GPU | best val mAP (epoch) |",
        "|---|---:|---|---:|---|---:|---:|---|",
    ]
    for r in runs:
        best = "—"
        if r.get("best_val_mAP") is not None:
            best = f"{r['best_val_mAP']:.4f}"
            if r.get("best_val_epoch") is not None:
                best += f" (@{r['best_val_epoch']})"
        ep = r.get("epochs_done")
        if r.get("max_epochs"):
            ep = f"{ep}/{r['max_epochs']}"
        elif r.get("last_iter") is not None:
            ep = f"ep{ep} iter {r['last_iter']}"
        mem = r.get("peak_memory_MiB_per_gpu")
        mem_s = f"{mem:.0f} MiB" if mem is not None else "—"
        if r.get("peak_memory_GiB") is not None and r.get("framework") == "ultralytics":
            mem_s = f"{r['peak_memory_GiB']:.1f} GiB"
        lines.append(
            f"| {r['name']} | {r['n_gpus']} | {r.get('wall_time') or '—'} | "
            f"{r.get('gpu_hours') if r.get('gpu_hours') is not None else '—'} | "
            f"{ep if ep is not None else '—'} | "
            f"{r.get('s_per_iter') if r.get('s_per_iter') is not None else '—'} | "
            f"{mem_s} | {best} |"
        )
    lines += ["", "## Per-run details", ""]
    for r in runs:
        lines.append(f"### `{r['name']}` ({r['framework']})")
        lines.append("")
        for k in ("work_dir", "log", "scalars", "results_csv", "start", "end_or_last", "status",
                  "last_step", "mean_data_time_s"):
            if r.get(k) is not None:
                lines.append(f"- **{k}**: `{r[k]}`")
        lines.append("")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mmdet", nargs="*", default=[], metavar="NAME:GPUS",
                   help="MMDetection work_dirs/<name> with GPU count, e.g. codino_swinl_ego:3")
    p.add_argument("--ultra", nargs="*", default=[], metavar="NAME:GPUS",
                   help="Ultralytics work_dirs/ultra/<name> with GPU count, e.g. y26x_1024_train:1")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--md", type=Path, default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    runs: list[dict[str, Any]] = []
    for spec in args.mmdet:
        name, gpus = parse_spec(spec)
        runs.append(parse_mmdet_run(name, gpus, resolve_work_dir("mmdet", name)))
    for spec in args.ultra:
        name, gpus = parse_spec(spec)
        runs.append(parse_ultra_run(name, gpus, resolve_work_dir("ultra", name)))

    md = to_markdown(runs)
    print(md)
    payload = {"generated": datetime.now().isoformat(sep=" "), "runs": runs}
    if args.out is not None:
        dump_json(payload, args.out, indent=2)
        print(f"wrote {args.out}", flush=True)
    if args.md is not None:
        args.md.parent.mkdir(parents=True, exist_ok=True)
        args.md.write_text(md, encoding="utf-8")
        print(f"wrote {args.md}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
