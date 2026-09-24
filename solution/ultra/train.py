#!/usr/bin/env python
"""Fine-tune Ultralytics detectors (YOLO26 / YOLO11 / RT-DETR) on the EgoObjects data.

  CUDA_VISIBLE_DEVICES=0 python ultra/train.py --model yolo26x.pt --imgsz 1024 --epochs 60 --batch 16
  CUDA_VISIBLE_DEVICES=0,1 python ultra/train.py --model yolo26x.pt --device 0,1 --batch 32 \
      --data ultra/data/ego_trainval.yaml --name y26x_1024_trainval mosaic=0.8 lr0=0.0008

Anything after the named flags is `key=value` passed straight to `model.train` (overrides the defaults below).
`--bench-iters N` times N training iterations (after warm-up), prints throughput / memory / GPU util and exits.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ego_ultra as E  # noqa: E402  (must precede other ultralytics imports)

import torch  # noqa: E402
from ultralytics.cfg import smart_value  # noqa: E402

COMMON_DEFAULTS = dict(
    cos_lr=True,
    warmup_epochs=2.0,  # starting from COCO weights, 2 epochs (~1250 iters at batch 16) of warm-up is plenty
    amp=True,
    cache="ram",  # resized frames kept in RAM (~18 GB/rank at 1024); avoids re-decoding 1080p/4K JPEGs every epoch
    workers=12,
    seed=0,
    deterministic=False,  # True forces deterministic cuDNN/cuBLAS kernels, which is noticeably slower
    plots=True,
    patience=100,
)
# optimizer='auto' silently switches AdamW(lr~7e-4) -> MuSGD(lr=0.01) once epochs*ceil(n/64) > 10k iterations
# (~64 epochs here), so pick the optimizer explicitly to keep runs of different length comparable.
YOLO_DEFAULTS = dict(optimizer="AdamW", lr0=0.001, lrf=0.01, momentum=0.9, weight_decay=0.0005, warmup_bias_lr=0.0)
# RT-DETR is a transformer: the usual DETR fine-tuning recipe (AdamW 1e-4, wd 1e-4); Ultralytics clips grads at 10.
RTDETR_DEFAULTS = dict(optimizer="AdamW", lr0=0.0001, lrf=0.01, momentum=0.9, weight_decay=0.0001, warmup_bias_lr=0.0)


class BenchDone(Exception):
    pass


class TrainBench:
    """Callbacks that time training iterations and sample GPU utilisation, then abort the run."""

    def __init__(self, iters: int, warmup: int = 10):
        self.iters, self.warmup = iters, warmup
        self.t_setup0 = self.t_setup1 = None
        self.starts, self.ends = [], []
        self.util, self.nvml_mem = [], []
        self._stop = threading.Event()
        self.result = None

    def _sample_gpu(self):
        try:
            import pynvml

            pynvml.nvmlInit()
            idx = int(os.environ["CUDA_VISIBLE_DEVICES"].split(",")[0])
            h = pynvml.nvmlDeviceGetHandleByIndex(idx)
            while not self._stop.wait(0.25):
                if len(self.ends) > self.warmup:
                    self.util.append(pynvml.nvmlDeviceGetUtilizationRates(h).gpu)
                    self.nvml_mem.append(pynvml.nvmlDeviceGetMemoryInfo(h).used / 2**30)
        except Exception as e:  # never let monitoring kill the benchmark
            print(f"[bench] GPU sampling disabled: {e}")

    def on_pretrain_routine_start(self, trainer):
        self.t_setup0 = time.time()

    def on_train_start(self, trainer):
        self.t_setup1 = time.time()
        torch.cuda.reset_peak_memory_stats()
        threading.Thread(target=self._sample_gpu, daemon=True).start()

    def on_train_batch_start(self, trainer):
        self.starts.append(time.perf_counter())

    def on_train_batch_end(self, trainer):
        torch.cuda.synchronize()
        self.ends.append(time.perf_counter())
        if len(self.ends) >= self.warmup + self.iters:
            self._stop.set()
            s, e, w = self.starts, self.ends, self.warmup
            wall = e[-1] - e[w - 1]
            compute = sum(e[i] - s[i] for i in range(w, len(e)))
            n = len(e) - w
            bs = trainer.batch_size
            self.result = dict(
                batch=bs,
                requested_batch=trainer.args.batch,
                imgsz=trainer.args.imgsz,
                workers=trainer.train_loader.num_workers,
                cache=trainer.args.cache,
                iters_measured=n,
                sec_per_iter=round(wall / n, 4),
                it_per_s=round(n / wall, 3),
                img_per_s=round(n * bs / wall, 2),
                data_wait_frac=round(1 - compute / wall, 3),
                est_epoch_min_10k=round(10000 / (n * bs / wall) / 60, 1),
                gpu_util_mean=round(sum(self.util) / max(len(self.util), 1), 1),
                gpu_util_samples=len(self.util),
                peak_alloc_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2),
                peak_reserved_gb=round(torch.cuda.max_memory_reserved() / 2**30, 2),
                nvml_used_max_gb=round(max(self.nvml_mem, default=0), 2),
                setup_s=round(self.t_setup1 - self.t_setup0, 1),
            )
            raise BenchDone


def parse_overrides(pairs: list[str]) -> dict:
    out = {}
    for p in pairs:
        if "=" not in p:
            raise SystemExit(f"override '{p}' is not key=value")
        k, v = p.split("=", 1)
        out[k.strip()] = smart_value(v.strip())
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="yolo26x.pt", help="weights name/path (resolved from solution/weights) or yaml")
    ap.add_argument("--data", default=str(E.DATA_YAMLS / "ego_train.yaml"))
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=float, default=16, help="total batch over all GPUs (-1/0.x = AutoBatch, 1 GPU)")
    ap.add_argument("--device", default="0", help="index into CUDA_VISIBLE_DEVICES, e.g. 0 or 0,1,2,3 (DDP)")
    ap.add_argument("--name", default=None, help="run name (default <model>_<imgsz>)")
    ap.add_argument("--project", default=str(E.WORK))
    ap.add_argument("--bench-iters", type=int, default=0, help="benchmark N iterations then exit (single GPU)")
    ap.add_argument("overrides", nargs="*", help="extra key=value passed to model.train")
    # allow `script.sh --bench-iters 20 cache=False` even when the shell injects key=value before "$@"
    args = ap.parse_intermixed_args()
    E.require_cuda_visible_devices()

    model = E.load_model(args.model)
    rtdetr = E.is_rtdetr(model)
    batch = int(args.batch) if args.batch >= 1 or args.batch == -1 else args.batch
    train_args = dict(
        COMMON_DEFAULTS,
        **(RTDETR_DEFAULTS if rtdetr else YOLO_DEFAULTS),
        # un-mosaicked full frames for the last ~1/6 of the run (<=10 epochs) to match the test distribution
        close_mosaic=min(10, max(1, args.epochs // 6)),
        data=str(Path(args.data).resolve()),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=batch,
        device=args.device,
        project=str(Path(args.project).resolve()),
        name=args.name or f"{Path(args.model).stem}_{args.imgsz}",
    )
    extra = parse_overrides(args.overrides)
    bench = None
    if args.bench_iters:
        bench = TrainBench(args.bench_iters)
        need = (bench.warmup + args.bench_iters + 5) * max(batch, 1)
        train_args.update(
            epochs=1, val=False, plots=False, save=False, exist_ok=True, name=f"bench_{train_args['name']}",
            fraction=min(1.0, round(need * 1.1 / 10000, 3)),
        )
        for ev in ("on_pretrain_routine_start", "on_train_start", "on_train_batch_start", "on_train_batch_end"):
            model.add_callback(ev, getattr(bench, ev))
    train_args.update(extra)

    print("[train.py] model class:", type(model).__name__, "| args:", json.dumps(train_args, default=str))
    trainer_cls = E.EgoRTDETRTrainer if rtdetr else E.EgoDetectionTrainer
    try:
        model.train(trainer=trainer_cls, **train_args)
    except BenchDone:
        r = dict(model=Path(args.model).name, **bench.result, overrides=extra)
        print("[bench] " + json.dumps(r), flush=True)
        out = Path(train_args["project"]) / "bench_results.jsonl"
        with open(out, "a") as f:
            f.write(json.dumps(r) + "\n")
        sys.stderr.flush()
        os._exit(0)  # skip interpreter teardown of persistent dataloader workers


if __name__ == "__main__":
    main()
