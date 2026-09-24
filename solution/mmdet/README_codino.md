# Co-DINO 5-scale Swin-L (MMDetection 3.3.0)

Primary detector for the ego competition. Objects365→COCO pretrained Co-DINO fine-tuned on the 10 ego classes.

## What was fetched

| Item | Source | Notes |
|---|---|---|
| Project code | [mmdetection v3.3.0](https://github.com/open-mmlab/mmdetection/archive/refs/tags/v3.3.0.tar.gz) → `projects/CO-DETR/` | Extracted into `solution/mmdet/projects/CO-DETR/` |
| Checkpoint | [co_dino_5scale_swin_large_16e_o365tococo-614254c9.pth](https://download.openmmlab.com/mmdetection/v3.0/codetr/co_dino_5scale_swin_large_16e_o365tococo-614254c9.pth) | Saved as `solution/weights/co_dino_5scale_swin_large_16e_o365tococo-614254c9.pth`, **md5 `5677b385a8fe90d2282dd801c629ac18`** |
| Dep | `fairscale` | Already in the shared env (needed only if using fairscale checkpointing; Swin `with_cp` uses torch checkpoint) |

## Patches / config fixes

### 1. AMP fp32 loss outs (`codetr/co_dino_head.py`)
Added `_outs_to_fp32` and wrap decoder/aux outs before loss. Under AMP, an image with no GT makes `bbox_overlaps` return an empty **fp16** tensor that is then index-put into an fp32 buffer → `RuntimeError`. Casting preds to fp32 restores the mmdet-2.x `@force_fp32` behaviour.

### 2. `rpn_proposal.nms.split_thr = 100000` (`codino_swinl_ego.py`)
With AMP, when the number of boxes ≥ default `split_thr=10000`, mmcv `batched_nms` takes a per-level loop that index-puts fp32 scores into an fp16 tensor and crashes. The aux RPN keeps ≤ 6×4000 boxes; raising `split_thr` forces the single-call (coordinate-offset) path, which is numerically identical and dtype-safe.

### 3. `dist_infer.sh` absolute paths
`CocoDataset` joins a *relative* `ann_file` with `data_root`. Custom subset jsons must be absolute; the script now absolutizes `CONFIG` / `CHECKPOINT` / `ANN` / `OUT_PREFIX`.

## Configs (`configs/`)

- `codino_swinl_ego.py` — main. Base: `projects/CO-DETR/configs/codino/co_dino_5scale_swin_l_16xb1_16e_o365tococo.py`. Dataset constants from `_base_/ego_dataset.py`.
  - **Resolution:** DETR multi-scale short 480..800 (step 32), long ≤ 1333; test `(1333, 800)` (1920×1080 → 1333×750).
  - **Optim:** `AmpOptimWrapper`, AdamW **lr 5e-5** (sqrt-scaled from 1e-4 @ bs16 → bs4; backbone ×0.1), wd 1e-4, clip 0.1. Linear warmup 500 iters, MultiStepLR ×0.1 at epoch **10**, **12** epochs, val every epoch, `save_best='coco/bbox_mAP'`, `max_keep_ckpts=1`.
  - **Heads:** `num_classes=10` on query / ROI / ATSS; `max_per_img=100`.
  - **load_from** skips (size mismatch 80/81→10/11): `query_head.cls_branches.*`, `roi_head.0.bbox_head.fc_cls/fc_reg`, `bbox_head.0.atss_cls`. Name mismatch: unexpected `query_head.label_embedding.weight`, missing `query_head.dn_generator.label_embedding.weight` (DN label emb random-init).
- `codino_swinl_ego_1920.py` — short 640..1088 / long ≤ 1920, test `(1920, 1088)`.
- `codino_swinl_ego_trainval.py` — same as main but `trainval.json`; no `save_best` (val is in-train).

## Commands

All from `solution/`. Python = `env/bin/python`. Prefer absolute paths for custom jsons.

```bash
cd /home/zhangyuming/Egoobjection/solution
S=$PWD
# --- train (example: 3 GPUs as currently running; 4 GPUs also fine) ---
CUDA_VISIBLE_DEVICES=0,1,2 PORT=29500 bash mmdet/tools/dist_train.sh \
  mmdet/configs/codino_swinl_ego.py 3
# train+val final model:
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 bash mmdet/tools/dist_train.sh \
  mmdet/configs/codino_swinl_ego_trainval.py 4

# --- val / test inference -> <OUT_PREFIX>.bbox.json ---
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29920 bash mmdet/tools/dist_infer.sh \
  mmdet/configs/codino_swinl_ego.py \
  work_dirs/codino_swinl_ego/best_coco_bbox_mAP_epoch_N.pth \
  4 val work_dirs/codino_swinl_ego/preds/val
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29920 bash mmdet/tools/dist_infer.sh \
  mmdet/configs/codino_swinl_ego.py \
  work_dirs/codino_swinl_ego/best_coco_bbox_mAP_epoch_N.pth \
  4 test work_dirs/codino_swinl_ego/preds/test
# offline eval (must match resource/eval.py):
env/bin/python tools/coco_eval.py --gt resource/data/val.json \
  --pred work_dirs/codino_swinl_ego/preds/val.bbox.json --per-class

# --- TTA (multi-scale + flip; WBF conf_type=max by default) ---
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29931 \
  env/bin/python mmdet/tools/tta_infer.py \
  --config mmdet/configs/codino_swinl_ego.py \
  --checkpoint work_dirs/codino_swinl_ego/best_coco_bbox_mAP_epoch_N.pth \
  --ann resource/data/val.json \
  --out-dir work_dirs/codino_swinl_ego/tta_val \
  --scales 1333x800,1600x960,1920x1152 --flip --gpus 4 \
  --method wbf --iou-thr 0.6 --conf-type max
```

**TTA notes:** Co-DINO only undoes `scale_factor`; it does **not** un-flip. `tta_infer.py` post-maps flipped xywh with `x' = W - x - w`. Prefer `--conf-type max` (default): WBF `avg` halves scores of singleton clusters and can *hurt* same-model TTA.

## Benchmarks

### 1-GPU smoke / bench (AMP, with_cp, bs1)
| setting | peak mem | s/iter | val infer |
|---|---|---|---|
| 1333 | ~9.3 GB | 0.88–0.95 | ~0.16–0.17 s/img |
| 1920 | ~13.4 GB | 1.25–1.33 | — |

Smoke (200 iters @ 1333, val_sub200): **mAP 0.258** (CocoMetric = `tools/coco_eval.py`). Ckpt: `work_dirs/codino_smoke/train_s1333/epoch_1.pth`.

### Live 3-GPU DDP (GPUs 0–1–2, `codino_swinl_ego.py`, log `logs/codino_swinl_ego_train.log`)
Snapshot around iter 600 / 3334 of epoch 1:

| metric | value |
|---|---|
| s/iter | ~1.11–1.20 (avg ≈ 1.15 after iter 200) |
| data_time | ~0.008–0.010 s |
| peak memory (logged) | ~9275–9291 MiB / GPU |
| ETA / epoch | ~1.06 h (3334 × 1.15 s) |
| ETA / 12 epochs | ~12.5–12.8 h (log ETA ≈ 12:25 at iter 600) |
| loss | 145 → 40 by iter 600 (warmup ends ~500) |
| `grad_norm: nan` | only iters 50 & 100 (AMP scale calibration); finite thereafter; **no loss nan/inf** |
| load_from | only class-head size mismatches + DN label-emb name mismatch (as above) |

Effective throughput ≈ 3 img / 1.15 s ≈ 2.6 img/s vs ~1.1 img/s single-GPU.

### Verify / TTA (smoke ckpt, `work_dirs/codino_verify/`)
| run | result |
|---|---|
| `dist_infer` val_sub200 | mAP **0.258**; `coco_eval.py` AP=0.2578 |
| `dist_infer` test_sub50 | format_only; exactly 50 image_ids; cats ∈ 0..9; boxes in-bounds |
| TTA 1333 orig / flip / fused (WBF max) | AP **0.2578 / 0.2549 / 0.2725** (raw flipped without unflip ≈ 0.008) |

## Layout reminder
- Outputs: `work_dirs/codino_swinl_ego/` (live train — do not touch while running)
- Predictions interchange: COCO results json (`image_id`, `category_id` 0..9, `bbox` xywh, `score`)
