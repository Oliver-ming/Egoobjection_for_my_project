# Experiment log

All val numbers are COCO mAP@0.5:0.95 on the full `val.json` (1000 images) computed with `tools/coco_eval.py`
(bit-identical to `resource/eval.py`) unless marked otherwise. "UL" = Ultralytics' built-in metric (not official).

## Smoke tests / benchmarks (2026-09-24)

| run | setting | result |
|---|---|---|
| Co-DINO Swin-L smoke | 200 iters, 1 GPU, 1333 | 0.258 on 200-img val subset (== eval.py) |
| Co-DINO TTA smoke | same ckpt, 1333 ± flip, WBF max | 0.2725 fused (orig 0.2578, flip 0.2549) |
| DINO Swin-L smoke | 150 iters, 1 GPU, 1333 | 0.037 on 150-img val subset (== eval.py) |
| YOLO26x smoke | 640, fraction 0.1, 2 ep | 0.2368 (UL 0.236); `--augment` 0.2768; `--nms false` 0.196 |
| RT-DETR-X smoke | 640, small fraction | 0.0970 (UL 0.0989) |

## Main runs

| run | data | epochs | best val (epoch) | last val | notes |
|---|---|---|---|---|---|
| codino_swinl_ego | train | 12 (step@10) | 0.884 (ep12) | 0.884 | bs2 from ep5; ep1–4: 0.739/0.814/0.840/0.854, then 0.864…0.884 |
| y26x_1024_train | train | 80 | 0.860 pycocotools @1024 (UL best 0.892 ep67) | UL 0.891 | 1280 TTA **0.887**; 1024 TTA 0.880; 1280 plain 0.856 |
| codino_swinl_ego_trainval | train+val | 12 | leaked 0.909 ep12 | leaked | ep4/8/12 leaked val 0.868/0.898/0.909. Use last epoch |
| y26x_1024_trainval | train+val | not run | | | |

## TTA ablation (Co-DINO train-only ep12, full val, pycocotools)

| view | AP |
|---|---|
| 1333 orig (matches training eval) | 0.884 |
| 1333 flip | 0.882 |
| 1600 orig / flip | 0.881 / 0.879 |
| 1920 orig / flip | 0.877 / 0.874 |
| four 1333+1600 views, WBF max iou 0.75 | 0.886 |

## Ensemble ablation (val, train-only models, WBF `conf_type=max`)

| members / weights | iou | AP |
|---|---|---|
| Co-DINO 1333 + YOLO 1280 TTA, w 1:2 | 0.75 | 0.899 |
| Co-DINO 1333 orig+flip + YOLO 1280 TTA, w 1:1:2 | 0.75 | 0.901 |
| Co-DINO 1333+1600, orig+flip + YOLO 1280 TTA, w 1:1:1:1:3 | 0.75 | 0.902 |
| same, WBF IoU 0.90 | 0.90 | **0.906** |

YOLO alone at 1280 TTA is 0.887. Higher-res Co-DINO views do not beat 1333 alone; they add a little inside the ensemble.

## Submissions

| file | models | notes |
|---|---|---|
| `submission.csv` (also `solution/submissions/final_submission.csv`) | Co-DINO trainval ep12 at 1333 and 1600, each with flip, plus YOLO26x 1280 TTA. WBF max, iou 0.90, weights 1,1,1,1,3 | Validated, 2000 rows. Honest val of this recipe on the train-only Co-DINO is 0.906. This file uses the trainval checkpoint instead |
