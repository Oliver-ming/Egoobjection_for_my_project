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
| codino_swinl_ego | train | 12 (step@10) | TODO | TODO | 3 GPUs × bs1, lr 5e-5, 1333 |
| y26x_1024_train | train | 80 | TODO | TODO | UL mAP50-95: ep1 0.399, ep2 0.503, ep3 0.596, ep4 0.654 |
| codino_swinl_ego_trainval | train+val | TODO | (leaked) | (leaked) | |
| y26x_1024_trainval | train+val | TODO | (leaked) | (leaked) | |

## TTA ablation (Co-DINO, val)

| variants | fusion | AP |
|---|---|---|

## Ensemble ablation (val, train-only models)

| members / weights | method / conf_type / iou_thr | AP |
|---|---|---|

## Submissions

| file | models | notes |
|---|---|---|
