# DINO-5scale Swin-L (MMDetection 3.3.0)

Fallback / ensemble member next to Co-DINO. COCO-pretrained DINO fine-tuned on the 10 ego classes.

## Checkpoint
- `solution/weights/dino-5scale_swin-l_8xb2-36e_coco-5486e051.pth` (COCO 58.4 box AP, sha256 prefix `5486e051` verified), from
  <https://github.com/RistoranteRist/mmlab-weights/releases/download/dino-swinl/dino-5scale_swin-l_8xb2-36e_coco-5486e051.pth>
  (URL in `mmdet/.mim/configs/dino/metafile.yml`).
- Loaded via `load_from`. Skipped (shape mismatch 80 -> 10 classes), nothing else missing/unexpected:
  `bbox_head.cls_branches.{0..6}.{weight,bias}`, `dn_query_generator.label_embedding.weight`.
- `backbone.init_cfg=None`, so no Swin ImageNet download (log warns "No pre-trained weights for SwinTransformer",
  expected: `load_from` overwrites it right after).

## Configs (`configs/`)
- `dino_swinl_ego.py`: main. Inherits `mmdet::dino/dino-5scale_swin-l_8xb2-36e_coco.py`; dataset constants read from
  `_base_/ego_dataset.py` (via `runpy`, since both bases define `data_root`). work_dir `work_dirs/dino_swinl_ego`.
  - Data: train.json (switch to trainval via `--cfg-options`, see below), val.json with CocoMetric bbox (classwise),
    test.json with `format_only=True` -> `<work_dir>/test.bbox.json`. `max_per_img=100`.
  - Resolution: the standard DETR aug (short 480-800, long <= 1333, + random crop branch), test at 1333x800
    (1920x1080 -> 1333x750). Same aug as the COCO recipe.
  - bs 2/GPU, `AmpOptimWrapper`, Swin `with_cp=True`, AdamW lr 5e-5 (backbone x0.1), wd 1e-4, clip 0.1.
    LR: COCO recipe 1e-4 @ total batch 16, scaled linearly to 4x2=8. That keeps the per-sample step of the recipe,
    and the lower LR is the safer choice when fine-tuning a converged model. For another total batch size, add
    `--auto-scale-lr` (linear from `base_batch_size=8`).
  - 12 epochs, 500-iter linear warmup (the new class heads start from scratch), MultiStepLR x0.1 at epochs [8, 11].
    Val every epoch, `save_best='coco/bbox_mAP'`, `max_keep_ckpts=2` (2.6 GB each with optimizer state; best is 0.9 GB).
- `dino_swinl_ego_val.py`: test.py on val -> metrics + `<work_dir>/val.bbox.json`.
- `dino_swinl_ego_hr.py`: higher-res variant (short 576-1024, long <= 2048, test 2048x1024 -> 1920x1080 becomes
  1820x1024). Needs bs 1/GPU; use lr 2.5e-5 (`--auto-scale-lr`).

## Commands (4 GPUs, from `solution/mmdet`)
```bash
cd /home/zhangyuming/Egoobjection/solution/mmdet
W=/home/zhangyuming/Egoobjection/solution/work_dirs
# train on train.json (val every epoch, best ckpt kept)
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29611 bash tools/dist_train.sh configs/dino_swinl_ego.py 4
# final model on train+val (val.json is then in-train; take epoch_12.pth)
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29611 bash tools/dist_train.sh configs/dino_swinl_ego.py 4 \
  --work-dir $W/dino_swinl_ego_trainval \
  --cfg-options train_dataloader.dataset.ann_file=/home/zhangyuming/Egoobjection/solution/data/trainval.json \
  train_cfg.val_interval=12 test_evaluator.outfile_prefix=$W/dino_swinl_ego_trainval/test
# val metrics + val.bbox.json
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29611 bash tools/dist_test.sh configs/dino_swinl_ego_val.py \
  $W/dino_swinl_ego/best_coco_bbox_mAP_epoch_N.pth 4
# test set -> $W/dino_swinl_ego/test.bbox.json
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29611 bash tools/dist_test.sh configs/dino_swinl_ego.py \
  $W/dino_swinl_ego/best_coco_bbox_mAP_epoch_N.pth 4
```
Note: `outfile_prefix` is derived from the config's `work_dir`. When you pass `--work-dir`, also override
`test_evaluator.outfile_prefix`. 2 GPUs: `dist_train.sh ... 2 --auto-scale-lr` (lr 2.5e-5).

## Benchmarks (1x RTX 4090 = GPU 2, AMP, with_cp; train subset of 300 random train images)
| setting | bs/GPU | peak mem (max_allocated / nvidia-smi) | s/iter | s/img | est. epoch, 4 GPUs | est. epoch, 2 GPUs |
|---|---|---|---|---|---|---|
| std 480-800 / 1333 | 2 | 20.5 GB / 22.2 GB | 0.81 | 0.40 | ~18-19 min | ~35-36 min |
| std, worst case (all 1333x800) | 2 | 19.5 GB / 22.2 GB | 1.07 | 0.53 | - | - |
| HR 576-1024 / 2048 | 1 | 17.7 GB / n.m. | 0.62 | 0.62 | ~27-29 min | ~53-55 min |

- Epoch estimates: 10000 imgs (train.json; trainval x1.1). They add +10-15% for DDP: GPUs 0-3 sit on separate NUMA
  nodes without P2P, and the gradient all-reduce is 875 MB fp32 per iteration. They exclude per-epoch val:
  1000 imgs at 0.155 s/img (std) is ~40 s on 4 GPUs, 0.28 s/img (HR) is ~70 s.
- Std 12 epochs on 4 GPUs: ~4 h (trainval ~4.3 h). On 2 GPUs: ~7.5 h.
- Test-set inference (2000 imgs, bs 1): std 0.166 s/img, i.e. ~5.5 min on 1 GPU or ~1.5 min on 4 GPUs (+~40 s startup).
  HR 0.284 s/img: ~9.5 min / ~2.5 min. `--tta` (flip) doubles this (untested).
- AMP: the first ~40 iterations log `grad_norm: nan/inf` while the dynamic loss scale calibrates (those steps are
  skipped); after that it is finite. Losses stay finite throughout.

## Smoke tests (GPU 2, outputs in `work_dirs/dino_swinl_ego_smoke/`)
- Train 150 iters (bs 2, 300-img subset, then val on a 150-img val subset): loss 41.1 -> 28.1 (still in warmup,
  lr <= 1.8e-5). Checkpoint, val, and save_best all ran; val-subset mAP 0.037 (bottle AP 0.31).
- `test.py` + `dino_swinl_ego_val.py` on the val subset: CocoMetric 0.0370 = `resource/eval.py` 0.037 (pycocotools
  0.037127). The json has int image_ids = the subset's val ids, category_id within 0..9, <= 100 dets/img, no
  out-of-bounds boxes. Top detection matches a GT bottle in original 2288x1080 pixels.
- Test format_only on 50 test images: `test.bbox.json` with exactly those 50 image_ids, 100 dets/img.
- `trainval.json` switch: dataset builds with 11000 images, all files resolve.

## Recommendation
`dino_swinl_ego.py` as is: std res, 4 GPUs x bs 2, lr 5e-5, 12 epochs, steps at [8, 11], about 4 h. Pick the epoch
on train.json, then retrain on trainval with the same schedule for the submission. HR costs ~1.5x train and ~1.7x
inference time. With ~80% COCO-large boxes its gain is uncertain, so it is only worth a run as an extra ensemble
member if GPU time allows. A cheap test first: evaluate the std model at the HR test scale
(`dino_swinl_ego_val.py` with `test_dataloader.dataset.pipeline.1.scale="(2048,1024)"`).
