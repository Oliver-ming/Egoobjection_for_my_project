# EgoObjects 检测竞赛 · solution

Egocentric 10 类目标检测（COCO mAP@0.5:0.95）。共享约定见 [`CONVENTIONS.md`](CONVENTIONS.md)。

## 最终提交

文件：[`submissions/final_submission.csv`](submissions/final_submission.csv)（说明见 [`submissions/README.md`](submissions/README.md)）。

- **Co-DINO Swin-L**，Objects365→COCO 权重，在 train+val 上训 12 轮，测试时用 `1333x800` 与 `1600x960`，各含原图和水平翻转。
- **YOLO26x**，只在训练集上训 80 轮（1024），测试用 `imgsz=1280` 和 Ultralytics TTA。
- 五路预测做 WBF：`conf_type=max`，`iou_thr=0.90`，权重 `1 1 1 1 3`。

这套融合在只训训练集的 Co-DINO 上、用官方 pycocotools 评验证集，mAP@0.5:0.95 为 **0.906**。DINO Swin-L 只做了冒烟测试，没有进入提交。

## 目录结构

```
solution/
├── CONVENTIONS.md          # 硬性约定（Python 路径、GPU、禁止改 resource/ 等）
├── env/                    # → 共享 uv 环境的符号链接
├── tools/                  # 数据、评测、集成、提交、参数/资源统计
├── mmdet/                  # Co-DINO / DINO（configs、projects/CO-DETR、dist_*）
├── ultra/                  # Ultralytics YOLO26 / YOLO11 / RT-DETR
├── weights/                # 预训练权重（git-ignored）
├── data/                   # 生成的 trainval.json、裁剪后 YOLO 树等（git-ignored）
├── work_dirs/              # 训练与推理输出（git-ignored）
├── logs/                   # 长任务 stdout（git-ignored）
└── docs/                   # 数据审计、参数量、资源、技术报告
```

## 环境

- Python：`/home/zhangyuming/Egoobjection/solution/env/bin/python`（共享 uv 环境，**不要**新建 venv / `pip install`）。
- 关键包版本：`torch 2.1.0+cu121`，`mmdet 3.3.0`，`mmcv 2.1.0`，`mmengine 0.10.7`，`ultralytics 8.4.160`。
- Ultralytics 相关命令前设置：`export YOLO_AUTOINSTALL=False`。
- 始终显式设置 `CUDA_VISIBLE_DEVICES`。

## 端到端复现

以下均在 `solution/` 下执行。`PY=env/bin/python`。

### 1. 准备数据

```bash
cd /home/zhangyuming/Egoobjection/solution
PY=env/bin/python
$PY tools/prepare_data.py --yolo-dir data/yolo
# 写出 data/trainval.json、data/val_mini.json，以及裁剪后的 YOLO labels
```

### 2. 训练

**Co-DINO Swin-L（主模型，示例 3 卡）：**

```bash
CUDA_VISIBLE_DEVICES=0,1,2 PORT=29500 bash mmdet/tools/dist_train.sh \
  mmdet/configs/codino_swinl_ego.py 3
# 最终 train+val：
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 bash mmdet/tools/dist_train.sh \
  mmdet/configs/codino_swinl_ego_trainval.py 4
```

**DINO Swin-L（4 卡）：**

```bash
cd mmdet
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29611 bash tools/dist_train.sh configs/dino_swinl_ego.py 4
# trainval：见 mmdet/README_dino.md（--work-dir + --cfg-options）
```

**YOLO26x（单卡）：**

```bash
export YOLO_AUTOINSTALL=False
CUDA_VISIBLE_DEVICES=3 bash ultra/run_y26x_1024_train.sh
# 或：CUDA_VISIBLE_DEVICES=3 bash ultra/run_y26x_1024_trainval.sh
```

细节与超参见：[`mmdet/README_codino.md`](mmdet/README_codino.md)、[`mmdet/README_dino.md`](mmdet/README_dino.md)、[`ultra/README.md`](ultra/README.md)。

### 3. 验证集推理与评测

```bash
# Co-DINO
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29920 bash mmdet/tools/dist_infer.sh \
  mmdet/configs/codino_swinl_ego.py \
  work_dirs/codino_swinl_ego/best_coco_bbox_mAP_epoch_N.pth \
  4 val work_dirs/codino_swinl_ego/preds/val
$PY tools/coco_eval.py --gt ../resource/data/val.json \
  --pred work_dirs/codino_swinl_ego/preds/val.bbox.json --per-class

# DINO（val 配置写出 val.bbox.json）
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29611 bash mmdet/tools/dist_test.sh \
  mmdet/configs/dino_swinl_ego_val.py \
  work_dirs/dino_swinl_ego/best_coco_bbox_mAP_epoch_N.pth 4

# YOLO
CUDA_VISIBLE_DEVICES=3 $PY ultra/predict_coco.py \
  --weights work_dirs/ultra/y26x_1024_train/weights/best.pt \
  --split val --imgsz 1024 --eval
```

### 4. 测试集推理与集成（与 `submissions/final_submission.csv` 一致）

```bash
# Co-DINO trainval 第 12 轮：1333 与 1600，原图 + 水平翻转
CUDA_VISIBLE_DEVICES=0,1,2,3 \
  $PY mmdet/tools/tta_infer.py \
  --config mmdet/configs/codino_swinl_ego.py \
  --checkpoint work_dirs/codino_swinl_ego_trainval/epoch_12.pth \
  --ann /home/zhangyuming/Egoobjection/resource/data/test.json \
  --out-dir work_dirs/codino_swinl_ego_trainval/tta_test \
  --scales 1333x800,1600x960 --flip --gpus 4 --port 29951

# YOLO26x：1280 + Ultralytics TTA
export YOLO_AUTOINSTALL=False
CUDA_VISIBLE_DEVICES=0 $PY ultra/predict_coco.py \
  --weights work_dirs/ultra/y26x_1024_train/weights/best.pt \
  --split test --imgsz 1280 --augment --device 0 \
  --out work_dirs/ultra/y26x_1024_train/preds/test_1280_tta.json

# WBF。tta_infer 自己的 fused.json 用的是默认 IoU，不要用它。
$PY tools/ensemble.py \
  --preds work_dirs/codino_swinl_ego_trainval/tta_test/1333x800_orig.json \
          work_dirs/codino_swinl_ego_trainval/tta_test/1333x800_flip.json \
          work_dirs/codino_swinl_ego_trainval/tta_test/1600x960_orig.json \
          work_dirs/codino_swinl_ego_trainval/tta_test/1600x960_flip.json \
          work_dirs/ultra/y26x_1024_train/preds/test_1280_tta.json \
  --weights 1 1 1 1 3 \
  --method wbf --iou-thr 0.9 --conf-type max --max-per-image 100 \
  --out work_dirs/tune/test_fused_iou90.json

$PY tools/make_submission.py \
  --pred work_dirs/tune/test_fused_iou90.json \
  --out submissions/final_submission.csv \
  --test /home/zhangyuming/Egoobjection/resource/data/test.json
```

### 5. 资源与参数台账

```bash
CUDA_VISIBLE_DEVICES="" YOLO_AUTOINSTALL=False \
  $PY tools/count_params.py --out docs/model_params.json --md docs/model_params.md

CUDA_VISIBLE_DEVICES="" \
  $PY tools/resource_report.py \
    --mmdet codino_swinl_ego:3 --ultra y26x_1024_train:1 \
    --out docs/resource_usage.json --md docs/resource_usage.md
```

## 文档

| 文件 | 内容 |
|---|---|
| [`docs/technical_report.md`](docs/technical_report.md) | 技术报告（中文） |
| [`submissions/README.md`](submissions/README.md) | 最终提交文件与融合配方 |
| [`docs/data_check.md`](docs/data_check.md) | 数据审计 |
| [`docs/model_params.md`](docs/model_params.md) | 参数量 |
| [`docs/resource_usage.md`](docs/resource_usage.md) | GPU / 训练时间（训练结束后重生成） |

## 预测交换格式

所有管线统一输出 COCO results JSON：

```json
[{"image_id": 123, "category_id": 0, "bbox": [x, y, w, h], "score": 0.9}, ...]
```

原图像素坐标，每图最多 100 个检测，评测用低分阈值（~0.001）。
