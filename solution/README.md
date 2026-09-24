# EgoObjects 检测竞赛 · solution

Egocentric 10 类目标检测（COCO mAP@0.5:0.95）。共享约定见 [`CONVENTIONS.md`](CONVENTIONS.md)。

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
- 始终显式设置 `CUDA_VISIBLE_DEVICES`（本项目仅使用 GPU 0–3）。

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

### 4. 测试集推理（含 TTA）

```bash
# 单尺度 test
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29920 bash mmdet/tools/dist_infer.sh \
  mmdet/configs/codino_swinl_ego.py CKPT 4 test \
  work_dirs/codino_swinl_ego/preds/test

# TTA：多尺度 + flip → WBF
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29931 \
  $PY mmdet/tools/tta_infer.py \
  --config mmdet/configs/codino_swinl_ego.py \
  --checkpoint CKPT \
  --ann ../resource/data/test.json \
  --out-dir work_dirs/codino_swinl_ego/tta_test \
  --scales 1333x800,1600x960,1920x1152 --flip --gpus 4 \
  --method wbf --iou-thr 0.6 --conf-type max

# YOLO test（可选 --augment）
CUDA_VISIBLE_DEVICES=3 $PY ultra/predict_coco.py \
  --weights work_dirs/ultra/y26x_1024_train/weights/best.pt \
  --split test --imgsz 1024 --out work_dirs/ultra/preds/y26x_test.json
```

### 5. 集成

```bash
$PY tools/ensemble.py \
  --preds work_dirs/codino_swinl_ego/tta_test/fused.json \
           work_dirs/dino_swinl_ego/test.bbox.json \
           work_dirs/ultra/preds/y26x_test.json \
  --weights 2 1 1 \
  --method wbf --iou-thr 0.6 --conf-type avg \
  --out work_dirs/ensemble/test_fused.json
```

### 6. 生成并校验提交

```bash
$PY tools/make_submission.py \
  --pred work_dirs/ensemble/test_fused.json \
  --out work_dirs/ensemble/submission.csv
# 或仅校验：
$PY tools/make_submission.py --validate-only work_dirs/ensemble/submission.csv
```

### 7. 资源与参数台账

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
| [`docs/technical_report.md`](docs/technical_report.md) | 技术报告草稿（中文，结果处为 `TODO(结果)`） |
| [`docs/data_check.md`](docs/data_check.md) | 数据审计 |
| [`docs/model_params.md`](docs/model_params.md) | 参数量 |
| [`docs/resource_usage.md`](docs/resource_usage.md) | GPU / 训练时间（训练结束后重生成） |

## 预测交换格式

所有管线统一输出 COCO results JSON：

```json
[{"image_id": 123, "category_id": 0, "bbox": [x, y, w, h], "score": 0.9}, ...]
```

原图像素坐标，每图最多 100 个检测，评测用低分阈值（~0.001）。
