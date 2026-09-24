# 技术报告（草稿）

> EgoObjects 子集 · 10 类 egocentric 目标检测 · 主指标 COCO mAP@0.5:0.95  
> 本文为可填结果的结构化草稿；凡需等训练/评测完成再写入的数字，均以 `TODO(结果)` 标明，**不臆造成绩**。

---

## 1. 任务与评测

- **任务**：在第一人称日常场景中检测 10 类物体（pillow, box, book, bottle, chair, mug, door, shelf, plate, sofa），`category_id` 为 0..9。
- **数据**：COCO 风格标注；train / val / test = 10000 / 1000 / 2000 张图（见 `resource/data/`）。
- **主指标**：COCO **mAP@0.5:0.95**（IoU 0.50–0.95，步长 0.05）；同时报告 AP50 / AP75、按尺度 APs/m/l、AR。
- **提交**：`submission.csv`（2000 行，每行一张测试图的 JSON 检测列表）；由官方 `resource/convert_to_submission.py` 生成，本仓库封装为 `tools/make_submission.py`（含格式校验）。
- **本地评测**：`tools/coco_eval.py`，与 `resource/eval.py` 同一 pycocotools 路径，保证与官方一致。

竞赛评分更看重**工程完成度与文档**，而非单纯刷分。

---

## 2. 数据分析

数据审计详见 `docs/data_check.md`（由 `tools/check_data.py` 生成）。要点如下。

### 2.1 划分与泄漏（帧级切分）

图像元数据含 `group_id` / `video_id` / `frame_id`。train / val / test 按**帧**从同一批视频中切分，而非按视频隔离：

| 统计 | 数值 |
|---|---|
| train / val / test 视频组数 | 300 / 245 / 273 |
| test 组与 train 重叠 | **272 / 273** |
| val 组与 train 重叠 | 244 / 245 |
| val 实例的 `(group_id, category_id)` 已在 train 出现 | **4184 / 4195 ≈ 99.7%** |

因此：**验证集是测试集的良好代理**；最终提交模型应在 **train+val** 上重训（`solution/data/trainval.json`），选模仍以 train-only 上的 val mAP 为准。

### 2.2 框尺度与图像尺寸

- 训练集约 **79.5% large**、19.2% medium、**1.3% small**（COCO 面积桶）；验证集分布接近。
- 主流分辨率 1920×1080，其余多为 720p / 方图 / 少量 4K。
- 约 15% 的 GT 框越界（负坐标或超出宽高）；官方 JSON **未裁剪**。评分仍相对官方 GT；训练时 mmdet `CocoDataset` 会丢掉无有效面积的框，YOLO 侧需自行裁剪。

### 2.3 标签问题与修复

- 官方 `resource/data/labels` 的 YOLO txt **未裁剪**到 [0,1]，与 COCO 原始框一致；Ultralytics 会把坐标越界的整图标为 corrupt（train 12 / val 2），并有 1 个零面积框。
- **修复**：`tools/prepare_data.py --yolo-dir solution/data/yolo` 按图像裁剪后重写标签；mmdet 直接读 COCO JSON。
- EXIF orientation：13000 张图均无异常旋转，解码尺寸与 JSON 一致。

### 2.4 类别分布（train / val 实例数）

| id | class | train | val |
|---:|---|---:|---:|
| 0 | pillow | 7138 | 662 |
| 1 | box | 9359 | 917 |
| 2 | book | 2967 | 287 |
| 3 | bottle | 6188 | 699 |
| 4 | chair | 5350 | 485 |
| 5 | mug | 3559 | 356 |
| 6 | door | 1407 | 154 |
| 7 | shelf | 1676 | 174 |
| 8 | plate | 2738 | 288 |
| 9 | sofa | 2111 | 173 |

---

## 3. 方法总览

| 角色 | 模型 | 预训练 | 管线 |
|---|---|---|---|
| **主模型** | Co-DINO 5-scale Swin-L | Objects365 → COCO | MMDetection `mmdet/` |
| 次要 / 集成 | DINO 5-scale Swin-L | COCO | MMDetection |
| 快速多样成员 | YOLO26x | COCO（Ultralytics） | `ultra/` |
| TTA | 多尺度 + 水平翻转 | — | `mmdet/tools/tta_infer.py` |
| 融合 | WBF（默认 `conf_type=max` 用于同模型 TTA） | — | `tools/ensemble.py` |

预测统一为 COCO results JSON（原图像素 xywh，`category_id` 0..9，每图 ≤100 框，低阈值 ~0.001），再经 ensemble / `make_submission.py`。

---

## 4. 架构选择与理由

1. **DETR 族 + 大规模预训练（Co-DINO / DINO）**  
   - 第一人称数据视角非常规、运动模糊、尺度跨度大，但 **~80% 为大目标**，更吃表征与预训练，而非极致小目标 detector。  
   - Co-DINO 带协同混合训练头（query + ATSS + RPN/RoI），预训练强（O365→COCO）；**推理只用 query/DETR 头**（`eval_module='detr'`），辅助头仅训练期存在（见 `docs/model_params.md`）。  
   - DINO Swin-L 作为同族、不同预训练与损失细节的成员，便于集成时提供相关但非同质的误差。

2. **YOLO26x 作为快速多样成员**  
   - 吞吐高（单卡 ~5–6 min/epoch @1024）、实现路径独立（卷积 NMS 头 vs Transformer），与 DETR 错误模式互补。  
   - 使用裁剪后的 YOLO 标签；训练显式 AdamW（避免 `optimizer=auto` 中途切 MuSGD）。

3. **TTA + WBF**  
   - 同模型多尺度/翻转用 WBF `conf_type=max`，避免 `avg` 对单视角簇过度降分（smoke 上已观察到该现象）。  
   - Co-DINO/DINO 头只反解 `scale_factor`、**不解 flip**，`tta_infer.py` 需后处理 `x' = W - x - w`。

---

## 5. 训练细节与超参数

### 5.1 Co-DINO Swin-L（`mmdet/configs/codino_swinl_ego.py`）

| 项 | 值 |
|---|---|
| 分辨率 | 训练短边 480..800（步长 32），长边 ≤1333；测试 (1333, 800) |
| 增强 | RandomFlip 0.5；RandomChoice（多尺度 resize / crop 分支） |
| 优化器 | AdamW，`lr=5e-5`（由原方 1e-4@bs16 **平方根**缩放到 bs4），backbone ×0.1，wd 1e-4 |
| 梯度裁剪 | max_norm 0.1 |
| AMP | `AmpOptimWrapper`，dynamic loss scale |
| 调度 | 12 epoch；Linear warmup 500 iter；MultiStepLR ×0.1 @ epoch 10 |
| Batch | 1 / GPU（当前 live：3 GPU → 有效 batch 3） |
| Checkpoint | 每 epoch；`save_best='coco/bbox_mAP'`；`max_keep_ckpts=1` |
| 预训练 | `co_dino_5scale_swin_large_16e_o365tococo-*.pth`；分类头 80/81→10/11 跳过加载 |

**lr 缩放理由**：Adam 类优化器常用 √规则；线性缩到 2.5e-5 会在 12 epoch 内欠训随机初始化的分类层。

最终模型：`codino_swinl_ego_trainval.py`（`trainval.json`，关闭依赖 val 的 `save_best` 语义）。

### 5.2 DINO Swin-L（`mmdet/configs/dino_swinl_ego.py`）

| 项 | 值 |
|---|---|
| 分辨率 / 增强 | 与 COCO 配方一致：短边 480–800，长边 ≤1333；测试 (1333, 800) |
| Batch | 2 / GPU |
| 优化器 | AdamW `lr=5e-5`（线性：1e-4@bs16 → 4×2=8），backbone ×0.1，wd 1e-4，clip 0.1 |
| AMP | 是；Swin `with_cp=True` |
| 调度 | 12 epoch；warmup 500 iter（`start_factor=0.1`）；MultiStepLR ×0.1 @ **[8, 11]** |
| 预训练 | `dino-5scale_swin-l_8xb2-36e_coco-*.pth` |

### 5.3 YOLO26x（`ultra/run_y26x_1024_train.sh`）

| 项 | 值 |
|---|---|
| imgsz | 1024 |
| epochs | 80；`close_mosaic=min(10, epochs//6)=10` |
| batch | 8（单卡） |
| 优化器 | AdamW，`lr0=1e-3`，`lrf=0.01`，wd 5e-4；`cos_lr`；warmup 2 epoch |
| AMP / cache | amp；`cache=ram`；workers=12 |
| 其它 | `patience=1000`（实质关闭早停）；`save_period=5` |

---

## 6. 工程实现

- **数据准备**：`tools/prepare_data.py` → `trainval.json`、`val_mini.json`、可选裁剪 YOLO 树。  
- **一致性评测**：`tools/coco_eval.py` ↔ `resource/eval.py`；smoke 上 Co-DINO / DINO 的 CocoMetric 与离线脚本对齐。  
- **提交校验**：`tools/make_submission.py`（过滤非法类别/越界框、截断 top-100、调用官方转换器、整表校验）。  
- **Co-DETR AMP 补丁**（见 `mmdet/README_codino.md`）：  
  1. `co_dino_head.py`：loss 前把 decoder/aux outs 转 fp32（无 GT 图在 AMP 下空 `bbox_overlaps` 会混入 fp16）。  
  2. `rpn_proposal.nms.split_thr=100000`：避免 mmcv `batched_nms` 在 AMP 下 fp32→fp16 index-put 崩溃。  
- **缓存隔离**：Ultralytics 缓存写到 `work_dirs/ultra/_cache/`，不污染 `resource/`。  
- **参数量统计**：`tools/count_params.py` → `docs/model_params.md`。  
- **资源台账**：`tools/resource_report.py` → `docs/resource_usage.md`（训练结束后重跑）。

---

## 7. 实验结果

> 下列表格待训练与评测完成后填写。**禁止编造数字。**

### 7.1 单模型验证集（train-only 训练）

| 模型 | mAP | AP50 | AP75 | APs | APm | APl |
|---|---:|---:|---:|---:|---:|---:|
| Co-DINO Swin-L | TODO(结果) | TODO(结果) | TODO(结果) | TODO(结果) | TODO(结果) | TODO(结果) |
| DINO Swin-L | TODO(结果) | TODO(结果) | TODO(结果) | TODO(结果) | TODO(结果) | TODO(结果) |
| YOLO26x @1024 | TODO(结果) | TODO(结果) | TODO(结果) | TODO(结果) | TODO(结果) | TODO(结果) |

### 7.2 按类 AP（主模型）

| class | AP |
|---|---:|
| pillow … sofa | TODO(结果) |

### 7.3 TTA 消融（Co-DINO）

| 设置 | mAP |
|---|---:|
| 单尺度 1333×800 | TODO(结果) |
| + flip | TODO(结果) |
| 多尺度 + flip + WBF(max) | TODO(结果) |

### 7.4 集成消融

| 成员 | mAP |
|---|---:|
| Co-DINO only | TODO(结果) |
| + DINO | TODO(结果) |
| + YOLO26x | TODO(结果) |
| 全量 WBF | TODO(结果) |

### 7.5 train vs trainval

| 模型 | train-only best val | trainval 同 epoch 协议 | 备注 |
|---|---:|---:|---|
| Co-DINO | TODO(结果) | TODO(结果) | trainval 的 val mAP 泄漏，仅作参考 |
| YOLO26x | TODO(结果) | TODO(结果) | 同上 |

基线参考（组委会，非本方法）：Faster R-CNN ~0.759，Deformable DETR ~0.788。

---

## 8. 资源说明

参数量见 `docs/model_params.md`（摘要）：

| 模型 | 总参数 (M) | 备注 |
|---|---:|---|
| Co-DINO Swin-L（训练完整） | 234.808 | 推理（backbone+neck+query）**219.043**；辅助头 15.765 |
| DINO Swin-L | 218.189 | |
| YOLO26x（nc=10） | 58.832 | COCO 80 类权重 58.993 |
| YOLO11x（nc=10） | 56.885 | |
| RT-DETR-X（nc=10） | 67.324 | |

训练资源（**部分完成时快照**，结束前用 `tools/resource_report.py` 重生成 `docs/resource_usage.md`）：

| run | GPUs | wall / GPU-h | 进度 | peak mem/GPU | best val mAP |
|---|---:|---|---|---|---|
| codino_swinl_ego | 3 | TODO(结果) | TODO(结果) | ~9.3 GiB（日志） | TODO(结果) |
| y26x_1024_train | 1 | TODO(结果) | TODO(结果) | ~20.9 GiB | TODO(结果) |
| dino_swinl_ego | TODO(结果) | TODO(结果) | TODO(结果) | TODO(结果) | TODO(结果) |

硬件：NVIDIA GeForce RTX 4090（本机 GPUs 0–3）。

---

## 9. 复现步骤

见仓库根目录 `solution/README.md`（prepare → train → val → TTA → ensemble → submission）。关键命令与脚本 argparse 一致，默认 Python：

`/home/zhangyuming/Egoobjection/solution/env/bin/python`

---

## 10. 局限与改进方向

- 帧级划分导致 train/val/test 强相关；公开榜乐观，真实跨视频泛化未测。  
- 小目标极少（~1.3%），高分辨率（1920 长边）收益不确定，可作为可选集成成员。  
- 越界 / 全图外 GT 会轻微召回上限；未改官方 GT。  
- Co-DINO 辅助头增加训练算力但不参与推理；可研究推理期蒸馏或关掉部分 aux。  
- 集成权重、TTA 尺度集合、YOLO26 one-to-many vs one-to-one 头仍待系统消融（`TODO(结果)`）。  
- 单卡 YOLO DDP 未在本环境充分验证。

---

*草稿版本与代码布局同步于 `solution/`；结果表在实验结束后一次性回填。*
