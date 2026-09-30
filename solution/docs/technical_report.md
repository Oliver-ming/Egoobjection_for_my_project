# 技术报告（草稿）

> EgoObjects 子集 · 10 类 egocentric 目标检测 · 主指标 COCO mAP@0.5:0.95  
> 分数均为 pycocotools 的 COCO mAP@0.5:0.95，与 `resource/eval.py` 一致。带「泄漏」的数字是模型在训练时见过验证集图像，只作记录，不用于选模。

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
| 多样成员 | YOLO26x | COCO（Ultralytics） | `ultra/` |
| TTA | 1333 与 1600，各加水平翻转 | — | `mmdet/tools/tta_infer.py` |
| 融合 | WBF，`conf_type=max`，IoU 0.90，权重 1,1,1,1,3 | — | `tools/ensemble.py` |

DINO Swin-L 的配置在 `mmdet/configs/dino_swinl_ego.py`，只做了冒烟测试，没有进入最终提交。

预测统一为 COCO results JSON（原图像素 xywh，`category_id` 0..9，每图 ≤100 框，低阈值 ~0.001），再经 ensemble / `make_submission.py`。

---

## 4. 架构选择与理由

1. **DETR 族 + 大规模预训练（Co-DINO）**  
   - 第一人称数据视角非常规、运动模糊、尺度跨度大，但 **~80% 为大目标**，更吃表征与预训练，而非极致小目标 detector。  
   - Co-DINO 带协同混合训练头（query + ATSS + RPN/RoI），预训练强（O365→COCO）；**推理只用 query/DETR 头**（`eval_module='detr'`），辅助头仅训练期存在（见 `docs/model_params.md`）。  
2. **YOLO26x 作为多样成员**  
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

最终提交用的融合，在只训训练集的模型上评验证集为 **0.906**。组委会基线：Faster R-CNN 约 0.759，Deformable DETR 约 0.788。DINO Swin-L 没有做完整训练，不在下表里。

### 7.1 单模型验证集（只训训练集）

| 模型 | mAP | AP50 | AP75 | APs | APm | APl |
|---|---:|---:|---:|---:|---:|---:|
| Co-DINO Swin-L，第 12 轮，1333 | 0.884 | 0.979 | 0.952 | 0.801 | 0.718 | 0.902 |
| YOLO26x @1024 | 0.860 | 0.975 | 0.926 | 0.749 | 0.729 | 0.875 |
| YOLO26x @1280 + TTA | 0.887 | 0.981 | 0.949 | — | — | — |

Co-DINO 验证曲线（1333）：第 1 / 4 / 8 / 12 轮为 0.739 / 0.854 / 0.876 / 0.884。第 5 轮起改为每卡 2 张图。YOLO 的 Ultralytics 自带指标在第 67 轮最高为 0.892，第 80 轮 0.891；上表是 pycocotools。1280 TTA 没有单独记下按尺度的 AP。

### 7.2 按类 AP（Co-DINO 第 12 轮，只训训练集）

| pillow | box | book | bottle | chair | mug | door | shelf | plate | sofa |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.903 | 0.858 | 0.884 | 0.835 | 0.869 | 0.899 | 0.894 | 0.890 | 0.870 | 0.934 |

### 7.3 TTA 消融（Co-DINO 第 12 轮，只训训练集）

| 设置 | mAP |
|---|---:|
| 1333×800 | 0.884 |
| 1333×800 水平翻转 | 0.882 |
| 1600×960 / 其翻转 | 0.881 / 0.879 |
| 1920×1088 / 其翻转 | 0.877 / 0.874 |
| 1333+1600，原图+翻转，WBF max、IoU 0.75 | 0.886 |

单独提高测试分辨率没有超过 1333。1920 再融进后面的集成会把 0.902 拉到 0.901。

### 7.4 集成消融（只训训练集的权重，WBF，`conf_type=max`）

| 成员 | IoU | mAP |
|---|---:|---:|
| Co-DINO 1333 原图 + 翻转 | 0.75 | 0.884 |
| 再加 YOLO26x 1280 TTA，权重 1:1:2 | 0.75 | 0.901 |
| 1333+1600 四路 + YOLO，权重 1:1:1:1:3 | 0.75 | 0.902 |
| 同上 | 0.90 | **0.906** |

Soft-NMS、按图预过滤、以及 YOLO 的 1152/1408 或 multi-label，都没有超过 0.906。提交文件使用同一融合，但 Co-DINO 换成 train+val 的第 12 轮。

### 7.5 train 与 trainval

| 模型 | 只训训练集 | train+val | 备注 |
|---|---:|---:|---|
| Co-DINO 第 12 轮 | 0.884 | 0.909 | 0.909 含泄漏。第 4 / 8 轮泄漏分数为 0.868 / 0.898 |
| YOLO26x | 0.860（1024，pycocotools） | 未训练 | Ultralytics 指标第 80 轮 0.891 |

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

训练资源由日志时间戳汇总（不是 `resource_report.py` 的自动表；该脚本的早期快照已过时）：

| run | GPUs | 墙钟 | GPU·时 | 峰值显存/卡 | 验证 mAP |
|---|---:|---|---:|---|---|
| codino_swinl_ego | 3 | 约 11.2 h（含中途改为每卡 2 张图） | 约 34 | 每卡 1 张图时 9.3 GiB；每卡 2 张图时约 19 GiB | 0.884 |
| codino_swinl_ego_trainval | 4 | 约 7.9 h | 约 32 | 约 23 GiB（余量很小） | 0.909（泄漏） |
| y26x_1024_train | 1 | 7.7 h | 7.7 | 约 21–23 GiB | 0.860（pycocotools，1024） |

硬件：NVIDIA GeForce RTX 4090 24 GiB。正式训练用过 GPU 0–7；0–3 与 4–7 都在不同 NUMA 节点上，卡间无直连。

---

## 9. 复现步骤

见仓库根目录 `solution/README.md`（prepare → train → val → TTA → ensemble → submission）。关键命令与脚本 argparse 一致，默认 Python：

`/home/zhangyuming/Egoobjection/solution/env/bin/python`

---

## 10. 局限与改进方向

- 帧级划分导致 train/val/test 强相关；验证集可以代表测试集，但不能代表换一批视频后的泛化。
- 小目标极少（约 1.3%）。把测试分辨率加到 1920 低于 1333，没有放进提交。
- 越界和完全在图外的标注会轻微压低召回上限；评测仍用官方标注。
- Co-DINO 的辅助头只在训练时使用。YOLO26 的端到端头在短程试验里比默认的 one-to-many 头低大约 4 个点，提交用默认头。
- 1600 分辨率上从第 12 轮再微调 5 轮，单模型验证集只到 0.888，没有替换提交里的权重。

---

*与 `submissions/final_submission.csv` 对应的结果写于 2026-09-30。*
