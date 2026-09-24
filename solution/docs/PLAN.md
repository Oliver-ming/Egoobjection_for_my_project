# 后续任务规划（交接用）

> 写于 2026-09-24 22:40。接手者先读本文件，再读 `solution/CONVENTIONS.md`（硬性规则）。
> 各流程细节不在此重复：`solution/README.md`、`ultra/README.md`、`mmdet/README_codino.md`、`mmdet/README_dino.md`、
> `docs/data_check.md`、`docs/technical_report.md`（草稿，结果处为 `TODO(结果)`）。代码历史见 `git log`（仓库根 `/home/zhangyuming/Egoobjection`）。

## 0. 用户要求与协作方式（必须遵守）

- 只用 uv 环境 `/home/zhangyuming/Embodied AI EgoObject Detection/env_embody_ai`（已软链为 `solution/env`），**不新建环境、不 pip 安装**（该环境同时被另一个训练任务使用）。
- `/home/zhangyuming/Embodied AI EgoObject Detection/` 目录除环境外**一律不读不写**；基线已由用户在那边完成，这里不做基线。
- GPU 只用 **0–3**；4–7 是用户另一个任务。
- 用户允许最多 **20 个 subagent** 并行；subagent **一律用 `grok-4.7-xhigh`**（用户要的 “fast” 版本不在可用列表里，已确认用这个）。
  写代码 / 审查 / 核验交给 subagent，主 agent 只做核心审核（坐标系、image_id/类别映射、提交格式、训练配置）和 GPU 调度。
  subagent 的 prompt 要自包含：先读 `CONVENTIONS.md`、明确分配的 GPU、禁止自行启动长时间训练、禁止 git 操作（由主 agent 提交）。
  多个 subagent 共用一张卡做短任务时用 `flock /home/zhangyuming/Egoobjection/solution/logs/gpu3.lock <cmd>` 串行化。
- 与用户用中文沟通。Kaggle 上传由用户手动完成（需填姓名学号），我们只产出并校验 `submission.csv`。
- `项目概览/项目要求.md` 在 git 里显示被修改，那是用户自己的改动，不要提交/还原。

## 1. 当前状态（22:40）

已完成并提交：公共工具（`tools/`：trainval 合并、数据检查、`coco_eval.py`、WBF `ensemble.py`、`make_submission.py`、参数量/资源统计）、
三条流程（Co-DINO Swin-L、DINO Swin-L、Ultralytics YOLO26x/YOLO11x/RT-DETR-X）均冒烟通过、TTA 工具 `mmdet/tools/tta_infer.py`、文档草稿。

正在运行（均为 nohup 后台进程，与会话无关）：

| 任务 | GPU | 启动 | 日志 / 输出 | 预计结束 |
|---|---|---|---|---|
| Co-DINO Swin-L，train-only，12 ep（step@10），1333 分辨率，3 卡×bs1，lr 5e-5 | 0,1,2 | 09-24 21:48 | `logs/codino_swinl_ego_train.log`；`work_dirs/codino_swinl_ego/`（`best_coco_bbox_mAP_epoch_N.pth`、`epoch_N.pth` 仅留最新；每轮 val 预测写 `preds/val.bbox.json`） | 09-25 ~10:30 |
| YOLO26x @1024，train-only，80 ep，bs8 | 3 | 09-24 22:02 | `logs/y26x_1024_train_*.log`；`work_dirs/ultra/y26x_1024_train/`（`weights/best.pt,last.pt,epoch*.pt`、`results.csv`） | 09-25 ~06:00 |

已知指标：YOLO26x 第 4 轮 val mAP50-95 = 0.654（**Ultralytics 自带指标**，与官方 pycocotools 略有差异）。
Co-DINO 冒烟 200 iter 在 200 张 val 子集上 0.258；正式训练第 1 轮 val 在 ~22:50 出。官方基线参考：Faster R-CNN ~0.759，Deformable DETR ~0.788。

监控：

```bash
cd /home/zhangyuming/Egoobjection/solution
rg "coco/bbox_mAP:" logs/codino_swinl_ego_train.log | tail          # Co-DINO 每轮 val
tail -c 600 logs/codino_swinl_ego_train.log                          # 当前 iter / eta
cut -d, -f1,8,9 work_dirs/ultra/y26x_1024_train/results.csv | tail  # YOLO mAP50 / mAP50-95
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
```

崩溃恢复（mmengine 自动从 work_dir 里最新 checkpoint 续训）：

```bash
CUDA_VISIBLE_DEVICES=0,1,2 PORT=29810 nohup bash mmdet/tools/dist_train.sh \
  mmdet/configs/codino_swinl_ego.py 3 --resume > logs/codino_swinl_ego_train_resume.log 2>&1 &
# YOLO: env/bin/python ultra/train.py ... resume=True（或直接 model=work_dirs/ultra/y26x_1024_train/weights/last.pt resume=True）
```

## 2. 关键决策（已定，除非有新证据不要推翻）

- **主模型 Co-DINO Swin-L**（Objects365→COCO 预训练，64.1 COCO AP）；冒烟时远强于 DINO（200 iter 0.258 vs 150 iter 0.037）。
  **YOLO26x** 作为便宜、差异化的集成成员；**DINO Swin-L 不训练**，仅在 GPU 空闲时作可选第三成员。
- **划分是按帧的**（test 273 个视频组中 272 个在 train 出现，val 99.7% 实例在 train 出现）⇒ val 是 test 的可靠代理；
  用 train-only 跑出的 val 曲线定轮数 / TTA / 集成参数，**最终模型在 train+val 上重训**，重训模型的 val 指标是泄漏的，不能用来选模型（用最后一轮）。
- 官方 YOLO 标签未裁剪（Ultralytics 会丢 14 张图）⇒ YOLO 一律用 `solution/data/yolo`（已裁剪）。EXIF 无问题。
- 预测交换格式：COCO results json，原图像素 xywh，`category_id` 0..9，每图 ≤100 框、低阈值（~0.001）。所有模型选择以 **pycocotools**（`tools/coco_eval.py`，与官方 `resource/eval.py` 逐位一致）为准。
- 同模型 TTA 融合用 WBF `conf_type=max`（DETR 输出大量低分重复框，`avg` 会稀释分数，冒烟时掉到 0.16）；多模型融合的 `conf_type` 要在 val 上网格搜索。

## 3. 时间线与任务

### 阶段 A（现在 → 09-25 ~06:00）：只监控

- 每轮检查 Co-DINO val mAP、loss 是否有 nan（前 ~100 iter 的 `grad_norm: nan` 是 AMP 校准，正常）。
- 可并行做（CPU，交给 subagent）：建 `docs/experiment_log.md` 的结果表并逐步填写；准备阶段 C/D 的一键脚本（见下）。

### 阶段 B（~06:00，YOLO train-only 结束，GPU 3 空出）

1. 官方指标评估（GPU 3，~15 min）：

   ```bash
   export YOLO_AUTOINSTALL=False; W=work_dirs/ultra/y26x_1024_train/weights
   CUDA_VISIBLE_DEVICES=3 env/bin/python ultra/predict_coco.py --weights $W/best.pt --split val --eval --out work_dirs/ultra/y26x_1024_train/preds/val_1024.json
   CUDA_VISIBLE_DEVICES=3 env/bin/python ultra/predict_coco.py --weights $W/best.pt --split val --eval --augment --out work_dirs/ultra/y26x_1024_train/preds/val_1024_tta.json
   CUDA_VISIBLE_DEVICES=3 env/bin/python ultra/predict_coco.py --weights $W/best.pt --split val --eval --imgsz 1280 --out work_dirs/ultra/y26x_1024_train/preds/val_1280.json
   ```
   同时看 `results.csv` 曲线：若最好轮明显早于 80（过拟合），trainval 重训轮数相应缩短。
2. 备份提交 #1（YOLO 单模，保证随时有可交文件）：`predict_coco.py --split test`（用 val 上最好的推理设置）→
   `env/bin/python tools/make_submission.py --pred <test.json> --out submissions/sub01_y26x_train.csv`。
3. 启动 YOLO trainval（~8 h，至 ~14:00）：
   `CUDA_VISIBLE_DEVICES=3 nohup bash ultra/run_y26x_1024_trainval.sh > logs/y26x_1024_trainval.nohup 2>&1 &`
   （如需改轮数：脚本末尾追加 `epochs=N` 覆盖。）提交时用 `last.pt`，不要用 `best.pt`（val 泄漏）。

### 阶段 C（~10:30，Co-DINO train-only 结束，GPU 0–2 空出）——核心

1. 从 `logs/codino_swinl_ego_train.log`（`coco/bbox_mAP`）记录每轮 val；确认最好轮与第 12 轮差距。
2. TTA 网格（GPU 0–2，`--gpus 3`，总时长控制在 ~45 min），用最好 checkpoint：

   ```bash
   CK=$(ls work_dirs/codino_swinl_ego/best_coco_bbox_mAP_epoch_*.pth)
   env CUDA_VISIBLE_DEVICES=0,1,2 env/bin/python mmdet/tools/tta_infer.py --config mmdet/configs/codino_swinl_ego.py \
     --checkpoint $CK --ann /home/zhangyuming/Egoobjection/resource/data/val.json \
     --out-dir work_dirs/codino_swinl_ego/tta_val --scales 1333x800,1600x960,1920x1152 --flip --gpus 3 --port 29941
   # 之后只重新融合不重跑推理：加 --skip-infer，改 --scales 子集 / --conf-type / --iou-thr / --method 比较
   ```
   单尺度 1333 无翻转就是基准；只保留在 val 上确实涨点的变体。
3. 用 train-only Co-DINO（选定 TTA）跑 test → 备份提交 #2（Co-DINO 单模）。
4. 集成调参（CPU，`tools/ensemble.py --gt val.json` 会直接打印输入/输出 AP）：Co-DINO(TTA) + YOLO(TTA) 的
   `--weights`（如 1:1、2:1、3:1）、`--iou-thr`（0.55–0.7）、`--conf-type`（avg / max / box_and_model_avg）、`--skip-box-thr`（0.001–0.05）。
   若集成不优于 Co-DINO 单模就不集成。用 train-only 的两个模型 test 预测按选定参数融合 → 备份提交 #3。
5. 启动 Co-DINO trainval（~12.5 h，3 卡，至 09-26 ~00:00；与 train-only 同 lr/batch，结论可直接迁移）：

   ```bash
   CUDA_VISIBLE_DEVICES=0,1,2 PORT=29811 nohup bash mmdet/tools/dist_train.sh \
     mmdet/configs/codino_swinl_ego_trainval.py 3 > logs/codino_swinl_ego_trainval_train.log 2>&1 &
   ```
   若 train-only 显示第 12 轮不是最好且差距明显，用 `--cfg-options train_cfg.max_epochs=N param_scheduler.1.milestones=[M]` 调整。
   时间紧就跳过 trainval，直接用备份提交 #3 作为最终结果。
6. GPU 3 在 YOLO trainval 结束（~14:00）后空闲：可选跑 DINO Swin-L（`mmdet/configs/dino_swinl_ego.py`，1 卡太慢——
   只有在确实空闲且时间充裕时，或改为评估 Co-DINO 1920 测试尺度）。不做也不影响主线。

### 阶段 D（09-26 ~00:00，最终提交）

1. Co-DINO trainval 用**最后一轮** checkpoint，按阶段 C 选定的 TTA 跑 test（可 4 卡：YOLO 已结束）；YOLO trainval `last.pt` 同设置跑 test。
2. 按阶段 C 选定的参数融合 → `tools/make_submission.py --pred <fused_test.json> --out submissions/final_submission.csv`（会自动校验：2000 行、id 与 test.json 一致、格式合法）。
3. 合理性检查：与备份 #3 比较每图框数 / 每类框数分布；把备份 #3 当伪 GT 算 mAP 应很高（明显偏低说明哪里错了）。
   把最终文件复制到 `/home/zhangyuming/Egoobjection/submission.csv`，告诉用户上传。

### 阶段 E（文档收尾，可交给 subagent，主 agent 审核）

- 填 `docs/technical_report.md` 全部 `TODO(结果)`：各模型 val mAP / AP50 / AP75 / APs/m/l、每类 AP、TTA 消融、集成消融、train vs trainval 说明。
- 资源：`env/bin/python tools/resource_report.py --mmdet codino_swinl_ego:3 codino_swinl_ego_trainval:3 --ultra y26x_1024_train:1 y26x_1024_trainval:1 --md docs/resource_usage.md --out docs/resource_usage.json`
  （另注明冒烟/测速的零散 GPU 时间）；参数量 `tools/count_params.py --ultra-ft <trainval last.pt> --md docs/model_params.md --out docs/model_params.json`；推理耗时写进报告。
- 更新 `solution/README.md` 的复现命令（与实际运行过的一致），`git commit`。

## 4. 实测性能（用于排期）

| 项目 | 数值 |
|---|---|
| Co-DINO 1333，3 卡 DDP | ~1.1–1.2 s/iter，3334 iter/epoch ≈ 1.05 h/epoch（含 val），9.3 GB/卡；推理 ~0.16 s/img/卡 |
| Co-DINO 1920 变体 | 13.4 GB，单卡 1.25–1.33 s/iter（~1.4× 时间），未完整验证 |
| DINO Swin-L 1333，bs2 | 20.5 GB，0.81 s/iter；4 卡 ~19 min/epoch |
| YOLO26x @1024 bs8 | ~5.9 min/epoch（含 val），20.8 GB |
| GPU 拓扑 | 0–3 各在不同 NUMA 节点、无 P2P，DDP 有 ~20% 通信开销 |

## 5. 坑（已踩过）

- Ultralytics 的 mAP ≠ pycocotools：决策一律用 `tools/coco_eval.py`。
- YOLO26 在 ultralytics 8.4.160 默认走 one-to-many 头 + NMS（`--nms false` 的端到端头低 ~4 AP）；RT-DETR 忽略 `--augment`。
- mmdet 手动传 `--work-dir` 时要同时覆盖 `test_evaluator.outfile_prefix`（`dist_infer.sh` 已处理）。
- Co-DETR 项目代码有一处 AMP 补丁（`codetr/co_dino_head.py::_outs_to_fp32`）和配置里的 `split_thr=100000`，原因见 `README_codino.md`，不要删。
- 端口：29810 已被 Co-DINO 训练占用；新任务用不同 `PORT`。
- Ultralytics 缓存被重定向到 `work_dirs/ultra/_cache`，不要让它往 `resource/` 写。
- Shell 工具偶尔会整体无响应（连 `echo` 都不返回），稍等重试即可；后台训练不受影响。
