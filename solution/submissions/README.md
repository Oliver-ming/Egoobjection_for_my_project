# 最终提交

`final_submission.csv` 是选定方案的测试集提交文件（2000 行，已通过 `tools/make_submission.py` 校验）。仓库根目录的 `submission.csv` 是同一份拷贝。

## 方案

在 **train+val** 上训完的 Co-DINO Swin-L（`work_dirs/codino_swinl_ego_trainval/epoch_12.pth`）做四路测试时增强：

- 尺度 `1333x800`、`1600x960`
- 每档含原图和水平翻转（翻转框用 `x' = W - x - w` 映回原图）

加上只在训练集上训的 YOLO26x（`work_dirs/ultra/y26x_1024_train/weights/best.pt`）在 `imgsz=1280`、`--augment` 下的测试集预测。

五路预测用 WBF 融合：`conf_type=max`，`iou_thr=0.90`，权重 `1 1 1 1 3`（四路 Co-DINO，最后一路是 YOLO），每图最多 100 框。

## 验证集依据

融合超参是在 **只训训练集** 的 Co-DINO（第 12 轮，验证集 mAP 0.884）上选的，不含验证集泄漏。同一配方在完整验证集上的 pycocotools mAP@0.5:0.95 为 **0.906**。提交文件把 Co-DINO 换成了 train+val 的第 12 轮权重；该权重在验证集上的 0.909 含泄漏，不能当作这份文件的分数。

融合命令（预测 json 已生成时）：

```bash
cd /home/zhangyuming/Egoobjection/solution
env/bin/python tools/ensemble.py \
  --preds work_dirs/codino_swinl_ego_trainval/tta_test/1333x800_orig.json \
          work_dirs/codino_swinl_ego_trainval/tta_test/1333x800_flip.json \
          work_dirs/codino_swinl_ego_trainval/tta_test/1600x960_orig.json \
          work_dirs/codino_swinl_ego_trainval/tta_test/1600x960_flip.json \
          work_dirs/ultra/y26x_1024_train/preds/test_1280_tta.json \
  --weights 1 1 1 1 3 \
  --method wbf --iou-thr 0.9 --conf-type max --max-per-image 100 \
  --out work_dirs/tune/test_fused_iou90.json

env/bin/python tools/make_submission.py \
  --pred work_dirs/tune/test_fused_iou90.json \
  --out submissions/final_submission.csv \
  --test /home/zhangyuming/Egoobjection/resource/data/test.json
```
