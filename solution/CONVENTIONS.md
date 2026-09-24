# Working conventions (read before touching anything)

## Hard rules
- Python: always `/home/zhangyuming/Egoobjection/solution/env/bin/python` (symlink to the shared uv env
  `/home/zhangyuming/Embodied AI EgoObject Detection/env_embody_ai`). Never create a new venv/conda env.
- Never `pip install` / `uv pip install` / upgrade anything in the env. Another project's training job is
  running from the same env. If a package is missing, stop and report it.
- Set `YOLO_AUTOINSTALL=False` for anything that imports ultralytics (it otherwise auto-installs packages).
- Never read or write anything under `/home/zhangyuming/Embodied AI EgoObject Detection/` except executing
  the env's python.
- GPUs: only 0-3 are ours. GPUs 4-7 belong to another job; never touch them. Always set
  `CUDA_VISIBLE_DEVICES` explicitly to the GPU(s) you were assigned. Kill every process you start before you
  finish; never leave background jobs running unless explicitly told to.
- Do not modify global config (e.g. `~/.config/Ultralytics/settings.json`); pass paths as arguments instead.
- Do not modify files under `/home/zhangyuming/Egoobjection/resource/` (official material). Copy if needed.

## Layout (all under /home/zhangyuming/Egoobjection/solution)
- `tools/`      shared tooling: data prep, COCO eval wrapper, WBF ensemble, submission builder/validator
- `ultra/`      Ultralytics pipeline (YOLO26 / YOLO11 / RT-DETR)
- `mmdet/`      MMDetection pipeline: `configs/`, `projects/` (third-party project code, e.g. CO-DETR)
- `weights/`    downloaded pretrained checkpoints (git-ignored)
- `work_dirs/`  all training / inference outputs, one sub-dir per experiment (git-ignored)
- `data/`       generated annotation files (trainval.json, ultralytics yaml, ...) (git-ignored)
- `logs/`       stdout logs of long runs (git-ignored)
- `docs/`       technical report and experiment log

## Data facts
- Data root: `/home/zhangyuming/Egoobjection/resource/data` (images/{train,val,test}, train/val/test.json,
  labels/{train,val} YOLO txt). `file_name` in json already includes the split prefix, e.g. `train/<id>.jpg`,
  and the file stem equals the COCO `image_id`.
- 10 classes, category_id 0..9 = pillow, box, book, bottle, chair, mug, door, shelf, plate, sofa.
- Train/val/test are split per frame from the same videos (val instances ~99.7% seen in train), so val mAP is
  a good proxy for test and the final model should be trained on train+val.
- Many boxes have slightly negative coords (clip to image), images are mostly 1920x1080, ~80% boxes are large.

## Prediction format (the single interchange format between all pipelines)
- COCO results json: flat list of `{"image_id": int, "category_id": int(0..9), "bbox": [x, y, w, h], "score": float}`
  in ORIGINAL image pixel coordinates, at most 100 detections per image (COCO maxDets=100), low score threshold
  (~0.001) for evaluation.
- Evaluate with `tools/coco_eval.py` (wraps pycocotools exactly like `resource/eval.py`).
- Submission: `tools/make_submission.py` (wraps official `resource/convert_to_submission.py` + validates).
