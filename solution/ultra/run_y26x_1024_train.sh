#!/usr/bin/env bash
# yolo26x @ 1024, train-only, 80 epochs, single GPU (set CUDA_VISIBLE_DEVICES before calling).
# Does NOT wrap flock — the caller owns GPU scheduling.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${ROOT}/env/bin/python"
LOG_DIR="${ROOT}/logs"
mkdir -p "${LOG_DIR}"

: "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES to the assigned GPU (e.g. 3)}"

export YOLO_AUTOINSTALL=False
NAME="y26x_1024_train"
LOG="${LOG_DIR}/${NAME}_$(date +%Y%m%d_%H%M%S).log"

# close_mosaic: train.py default = min(10, epochs//6) -> 10 for 80 epochs
# patience=1000: effectively disable early stop (Ultralytics patience=0 stops immediately after best)
# save_period=5: keep epoch checkpoints for later selection
# cache=ram: ~18 GB for 1024 resized frames; needs a free GPU host with enough RAM
cd "${ROOT}"
echo "[run] logging to ${LOG}"
# "$@" after fixed overrides so e.g. --bench-iters 20 still works; trailing key=value can override
"${PY}" ultra/train.py \
  --model yolo26x.pt \
  --data ultra/data/ego_train.yaml \
  --imgsz 1024 \
  --epochs 80 \
  --batch 8 \
  --device 0 \
  --name "${NAME}" \
  patience=1000 \
  save_period=5 \
  cache=ram \
  workers=12 \
  "$@" \
  2>&1 | tee "${LOG}"
