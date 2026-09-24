#!/usr/bin/env bash
# yolo26x @ 1024, train+val, 80 epochs, single GPU (set CUDA_VISIBLE_DEVICES before calling).
# Val metrics are optimistic (val is in the training set) — do not use them for model selection.
# Does NOT wrap flock — the caller owns GPU scheduling.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${ROOT}/env/bin/python"
LOG_DIR="${ROOT}/logs"
mkdir -p "${LOG_DIR}"

: "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES to the assigned GPU (e.g. 3)}"

export YOLO_AUTOINSTALL=False
NAME="y26x_1024_trainval"
LOG="${LOG_DIR}/${NAME}_$(date +%Y%m%d_%H%M%S).log"

cd "${ROOT}"
echo "[run] logging to ${LOG}"
"${PY}" ultra/train.py \
  --model yolo26x.pt \
  --data ultra/data/ego_trainval.yaml \
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
