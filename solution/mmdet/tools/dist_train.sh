#!/usr/bin/env bash
# Usage: CUDA_VISIBLE_DEVICES=0,1,2,3 bash dist_train.sh CONFIG NUM_GPUS [extra train.py args...]
set -euo pipefail

CONFIG=$1
GPUS=$2
PORT=${PORT:-29510}
SOLUTION_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
PY="$SOLUTION_DIR/env/bin/python"

# `solution/mmdet` must be importable so that `custom_imports=['projects.CO-DETR.codetr']` resolves.
export PYTHONPATH="$SOLUTION_DIR/mmdet:${PYTHONPATH:-}"

"$PY" -m torch.distributed.run \
    --nnodes=1 \
    --nproc_per_node="$GPUS" \
    --master_port="$PORT" \
    "$SOLUTION_DIR/mmdet/tools/train.py" \
    "$CONFIG" \
    --launcher pytorch "${@:3}"
