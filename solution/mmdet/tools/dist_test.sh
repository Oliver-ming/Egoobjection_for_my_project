#!/usr/bin/env bash
# Usage: CUDA_VISIBLE_DEVICES=0 bash dist_test.sh CONFIG CHECKPOINT NUM_GPUS [extra test.py args...]
set -euo pipefail

CONFIG=$1
CHECKPOINT=$2
GPUS=$3
PORT=${PORT:-29520}
SOLUTION_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
PY="$SOLUTION_DIR/env/bin/python"

export PYTHONPATH="$SOLUTION_DIR/mmdet:${PYTHONPATH:-}"

"$PY" -m torch.distributed.run \
    --nnodes=1 \
    --nproc_per_node="$GPUS" \
    --master_port="$PORT" \
    "$SOLUTION_DIR/mmdet/tools/test.py" \
    "$CONFIG" \
    "$CHECKPOINT" \
    --launcher pytorch "${@:4}"
