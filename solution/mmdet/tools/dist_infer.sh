#!/usr/bin/env bash
# Run a config + checkpoint on any COCO json and write <OUT_PREFIX>.bbox.json (COCO results format,
# original image coordinates, category_id = json category ids).
# Usage: CUDA_VISIBLE_DEVICES=0,1,2,3 bash dist_infer.sh CONFIG CHECKPOINT NUM_GPUS ANN OUT_PREFIX [extra test.py args...]
#   ANN = val | test | /path/to/coco.json
#   If the json has annotations, mmdet's CocoMetric also prints COCO mAP; otherwise format_only.
set -euo pipefail

CONFIG=$1
CHECKPOINT=$2
GPUS=$3
ANN=$4
OUT_PREFIX=$5
SOLUTION_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
PY="$SOLUTION_DIR/env/bin/python"
DATA_DIR=/home/zhangyuming/Egoobjection/resource/data

case "$ANN" in
    val|test) ANN="$DATA_DIR/$ANN.json" ;;
esac
if "$PY" -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get('annotations') else 1)" "$ANN"; then
    FORMAT_ONLY=False
else
    FORMAT_ONLY=True
fi
mkdir -p "$(dirname "$OUT_PREFIX")"

bash "$SOLUTION_DIR/mmdet/tools/dist_test.sh" "$CONFIG" "$CHECKPOINT" "$GPUS" \
    --work-dir "$(dirname "$OUT_PREFIX")" \
    --cfg-options \
    test_dataloader.dataset.ann_file="$ANN" \
    test_evaluator.ann_file="$ANN" \
    test_evaluator.format_only="$FORMAT_ONLY" \
    test_evaluator.outfile_prefix="$OUT_PREFIX" \
    "${@:6}"
