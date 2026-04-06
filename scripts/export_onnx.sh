#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: bash scripts/export_onnx.sh <checkpoint> [output_onnx]"
  exit 1
fi

CHECKPOINT=$1
OUTPUT=${2:-outputs/live_portrait_deeplabv3plus.onnx}
CONFIG=${CONFIG:-configs/live_portrait/deeplabv3plus_r50_4xb4-80k_live_portrait-512x512.py}

mkdir -p "$(dirname "$OUTPUT")"

source .venv/bin/activate

python tools/export_onnx.py \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --output "$OUTPUT" \
  --shape 512 512
