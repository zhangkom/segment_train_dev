#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: bash scripts/test.sh <checkpoint> [config]"
  exit 1
fi

CHECKPOINT=$1
CONFIG=${2:-configs/live_portrait/deeplabv3plus_r50_4xb4-80k_live_portrait-512x512.py}

source .venv/bin/activate

python tools/test_mmseg.py --config "$CONFIG" --checkpoint "$CHECKPOINT"
