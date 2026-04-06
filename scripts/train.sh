#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-configs/live_portrait/deeplabv3plus_r50_4xb4-80k_live_portrait-512x512.py}
WORK_DIR=${WORK_DIR:-}

source .venv/bin/activate

python tools/check_dataset.py --root data/live_portrait

python tools/train_mmseg.py --config "$CONFIG" ${WORK_DIR:+--work-dir "$WORK_DIR"}
