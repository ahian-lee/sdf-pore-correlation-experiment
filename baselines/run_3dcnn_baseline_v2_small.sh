#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

LIMIT=5000 \
EPOCHS=80 \
BATCH_SIZE=16 \
LR=8e-4 \
PATIENCE=12 \
DEVICE="${DEVICE:-cuda}" \
OUTPUT_DIR="${OUTPUT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)/baseline_runs/3dcnn_pld_lcd_v2_small}" \
bash "${SCRIPT_DIR}/run_3dcnn_baseline_v2.sh"
