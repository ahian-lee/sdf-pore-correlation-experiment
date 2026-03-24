#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

LIMIT=90130 \
EPOCHS=120 \
BATCH_SIZE=12 \
LR=5e-4 \
PATIENCE=20 \
DEVICE="${DEVICE:-cuda}" \
OUTPUT_DIR="${OUTPUT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)/baseline_runs/3dcnn_pld_lcd_v2_full}" \
bash "${SCRIPT_DIR}/run_3dcnn_baseline_v2.sh"
