#!/usr/bin/env bash
set -euo pipefail

SDF_DIR="${SDF_DIR:-/opt/data/private/moffusion/autofusion/data/resolution_32}"
PROPERTY_CSV="${PROPERTY_CSV:-/opt/data/private/moffusion/data/properties/250k_pld_processed.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-./baseline_runs/3dcnn_pld_lcd}"
LIMIT="${LIMIT:-5000}"
EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LR="${LR:-1e-3}"
DEVICE="${DEVICE:-cuda}"

mkdir -p "${OUTPUT_DIR}"

python baselines/train_3dcnn_regressor.py \
  --sdf_dir "${SDF_DIR}" \
  --property_csv "${PROPERTY_CSV}" \
  --output_dir "${OUTPUT_DIR}" \
  --targets PLD LCD \
  --limit "${LIMIT}" \
  --epochs "${EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --device "${DEVICE}"
