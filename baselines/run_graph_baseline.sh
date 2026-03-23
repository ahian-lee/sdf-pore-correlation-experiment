#!/usr/bin/env bash
set -euo pipefail

CIF_DIR="${CIF_DIR:-/opt/data/private/moffusion/250k_cif}"
PROPERTY_CSV="${PROPERTY_CSV:-/opt/data/private/moffusion/data/properties/250k_pld_processed.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-./baseline_runs/graph_pld_lcd}"
LIMIT="${LIMIT:-3000}"
EPOCHS="${EPOCHS:-40}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LR="${LR:-1e-3}"
DEVICE="${DEVICE:-cuda}"
CUTOFF="${CUTOFF:-4.5}"

mkdir -p "${OUTPUT_DIR}"

python baselines/train_graph_mpnn_regressor.py \
  --cif_dir "${CIF_DIR}" \
  --property_csv "${PROPERTY_CSV}" \
  --output_dir "${OUTPUT_DIR}" \
  --targets PLD LCD \
  --limit "${LIMIT}" \
  --epochs "${EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --cutoff "${CUTOFF}" \
  --device "${DEVICE}"
