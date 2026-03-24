#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

SDF_DIR="${SDF_DIR:-/opt/data/private/moffusion/autofusion/data/resolution_32}"
PROPERTY_CSV="${PROPERTY_CSV:-/opt/data/private/moffusion/data/properties/250k_pld_processed.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/baseline_runs/3dcnn_pld_lcd}"
LIMIT="${LIMIT:-5000}"
EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LR="${LR:-1e-3}"
DEVICE="${DEVICE:-cuda}"

mkdir -p "${OUTPUT_DIR}"
LOG_FILE="${LOG_FILE:-${OUTPUT_DIR}/train.log}"

{
echo "[INFO] start_time=$(date '+%Y-%m-%d %H:%M:%S')"
echo "[INFO] script=run_3dcnn_baseline.sh"
echo "[INFO] sdf_dir=${SDF_DIR}"
echo "[INFO] property_csv=${PROPERTY_CSV}"
echo "[INFO] output_dir=${OUTPUT_DIR}"
echo "[INFO] limit=${LIMIT} epochs=${EPOCHS} batch_size=${BATCH_SIZE} lr=${LR} device=${DEVICE}"
python "${SCRIPT_DIR}/train_3dcnn_regressor.py" \
  --sdf_dir "${SDF_DIR}" \
  --property_csv "${PROPERTY_CSV}" \
  --output_dir "${OUTPUT_DIR}" \
  --targets PLD LCD \
  --limit "${LIMIT}" \
  --epochs "${EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --device "${DEVICE}"
echo "[INFO] end_time=$(date '+%Y-%m-%d %H:%M:%S')"
} 2>&1 | tee -a "${LOG_FILE}"
