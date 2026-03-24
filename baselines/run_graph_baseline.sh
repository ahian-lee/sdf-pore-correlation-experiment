#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

CIF_DIR="${CIF_DIR:-/opt/data/private/moffusion/250k_cif}"
PROPERTY_CSV="${PROPERTY_CSV:-/opt/data/private/moffusion/data/properties/250k_pld_processed.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/baseline_runs/graph_pld_lcd}"
LIMIT="${LIMIT:-3000}"
EPOCHS="${EPOCHS:-40}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LR="${LR:-1e-3}"
DEVICE="${DEVICE:-cuda}"
CUTOFF="${CUTOFF:-4.5}"

mkdir -p "${OUTPUT_DIR}"
LOG_FILE="${LOG_FILE:-${OUTPUT_DIR}/train.log}"

{
echo "[INFO] start_time=$(date '+%Y-%m-%d %H:%M:%S')"
echo "[INFO] script=run_graph_baseline.sh"
echo "[INFO] cif_dir=${CIF_DIR}"
echo "[INFO] property_csv=${PROPERTY_CSV}"
echo "[INFO] output_dir=${OUTPUT_DIR}"
echo "[INFO] limit=${LIMIT} epochs=${EPOCHS} batch_size=${BATCH_SIZE} lr=${LR} cutoff=${CUTOFF} device=${DEVICE}"
python "${SCRIPT_DIR}/train_graph_mpnn_regressor.py" \
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
echo "[INFO] end_time=$(date '+%Y-%m-%d %H:%M:%S')"
} 2>&1 | tee -a "${LOG_FILE}"
