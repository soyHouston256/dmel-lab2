#!/usr/bin/env bash
#SBATCH --job-name=dmel_precompute
#SBATCH --partition=standard
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --open-mode=append
#SBATCH --requeue
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Enviar desde raíz DMEL}"
source "${VENV:-$PWD/.venv}/bin/activate"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
DATA="${DATA:-dataset/train.h5}"
CONFIG="${CONFIG:-configs/decomposition.json}"
# Por defecto: medir 32 ventanas antes de dimensionar todo el dataset.
LIMIT="${LIMIT:-32}"
OUT="${OUT:-cache/pilot-${SLURM_JOB_ID}.h5}"
MAX_SECONDS="${MAX_SECONDS:-0}"
AUTO_REQUEUE="${AUTO_REQUEUE:-0}"
MAX_RESTARTS="${MAX_RESTARTS:-4}"
RESTARTS="${SLURM_RESTART_COUNT:-0}"
[[ "$MAX_SECONDS" =~ ^[0-9]+$ && "$MAX_RESTARTS" =~ ^[0-9]+$ && "$RESTARTS" =~ ^[0-9]+$ ]] || { echo 'Presupuesto/reinicios deben ser enteros >=0' >&2; exit 2; }
[[ "$AUTO_REQUEUE" == 0 || "$AUTO_REQUEUE" == 1 ]] || { echo 'AUTO_REQUEUE debe ser 0 o 1' >&2; exit 2; }
[[ "$AUTO_REQUEUE" == 0 || "$MAX_SECONDS" -gt 0 ]] || { echo 'AUTO_REQUEUE exige MAX_SECONDS>0 y margen antes del walltime' >&2; exit 2; }
# Si Slurm informa su deadline, conservar al menos 10 minutos para el cierre.
if [[ "$MAX_SECONDS" -gt 0 && "${SLURM_JOB_END_TIME:-}" =~ ^[0-9]+$ ]]; then
    available=$((SLURM_JOB_END_TIME - $(date +%s) - 600))
    [[ "$available" -gt 0 ]] || { echo 'No queda margen suficiente para este ciclo' >&2; exit 2; }
    if [[ "$MAX_SECONDS" -gt "$available" ]]; then MAX_SECONDS="$available"; fi
fi
echo "date=$(date -Iseconds) job=$SLURM_JOB_ID node=$(hostname) cpus=$SLURM_CPUS_PER_TASK restart=$RESTARTS max_seconds=$MAX_SECONDS"
python -m dmel.cli environment
args=(--data "$DATA" --config "$CONFIG" --out "$OUT" --workers "$SLURM_CPUS_PER_TASK" --limit "$LIMIT" --split "${SPLIT:-all}")
args+=(--max-seconds "$MAX_SECONDS")
[[ -z "${SELECTION_CONFIG:-}" ]] || args+=(--selection-config "$SELECTION_CONFIG")
status=0
python -m dmel.cli precompute "${args[@]}" || status=$?
if [[ "$status" == 75 ]]; then
    echo "Cache parcial cerrada: ${OUT}.partial"
    if [[ "$AUTO_REQUEUE" == 1 && "$RESTARTS" -lt "$MAX_RESTARTS" ]]; then
        echo "Solicitando siguiente ciclo del job $SLURM_JOB_ID ($((RESTARTS + 1))/$MAX_RESTARTS reinicios)"
        scontrol requeue "$SLURM_JOB_ID"
        exit 0
    fi
    echo 'Quedan filas; reanudación manual necesaria o máximo de reinicios alcanzado.'
fi
[[ "$status" == 0 ]] || exit "$status"
sha256sum "$OUT"
