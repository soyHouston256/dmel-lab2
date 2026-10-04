#!/usr/bin/env bash
#SBATCH --job-name=dmel_predict
#SBATCH --partition=gpu
#SBATCH --gres=shard:rtxa6000:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Enviar desde raíz DMEL}"
source "${VENV:-$PWD/.venv}/bin/activate"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" OPENBLAS_NUM_THREADS=1
args=(--data "${DATA:-dataset/test.h5}" --run "${RUN:?Indicar RUN}" --device cuda)
[[ -z "${CACHE:-}" ]] || args+=(--cache "$CACHE")
python -m dmel.cli predict "${args[@]}"
sha256sum "$RUN/test/predictions.csv"
