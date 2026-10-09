#!/usr/bin/env bash
#SBATCH --job-name=lab2_test_dmel4h
#SBATCH --partition=gpu
#SBATCH --gres=shard:rtxa6000:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --no-requeue
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Enviar desde raíz DMEL}"
source "${VENV:-$PWD/.venv}/bin/activate"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1
out="runs/cierre-4h-seed42-20261008/test-dmel"
[[ ! -e "$out" ]] || { echo "Salida ya existente: $out" >&2; exit 2; }
python -m dmel.cli predict --data dataset/test.h5 --run runs/dmel-seed42 \
    --cache cache/test-cierre-4h-20261008.h5 --out "$out" --device cuda
python -m scripts.lab2_closure verify-test --data dataset/test.h5 --run runs/dmel-seed42 \
    --predictions "$out" --out outputs/verification/cierre-4h-seed42-20261008/test-dmel.json
