#!/usr/bin/env bash
#SBATCH --job-name=lab2_cierre4h
#SBATCH --partition=gpu
#SBATCH --gres=shard:rtxa6000:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --open-mode=append
#SBATCH --no-requeue
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Enviar desde raíz DMEL}"
source "${VENV:-$PWD/.venv}/bin/activate"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=8
out="runs/cierre-4h-seed42-20261008"
reports="outputs/verification/cierre-4h-seed42-20261008"
for target in "$out/test-lstm" "$out/dmel-3epochs" "$out/mrs-informer-3epochs"; do
    [[ ! -e "$target" ]] || { echo "Salida ya existente: $target; revisar antes de repetir" >&2; exit 2; }
done
mkdir -p "$out" "$reports"
echo "stage=preflight date=$(date -Iseconds)"
python -m scripts.lab2_closure preflight --root . --out "$reports/preflight.json"

# Asegurar primero la exportación completa del mejor modelo de validación.
echo "stage=test_lstm date=$(date -Iseconds)"
python -m dmel.cli predict --data dataset/test.h5 --run runs/lstm-seed42-fd-fix \
    --out "$out/test-lstm" --device cuda
python -m scripts.lab2_closure verify-test --data dataset/test.h5 --run runs/lstm-seed42-fd-fix \
    --predictions "$out/test-lstm" --out "$reports/test-lstm.json"

# Control y ablación desde cero, mismos datos, semilla y presupuesto.
# Sin requeue entre épocas: una sola asignación para minimizar esperas.
echo "stage=control_dmel_3epochs date=$(date -Iseconds)"
python -m dmel.cli train --data dataset/train.h5 --cache cache/train.h5 \
    --config configs/dmel-3epochs.json --out "$out/dmel-3epochs" --device cuda --epochs-per-cycle 0
echo "stage=ablation_mrs_informer_3epochs date=$(date -Iseconds)"
python -m dmel.cli train --data dataset/train.h5 --cache cache/train.h5 \
    --config configs/mrs_informer-3epochs.json --out "$out/mrs-informer-3epochs" --device cuda --epochs-per-cycle 0
python -m scripts.lab2_closure compare-ablation --reference "$out/dmel-3epochs" \
    --ablation "$out/mrs-informer-3epochs" --out "$reports/ablation-3epochs.json"
echo "stage=complete_test_lstm_and_ablation_3epochs date=$(date -Iseconds)"
