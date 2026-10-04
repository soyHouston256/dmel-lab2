#!/usr/bin/env bash
#SBATCH --job-name=dmel_smoke
#SBATCH --partition=gpu
#SBATCH --gres=shard:rtxa6000:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --open-mode=append
#SBATCH --requeue
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Enviar desde raíz DMEL}"
source "${VENV:-$PWD/.venv}/bin/activate"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
CONFIG="${CONFIG:-configs/smoke.json}"
OUT="${OUT:-runs/smoke-${SLURM_JOB_ID}}"
EPOCHS_PER_CYCLE="${EPOCHS_PER_CYCLE:-0}"
AUTO_REQUEUE="${AUTO_REQUEUE:-0}"
MAX_RESTARTS="${MAX_RESTARTS:-99}"
RESTARTS="${SLURM_RESTART_COUNT:-0}"
RESUME="${RESUME:-0}"
[[ "$EPOCHS_PER_CYCLE" =~ ^[0-9]+$ && "$MAX_RESTARTS" =~ ^[0-9]+$ && "$RESTARTS" =~ ^[0-9]+$ ]] || { echo 'Épocas/reinicios deben ser enteros >=0' >&2; exit 2; }
[[ "$AUTO_REQUEUE" == 0 || "$AUTO_REQUEUE" == 1 ]] || { echo 'AUTO_REQUEUE debe ser 0 o 1' >&2; exit 2; }
[[ "$RESUME" == 0 || "$RESUME" == 1 ]] || { echo 'RESUME debe ser 0 o 1' >&2; exit 2; }
[[ "$AUTO_REQUEUE" == 0 || "$EPOCHS_PER_CYCLE" -gt 0 ]] || { echo 'AUTO_REQUEUE exige EPOCHS_PER_CYCLE>0' >&2; exit 2; }
if [[ "$RESTARTS" -gt 0 ]]; then
    [[ -f "$OUT/last.pt" ]] || { echo 'Reinicio sin checkpoint: revisar causa antes de continuar' >&2; exit 2; }
    RESUME=1
fi
echo "date=$(date -Iseconds) job=$SLURM_JOB_ID node=$(hostname) cuda_devices=${CUDA_VISIBLE_DEVICES:-unset} shards=${SLURM_SHARDS_ON_NODE:-unset} restart=$RESTARTS resume=$RESUME epochs_per_cycle=$EPOCHS_PER_CYCLE"
python -m dmel.cli environment
# Operación real dentro del recurso asignado; sbatch establece CUDA_VISIBLE_DEVICES.
python -c 'import torch; assert torch.cuda.is_available(); x=torch.ones(32,32,device="cuda"); print("GPU smoke:", (x@x).sum().item(),torch.cuda.get_device_name(0))'
args=(--data "${DATA:-dataset/train.h5}" --config "$CONFIG" --out "$OUT" --device cuda)
args+=(--epochs-per-cycle "$EPOCHS_PER_CYCLE")
[[ -z "${CACHE:-}" ]] || args+=(--cache "$CACHE")
[[ "${RESUME:-0}" != 1 ]] || args+=(--resume)
status=0
python -m dmel.cli train "${args[@]}" || status=$?
if [[ "$status" == 75 ]]; then
    [[ -f "$OUT/last.pt" && -f "$OUT/status.json" ]] || { echo 'Pausa sin checkpoint/estado: no reencolar' >&2; exit 2; }
    echo "Época guardada en $OUT/last.pt"
    if [[ "$AUTO_REQUEUE" == 1 && "$RESTARTS" -lt "$MAX_RESTARTS" ]]; then
        echo "Solicitando siguiente ciclo del job $SLURM_JOB_ID ($((RESTARTS + 1))/$MAX_RESTARTS reinicios)"
        scontrol requeue "$SLURM_JOB_ID"
        exit 0
    fi
    echo 'Entrenamiento parcial; reanudar con RESUME=1 tras revisar status.json.'
fi
[[ "$status" == 0 ]] || exit "$status"
sha256sum "$OUT/best.pt" "$OUT/config.json"
