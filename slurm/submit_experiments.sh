#!/usr/bin/env bash
# Encola baseline -> DMEL -> ablación; recursos definitivos tras medir piloto.
# bash slurm/submit_experiments.sh [--model lstm|dmel|mrs_informer] --dry-run|--submit
set -euo pipefail
cd "$(dirname "$0")/.."
MODE=--dry-run
models=(lstm dmel mrs_informer)
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --dry-run|--submit) MODE="$1"; shift ;;
        --model)
            [[ "$#" -ge 2 ]] || { echo 'Falta modelo después de --model' >&2; exit 2; }
            case "$2" in lstm|dmel|mrs_informer) models=("$2") ;; *) echo 'Modelo inválido' >&2; exit 2 ;; esac
            shift 2 ;;
        *) echo 'Usar [--model lstm|dmel|mrs_informer] --dry-run|--submit' >&2; exit 2 ;;
    esac
done
DECOMP_CACHE="${CACHE:-cache/train.h5}"
if [[ "$MODE" == --submit ]]; then
    [[ -f dataset/train.h5 ]] || { echo 'Falta dataset/train.h5' >&2; exit 1; }
    for model in "${models[@]}"; do
        [[ "$model" == lstm || -f "$DECOMP_CACHE" ]] || { echo 'Falta cache completa' >&2; exit 1; }
    done
    # Detectar una subida incompleta antes de enviar un batch sin ciclos.
    echo 'Comprobando opciones de entrenamiento (sin cargar PyTorch)...'
    grep -q -- '--epochs-per-cycle' slurm/train.sbatch || { echo 'Actualizar slurm/train.sbatch desde el Mac' >&2; exit 1; }
    # -S omite site-packages: la ayuda debe depender solo de la biblioteca estándar.
    # Capturar la salida completa evita cerrar un pipe con grep -q prematuramente.
    if ! cli_help=$("${VENV:-$PWD/.venv}/bin/python" -S -m dmel.cli train --help); then
        echo 'Falló la comprobación de la CLI. Actualizar el código desde el Mac; no se enviaron jobs.' >&2
        exit 1
    fi
    [[ "$cli_help" == *"--epochs-per-cycle"* ]] || { echo 'Actualizar el código Python desde el Mac; no se enviaron jobs.' >&2; exit 1; }
    mkdir -p logs
fi
previous=""
for model in "${models[@]}"; do
    args=(sbatch --parsable --job-name="lab2_$model" --time="${TRAIN_TIME:-08:00:00}" --gres="${TRAIN_GRES:-shard:rtxa6000:8}")
    [[ -z "$previous" ]] || args+=(--dependency="afterok:$previous")
    export CONFIG="configs/$model.json" OUT="runs/${model}-seed42" CACHE="$DECOMP_CACHE"
    export DATA="dataset/train.h5" RESUME=0 AUTO_REQUEUE=1
    export EPOCHS_PER_CYCLE="${TRAIN_EPOCHS_PER_CYCLE:-1}" MAX_RESTARTS="${TRAIN_MAX_RESTARTS:-99}"
    [[ "$model" != lstm ]] || CACHE=""
    args+=(--export=ALL slurm/train.sbatch)
    if [[ "$MODE" == --dry-run ]]; then
        printf '[dry-run] CONFIG=%q OUT=%q CACHE=%q EPOCHS_PER_CYCLE=%q AUTO_REQUEUE=%q MAX_RESTARTS=%q ' "$CONFIG" "$OUT" "$CACHE" "$EPOCHS_PER_CYCLE" "$AUTO_REQUEUE" "$MAX_RESTARTS"
        printf '%q ' "${args[@]}"; printf '\n'
        previous="DRY_$model"
    else
        echo "Enviando $model a Slurm..."
        previous="$("${args[@]}")"
        previous="${previous%%;*}"
        [[ "$previous" =~ ^[0-9]+$ ]] || { echo 'sbatch no devolvió job id válido' >&2; exit 1; }
        printf '%s\t%s\n' "$previous" "$model" >> logs/submissions.tsv
        echo "$model: job $previous"
    fi
done
