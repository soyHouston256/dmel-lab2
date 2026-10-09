#!/usr/bin/env bash
# Ejecución local en el equipo elegido; no abre SSH ni envía jobs Slurm.
set -euo pipefail
cd "$(dirname "$0")/.."
stage="${1:?Usar: ablation | test-lstm | test-dmel}"
case "$stage" in ablation|test-lstm|test-dmel) ;; *) echo 'Etapa inválida' >&2; exit 2;; esac
PYTHON="${DMEL_PYTHON:-python}"
device="${DMEL_DEVICE:-cuda}"
input="${DMEL_INPUT_ROOT:-$PWD/migration/khipu}"
tag="${DMEL_CLOSURE_TAG:-closure-seed42-20261008}"
out="$PWD/runs/$tag"
reports="$PWD/outputs/verification/$tag"
mkdir -p "$out" "$reports" logs
exec > >(tee -a "logs/${tag}-${stage}.log") 2>&1
"$PYTHON" -c 'import sys, torch; from dmel.engine import device_for; device_for(sys.argv[1]); print("device:", sys.argv[1])' "$device"
"$PYTHON" -m scripts.lab2_closure preflight --root "$input" --out "$reports/preflight-${stage}.json"

if [[ "$stage" == ablation ]]; then
    resume=()
    [[ ! -f "$out/mrs-informer/last.pt" ]] || resume=(--resume)
    "$PYTHON" -m dmel.cli train --data "$input/dataset/train.h5" --cache "$input/cache/train.h5" \
        --config configs/mrs_informer.json --out "$out/mrs-informer" --device "$device" "${resume[@]}" \
        --epochs-per-cycle "${DMEL_EPOCHS_PER_CYCLE:-0}"
    "$PYTHON" -m scripts.lab2_closure compare-ablation --reference "$input/runs/dmel-seed42" \
        --ablation "$out/mrs-informer" --out "$reports/ablation.json"
else
    cache_args=()
    model=lstm
    run="$input/runs/lstm-seed42-fd-fix"
    if [[ "$stage" == test-dmel ]]; then
        model=dmel
        run="$input/runs/dmel-seed42"
        cache="$PWD/cache/test-${tag}.h5"
        if [[ ! -f "$cache" ]]; then
            OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "$PYTHON" -m dmel.cli precompute \
                --data "$input/dataset/test.h5" --out "$cache" --config configs/decomposition.json \
                --split test --limit 0 --workers "${DMEL_PRECOMPUTE_WORKERS:-8}" \
                --max-seconds "${DMEL_PRECOMPUTE_MAX_SECONDS:-0}"
        fi
        cache_args=(--cache "$cache")
    fi
    destination="$out/test-$model"
    # Una exportación existente se verifica; no se sobreescribe silenciosamente.
    if [[ ! -d "$destination" ]]; then
        "$PYTHON" -m dmel.cli predict --data "$input/dataset/test.h5" --run "$run" \
            --out "$destination" --device "$device" "${cache_args[@]}"
    fi
    "$PYTHON" -m scripts.lab2_closure verify-test --data "$input/dataset/test.h5" --run "$run" \
        --predictions "$destination" --out "$reports/test-${model}.json"
fi
