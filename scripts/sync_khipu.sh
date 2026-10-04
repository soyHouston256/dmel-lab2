#!/usr/bin/env bash
# bash scripts/sync_khipu.sh --check|--code|--data|--pull [--dry-run]
# Adaptado del proyecto Sonidos. Sin --delete; test_targets.csv no se transfiere.
set -euo pipefail
REMOTE="${KHIPU_REMOTE:-max.ramirez@khipu.utec.edu.pe}"
RDIR="${KHIPU_DIR:-/home/max.ramirez/dmel}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODE=check
DRY=0
for arg in "$@"; do
    case "$arg" in
        --check|--code|--data|--pull) MODE="${arg#--}" ;;
        --dry-run) DRY=1 ;;
        *) echo "Uso: $0 --check|--code|--data|--pull [--dry-run]" >&2; exit 1 ;;
    esac
done
[[ "$RDIR" =~ ^/[a-zA-Z0-9_./-]+$ && "$RDIR" != / && "$RDIR" != /home/max.ramirez/deep ]] || { echo 'Ruta remota inválida o de Sonidos' >&2; exit 1; }
[[ "$REMOTE" =~ ^[a-zA-Z0-9_][a-zA-Z0-9_.@-]*$ ]] || { echo 'Host inválido' >&2; exit 1; }
run() {
    if [[ "$DRY" == 1 ]]; then
        printf '[dry-run]'; printf ' %q' "$@"; printf '\n'
    else
        "$@"
    fi
}
prepare_remote() {
    # Exige directorio nuevo/vacío o marcado como proyecto DMEL.
    run ssh "$REMOTE" "mkdir -p '$RDIR' && if test -f '$RDIR/.dmel-project' || test -z \"\$(ls -A '$RDIR')\"; then touch '$RDIR/.dmel-project'; else echo 'Directorio no vacío sin marcador DMEL' >&2; exit 1; fi"
}
cd "$ROOT"
case "$MODE" in
    check)
        run ssh -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=12 "$REMOTE" \
            'hostname; command -v sbatch; sinfo -s; sinfo -p gpu -N -o "%N %G"; squeue -u "$USER" -o "%.10i %.28j %.9P %.8T %.10M %R"'
        ;;
    code)
        prepare_remote
        run rsync -avP --partial --exclude '__pycache__/' --exclude '*.pyc' \
            dmel configs scripts slurm tests docs requirements.txt requirements-khipu.txt README.md "$REMOTE:$RDIR/"
        ;;
    data)
        for f in train.h5 test.h5 metadata.json leer_datos.py; do
            [[ -f "dataset/$f" ]] || { echo "Falta dataset/$f" >&2; exit 1; }
        done
        prepare_remote
        run ssh "$REMOTE" "mkdir -p '$RDIR/dataset'"
        run rsync -avP --partial dataset/train.h5 dataset/test.h5 dataset/metadata.json dataset/leer_datos.py "$REMOTE:$RDIR/dataset/"
        # SHA-256 del archivo debe coincidir con el registrado por inspect local.
        run ssh "$REMOTE" "cd '$RDIR' && sha256sum dataset/train.h5 dataset/test.h5"
        ;;
    pull)
        run mkdir -p runs logs
        run rsync -avP --partial --include '*/' --include 'best.pt' --include '*.json' \
            --include '*.csv' --include '*.npz' --exclude '*' "$REMOTE:$RDIR/runs/" runs/
        run rsync -avP --partial "$REMOTE:$RDIR/logs/" logs/
        ;;
esac
