#!/usr/bin/env bash
# Lo ejecuta el usuario en su terminal. No envía ni cancela trabajos.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
if [[ -n "${MIGRATION_PYTHON:-}" ]]; then
    PYTHON="$MIGRATION_PYTHON"
elif [[ -x /opt/anaconda3/bin/python ]]; then
    PYTHON=/opt/anaconda3/bin/python
else
    PYTHON=python3
fi
exec "$PYTHON" "$ROOT/scripts/pull_khipu.py" "$@"
