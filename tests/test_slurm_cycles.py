"""Prueba local del contrato del batch; no contacta un scheduler."""
import os
from pathlib import Path
import shlex
import subprocess
import time

import pytest


@pytest.mark.parametrize("compute_status,auto,restarts,requeue_status,expected,action", [
    (0, 1, 0, 0, 0, "sha256sum"),
    (75, 1, 0, 0, 0, "scontrol requeue 123"),
    (75, 1, 4, 0, 75, None),
    (75, 0, 0, 0, 75, None),
    (1, 1, 0, 0, 1, None),
    (75, 1, 0, 1, 1, "scontrol requeue 123"),
])
@pytest.mark.parametrize("command", ["precompute", "train"])
def test_batch_requeues_only_clean_pauses(tmp_path, command, compute_status, auto, restarts,
                                         requeue_status, expected, action):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    activate = tmp_path / "venv/bin/activate"
    activate.parent.mkdir(parents=True)
    activate.write_text(f"export PATH={shlex.quote(str(bin_dir))}:\"$PATH\"\n")
    programs = {
        "python": 'if [[ "${3:-}" == "$COMMAND" ]]; then printf "%s\\n" "$*" > "$PYTHON_ARGS"; exit "$PRECOMPUTE_STATUS"; fi\n',
        "scontrol": 'echo "scontrol $*" >> "$CALLS"\nexit "$REQUEUE_STATUS"\n',
        "sha256sum": 'echo sha256sum >> "$CALLS"\n',
    }
    for name, body in programs.items():
        path = bin_dir / name
        path.write_text("#!/bin/bash\n" + body)
        path.chmod(0o755)
    calls = tmp_path / "calls.txt"
    out = tmp_path / "run"
    out.mkdir()
    (out / "last.pt").touch()
    (out / "status.json").write_text('{"complete":false}')
    python_args = tmp_path / "python-args.txt"
    env = {**os.environ, "SLURM_SUBMIT_DIR": str(tmp_path), "SLURM_JOB_ID": "123",
           "SLURM_CPUS_PER_TASK": "16", "VENV": str(activate.parents[1]),
           "SLURM_JOB_END_TIME": str(int(time.time()) + 28800),
           "SLURM_RESTART_COUNT": str(restarts), "MAX_SECONDS": "25200",
           "AUTO_REQUEUE": str(auto), "MAX_RESTARTS": "4", "CALLS": str(calls),
           "PRECOMPUTE_STATUS": str(compute_status), "REQUEUE_STATUS": str(requeue_status),
           "COMMAND": command, "PYTHON_ARGS": str(python_args), "OUT": str(out),
           "EPOCHS_PER_CYCLE": "1", "RESUME": "0", "CACHE": ""}
    script = Path(__file__).resolve().parents[1] / f"slurm/{command}.sbatch"
    result = subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True)
    assert result.returncode == expected, result.stdout + result.stderr
    assert (calls.read_text().strip() if calls.exists() else None) == action
    if command == "train":
        assert "--epochs-per-cycle 1" in python_args.read_text()
        assert ("--resume" in python_args.read_text()) == (restarts > 0)
