"""La ayuda y la validación del envío no necesitan paquetes científicos."""
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("command", [None, "train", "precompute", "inspect", "environment",
                                    "evaluate", "predict", "persistence"])
def test_help_without_site_packages(command):
    args = [sys.executable, "-S", "-m", "dmel.cli"]
    if command:
        args.append(command)
    result = subprocess.run(args + ["--help"], cwd=ROOT, capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout
    if command == "train":
        assert "--epochs-per-cycle" in result.stdout


def test_invalid_args_are_reported_without_loading_dependencies():
    result = subprocess.run([sys.executable, "-S", "-m", "dmel.cli", "train", "--bad-option"],
                            cwd=ROOT, capture_output=True, text=True, timeout=15)
    assert result.returncode == 2
    assert "error:" in result.stderr and "Traceback" not in result.stderr


@pytest.mark.parametrize("stale_cli", [False, True])
def test_submission_preflight_without_ml_dependencies(tmp_path, stale_cli):
    # Ejecuta la CLI real sin site-packages y un sbatch simulado; nunca usa Slurm.
    (tmp_path / "slurm").mkdir()
    (tmp_path / "dmel").mkdir()
    (tmp_path / "dataset").mkdir()
    (tmp_path / "dataset/train.h5").touch()
    for name in ("submit_experiments.sh", "train.sbatch"):
        shutil.copy(ROOT / "slurm" / name, tmp_path / "slurm" / name)
    for name in ("__init__.py", "cli.py"):
        shutil.copy(ROOT / "dmel" / name, tmp_path / "dmel" / name)
    # No se copian data/engine: intentar importarlos durante --help debe fallar.
    if stale_cli:
        (tmp_path / "dmel/cli.py").write_text("from .data import schema\n")
    venv_bin = tmp_path / ".venv/bin"
    venv_bin.mkdir(parents=True)
    (venv_bin / "python").symlink_to(sys.executable)
    sbatch = venv_bin / "sbatch"
    sbatch.write_text('#!/bin/bash\nprintf "%s\\n" "$*" >> "$SUBMIT_CALLS"\necho 54321\n')
    sbatch.chmod(0o755)
    calls = tmp_path / "calls.txt"
    env = {**os.environ, "PATH": str(venv_bin) + os.pathsep + os.environ["PATH"],
           "VENV": str(venv_bin.parent), "SUBMIT_CALLS": str(calls)}
    result = subprocess.run(["bash", "slurm/submit_experiments.sh", "--model", "lstm", "--submit"],
                            cwd=tmp_path, env=env, capture_output=True, text=True, timeout=15)
    if stale_cli:
        assert result.returncode != 0
        assert "no se enviaron jobs" in result.stderr
        assert not calls.exists()
    else:
        assert result.returncode == 0, result.stderr
        assert "lstm: job 54321" in result.stdout
        assert calls.read_text().count("--job-name=lab2_lstm") == 1
        assert (tmp_path / "logs/submissions.tsv").read_text() == "54321\tlstm\n"
