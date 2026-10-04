"""Prueba local sintética reproducible; ejecutar python -m scripts.smoke."""
from dataclasses import asdict
from pathlib import Path
import argparse

import h5py
import numpy as np
import torch

from dmel.decomposition import DecompositionConfig
from dmel.precompute import precompute
from dmel.engine import evaluate, train, write_json, environment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="runs/synthetic-smoke")
    args = parser.parse_args()
    root = Path(args.out)
    if root.exists():
        raise FileExistsError("Usar directorio de smoke nuevo")
    root.mkdir(parents=True)
    rng = np.random.default_rng(42)
    x = rng.normal(size=(24, 336, 12)).astype("f4")
    t = np.arange(336)
    x[:, :, 11] = 2 + .4 * np.sin(t[None] / 16 + np.arange(24)[:, None] / 5)
    y = np.repeat(x[:, -1:, 11], 48, axis=1).copy()
    train_file, test_file = root / "synthetic_train.h5", root / "synthetic_test.h5"
    with h5py.File(train_file, "w") as f:
        f["X"], f["y"] = x, y
        f["split"], f["basin_id"] = np.array([0] * 16 + [1] * 8), np.arange(24) % 4
        f.attrs["synthetic"] = True
    with h5py.File(test_file, "w") as f:
        f["X"], f["basin_id"] = x[:5], np.arange(5) % 4
        f.attrs["synthetic"] = True
    cfg = DecompositionConfig(trials=4, max_imfs=3, max_siftings=10)
    precompute(train_file, root / "train-cache.h5", cfg)
    precompute(test_file, root / "test-cache.h5", cfg, split="test")
    trained = train(train_file, "configs/smoke.json", root / "model", root / "train-cache.h5")
    predicted = evaluate(test_file, root / "model", root / "test-cache.h5", split="test")
    write_json(root / "smoke-summary.json", {"synthetic": True, "purpose": "integración, no resultado hidrológico",
               "environment": environment(), "training": trained, "prediction": predicted})


if __name__ == "__main__":
    main()
