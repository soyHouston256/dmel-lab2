"""Costo de ventanas históricas; no lee ni evalúa etiquetas de test."""
from dataclasses import asdict
import argparse
import json
from pathlib import Path
import time

import h5py
import numpy as np

from dmel.decomposition import DecompositionConfig, decompose_window
from dmel.data import sha256
from dmel.engine import write_json, environment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="dataset/test.h5")
    parser.add_argument("--ids", nargs="+", type=int, default=[0, 1, 100, 500])
    parser.add_argument("--out", default="outputs/decomposition-benchmark.json")
    args = parser.parse_args()
    cfg, rows = DecompositionConfig(), []
    with h5py.File(args.data) as f:
        for row_id in args.ids:
            q = f["X"][row_id, :, 11]
            tick = time.perf_counter()
            components, diagnostic = decompose_window(q, cfg, sample_id=row_id)
            seconds = time.perf_counter() - tick
            row = {"Id": row_id, "seconds": seconds, "max_abs_reconstruction_error": float(np.max(abs(components.sum(-1) - q))), "diagnostics": diagnostic}
            rows.append(row)
            print(json.dumps({"Id": row_id, "seconds": seconds}), flush=True)
    write_json(args.out, {"purpose": "costo e integridad de historia; sin objetivos ni selección de modelo",
               "environment": environment(), "data_sha256": sha256(args.data), "config": asdict(cfg), "samples": rows,
               "median_seconds_per_window": float(np.median([r["seconds"] for r in rows])),
               "extrapolation_warning": "4 ventanas locales no predicen tiempos de cola ni rendimiento de Khipu"})


if __name__ == "__main__":
    main()
