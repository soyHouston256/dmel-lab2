"""HDF5 por proceso; particiones del laboratorio y escala ajustada solo en train."""
import hashlib
import json
import os
from pathlib import Path

import h5py
import numpy as np
from torch.utils.data import Dataset


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def schema(path):
    with h5py.File(path, "r") as f:
        if "X" not in f or f["X"].ndim != 3 or f["X"].shape[1:] != (336, 12):
            raise ValueError("Se requiere X [N,336,12]")
        n = len(f["X"])
        if n == 0 or "basin_id" not in f or f["basin_id"].shape != (n,):
            raise ValueError("Se requiere basin_id [N] y N>0")
        if "y" in f:
            if f["y"].shape != (n, 48) or "split" not in f or f["split"].shape != (n,):
                raise ValueError("Se requiere y [N,48] y split [N]")
            flags = f["split"][:]
            if not np.isin(flags, [0, 1]).all():
                raise ValueError("split solo admite 0=train y 1=validation")
            if not all(np.any(flags == k) for k in (0, 1)):
                raise ValueError("Deben existir entrenamiento y validación")
        elif "split" in f:
            raise ValueError("Un archivo con split necesita y")
        return {"n": n, "supervised": "y" in f,
                "datasets": {k: {"shape": list(v.shape), "dtype": str(v.dtype)}
                             for k, v in f.items() if isinstance(v, h5py.Dataset)}}


def select_ids(path, split, limit=0, seed=42):
    info = schema(path)
    with h5py.File(path, "r") as f:
        if split == "test":
            if info["supervised"]:
                raise ValueError("test requiere archivo sin etiquetas")
            ids = np.arange(info["n"])
        elif split in ("train", "validation"):
            if not info["supervised"]:
                raise ValueError("Falta train.h5 con y y split; test no sirve para entrenar")
            ids = np.flatnonzero(f["split"][:] == (0 if split == "train" else 1))
        elif split == "all":
            ids = np.arange(info["n"])
        else:
            raise ValueError(f"Partición desconocida: {split}")
    if limit < 0:
        raise ValueError("limit debe ser >=0")
    if limit and limit < len(ids):
        ids = np.sort(np.random.default_rng(seed).choice(ids, limit, replace=False))
    return ids.astype(np.int64)


def fit_normalizer(path, ids, block=128):
    """Welford combinado en float64 sobre historias de entrenamiento únicamente."""
    count, mean, m2 = 0, np.zeros(12), np.zeros(12)
    with h5py.File(path, "r") as f:
        if "split" not in f or not np.all(f["split"][np.sort(ids)] == 0):
            raise ValueError("La normalización solo puede ajustarse con split=0")
        for start in range(0, len(ids), block):
            x = f["X"][np.sort(ids[start:start + block])].astype(np.float64).reshape(-1, 12)
            if not np.isfinite(x).all():
                raise ValueError("Train contiene NaN/Inf: definir política de imputación antes de entrenar")
            bn, bm = len(x), x.mean(0)
            delta = bm - mean
            m2 += ((x - bm) ** 2).sum(0) + delta ** 2 * count * bn / (count + bn)
            mean += delta * bn / (count + bn)
            count += bn
    if not count:
        raise ValueError("No hay filas para normalización")
    std = np.sqrt(m2 / count)
    std[std < 1e-6] = 1.0
    return {"mean": mean.tolist(), "std": std.tolist(), "count": count,
            "fit_split": "train", "train_ids_sha256": hashlib.sha256(ids.astype("<i8").tobytes()).hexdigest()}


class RunoffDataset(Dataset):
    def __init__(self, path, split, normalizer=None, cache=None, limit=0, seed=42, source_sha256=None):
        self.path, self.cache = str(Path(path).resolve()), str(Path(cache).resolve()) if cache else None
        self.ids = select_ids(path, split, limit, seed)
        self.normalizer = normalizer
        self._file = self._cache_file = None
        self._pid = None
        self.cache_config = None
        if cache:
            with h5py.File(cache, "r") as f:
                if not f.attrs.get("complete", False):
                    raise ValueError("Cache incompleta")
                if f.attrs["source_sha256"] != (source_sha256 or sha256(path)):
                    raise ValueError("Cache no corresponde al HDF5 fuente")
                cache_ids = f["ids"][:]
                if len(np.unique(cache_ids)) != len(cache_ids):
                    raise ValueError("Cache tiene ids duplicados")
                lookup = {int(v): i for i, v in enumerate(cache_ids)}
                if any(int(i) not in lookup for i in self.ids):
                    raise ValueError("Cache no cubre todas las muestras solicitadas")
                self.cache_rows = np.array([lookup[int(i)] for i in self.ids])
                if f["components"].shape != (len(cache_ids), 336, 2):
                    raise ValueError("Cache debe tener components [N,336,2]")
                self.cache_config = json.loads(f.attrs["config"])

    def _open(self):
        if self._pid != os.getpid() or self._file is None:
            self.close()
            self._file = h5py.File(self.path, "r")
            self._cache_file = h5py.File(self.cache, "r") if self.cache else None
            self._pid = os.getpid()

    def __getitem__(self, index):
        self._open()
        row = int(self.ids[index])
        x = np.asarray(self._file["X"][row], dtype=np.float32)
        if not np.isfinite(x).all():
            raise ValueError(f"X no finita en fila {row}")
        item = {"Id": row, "basin_id": int(self._file["basin_id"][row]), "last_q": x[-1, 11].copy()}
        if self._cache_file is not None:
            c = self._cache_file["components"][self.cache_rows[index]].astype(np.float32)
            if not np.isfinite(c).all() or not np.allclose(c.sum(-1), x[:, 11], atol=1e-5, rtol=2e-5):
                raise ValueError(f"Cache no reconstruye caudal en fila {row}")
            if self.normalizer:
                c[:, 0] /= self.normalizer["std"][11]
                c[:, 1] = (c[:, 1] - self.normalizer["mean"][11]) / self.normalizer["std"][11]
            item["components"] = c
        if self.normalizer:
            x = (x - np.asarray(self.normalizer["mean"], np.float32)) / np.asarray(self.normalizer["std"], np.float32)
        item["X"] = x
        if "y" in self._file:
            y = self._file["y"][row].astype(np.float32)
            if not np.isfinite(y).all():
                raise ValueError(f"y no finita en fila {row}")
            item["y"] = y  # unidades físicas; y_aux no se utiliza como entrada
        return item

    def __len__(self):
        return len(self.ids)

    def close(self):
        for f in (self._file, self._cache_file):
            if f is not None:
                f.close()
        self._file = self._cache_file = None
        self._pid = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state.update(_file=None, _cache_file=None, _pid=None)
        return state
