"""Cache de descomposiciones por ventana, con un único escritor HDF5."""
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import asdict
import json
import math
import multiprocessing
from pathlib import Path
import time

import h5py
import numpy as np
import scipy

from .data import select_ids, sha256
from . import decomposition
from .decomposition import DecompositionConfig, decompose_window


def _window(task):
    path, row, config = task
    with h5py.File(path, "r") as f:
        q = f["X"][row, :, 11]  # No se abre y ni y_aux.
    start = time.perf_counter()
    components, diagnostic = decompose_window(q, DecompositionConfig(**config), sample_id=int(row))
    diagnostic["seconds"] = time.perf_counter() - start
    return row, components, diagnostic


def _bounded_results(pool, tasks, workers, should_stop=lambda: False):
    """Python 3.12 Executor.map encola todo; aquí hay <=2*workers futuros."""
    pending = {}
    iterator = iter(tasks)
    for _ in range(2 * workers):
        if should_stop():
            break
        item = next(iterator, None)
        if item is None:
            break
        index, task = item
        pending[pool.submit(_window, task)] = index
    while pending:
        ready, _ = wait(pending, return_when=FIRST_COMPLETED)
        for future in ready:
            index = pending.pop(future)
            yield index, future.result()
            # Al vencer el presupuesto se drenan los trabajos ya enviados.
            item = None if should_stop() else next(iterator, None)
            if item is not None:
                next_index, task = item
                pending[pool.submit(_window, task)] = next_index


def _serial_results(tasks, should_stop):
    for index, task in tasks:
        if should_stop():
            break
        yield index, _window(task)


def precompute(source, output, cfg, split="all", limit=0, workers=1, selection_config=None,
               max_seconds=0):
    """Presupuesto suave: incluye lectura/hash; drena tareas antes de cerrar HDF5.

    max_seconds=0 no limita el tiempo. Dejar margen en Slurm para las ventanas
    en curso y el cierre; esto no protege contra SIGKILL o fallos del nodo.
    """
    if workers < 1:
        raise ValueError("workers>=1")
    if not math.isfinite(max_seconds) or max_seconds < 0:
        raise ValueError("max_seconds debe ser finito y >=0")
    deadline = time.monotonic() + max_seconds if max_seconds else None
    def should_stop():
        return deadline is not None and time.monotonic() >= deadline
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Cache existente: {output}; usar otro nombre o la cache existente")
    if selection_config:
        if split != "all" or limit:
            raise ValueError("selection_config exige split=all y limit=0")
        training = json.loads(Path(selection_config).read_text()).get("training", {})
        seed = training.get("seed", 42)
        ids = np.sort(np.concatenate([
            select_ids(source, "train", training.get("max_train", 0), seed),
            select_ids(source, "validation", training.get("max_val", 0), seed)]))
    else:
        ids = select_ids(source, split, limit, cfg.seed)
    signature, config = sha256(source), asdict(cfg)
    cache_config = {"parameters": config, "implementation_sha256": sha256(decomposition.__file__),
                    "numpy": np.__version__, "scipy": scipy.__version__}
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial")
    with h5py.File(partial, "a") as f:
        if "components" in f:
            if f.attrs["source_sha256"] != signature or json.loads(f.attrs["config"]) != cache_config or not np.array_equal(f["ids"][:], ids):
                raise ValueError("Cache parcial incompatible: usar otro nombre de salida")
        else:
            f.attrs.update(source_sha256=signature, config=json.dumps(cache_config, sort_keys=True), complete=False)
            f.create_dataset("ids", data=ids)
            f.create_dataset("components", shape=(len(ids), 336, 2), dtype="f4", chunks=(1, 336, 2))
            f.create_dataset("done", data=np.zeros(len(ids), dtype=bool))
            f.create_dataset("diagnostics", (len(ids),), dtype=h5py.string_dtype())
        pending = np.flatnonzero(~f["done"][:])
        tasks = ((i, (str(source), int(ids[i]), config)) for i in pending)
        started = time.perf_counter()
        pool = ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn")) if workers > 1 else None
        try:
            results = (_bounded_results(pool, tasks, workers, should_stop) if pool
                       else _serial_results(tasks, should_stop))
            for position, (index, result) in enumerate(results, 1):
                row, components, diagnostic = result
                if row != ids[index] or components.shape != (336, 2) or not np.isfinite(components).all():
                    raise ValueError("Resultado de descomposición inválido")
                f["components"][index] = components
                f["diagnostics"][index] = json.dumps(diagnostic, allow_nan=False)
                f["done"][index] = True
                if position % 16 == 0 or position == len(pending):
                    f.flush()
                    print(json.dumps({"done_now": position, "pending_at_start": len(pending),
                                      "seconds": round(time.perf_counter() - started, 3)}), flush=True)
        finally:
            if pool:
                pool.shutdown(wait=True, cancel_futures=True)
        done = int(np.count_nonzero(f["done"][:]))
        complete = done == len(ids)
        f.attrs["complete"] = complete
        f.flush()
    if complete:
        partial.replace(output)
    return {"output": str(output if complete else partial), "n": len(ids),
            "complete": complete, "done": done, "remaining": len(ids) - done,
            "done_this_cycle": done - (len(ids) - len(pending)),
            "source_sha256": signature, "config": cache_config}
