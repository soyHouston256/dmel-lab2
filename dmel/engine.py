"""Entrenamiento supervisado, reanudación y exportación reproducible."""
from dataclasses import asdict, dataclass
import csv
import json
import math
import platform
from pathlib import Path
import random
import time

import h5py
import numpy as np
import scipy
import torch
from torch.utils.data import DataLoader

from .data import RunoffDataset, fit_normalizer, select_ids, sha256
from .metrics import report
from .models import ForecastModel, ModelConfig


@dataclass
class TrainConfig:
    epochs: int = 100
    batch_size: int = 32
    lr: float = 1e-4
    patience: int = 10
    seed: int = 42
    workers: int = 0
    threads: int = 4
    max_train: int = 0
    max_val: int = 0
    grad_clip: float = 1.0


def read_config(path):
    config = json.loads(Path(path).read_text())
    if set(config) - {"model", "training", "description"}:
        raise ValueError("Claves desconocidas en configuración")
    model, training = ModelConfig(**config["model"]), TrainConfig(**config.get("training", {}))
    if min(training.epochs, training.batch_size, training.threads) < 1 or training.workers < 0 or training.patience < 1 or not all(math.isfinite(x) and x > 0 for x in (training.lr, training.grad_clip)):
        raise ValueError("Configuración de entrenamiento inválida")
    return model, training


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)


def environment():
    return {"python": platform.python_version(), "platform": platform.platform(), "torch": torch.__version__,
            "numpy": np.__version__, "scipy": scipy.__version__, "h5py": h5py.__version__,
            "cuda_build": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


def device_for(name):
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA no disponible; verificar dentro del job Slurm")
    return torch.device(name)


def loader(dataset, cfg, shuffle=False, generator=None):
    # HDF5 se abre después de iniciar el worker; spawn evita heredar handles.
    return DataLoader(dataset, batch_size=cfg.batch_size, shuffle=shuffle, num_workers=cfg.workers,
                      generator=generator, **({"multiprocessing_context": "spawn"} if cfg.workers else {}))


def forward(model, batch, device):
    return model(batch["X"].to(device), batch["components"].to(device) if "components" in batch else None)


@torch.no_grad()
def infer(model, batches, normalizer, device):
    if model is not None:
        model.eval()
    predictions, truths, ids, basins = [], [], [], []
    for batch in batches:
        if model is None:
            pred = batch["last_q"].numpy()[:, None].repeat(48, axis=1)
        else:
            pred = (forward(model, batch, device) * normalizer["std"][11] + normalizer["mean"][11]).cpu().numpy()
        if not np.isfinite(pred).all():
            raise ValueError("Predicción no finita")
        predictions.append(pred)
        # numpy() comparte el almacenamiento del tensor recibido del worker.
        # Copiar evita retener memoria compartida/descriptores por cada lote
        # hasta concatenate (Too many open files en validaciones completas).
        ids.append(batch["Id"].numpy().copy())
        basins.append(batch["basin_id"].numpy().copy())
        if "y" in batch:
            truths.append(batch["y"].numpy().copy())
    return {"pred": np.concatenate(predictions), "ids": np.concatenate(ids), "basins": np.concatenate(basins),
            "y": np.concatenate(truths) if truths else None}


def save_checkpoint(path, state):
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def train(data_path, config_path, output, cache=None, device="cpu", resume=False,
          epochs_per_cycle=0):
    # Es un límite operativo, no cambia epochs ni la configuración científica.
    if not isinstance(epochs_per_cycle, int) or epochs_per_cycle < 0:
        raise ValueError("epochs_per_cycle debe ser un entero >=0")
    model_cfg, cfg = read_config(config_path)
    device = device_for(device)
    needs_cache = model_cfg.name in ("dmel", "mrs_informer", "mrs_lstm")
    if needs_cache and not cache:
        raise ValueError("DMEL y ablaciones requieren --cache; ejecutar precompute primero")
    if not needs_cache and cache:
        raise ValueError("Baseline sobre señal cruda no usa --cache")
    source_hash = sha256(data_path)
    ids = select_ids(data_path, "train", cfg.max_train, cfg.seed)
    output = Path(output)
    if output.exists() and any(output.iterdir()) and not resume:
        raise FileExistsError("La corrida ya existe: usar --resume o un nombre nuevo")
    effective = {"model": asdict(model_cfg), "training": asdict(cfg), "data_sha256": source_hash,
                 "data_path": str(Path(data_path).resolve()), "cache_sha256": sha256(cache) if cache else None,
                 "implementation": {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
                 "diagnostic_subset": bool(cfg.max_train or cfg.max_val)}
    torch.set_num_threads(cfg.threads)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    generator = torch.Generator().manual_seed(cfg.seed)
    start_epoch, best, bad = 0, float("inf"), 0
    model = ForecastModel(model_cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    if resume:
        saved_cfg = json.loads((output / "config.json").read_text())
        if saved_cfg != effective:
            raise ValueError("Config/datos/cache cambiaron: no es una reanudación compatible")
        checkpoint = torch.load(output / "last.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch, best, bad = checkpoint["epoch"] + 1, checkpoint["best"], checkpoint["bad"]
        torch.set_rng_state(checkpoint["torch_rng"])
        if device.type == "cuda" and checkpoint["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        generator.set_state(checkpoint["loader_rng"])
        normalizer = json.loads((output / "normalizer.json").read_text())
        history = json.loads((output / "history.json").read_text())
        history = [h for h in history if h["epoch"] < start_epoch]  # el checkpoint define el último commit
    else:
        normalizer = fit_normalizer(data_path, ids)
        history = []
    train_ds = RunoffDataset(data_path, "train", normalizer, cache, cfg.max_train, cfg.seed, source_hash)
    val_ds = RunoffDataset(data_path, "validation", normalizer, cache, cfg.max_val, cfg.seed, source_hash)
    output.mkdir(parents=True, exist_ok=True)
    if not resume:
        write_json(output / "config.json", effective)
        write_json(output / "normalizer.json", normalizer)
        write_json(output / "environment.json", environment())
        write_json(output / "decomposition.json", train_ds.cache_config)
        np.savez(output / "split_ids.npz", train=train_ds.ids, validation=val_ds.ids)
    train_loader, val_loader = loader(train_ds, cfg, True, generator), loader(val_ds, cfg)
    print(json.dumps({"event": "training_start", "model": model_cfg.name,
                      "resume": resume, "start_epoch": start_epoch,
                      "epochs_target": cfg.epochs, "epochs_per_cycle": epochs_per_cycle,
                      "n_train": len(train_ds), "n_validation": len(val_ds)}), flush=True)
    try:
        for epoch in range(start_epoch, cfg.epochs):
            if bad >= cfg.patience:
                break
            tick, total, count = time.perf_counter(), 0.0, 0
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            model.train()
            for batch_index, batch in enumerate(train_loader, 1):
                optimizer.zero_grad(set_to_none=True)
                y = (batch["y"].to(device) - normalizer["mean"][11]) / normalizer["std"][11]
                pred = forward(model, batch, device)
                loss = torch.mean((pred - y) ** 2)
                if not torch.isfinite(loss):
                    raise ValueError("Pérdida no finita")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip, error_if_nonfinite=True)
                optimizer.step()
                total += loss.item() * len(y)
                count += len(y)
                if batch_index % 500 == 0:
                    print(json.dumps({"event": "train_progress", "epoch": epoch,
                                      "batches_done": batch_index, "batches_total": len(train_loader),
                                      "seconds": round(time.perf_counter() - tick, 3)}), flush=True)
            result = infer(model, val_loader, normalizer, device)
            rmse = float(np.sqrt(np.mean((result["pred"].astype(np.float64) - result["y"]) ** 2)))
            improved = rmse < best
            best, bad = (rmse, 0) if improved else (best, bad + 1)
            row = {"epoch": epoch, "train_mse_normalized": total / count, "val_rmse_mm_h": rmse,
                   "seconds": time.perf_counter() - tick, "peak_cuda_mib": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None}
            history.append(row)
            state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch,
                     "best": best, "bad": bad, "torch_rng": torch.get_rng_state(),
                     "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
                     "loader_rng": generator.get_state()}
            # history antes de checkpoint; al reanudar se descartan filas no comprometidas.
            write_json(output / "history.json", history)
            if improved:
                save_checkpoint(output / "best.pt", state)
            save_checkpoint(output / "last.pt", state)
            print(json.dumps(row), flush=True)
            if (epochs_per_cycle and epoch + 1 - start_epoch >= epochs_per_cycle
                    and epoch + 1 < cfg.epochs and bad < cfg.patience):
                result = {"output": str(output), "complete": False,
                          "reason": "epoch_cycle_limit", "epochs_completed": epoch + 1,
                          "epochs_target": cfg.epochs, "best_val_rmse_mm_h": best}
                write_json(output / "status.json", result)
                return result  # finally cierra los datasets antes de salir con 75.
    finally:
        train_ds.close()
        val_ds.close()
    result = evaluate(data_path, output, cache=cache, device=str(device), split="validation")
    result.update(complete=True, epochs_completed=len(history), epochs_target=cfg.epochs,
                  stop_reason="early_stopping" if bad >= cfg.patience else "epochs_reached")
    write_json(output / "status.json", result)
    return result


def evaluate(data_path, run, cache=None, device="cpu", split="validation", output=None, persistence=False):
    device = device_for(device)
    normalizer = model = None
    if persistence:
        cfg = TrainConfig()
        cfg.threads = 2
        if cache:
            raise ValueError("Persistencia no necesita cache")
    else:
        run = Path(run)
        effective = json.loads((run / "config.json").read_text())
        cfg = TrainConfig(**effective["training"])
        torch.set_num_threads(cfg.threads)
        if split == "validation" and sha256(data_path) != effective["data_sha256"]:
            raise ValueError("Validación debe usar el mismo archivo de entrenamiento")
        normalizer = json.loads((run / "normalizer.json").read_text())
        model = ForecastModel(ModelConfig(**effective["model"])).to(device)
        saved = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(saved["model"])
    dataset = RunoffDataset(data_path, split, normalizer, cache, cfg.max_val if split == "validation" else 0, cfg.seed)
    if not persistence:
        expected = json.loads((run / "decomposition.json").read_text())
        if dataset.cache_config != expected:
            raise ValueError("Usar la misma configuración de descomposición de entrenamiento")
    try:
        result = infer(model, loader(dataset, cfg), normalizer, device)
    finally:
        dataset.close()
    destination = Path(output) if output else Path(run) / ("test" if split == "test" else "validation")
    destination.mkdir(parents=True, exist_ok=True)
    csv_path = destination / "predictions.csv"
    with open(csv_path.with_suffix(".tmp"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Id"] + [f"q_{h:02d}" for h in range(1, 49)])
        for row_id, values in zip(result["ids"], result["pred"]):
            writer.writerow([int(row_id)] + values.tolist())
    csv_path.with_suffix(".tmp").replace(csv_path)
    np.savez_compressed(destination / "predictions.npz", **{k: v for k, v in result.items() if v is not None})
    metadata = {"n": len(result["ids"]), "source_sha256": sha256(data_path), "split": split,
                "units": "mm/h", "clipping": False, "persistence": persistence,
                "diagnostic_subset": bool(cfg.max_val) if split == "validation" else False}
    write_json(destination / "metadata.json", metadata)
    if result["y"] is not None:
        metrics = report(result["y"], result["pred"], result["basins"])
        write_json(destination / "metrics.json", metrics)
        return {"output": str(destination), "global": metrics["global"], **metadata}
    return {"output": str(destination), **metadata}
