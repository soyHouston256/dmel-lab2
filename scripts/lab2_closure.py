"""Auditoría del cierre real del laboratorio; no conecta a Khipu ni lee test_targets."""
import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path

import h5py
import numpy as np
import scipy
import torch

from dmel import decomposition
from dmel.data import RunoffDataset, schema, select_ids, sha256
from dmel.decomposition import DecompositionConfig
from dmel.engine import environment, read_config, write_json
from dmel.metrics import report
from dmel.models import ForecastModel, ModelConfig


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def model_info(run):
    run = Path(run)
    cfg = read(run / "config.json")
    model = ForecastModel(ModelConfig(**cfg["model"]))
    checkpoint = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"], strict=True)
    return {"model": cfg["model"]["name"], "parameters": sum(p.numel() for p in model.parameters()),
            "best_epoch_zero_based": checkpoint["epoch"], "checkpoint_sha256": sha256(run / "best.pt")}


def preflight(root, ablation_config):
    root = Path(root)
    train, test, cache = root / "dataset/train.h5", root / "dataset/test.h5", root / "cache/train.h5"
    reference = root / "runs/dmel-seed42"
    cfg = read(reference / "config.json")
    train_schema, test_schema = schema(train), schema(test)
    require(train_schema["n"] == 272142 and train_schema["supervised"], "Train real incorrecto")
    require(test_schema["n"] == 27983 and not test_schema["supervised"], "Test real incorrecto")
    train_hash, cache_hash, test_hash = sha256(train), sha256(cache), sha256(test)
    require(train_hash == cfg["data_sha256"], "Train difiere del entrenamiento original")
    require(cache_hash == cfg["cache_sha256"], "Cache train difiere del entrenamiento original")
    implementation = {p.name: sha256(p) for p in Path(decomposition.__file__).parent.glob("*.py")}
    require(implementation == cfg["implementation"], "Código dmel distinto del experimento original")
    expected_decomp = read(reference / "decomposition.json")
    current_decomp = {"parameters": asdict(DecompositionConfig()),
                      "implementation_sha256": sha256(decomposition.__file__),
                      "numpy": np.__version__, "scipy": scipy.__version__}
    require(current_decomp == expected_decomp, "Descomposición/versiones distintas del entrenamiento")
    require(read("configs/decomposition.json") == expected_decomp["parameters"], "Config de test incompatible")
    ids_train, ids_val = select_ids(train, "train"), select_ids(train, "validation")
    require((len(ids_train), len(ids_val)) == (254000, 18142), "Partición real incorrecta")
    with np.load(reference / "split_ids.npz", allow_pickle=False) as ids:
        require(np.array_equal(ids["train"], ids_train) and np.array_equal(ids["validation"], ids_val),
                "Particiones distintas del experimento original")
    ds = RunoffDataset(train, "all", cache=cache, source_sha256=train_hash)
    try:
        require(ds.cache_config == expected_decomp, "Cache con otra descomposición")
        for row in (0, len(ds) // 2, len(ds) - 1):
            ds[row]  # Integridad de reconstrucción en tres filas, además del hash completo.
    finally:
        ds.close()
    model, training = read_config(ablation_config)
    candidate = asdict(model)
    require(candidate.pop("name") == "mrs_informer", "Ablación incorrecta")
    require(candidate == {k: v for k, v in cfg["model"].items() if k != "name"}, "Cambian otros hiperparámetros")
    require(asdict(training) == cfg["training"], "Presupuesto/semilla distintos de DMEL")
    runs = {}
    for name in ("dmel-seed42", "lstm-seed42-fd-fix"):
        run = root / "runs" / name
        require(read(run / "status.json")["complete"] is True, f"Corrida incompleta: {name}")
        runs[name] = model_info(run)
    return {"ready": True, "environment": environment(), "train_sha256": train_hash,
            "train_cache_sha256": cache_hash, "test_sha256": test_hash,
            "n_train": len(ids_train), "n_validation": len(ids_val), "n_test": test_schema["n"],
            "existing_runs": runs, "ablation_parameters": sum(p.numel() for p in ForecastModel(model).parameters()),
            "ablation_config_sha256": sha256(ablation_config),
            "scope": "Preflight solamente; no acredita ejecución de ablación ni exportación de test"}


def verify_test(data, run, predictions):
    data, run, predictions = Path(data), Path(run), Path(predictions)
    info, metadata = schema(data), read(predictions / "metadata.json")
    require(info["n"] == 27983 and not info["supervised"], "Se requiere el test real de 27983 filas sin etiquetas")
    expected_ids = np.arange(info["n"])
    data_hash = sha256(data)
    require(metadata.get("n") == len(expected_ids) and metadata.get("split") == "test"
            and metadata.get("source_sha256") == data_hash and metadata.get("units") == "mm/h"
            and metadata.get("diagnostic_subset") is False and metadata.get("clipping") is False
            and metadata.get("persistence") is False, "Metadatos de test incompatibles")
    with h5py.File(data) as f:
        expected_basins = f["basin_id"][:]
    with np.load(predictions / "predictions.npz", allow_pickle=False) as archive:
        require("y" not in archive.files, "El test exportado no debe contener etiquetas")
        pred = archive["pred"]
        require(pred.shape == (27983, 48) and np.isfinite(pred).all(), "Forma o valores del NPZ incorrectos")
        require(np.array_equal(archive["ids"], expected_ids), "Ids duplicados, omitidos o desordenados")
        require(np.array_equal(archive["basins"], expected_basins), "Cuencas fuera del orden del HDF5")
    count = 0
    with (predictions / "predictions.csv").open(newline="") as stream:
        reader = csv.reader(stream)
        require(next(reader) == ["Id"] + [f"q_{h:02d}" for h in range(1, 49)], "Cabecera CSV incorrecta")
        for count, row in enumerate(reader, 1):
            require(count <= 27983 and len(row) == 49 and row[0] == str(count - 1), "Filas/Ids CSV incorrectos")
            values = np.asarray(row[1:], dtype=np.float64)
            require(np.isfinite(values).all() and np.array_equal(values, pred[count - 1].astype(np.float64)),
                    f"CSV y NPZ difieren en fila {count - 1}")
    require(count == 27983, "CSV incompleto")
    require(not (predictions / "metrics.json").exists(), "Hay métricas inesperadas de test sin etiquetas")
    return {"complete": True, "n": count, "horizons": 48, "prediction_values": count * 48,
            "units": "mm/h", "id_range": [0, count - 1], "source_sha256": data_hash,
            "csv_sha256": sha256(predictions / "predictions.csv"),
            "npz_sha256": sha256(predictions / "predictions.npz"),
            "run": str(run), "checkpoint_present_at_verification": model_info(run),
            "negative_prediction_fraction": float((pred < 0).mean()),
            "scope": "Integridad de exportación; sin métricas ni selección con objetivos de test"}


def compare_ablation(reference, ablation):
    reference, ablation = Path(reference), Path(ablation)
    a, b = read(reference / "config.json"), read(ablation / "config.json")
    require(a["model"]["name"] == "dmel" and b["model"]["name"] == "mrs_informer", "Par de modelos incorrecto")
    for key in ("training", "data_sha256", "cache_sha256", "implementation"):
        require(a[key] == b[key], f"Comparación incompatible: {key}")
    require({k: v for k, v in a["model"].items() if k != "name"}
            == {k: v for k, v in b["model"].items() if k != "name"}, "Cambian otros hiperparámetros")
    for file in ("normalizer.json", "decomposition.json"):
        require(read(reference / file) == read(ablation / file), f"Comparación incompatible: {file}")
    with np.load(reference / "split_ids.npz") as sa, np.load(ablation / "split_ids.npz") as sb:
        for key in ("train", "validation"):
            require(np.array_equal(sa[key], sb[key]), "Particiones diferentes")
    metrics, model_details, arrays = {}, {}, []
    for name, run in (("dmel", reference), ("mrs_informer", ablation)):
        status, meta = read(run / "status.json"), read(run / "validation/metadata.json")
        require(status.get("complete") is True and not meta["diagnostic_subset"]
                and meta["split"] == "validation" and meta["n"] == 18142
                and meta["source_sha256"] == a["data_sha256"], f"Validación incompleta/incorrecta: {name}")
        with np.load(run / "validation/predictions.npz", allow_pickle=False) as archive:
            arr = {k: archive[k] for k in ("pred", "y", "ids", "basins")}
        require(arr["pred"].shape == arr["y"].shape == (18142, 48), "Validación con dimensiones incorrectas")
        with np.load(run / "split_ids.npz") as ids:
            require(np.array_equal(arr["ids"], ids["validation"]), "Ids de validación incompletos")
        arrays.append(arr)
        metrics[name] = report(arr["y"], arr["pred"], arr["basins"])
        model_details[name] = {**model_info(run), "epochs_completed": status["epochs_completed"]}
    for key in ("y", "ids", "basins"):
        require(np.array_equal(arrays[0][key], arrays[1][key]), f"Validación no emparejada: {key}")
    delta = metrics["mrs_informer"]["global"]["rmse"] - metrics["dmel"]["global"]["rmse"]
    return {"complete": True, "contrast": "Rama baja LSTM (DMEL) vs. Informer (MRS–Informer)",
            "training_budget_epochs": a["training"]["epochs"],
            "model_details": model_details, "metrics": metrics, "rmse_delta_ablation_minus_dmel": delta,
            "rmse_change_percent": 100 * delta / metrics["dmel"]["global"]["rmse"],
            "interpretation": "Delta positivo favorece DMEL; negativo favorece la ablación.",
            "limitations": ["El resultado solo describe el presupuesto de épocas indicado; no demuestra convergencia.",
                            "Una semilla; sin prueba de significancia.",
                            "Los checkpoints se seleccionaron en esta validación.",
                            "Cambiar la rama cambia también el número de parámetros.",
                            "Agregación de pares muestra/horizonte; no serie cronológica reconstruida."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("preflight")
    p.add_argument("--root", default="migration/khipu")
    p.add_argument("--config", default="configs/mrs_informer.json")
    p = sub.add_parser("verify-test")
    p.add_argument("--data", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--predictions", required=True)
    p = sub.add_parser("compare-ablation")
    p.add_argument("--reference", required=True)
    p.add_argument("--ablation", required=True)
    for p in sub.choices.values():
        p.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.command == "preflight":
        result = preflight(args.root, args.config)
    elif args.command == "verify-test":
        result = verify_test(args.data, args.run, args.predictions)
    else:
        result = compare_ablation(args.reference, args.ablation)
    result["verified_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(args.out, result)
    print(json.dumps({"report": args.out, "ready": result.get("ready"), "complete": result.get("complete")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
