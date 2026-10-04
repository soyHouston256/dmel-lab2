"""Métricas en mm/h. Las ventanas se evalúan como pares muestra/horizonte."""
import numpy as np


def scores(y, pred):
    y, pred = np.asarray(y, np.float64).ravel(), np.asarray(pred, np.float64).ravel()
    if y.shape != pred.shape or not len(y) or not np.isfinite(y).all() or not np.isfinite(pred).all():
        raise ValueError("Métricas requieren predicciones/objetivos finitos y del mismo tamaño")
    error = pred - y
    sy, sp, my, mp = y.std(), pred.std(), y.mean(), pred.mean()
    nse = float(1 - np.sum(error ** 2) / np.sum((y - my) ** 2)) if sy > 1e-12 else None
    r = float(np.corrcoef(y, pred)[0, 1]) if min(sy, sp) > 1e-12 else None
    alpha = float(sp / sy) if sy > 1e-12 else None
    beta = float(mp / my) if abs(my) > 1e-12 else None
    kge = float(1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)) if None not in (r, alpha, beta) else None
    return {"n": len(y), "rmse": float(np.sqrt(np.mean(error ** 2))), "mae": float(np.mean(abs(error))),
            "bias": float(error.mean()), "nse": nse, "kge2009": kge, "r": r, "alpha": alpha, "beta": beta,
            "negative_prediction_fraction": float(np.mean(pred < 0))}


def report(y, pred, basins):
    if y.shape != pred.shape or y.ndim != 2 or y.shape[1] != 48 or len(basins) != len(y):
        raise ValueError("Se requiere y/pred [N,48] y basins [N]")
    by_basin = {str(int(b)): scores(y[basins == b], pred[basins == b]) for b in np.unique(basins)}
    macro = {}
    for metric in ("rmse", "mae", "nse", "kge2009"):
        values = [row[metric] for row in by_basin.values() if row[metric] is not None]
        macro[metric] = {"mean": float(np.mean(values)) if values else None, "valid_basins": len(values)}
    return {"units": "mm/h", "aggregation": "pares muestra/horizonte; no serie cronológica reconstruida",
            "global": scores(y, pred), "per_horizon": {str(h + 1): scores(y[:, h], pred[:, h]) for h in range(48)},
            "per_basin": by_basin, "macro_basin": macro}
