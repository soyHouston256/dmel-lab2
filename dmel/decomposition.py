"""Own finite-window EMD/CEEMDAN and entropy routing for the DMEL adaptation.

This is a bounded, Torres-2011-style CEEMDAN implementation, not a call to
PyEMD and not a claim of numerical equivalence to the unpublished DMEL code.
Only samples explicitly supplied in ``q`` are used. See docs/decomposition.md.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from numbers import Integral, Real

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import find_peaks
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class DecompositionConfig:
    trials: int = 100
    noise_scale: float = 0.2
    seed: int = 42
    max_imfs: int = 8
    max_siftings: int = 50
    sift_tolerance: float = 0.05
    entropy_m: int = 2
    entropy_r: float = 0.2

    def __post_init__(self) -> None:
        for name in ("trials", "max_imfs", "max_siftings", "entropy_m"):
            _validate_integer(name, getattr(self, name), minimum=1)
        _validate_integer("seed", self.seed, minimum=0)
        for name in ("noise_scale", "sift_tolerance", "entropy_r"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError(f"{name} must be a finite real number")
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
            if name != "noise_scale" and value == 0:
                raise ValueError(f"{name} must be positive")


def _validate_integer(name: str, value: int, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _series(q: np.ndarray, minimum_length: int = 1) -> np.ndarray:
    if np.iscomplexobj(q):
        raise ValueError("q must be real-valued")
    try:
        result = np.asarray(q, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("q must be a one-dimensional numeric sequence") from exc
    if result.ndim != 1 or result.size < minimum_length:
        raise ValueError(f"q must be one-dimensional with at least {minimum_length} samples")
    if not np.isfinite(result).all():
        raise ValueError("q must contain only finite samples; handle missing data upstream")
    if np.max(np.abs(result)) > np.finfo(np.float32).max:
        raise ValueError("q exceeds the float32 output range")
    return result.copy()


def _extrema(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # find_peaks uses the midpoint of a flat extremum; endpoints are excluded.
    return find_peaks(x)[0], find_peaks(-x)[0]


def _oscillatory(x: np.ndarray) -> bool:
    maxima, minima = _extrema(x)
    return maxima.size > 0 and minima.size > 0


def _envelope(x: np.ndarray, extrema: np.ndarray) -> np.ndarray:
    """Natural cubic envelope with two mirrored extrema at each boundary."""
    n = x.size
    left = extrema[:2][::-1]
    right = extrema[-2:][::-1]
    knots = np.concatenate((-left, extrema, 2 * (n - 1) - right))
    values = np.concatenate((x[left], x[extrema], x[right]))
    return CubicSpline(knots, values, bc_type="natural")(np.arange(n))


def _zero_crossings(x: np.ndarray) -> int:
    nonzero = x[x != 0]
    return int(np.count_nonzero(np.signbit(nonzero[1:]) != np.signbit(nonzero[:-1])))


def _first_imf(x: np.ndarray, config: DecompositionConfig) -> np.ndarray | None:
    h = x.copy()
    if not _oscillatory(h):
        return None
    for _ in range(config.max_siftings):
        maxima, minima = _extrema(h)
        if not maxima.size or not minima.size:
            return None
        mean_envelope = (_envelope(h, maxima) + _envelope(h, minima)) * 0.5
        updated = h - mean_envelope
        if not np.isfinite(updated).all():
            raise FloatingPointError("non-finite EMD envelope; input decomposition failed")
        energy = float(np.dot(h, h))
        relative_change = float(np.dot(mean_envelope, mean_envelope)) / max(
            energy, np.finfo(np.float64).tiny
        )
        maxima, minima = _extrema(updated)
        counts_agree = abs(maxima.size + minima.size - _zero_crossings(updated)) <= 1
        h = updated
        if relative_change <= config.sift_tolerance and counts_agree:
            break
    return h


def _emd(x: np.ndarray, config: DecompositionConfig, max_imfs: int) -> list[np.ndarray]:
    residual = x.copy()
    modes: list[np.ndarray] = []
    minimum_energy = np.finfo(np.float64).eps**2 * max(float(np.dot(x, x)), 1.0)
    for _ in range(max_imfs):
        mode = _first_imf(residual, config)
        if mode is None or float(np.dot(mode, mode)) <= minimum_energy:
            break
        modes.append(mode)
        residual = residual - mode
        if not _oscillatory(residual):
            break
    return modes


def _unit_std(x: np.ndarray) -> np.ndarray:
    scale = float(np.std(x, ddof=0))
    return x / scale if scale > np.finfo(np.float64).eps else np.zeros_like(x)


def ceemdan(
    q: np.ndarray,
    config: DecompositionConfig,
    sample_id: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return float64 ``(imfs [K,L], residual [L])`` using only this window.

    At stage zero white noise is used. Later stages use the corresponding
    noise IMF normalized to unit standard deviation. At every stage the noise
    amplitude is ``noise_scale * std(current residual)``. Missing noise IMFs
    contribute zero noise. The final residual is retained even at ``max_imfs``.
    """
    if not isinstance(config, DecompositionConfig):
        raise TypeError("config must be a DecompositionConfig")
    _validate_integer("sample_id", sample_id, minimum=0)
    x = _series(q)
    empty = np.empty((0, x.size), dtype=np.float64)
    scale = float(np.std(x, ddof=0))
    if scale == 0 or not _oscillatory(x):
        return empty, x

    offset = float(np.mean(x))
    standardized = (x - offset) / scale
    if config.noise_scale == 0:
        modes = _emd(standardized, config, config.max_imfs)
    else:
        rng = np.random.default_rng(np.random.SeedSequence([int(config.seed), int(sample_id)]))
        noises = rng.standard_normal((config.trials, x.size))
        noises -= noises.mean(axis=1, keepdims=True)
        noises = np.asarray([_unit_std(noise) for noise in noises])
        noise_modes = [
            _emd(noise, config, config.max_imfs - 1) for noise in noises
        ]
        modes = []
        residual = standardized.copy()
        for stage in range(config.max_imfs):
            if not _oscillatory(residual):
                break
            amplitude = config.noise_scale * float(np.std(residual, ddof=0))
            average = np.zeros_like(residual)
            for trial in range(config.trials):
                if stage == 0:
                    noise = noises[trial]
                elif stage - 1 < len(noise_modes[trial]):
                    noise = _unit_std(noise_modes[trial][stage - 1])
                else:
                    noise = np.zeros_like(residual)
                mode = _first_imf(residual + amplitude * noise, config)
                if mode is not None:
                    average += mode / config.trials
            if float(np.dot(average, average)) <= np.finfo(np.float64).eps**2:
                break
            modes.append(average)
            residual = residual - average

    imfs = np.asarray(modes, dtype=np.float64).reshape(-1, x.size) * scale
    # Subtract from original input, so trend, offset and unextracted content
    # remain in the residual regardless of the stopping rule.
    residual = x - imfs.sum(axis=0)
    if not np.isfinite(imfs).all() or not np.isfinite(residual).all():
        raise FloatingPointError("CEEMDAN produced non-finite components")
    return imfs, residual


def _entropy_details(x: np.ndarray, m: int, r: float) -> tuple[float, str, int, int]:
    """SampEn on equal-sized template sets; unordered pairs, no self-matches."""
    n_templates = x.size - m
    if n_templates < 2:
        return math.inf, "insufficient_samples", 0, 0
    scale = float(np.std(x, ddof=0))
    if scale == 0:
        pairs = n_templates * (n_templates - 1) // 2
        return 0.0, "constant", pairs, pairs
    # Work in standard-deviation units, to avoid overflow and unit sensitivity.
    normalized = (x - float(np.mean(x))) / scale
    templates = np.lib.stride_tricks.sliding_window_view(normalized, m + 1)

    def count_pairs(vectors: np.ndarray) -> int:
        tree = cKDTree(np.ascontiguousarray(vectors))
        ordered_with_self = int(tree.count_neighbors(tree, r, p=np.inf))
        return (ordered_with_self - n_templates) // 2

    b = count_pairs(templates[:, :m])
    a = count_pairs(templates)
    if b == 0:
        return math.inf, "no_template_matches", a, b
    if a == 0:
        return math.inf, "no_extended_matches", a, b
    return float(-math.log(a / b)), "finite", a, b


def sample_entropy(q: np.ndarray, m: int = 2, r: float = 0.2) -> float:
    """SampEn with Chebyshev <= r*SD(ddof=0); infinity if no usable matches.

    ``r`` is a dimensionless standard-deviation multiplier, not an absolute
    tolerance. Constants yield zero. At least m+2 points are required.
    """
    _validate_integer("m", m, minimum=1)
    if isinstance(r, bool) or not isinstance(r, Real) or not math.isfinite(r) or r <= 0:
        raise ValueError("r must be finite and positive")
    x = _series(q, minimum_length=m + 2)
    return _entropy_details(x, m, r)[0]


def _json_entropy(details: tuple[float, str, int, int]) -> dict:
    value, status, a, b = details
    return {"value": value if math.isfinite(value) else None,
            "status": status, "matches_m_plus_1": int(a), "matches_m": int(b)}


def decompose_window(
    q: np.ndarray,
    config: DecompositionConfig,
    sample_id: int = 0,
) -> tuple[np.ndarray, dict]:
    """Return ``(components [L,2] float32, JSON-safe diagnostics)``.

    Column 0 sums IMFs whose SampEn is strictly greater than SampEn(q).
    Column 1 contains every other IMF and the residual. Infinite entropies
    use extended-real ordering, so infinity ties route low. The caller must
    slice q at the forecast origin; this function cannot inspect timestamps.
    """
    if not isinstance(config, DecompositionConfig):
        raise TypeError("config must be a DecompositionConfig")
    x = _series(q, minimum_length=config.entropy_m + 2)
    imfs, residual = ceemdan(x, config, sample_id=sample_id)
    original = _entropy_details(x, config.entropy_m, config.entropy_r)
    mode_entropies = [_entropy_details(mode, config.entropy_m, config.entropy_r)
                      for mode in imfs]
    high_indices = [i for i, entropy in enumerate(mode_entropies) if entropy[0] > original[0]]
    assignments = ["high" if i in high_indices else "low" for i in range(len(imfs))]
    high = imfs[high_indices].sum(axis=0) if high_indices else np.zeros_like(x)
    low = x - high  # Exactly includes low IMFs, residual and numerical remainder.
    components64 = np.column_stack((high, low))
    if np.max(np.abs(components64)) > np.finfo(np.float32).max:
        raise FloatingPointError("decomposed components exceed float32 range")
    components = components64.astype(np.float32)
    reconstruction_error = float(np.max(np.abs(components.astype(np.float64).sum(axis=1) - x)))
    diagnostics = {
        "algorithm": "own-ceemdan-torres2011-fixed-stage-snr-v1",
        "routing": "sampen-imf-strictly-greater-than-window;residual-low;infinity-ties-low",
        "n_samples": int(x.size),
        "n_imfs": int(len(imfs)),
        "seed": int(config.seed),
        "sample_id": int(sample_id),
        "entropy_original": original[0] if math.isfinite(original[0]) else None,
        "entropy_imfs": [e[0] if math.isfinite(e[0]) else None for e in mode_entropies],
        "entropy_status": {"original": original[1], "imfs": [e[1] for e in mode_entropies]},
        "entropy_details": {"original": _json_entropy(original),
                            "imfs": [_json_entropy(e) for e in mode_entropies]},
        "assignments": assignments,
        "residual_channel": "low",
        "residual_std": float(np.std(residual)),
        "reconstruction_error": reconstruction_error,
        "relative_reconstruction_error": reconstruction_error / max(float(np.max(np.abs(x))), 1.0),
        "config": {key: int(value) if isinstance(value, Integral) else float(value)
                   for key, value in asdict(config).items()},
    }
    return components, diagnostics
