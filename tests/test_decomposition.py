"""Scientific invariants and edge cases of our causal-window decomposition."""

import json
import math

import numpy as np
import pytest

from dmel.decomposition import DecompositionConfig, ceemdan, decompose_window, sample_entropy


def signal(n=128):
    t = np.arange(n, dtype=np.float64)
    return 20 + 0.02 * t + 2 * np.sin(2 * np.pi * t / 8) + 4 * np.sin(2 * np.pi * t / 40)


@pytest.fixture
def config():
    return DecompositionConfig(trials=5, max_imfs=4, max_siftings=25)


def test_ceemdan_reconstructs_from_modes_and_retained_residual(config):
    q = signal()
    imfs, residual = ceemdan(q, config)
    assert 1 <= imfs.shape[0] <= config.max_imfs
    assert imfs.shape[1] == q.size
    np.testing.assert_allclose(imfs.sum(axis=0) + residual, q, rtol=1e-13, atol=1e-13)
    assert np.std(residual) > 0  # Do not silently discard the remaining trend.


def test_first_emd_mode_recovers_known_fast_oscillation():
    q = signal(256)
    imfs, _ = ceemdan(q, DecompositionConfig(noise_scale=0, max_imfs=4))
    fast = 2 * np.sin(2 * np.pi * np.arange(q.size) / 8)
    # End effects are real; evaluate the known interior oscillation separately.
    assert np.corrcoef(imfs[0, 16:-16], fast[16:-16])[0, 1] > 0.98


def test_two_channel_output_preserves_input_and_matches_routing(config):
    q = signal()
    before = q.copy()
    components, diagnostic = decompose_window(q, config, sample_id=17)
    imfs, residual = ceemdan(q, config, sample_id=17)
    assert components.shape == (q.size, 2)
    assert components.dtype == np.float32
    np.testing.assert_array_equal(q, before)
    np.testing.assert_allclose(components.sum(axis=1), q, rtol=2e-7, atol=3e-6)
    high = [i for i, value in enumerate(diagnostic["assignments"]) if value == "high"]
    expected = imfs[high].sum(axis=0) if high else np.zeros_like(q)
    np.testing.assert_allclose(components[:, 0], expected, rtol=2e-7, atol=3e-6)
    np.testing.assert_allclose(components[:, 1], q - expected, rtol=2e-7, atol=3e-6)
    assert np.isfinite(residual).all()
    json.dumps(diagnostic, allow_nan=False)


@pytest.mark.parametrize("q", [np.ones(16) * 3.5, np.zeros(16), np.arange(16), -np.arange(16)])
def test_constant_and_monotone_series_are_entirely_low(config, q):
    components, diagnostic = decompose_window(q, config)
    assert diagnostic["n_imfs"] == 0
    np.testing.assert_array_equal(components[:, 0], np.zeros(len(q), dtype=np.float32))
    np.testing.assert_array_equal(components[:, 1], q.astype(np.float32))
    json.dumps(diagnostic, allow_nan=False)


def test_seed_and_sample_id_are_deterministic_and_local(config):
    q = signal()
    np.random.seed(13)
    expected_global_draw = np.random.random(4)
    np.random.seed(13)
    a, da = decompose_window(q, config, sample_id=9)
    np.testing.assert_array_equal(np.random.random(4), expected_global_draw)
    b, db = decompose_window(q, config, sample_id=9)
    np.testing.assert_array_equal(a, b)
    assert da == db
    imfs_a, _ = ceemdan(q, config, sample_id=9)
    imfs_b, _ = ceemdan(q, config, sample_id=10)
    assert not np.array_equal(imfs_a, imfs_b)


def test_same_visible_prefix_ignores_arbitrary_future(config):
    prefix = signal(96)
    stream_a = np.concatenate([prefix, np.zeros(30)])
    stream_b = np.concatenate([prefix, np.full(30, 1e8)])
    a, da = decompose_window(stream_a[:96], config, sample_id=95)
    b, db = decompose_window(stream_b[:96], config, sample_id=95)
    np.testing.assert_array_equal(a, b)
    assert da == db


def brute_force_sampen(q, m, r):
    x = np.asarray(q, dtype=np.float64)
    tolerance = r * x.std(ddof=0)
    a = b = 0
    for i in range(len(x) - m):
        for j in range(i + 1, len(x) - m):
            if np.max(np.abs(x[i:i + m] - x[j:j + m])) <= tolerance:
                b += 1
                if abs(x[i + m] - x[j + m]) <= tolerance:
                    a += 1
    return -math.log(a / b) if a and b else math.inf


def test_sample_entropy_matches_independent_pair_count_and_affine_invariance():
    q = np.array([0, 1, 0, 1, 0, 2, 0, 1, 0, 1, 0, 1], dtype=float)
    expected = brute_force_sampen(q, m=2, r=0.2)
    assert math.isfinite(expected) and expected > 0
    assert sample_entropy(q) == pytest.approx(expected, abs=1e-12)
    assert sample_entropy(7 * q + 100) == pytest.approx(expected, abs=1e-12)
    assert sample_entropy(np.ones(10)) == 0


def test_infinite_entropy_is_explicit_and_json_safe(config):
    q = np.array([0, 1, 4, 9, 16, 25], dtype=float)
    assert math.isinf(sample_entropy(q, r=1e-6))
    tiny_r = DecompositionConfig(trials=3, max_imfs=2, entropy_r=1e-6)
    components, diagnostic = decompose_window(q, tiny_r)
    assert diagnostic["entropy_original"] is None
    assert diagnostic["entropy_status"]["original"] == "no_template_matches"
    assert np.isfinite(components).all()
    json.dumps(diagnostic, allow_nan=False)


def test_infinity_ties_route_all_imfs_low():
    q = np.array([0.7, 4.2, 1.1, 3.3, -2.8, 7.6, 0.01, 8.9])
    config = DecompositionConfig(trials=3, max_imfs=2, entropy_r=1e-10)
    result, diagnostic = decompose_window(q, config)
    assert diagnostic["n_imfs"] > 0
    assert diagnostic["entropy_original"] is None
    assert diagnostic["assignments"] == ["low"] * diagnostic["n_imfs"]
    np.testing.assert_array_equal(result[:, 0], np.zeros(len(q), dtype=np.float32))


@pytest.mark.parametrize("q", [[1, 2, np.nan, 4], [1, 2, np.inf, 4], [], [1, 2, 3], [[1, 2], [3, 4]], [1j] * 8])
def test_invalid_input_rejected(config, q):
    with pytest.raises(ValueError):
        decompose_window(q, config)


@pytest.mark.parametrize("change", [{"trials": 0}, {"seed": -1}, {"max_imfs": 0}, {"max_siftings": 0},
                                     {"entropy_m": 0}, {"entropy_r": 0}, {"noise_scale": -0.2},
                                     {"sift_tolerance": float("nan")}, {"trials": 2.5}, {"seed": True}])
def test_invalid_configuration_rejected(change):
    with pytest.raises(ValueError):
        DecompositionConfig(**change)


def test_max_modes_keeps_all_unextracted_content_in_residual():
    q = signal()
    imfs, residual = ceemdan(q, DecompositionConfig(trials=3, max_imfs=1))
    assert imfs.shape == (1, len(q))
    np.testing.assert_allclose(imfs.sum(axis=0) + residual, q, atol=1e-13)
