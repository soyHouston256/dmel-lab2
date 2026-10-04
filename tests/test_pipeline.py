from dataclasses import asdict
import json

import h5py
import numpy as np
import pytest
import torch

from dmel.data import RunoffDataset, fit_normalizer, select_ids
from dmel.engine import evaluate, train, write_json
from dmel.metrics import scores
from dmel.models import ForecastModel, ModelConfig, ProbSparse
from dmel.decomposition import DecompositionConfig
from dmel.precompute import precompute


@pytest.fixture
def hdf(tmp_path):
    rng = np.random.default_rng(5)
    x = rng.normal(size=(12, 336, 12)).astype("f4")
    x[:, :, 11] = np.maximum(0, 2 + 0.5 * np.sin(np.arange(336) / 13))[None]
    y = np.repeat(x[:, -1, 11, None], 48, axis=1)
    path = tmp_path / "train.h5"
    with h5py.File(path, "w") as f:
        f["X"], f["y"] = x, y
        f["split"] = np.array([0] * 8 + [1] * 4)
        f["basin_id"] = np.arange(12) % 3
    return path


def tiny(name="dmel"):
    return ModelConfig(name=name, d_model=8, heads=2, ff=16, lstm_hidden=8,
                       lstm_layers=1, label_len=8, factor=1, dropout=0)


def test_train_only_scaling_and_hdf_ids(hdf):
    ids = select_ids(hdf, "train")
    before = fit_normalizer(hdf, ids)
    with h5py.File(hdf, "r+") as f:
        f["X"][8:] = 1e8
        f["y"][8:] = -1e8
    assert fit_normalizer(hdf, ids) == before
    with pytest.raises(ValueError, match="split=0"):
        fit_normalizer(hdf, np.arange(12))
    ds = RunoffDataset(hdf, "validation", before)
    assert ds[0]["Id"] == 8
    assert "y_aux" not in ds[0]
    ds.close()


def test_entropy_cache_roundtrip_and_future_independence(hdf, tmp_path):
    cfg = DecompositionConfig(trials=2, max_imfs=2, max_siftings=5)
    first, second = tmp_path / "first.h5", tmp_path / "second.h5"
    precompute(hdf, first, cfg, limit=2)
    with h5py.File(hdf, "r+") as f:
        f["y"][:] = 1000
        f["y_aux"] = np.ones((12, 48, 11)) * 1e6
    precompute(hdf, second, cfg, limit=2)
    with h5py.File(first) as a, h5py.File(second) as b:
        np.testing.assert_array_equal(a["components"][:], b["components"][:])
    with pytest.raises(ValueError, match="fuente"):
        RunoffDataset(hdf, "all", cache=first, limit=2)
    norm = fit_normalizer(hdf, select_ids(hdf, "train"))
    ds = RunoffDataset(hdf, "all", norm, second, limit=2)
    item = ds[0]
    np.testing.assert_allclose(item["components"].sum(-1), item["X"][:, 11], atol=1e-6)
    ds.close()


@pytest.mark.parametrize("name", ["lstm", "informer", "dmel", "mrs_informer", "mrs_lstm"])
def test_model_gradient_shape_and_batch_invariance(name):
    torch.set_num_threads(2)
    torch.manual_seed(9)
    model = ForecastModel(tiny(name))
    x = torch.randn(2, 336, 12)
    components = torch.stack((x[:, :, 11] * .2, x[:, :, 11] * .8), dim=-1)
    model.eval()
    pred = model(x, components)
    assert pred.shape == (2, 48)
    torch.testing.assert_close(pred[:1], model(x[:1], components[:1]), atol=2e-6, rtol=1e-5)
    pred.square().mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    if name.startswith("mrs") or name == "dmel":
        assert sum(p.grad.abs().sum() for p in model.high.parameters() if p.grad is not None) > 0
        assert sum(p.grad.abs().sum() for p in model.low.parameters() if p.grad is not None) > 0


def test_probsparse_dense_limit_matches_attention():
    torch.manual_seed(12)
    attention = ProbSparse(8, 2, factor=100, dropout=0)
    x = torch.randn(2, 5, 8)
    q, k, v = attention.qkv(x).reshape(2, 5, 3, 2, 4).permute(2, 0, 3, 1, 4)
    expected = attention.projection(((q @ k.transpose(-1, -2) / 2).softmax(-1) @ v).transpose(1, 2).reshape(2, 5, 8))
    torch.testing.assert_close(attention(x), expected)


def test_metrics_perfect_and_degenerate():
    y = np.arange(1, 10)
    result = scores(y, y)
    assert result["rmse"] == 0
    assert result["nse"] == 1
    assert result["kge2009"] == pytest.approx(1)
    constant = scores(np.ones(5), np.ones(5))
    assert constant["nse"] is None and constant["kge2009"] is None
    with pytest.raises(ValueError):
        scores([1, np.nan], [1, 2])


def test_train_reload_resume_and_prediction(hdf, tmp_path):
    cfg = {"model": asdict(tiny("lstm")), "training": {"epochs": 2, "batch_size": 4,
           "threads": 2, "workers": 0, "patience": 2, "lr": 0.001}}
    config = tmp_path / "config.json"
    write_json(config, cfg)
    out = tmp_path / "run"
    result = train(hdf, config, out)
    assert result["n"] == 4
    before = np.load(out / "validation/predictions.npz")["pred"]
    train(hdf, config, out, resume=True)
    after = np.load(out / "validation/predictions.npz")["pred"]
    np.testing.assert_array_equal(before, after)
    # Repetir una corrida terminada no añade épocas ni cambia predicciones.
    saved = torch.load(out / "last.pt", weights_only=True)
    assert saved["epoch"] == 1
    assert len(json.loads((out / "history.json").read_text())) == 2
    with pytest.raises(FileExistsError):
        train(hdf, config, out)
    test = tmp_path / "test.h5"
    with h5py.File(hdf) as f, h5py.File(test, "w") as g:
        g["X"], g["basin_id"] = f["X"][:3], f["basin_id"][:3]
    prediction = evaluate(test, out, split="test")
    assert prediction["n"] == 3
    assert not (out / "test/metrics.json").exists()
    baseline = evaluate(hdf, None, output=tmp_path / "persistence", persistence=True)
    assert baseline["global"]["rmse"] == 0


def test_interrupted_resume_matches_uninterrupted(hdf, tmp_path, monkeypatch):
    from dmel import engine
    config = tmp_path / "resume.json"
    write_json(config, {"model": asdict(tiny("lstm")), "training": {
        "epochs": 2, "batch_size": 4, "threads": 2, "patience": 3}})
    train(hdf, config, tmp_path / "reference")
    save = engine.save_checkpoint

    def interrupt_after_committed_epoch(path, state):
        save(path, state)
        if path.name == "last.pt" and state["epoch"] == 0:
            raise RuntimeError("interrupción simulada")

    monkeypatch.setattr(engine, "save_checkpoint", interrupt_after_committed_epoch)
    with pytest.raises(RuntimeError, match="simulada"):
        train(hdf, config, tmp_path / "interrupted")
    monkeypatch.setattr(engine, "save_checkpoint", save)
    train(hdf, config, tmp_path / "interrupted", resume=True)
    ref = torch.load(tmp_path / "reference/last.pt", weights_only=True)["model"]
    resumed = torch.load(tmp_path / "interrupted/last.pt", weights_only=True)["model"]
    for key in ref:
        torch.testing.assert_close(ref[key], resumed[key], rtol=0, atol=0)


@pytest.mark.parametrize("name", ["lstm", "dmel"])
def test_epoch_cycles_match_continuous_training(hdf, tmp_path, name):
    model = asdict(tiny(name))
    model["dropout"] = 0.2
    config = tmp_path / "cycles.json"
    write_json(config, {"model": model, "training": {
        "epochs": 3, "batch_size": 4, "threads": 2, "patience": 10}})
    cache = None
    if name == "dmel":
        cache = tmp_path / "cache.h5"
        precompute(hdf, cache, DecompositionConfig(trials=2, max_imfs=2, max_siftings=5))
    reference, cycled = tmp_path / "reference", tmp_path / "cycled"
    train(hdf, config, reference, cache)
    first = train(hdf, config, cycled, cache, epochs_per_cycle=1)
    assert first["complete"] is False and first["epochs_completed"] == 1
    assert (cycled / "last.pt").exists() and not (cycled / "validation/metrics.json").exists()
    second = train(hdf, config, cycled, cache, resume=True, epochs_per_cycle=1)
    assert second["complete"] is False and second["epochs_completed"] == 2
    final = train(hdf, config, cycled, cache, resume=True, epochs_per_cycle=1)
    assert final["complete"] and final["epochs_completed"] == 3
    assert json.loads((cycled / "status.json").read_text())["complete"]
    ref = torch.load(reference / "last.pt", weights_only=True)
    resumed = torch.load(cycled / "last.pt", weights_only=True)
    for key in ref["model"]:
        torch.testing.assert_close(ref["model"][key], resumed["model"][key], rtol=0, atol=0)
    for name in ("torch_rng", "loader_rng"):
        torch.testing.assert_close(ref[name], resumed[name], rtol=0, atol=0)
    np.testing.assert_array_equal(np.load(reference / "validation/predictions.npz")["pred"],
                                  np.load(cycled / "validation/predictions.npz")["pred"])


def test_epoch_cycle_honors_early_stopping(hdf, tmp_path, monkeypatch):
    from dmel import engine

    config = tmp_path / "early.json"
    write_json(config, {"model": asdict(tiny("lstm")), "training": {
        "epochs": 10, "batch_size": 4, "threads": 2, "patience": 1}})
    infer = engine.infer
    def constant_validation(*args):
        result = infer(*args)
        result["pred"] = result["y"] + 1
        return result
    monkeypatch.setattr(engine, "infer", constant_validation)
    out = tmp_path / "early"
    assert not train(hdf, config, out, epochs_per_cycle=1)["complete"]
    final = train(hdf, config, out, resume=True, epochs_per_cycle=1)
    assert final["complete"] and final["epochs_completed"] == 2
    assert final["stop_reason"] == "early_stopping"


def test_train_cli_pause_and_resume(hdf, tmp_path, monkeypatch, capsys):
    from dmel import cli

    config, out = tmp_path / "train.json", tmp_path / "cli-run"
    write_json(config, {"model": asdict(tiny("lstm")), "training": {
        "epochs": 2, "batch_size": 4, "threads": 2}})
    argv = ["dmel", "train", "--data", str(hdf), "--out", str(out),
            "--config", str(config), "--epochs-per-cycle", "1"]
    monkeypatch.setattr("sys.argv", argv)
    with pytest.raises(SystemExit) as paused:
        cli.main()
    assert paused.value.code == 75
    assert '"complete": false' in capsys.readouterr().out
    monkeypatch.setattr("sys.argv", argv + ["--resume"])
    cli.main()
    assert '"complete": true' in capsys.readouterr().out


@pytest.mark.parametrize("limit", [-1, 1.5])
def test_train_rejects_invalid_epoch_cycle(hdf, tmp_path, limit):
    with pytest.raises(ValueError, match="epochs_per_cycle"):
        train(hdf, "unused.json", tmp_path / "unused", epochs_per_cycle=limit)


def test_parallel_precompute_matches_serial(hdf, tmp_path):
    cfg = DecompositionConfig(trials=2, max_imfs=2, max_siftings=5)
    precompute(hdf, tmp_path / "serial.h5", cfg, limit=3)
    precompute(hdf, tmp_path / "parallel.h5", cfg, limit=3, workers=2)
    with h5py.File(tmp_path / "serial.h5") as a, h5py.File(tmp_path / "parallel.h5") as b:
        np.testing.assert_array_equal(a["components"][:], b["components"][:])


def test_precompute_deadline_closes_partial_and_resumes_exactly(hdf, tmp_path, monkeypatch):
    from types import SimpleNamespace
    from dmel import precompute as module

    cfg = DecompositionConfig(trials=2, max_imfs=2, max_siftings=5)
    reference, resumed = tmp_path / "reference.h5", tmp_path / "resumed.h5"
    precompute(hdf, reference, cfg)
    clock, calls = [0], []
    window = module._window

    def timed_window(task):
        result = window(task)
        clock[0] += 1
        calls.append(task[1])
        return result

    monkeypatch.setattr(module, "time", SimpleNamespace(
        monotonic=lambda: clock[0], perf_counter=module.time.perf_counter))
    monkeypatch.setattr(module, "_window", timed_window)
    result = precompute(hdf, resumed, cfg, max_seconds=3)
    assert result["complete"] is False and result["done"] == 3
    assert not resumed.exists()
    with h5py.File(result["output"]) as f:
        assert not f.attrs["complete"]
        assert np.count_nonzero(f["done"][:]) == 3
    # Las filas confirmadas no deben volver a calcularse en el siguiente ciclo.
    first_cycle = set(calls)
    calls.clear()
    result = precompute(hdf, resumed, cfg)
    assert result["complete"] and result["remaining"] == 0
    assert result["done_this_cycle"] == 9 and not first_cycle.intersection(calls)
    with h5py.File(reference) as a, h5py.File(resumed) as b:
        np.testing.assert_array_equal(a["ids"][:], b["ids"][:])
        np.testing.assert_array_equal(a["components"][:], b["components"][:])


def test_parallel_pause_drains_submitted_windows_then_resumes(hdf, tmp_path, monkeypatch):
    from dmel import precompute as module

    cfg = DecompositionConfig(trials=2, max_imfs=2, max_siftings=5)
    output = tmp_path / "parallel-resume.h5"
    bounded = module._bounded_results

    def stop_after_initial_batch(pool, tasks, workers, should_stop):
        submitted = [0]
        def counted_tasks():
            for item in tasks:
                submitted[0] += 1
                yield item
        return bounded(pool, counted_tasks(), workers,
                       lambda: submitted[0] >= 2 * workers)

    monkeypatch.setattr(module, "_bounded_results", stop_after_initial_batch)
    paused = precompute(hdf, output, cfg, workers=2)
    assert not paused["complete"] and paused["done"] == 4
    assert not output.exists()
    monkeypatch.setattr(module, "_bounded_results", bounded)
    finished = precompute(hdf, output, cfg, workers=2)
    assert finished["complete"] and finished["done_this_cycle"] == 8
    precompute(hdf, tmp_path / "reference.h5", cfg)
    with h5py.File(output) as a, h5py.File(tmp_path / "reference.h5") as b:
        np.testing.assert_array_equal(a["components"][:], b["components"][:])


def test_precompute_cli_distinguishes_pause_from_completion(hdf, tmp_path, monkeypatch, capsys):
    from dmel import cli

    output, config = tmp_path / "cli.h5", tmp_path / "decomp.json"
    config.write_text(json.dumps({"trials": 2, "max_imfs": 1, "max_siftings": 5}))
    argv = ["dmel", "precompute", "--data", str(hdf), "--out", str(output),
            "--config", str(config), "--limit", "2"]
    monkeypatch.setattr("sys.argv", argv + ["--max-seconds", "1e-9"])
    with pytest.raises(SystemExit) as paused:
        cli.main()
    assert paused.value.code == 75
    assert json.loads(capsys.readouterr().out)["complete"] is False
    assert not output.exists()
    monkeypatch.setattr("sys.argv", argv)
    cli.main()
    assert output.exists()
    assert '"complete": true' in capsys.readouterr().out


@pytest.mark.parametrize("seconds", [-1, float("nan"), float("inf")])
def test_precompute_rejects_invalid_time_budget(hdf, tmp_path, seconds):
    with pytest.raises(ValueError, match="max_seconds"):
        precompute(hdf, tmp_path / "bad.h5", DecompositionConfig(), max_seconds=seconds)


def test_smoke_cache_covers_exact_train_and_validation_subsets(hdf, tmp_path):
    selection = tmp_path / "selection.json"
    write_json(selection, {"training": {"max_train": 3, "max_val": 2, "seed": 91}})
    cache = tmp_path / "selected.h5"
    precompute(hdf, cache, DecompositionConfig(trials=2, max_imfs=1), selection_config=selection)
    for split, count in [("train", 3), ("validation", 2)]:
        ds = RunoffDataset(hdf, split, cache=cache, limit=count, seed=91)
        assert len(ds) == count
        ds[0]
        ds.close()


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf")])
def test_rejects_invalid_gradient_clip(tmp_path, value):
    from dmel.engine import read_config
    config = tmp_path / "invalid.json"
    config.write_text(json.dumps({"model": {"name": "lstm"}, "training": {"grad_clip": value}}))
    with pytest.raises(ValueError, match="inválida"):
        read_config(config)


def test_full_dmel_training_integration(hdf, tmp_path):
    cache = tmp_path / "cache.h5"
    precompute(hdf, cache, DecompositionConfig(trials=2, max_imfs=2, max_siftings=5))
    config = tmp_path / "dm.json"
    write_json(config, {"model": asdict(tiny()), "training": {"epochs": 1, "batch_size": 4, "threads": 2}})
    result = train(hdf, config, tmp_path / "dmel", cache=cache)
    assert np.isfinite(result["global"]["rmse"])
