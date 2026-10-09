"""La entrega de test debe estar completa y preservar la identidad de las filas."""
import csv
import json

import h5py
import numpy as np
import pytest

from scripts import lab2_closure as closure


@pytest.fixture
def export(tmp_path, monkeypatch):
    n = 27983
    data = tmp_path / "test.h5"
    with h5py.File(data, "w") as f:
        f.create_dataset("X", shape=(n, 336, 12), dtype="f4", chunks=(1, 336, 12))
        f["basin_id"] = np.arange(n) % 508
    out = tmp_path / "export"
    out.mkdir()
    pred = np.repeat((np.arange(n, dtype="f4") / 1024)[:, None], 48, axis=1)
    np.savez_compressed(out / "predictions.npz", pred=pred, ids=np.arange(n), basins=np.arange(n) % 508)
    with (out / "predictions.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Id"] + [f"q_{i:02d}" for i in range(1, 49)])
        for i, row in enumerate(pred):
            writer.writerow([i] + row.tolist())
    (out / "metadata.json").write_text(json.dumps({"n": n, "split": "test", "source_sha256": closure.sha256(data),
        "units": "mm/h", "diagnostic_subset": False, "clipping": False, "persistence": False}))
    monkeypatch.setattr(closure, "model_info", lambda _: {"model": "lstm", "test_stub": True})
    return data, tmp_path / "run", out


def test_complete_export(export):
    result = closure.verify_test(*export)
    assert result["complete"] and result["prediction_values"] == 1343184
    assert result["id_range"] == [0, 27982]


def test_truncated_csv_is_not_complete(export):
    data, run, out = export
    path = out / "predictions.csv"
    lines = path.read_text().splitlines(keepends=True)
    path.write_text("".join(lines[:-1]))
    with pytest.raises(ValueError, match="CSV incompleto"):
        closure.verify_test(data, run, out)


def test_csv_prediction_must_match_npz(export):
    data, run, out = export
    path = out / "predictions.csv"
    content = path.read_text()
    path.write_text(content.replace("0,0.0,", "0,999.0,", 1))
    with pytest.raises(ValueError, match="CSV y NPZ difieren"):
        closure.verify_test(data, run, out)


def test_duplicate_id_rejected(export):
    data, run, out = export
    path = out / "predictions.npz"
    with np.load(path) as f:
        values = {k: f[k] for k in f.files}
    values["ids"][-1] = 0
    np.savez_compressed(path, **values)
    with pytest.raises(ValueError, match="Ids duplicados"):
        closure.verify_test(data, run, out)
