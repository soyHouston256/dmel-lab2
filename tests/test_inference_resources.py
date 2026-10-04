"""La inferencia debe liberar el almacenamiento de los lotes consumidos."""
import weakref

import numpy as np
import pytest
import torch

from dmel.engine import TrainConfig, infer, loader


class LastValueModel(torch.nn.Module):
    def forward(self, x, components=None):
        return x[:, -1, 11, None].repeat(1, 48)


class ValidationDataset(torch.utils.data.Dataset):
    """567 lotes, como la validación completa en Khipu, sin datos privados."""
    def __len__(self):
        return 18142

    def __getitem__(self, index):
        q = np.float32(index % 100 / 10)
        return {"Id": index, "basin_id": index % 3, "last_q": q,
                "X": np.full((1, 12), q, dtype=np.float32),
                "y": np.full(48, q, dtype=np.float32)}


@pytest.mark.parametrize("supervised", [False, True])
@pytest.mark.parametrize("persistence", [False, True])
def test_inference_releases_consumed_batch_storage(supervised, persistence):
    # from_numpy conserva el array propietario mientras cualquier vista del
    # tensor siga viva. Así detectamos la retención que agota descriptores con
    # workers Linux, sin depender de su límite ni de una API privada de PyTorch.
    owners = []

    def batches():
        for i in range(64):
            if i >= 2:
                # infer aún puede tener el lote anterior al pedir el siguiente,
                # pero ya no debe retener propietarios de lotes más antiguos.
                assert all(ref() is None for ref in owners[i - 2].values()), (
                    "La inferencia retiene almacenamiento de lotes ya consumidos"
                )
            row_ids = np.arange(2 * i, 2 * i + 2, dtype=np.int64)
            arrays = {
                "Id": row_ids,
                "basin_id": row_ids % 3,
                "last_q": row_ids.astype(np.float32),
                "X": np.broadcast_to(row_ids[:, None, None], (2, 336, 12)).astype(np.float32),
            }
            if supervised:
                arrays["y"] = np.repeat(row_ids[:, None], 48, axis=1).astype(np.float32)
            owners.append({key: weakref.ref(value) for key, value in arrays.items()})
            yield {key: torch.from_numpy(value) for key, value in arrays.items()}
            del arrays, row_ids

    normalizer = {"mean": [0.] * 12, "std": [1.] * 12}
    result = infer(None if persistence else LastValueModel(), batches(), normalizer, torch.device("cpu"))
    expected_ids = np.arange(128)
    expected = np.repeat(expected_ids[:, None], 48, axis=1).astype(np.float32)
    np.testing.assert_array_equal(result["pred"], expected)
    np.testing.assert_array_equal(result["ids"], expected_ids)
    np.testing.assert_array_equal(result["basins"], expected_ids % 3)
    if supervised:
        np.testing.assert_array_equal(result["y"], expected)
    else:
        assert result["y"] is None
    assert all(ref() is None for batch in owners for ref in batch.values())


@pytest.mark.skipif("file_descriptor" not in torch.multiprocessing.get_all_sharing_strategies(),
                    reason="La regresión de descriptores requiere file_descriptor (Linux).")
def test_full_validation_with_spawn_workers_matches_serial():
    dataset = ValidationDataset()
    normalizer = {"mean": [0.] * 12, "std": [1.] * 12}
    model = LastValueModel()
    serial = infer(model, loader(dataset, TrainConfig(workers=0)), normalizer, torch.device("cpu"))
    parallel = infer(model, loader(dataset, TrainConfig(workers=2)), normalizer, torch.device("cpu"))
    assert parallel["pred"].shape == (18142, 48)
    for key in serial:
        np.testing.assert_array_equal(parallel[key], serial[key])
