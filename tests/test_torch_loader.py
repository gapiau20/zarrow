import zarr

torch_import_error = None
try:
    import torch
except ImportError as err:
    torch_import_error = err

from modules.torch_loader import MultimodalDataset


def test_multimodal_dataset_loads_zarr(tmp_path):
    assert torch_import_error is None, f"torch is required for this test: {torch_import_error}"
    import numpy as np

    path = tmp_path / "data.zarr"
    store = zarr.open(str(path), mode="w")
    group = store.require_group("0")
    group.create_array("a", data=np.array([1, 2, 3], dtype=np.int32))
    group.create_array("b", data=np.array([4.0, 5.0, 6.0], dtype=np.float32))
    dataset = MultimodalDataset(str(path))

    assert len(dataset) == 1
    item = dataset[0]
    assert item["a"].tolist() == [1, 2, 3]
    assert item["b"].tolist() == [4.0, 5.0, 6.0]
