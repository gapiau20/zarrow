import numpy as np
import pandas as pd

torch_import_error = None
try:
    import torch
except ImportError as err:
    torch_import_error = err

from modules.torch_loader import MultimodalDataset
from modules.zarr_tools import ZarrWriter


def test_multimodal_dataset_one_item_per_subject(tmp_path):
    assert torch_import_error is None, f"torch is required for this test: {torch_import_error}"

    path = str(tmp_path / "data.zarr")
    writer = ZarrWriter(path, "subject_id")
    writer.write_dataframe(pd.DataFrame({"subject_id": [1, 2], "age": [50, 60], "sex": ["M", "F"]}), "patient")
    writer.write_dataframe(pd.DataFrame({"subject_id": [1, 1], "value": [4.0, 5.0]}), "labs")
    writer.write_index(np.array([1, 2]))
    dataset = MultimodalDataset(path, "subject_id", ["patient", "labs"])

    assert len(dataset) == 2
    first = dataset[0]
    assert first["patient"].tolist() == [[1.0, 50.0]]  # string column dropped
    assert first["labs"].tolist() == [[1.0, 4.0], [1.0, 5.0]]
    assert dataset[1]["labs"].numel() == 0  # subject without labs
