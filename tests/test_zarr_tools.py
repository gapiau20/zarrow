import numpy as np
import pandas as pd

from modules.zarr_tools import ZarrWriter, ZarrLoader


def test_zarr_writer_and_loader_roundtrip(tmp_path):
    zarr_path = tmp_path / "cohort.zarr"
    df = pd.DataFrame({"subject_id": [1, 2], "label": ["A", "B"], "score": [0.1, 0.2]})
    writer = ZarrWriter(str(zarr_path), index_id="subject_id")
    writer.write_dataframe(df, "patient")
    writer.write_index(np.array([1, 2], dtype=np.int32))

    loader = ZarrLoader(str(zarr_path))
    loaded = loader.load_group("patient")
    assert loaded["label"].tolist() == ["A", "B"]
    assert np.allclose(loaded["score"].astype(float), np.array([0.1, 0.2]))

    loaded_df = loader.load_group("patient", as_df=True)
    assert set(loaded_df.columns) == {"subject_id", "label", "score"}
    np.testing.assert_array_equal(loader.load_index(), np.array([1, 2], dtype=np.int32))


def test_zarr_writer_unicode_missing_overflow_and_empty(tmp_path):
    zarr_path = tmp_path / "edge.zarr"
    df = pd.DataFrame({"name": ["M\u00fcller", None], "big": [3_000_000_000, 1]})
    writer = ZarrWriter(str(zarr_path), index_id="subject_id")
    writer.write_dataframe(df, "g")
    writer.write_dataframe(pd.DataFrame({"name": pd.Series([], dtype=object)}), "empty")

    loaded = ZarrLoader(str(zarr_path)).load_group("g", as_df=True)
    assert loaded["name"].tolist() == ["M\u00fcller", ""]
    assert loaded["big"].tolist() == [3_000_000_000, 1]
