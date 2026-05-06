import os
import yaml
import pandas as pd

from modules.cohort import (
    get_columns_from_dataframe,
    read_tabular_file,
    get_schema_from_tabular_file,
    build_default_schema_from_tabular_file,
    get_schema_from_config,
    TabularCohort,
)


def test_get_columns_from_dataframe():
    df = pd.DataFrame({"a": [1], "b": [2]})
    assert get_columns_from_dataframe(df) == ["a", "b"]


def test_read_tabular_file_csv_and_json(tmp_path):
    csv_path = tmp_path / "data.csv"
    pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}).to_csv(csv_path, index=False)
    csv_df = read_tabular_file(str(csv_path))
    assert csv_df["a"].tolist() == [1, 2]

    json_path = tmp_path / "data.json"
    pd.DataFrame({"a": [3, 4], "b": ["m", "n"]}).to_json(json_path, orient="records")
    json_df = read_tabular_file(str(json_path))
    assert json_df["a"].tolist() == [3, 4]


def test_get_schema_from_tabular_file_and_build_default_schema(tmp_path):
    csv_path = tmp_path / "data.csv"
    pd.DataFrame({"subject_id": [1], "value": [10]}).to_csv(csv_path, index=False)
    schema = get_schema_from_tabular_file(str(csv_path))
    assert schema["file"] == str(csv_path)
    assert schema["columns"] == ["subject_id", "value"]

    yaml_path = tmp_path / "schema.yaml"
    build_default_schema_from_tabular_file(str(csv_path), str(yaml_path))
    loaded = yaml.safe_load(yaml_path.read_text())
    assert loaded["file"] == str(csv_path)
    assert loaded["columns"] == ["subject_id", "value"]


def test_get_schema_from_config(tmp_path):
    config = {
        "dataset": {
            "name": "demo",
            "patient": {
                "file": "data.csv",
                "columns": ["subject_id"],
                "processor": {"name": "EHRDataFrameProcessor"},
                "filters": [],
            }
        }
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    loaded = get_schema_from_config(str(config_path))
    assert loaded["name"] == "demo"


class DummyCohort(TabularCohort):
    def download_file(self, file):
        return


def test_tabular_cohort_process_and_load(tmp_path):
    csv_path = tmp_path / "data.csv"
    pd.DataFrame({"subject_id": [1, 2], "value": [100, 200]}).to_csv(csv_path, index=False)
    schema = {
        "name": "demo",
        "patient": {
            "file": "data.csv",
            "columns": ["subject_id", "value"],
            "processor": {"name": "EHRDataFrameProcessor"},
            "filters": [],
        },
    }
    cohort = DummyCohort(str(tmp_path), "subject_id", schema, chunk_size=1)
    built = cohort.build_cohort()
    assert "patient" in built
    assert built["patient"]["value"].tolist() == [100, 200]

    zarr_path = tmp_path / "output.zarr"
    loaded = cohort.load(["patient"], str(zarr_path))
    assert "patient" in loaded
