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


def test_read_tabular_file_chunked_cp1252(tmp_path):
    csv_path = tmp_path / "latin.csv"
    csv_path.write_bytes("a,b\n1,M\u00fcller\n".encode("cp1252"))
    chunks = list(read_tabular_file(str(csv_path), chunksize=1))
    assert chunks[0]["b"].tolist() == ["M\u00fcller"]


def test_read_tabular_file_chunked_non_csv(tmp_path):
    json_path = tmp_path / "data.json"
    pd.DataFrame({"a": [1, 2]}).to_json(json_path, orient="records")
    chunks = read_tabular_file(str(json_path), chunksize=1)
    assert [c["a"].tolist() for c in chunks] == [[1, 2]]


def _write_csv(tmp_path, name, df):
    df.to_csv(tmp_path / name, index=False)
    return {"file": name, "columns": list(df.columns), "processor": {"name": "EHRDataFrameProcessor"}}


def test_inclusion_groups_restrict_every_group(tmp_path):
    admission = _write_csv(tmp_path, "adm.csv", pd.DataFrame({"subject_id": [1, 2, 3], "hadm_id": [10, 20, 30]}))
    demo = _write_csv(tmp_path, "demo.csv", pd.DataFrame({"subject_id": [1, 2, 3], "age": [15, 40, 70]}))
    demo["inclusion"] = True
    demo["filters"] = [{"name": "AgeFilter", "parameters": {"age_column": "age", "age_min": 18}}]
    cohort = DummyCohort(str(tmp_path), "subject_id", {"name": "demo", "admission": admission, "demographics": demo})
    built = cohort.build_cohort()
    assert built["admission"]["subject_id"].tolist() == [2, 3]
    assert built["demographics"]["subject_id"].tolist() == [2, 3]


def test_build_cohort_all_rows_filtered_out(tmp_path):
    group = _write_csv(tmp_path, "sex.csv", pd.DataFrame({"subject_id": [1], "sex": ["F"]}))
    group["filters"] = [{"name": "SexFilter", "parameters": {"sex_column": "sex", "sex": "M"}}]
    built = DummyCohort(str(tmp_path), "subject_id", {"name": "demo", "patient": group}).build_cohort()
    assert built["patient"].empty


def test_build_and_save_without_patient_group(tmp_path):
    group = _write_csv(tmp_path, "global.csv", pd.DataFrame({"subject_id": [2, 1, 2], "value": [1, 2, 3]}))
    cohort = DummyCohort(str(tmp_path), "subject_id", {"name": "demo", "global": group})
    zarr_path = str(tmp_path / "out.zarr")
    cohort.build_and_save(zarr_path)
    from modules.zarr_tools import ZarrLoader
    assert ZarrLoader(zarr_path).store["subject_id"][:].tolist() == [1, 2]
