import os

from modules.physionet_cohort import PhysioNetPatientCohort, MIMICPatientCohort


def test_physionet_patient_cohort_download_file_existing(tmp_path):
    csv_file = tmp_path / "data.csv"
    csv_file.write_text("subject_id\n1\n")

    cohort = PhysioNetPatientCohort.__new__(PhysioNetPatientCohort)
    cohort.tmp_dir = str(tmp_path)
    cohort.db = "mimic"
    cohort.download_file(str(csv_file))

    assert csv_file.exists()


def test_mimic_patient_cohort_inherits_download():
    cohort = MIMICPatientCohort.__new__(MIMICPatientCohort)
    assert hasattr(cohort, "download_file")


def test_physionet_download_when_file_only_in_cwd(tmp_path, monkeypatch):
    import modules.physionet_cohort as pc
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data.csv").write_text("subject_id\n1\n")
    calls = []
    monkeypatch.setattr(pc, "download_physionet_file", lambda *args, **kwargs: calls.append(args))

    cohort = PhysioNetPatientCohort.__new__(PhysioNetPatientCohort)
    cohort.tmp_dir = str(tmp_path / "tmp")
    cohort.db = "mimic"
    cohort.schema = {"name": "mimic", "version": "3.1"}
    cohort.download_file("data.csv")
    assert calls == [("mimic", "3.1", "data.csv", str(tmp_path / "tmp"))]  # not in tmp_dir, so downloaded


def test_physionet_download_requires_version(tmp_path):
    import pytest
    cohort = PhysioNetPatientCohort.__new__(PhysioNetPatientCohort)
    cohort.tmp_dir = str(tmp_path)
    cohort.db = "mimic"
    cohort.schema = {"name": "mimic"}
    with pytest.raises(ValueError, match="version"):
        cohort.download_file("missing.csv")


def test_schema_metadata_keys_are_not_groups(tmp_path):
    schema = {"name": "mimic", "version": "3.1",
              "patient": {"file": "p.csv", "columns": ["subject_id"], "processor": {"name": "EHRDataFrameProcessor"}}}
    cohort = MIMICPatientCohort(str(tmp_path), "subject_id", schema)
    assert list(cohort.processors) == ["patient"]
