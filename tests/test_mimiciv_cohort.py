
# Permet l'import du dossier modules lors d'une exécution directe
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import pandas as pd
from modules.mimiciv_cohort import MIMICIVPatientCohort
import tempfile
import os

class DummyFilter:
    def apply(self, df):
        return df  # no filtering

def make_dummy_csv(path, data):
    df = pd.DataFrame(data)
    df.to_csv(path, index=False)

@pytest.fixture
def dummy_files(tmp_path):
    patients = tmp_path / "patients.csv"
    admissions = tmp_path / "admissions.csv"
    make_dummy_csv(patients, [
        {"subject_id": 1, "anchor_age": 60, "gender": "M"},
        {"subject_id": 2, "anchor_age": 70, "gender": "F"},
    ])
    make_dummy_csv(admissions, [
        {"hadm_id": 10, "subject_id": 1, "admittime": "2020-01-01", "insurance": "A", "language": "EN", "marital_status": "S", "race": "W"},
        {"hadm_id": 20, "subject_id": 2, "admittime": "2020-02-01", "insurance": "B", "language": "FR", "marital_status": "M", "race": "B"},
    ])
    return str(patients), str(admissions), str(tmp_path)

def test_build_patients(dummy_files):
    patients_file, admissions_file, tmp_dir = dummy_files
    cohort = MIMICIVPatientCohort(
        db=None,
        patients_file=patients_file,
        admission_file=admissions_file,
        tmp_dir=tmp_dir,
        zarr_index="subject_id",
        filters=[DummyFilter()]
    )
    df = cohort.build_patients()
    assert len(df) == 2
    assert set(df["subject_id"]) == {1, 2}

def test_build_admissions(dummy_files):
    patients_file, admissions_file, tmp_dir = dummy_files
    cohort = MIMICIVPatientCohort(
        db=None,
        patients_file=patients_file,
        admission_file=admissions_file,
        tmp_dir=tmp_dir,
        zarr_index="subject_id",
        filters=[]
    )
    patient_ids = [1]
    df = cohort.build_admissions(patient_ids)
    assert len(df) == 1
    assert df.iloc[0]["subject_id"] == 1

def test_build_cohort(dummy_files):
    patients_file, admissions_file, tmp_dir = dummy_files
    cohort = MIMICIVPatientCohort(
        db=None,
        patients_file=patients_file,
        admission_file=admissions_file,
        tmp_dir=tmp_dir,
        zarr_index="subject_id",
        filters=[]
    )
    result = cohort.build_cohort()
    assert "patient" in result
    assert "admission" in result
    assert "patient_demographics" in result
    assert len(result["patient"]) == 2
    assert len(result["admission"]) == 2
    assert len(result["patient_demographics"]) == 2
