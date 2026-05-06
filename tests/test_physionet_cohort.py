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
