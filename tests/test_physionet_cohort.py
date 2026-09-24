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
    monkeypatch.setattr(pc, "dl_files", lambda *args, **kwargs: calls.append(args))

    cohort = PhysioNetPatientCohort.__new__(PhysioNetPatientCohort)
    cohort.tmp_dir = str(tmp_path / "tmp")
    cohort.db = "mimic"
    cohort.download_file("data.csv")
    assert len(calls) == 1  # file is not in tmp_dir, so it must be downloaded
