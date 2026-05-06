import os

from modules.improve_cohort import IMPROVECohort, IMPROVETAVICohort


def test_improvecohort_download_file_copies_existing_file(tmp_path):
    src = tmp_path / "input.csv"
    src.write_text("subject_id\n1\n")

    cohort = IMPROVECohort.__new__(IMPROVECohort)
    cohort.tmp_dir = str(tmp_path / "tmp")
    os.makedirs(cohort.tmp_dir, exist_ok=True)

    cohort.download_file(str(src))
    copied = tmp_path / "tmp" / str(src)
    assert copied.exists()
    assert copied.read_text() == src.read_text()


def test_improvetavicohort_inherits_download():
    cohort = IMPROVETAVICohort.__new__(IMPROVETAVICohort)
    assert hasattr(cohort, "download_file")
