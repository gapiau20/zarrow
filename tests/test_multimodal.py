import pickle

import numpy as np
import pandas as pd
import pytest
import wfdb

import modules.multimodal as multimodal
from modules.cohort import TabularCohort
from modules.multimodal import MIMICECGAdapter, add_modality, select_records
from modules.zarr_tools import ZarrLoader, ZarrWriter

LEADS = MIMICECGAdapter.LEADS


def test_multimodal_module_has_docstring():
    assert multimodal.__doc__ is not None


def write_ecg(source_dir, subject, study, value, n_samples=5000, gain=200, leads=LEADS):
    '''Synthetic MIMIC-IV-ECG record: lead j holds value + j (in mV), at files/p<subject>/s<study>/<study>.'''
    rel = f"files/p{subject}/s{study}/{study}"
    (source_dir / rel).parent.mkdir(parents=True, exist_ok=True)
    mv = np.tile(value + np.arange(len(leads), dtype=float), (n_samples, 1))
    wfdb.wrsamp(str(study), fs=500, units=["mV"] * len(leads), sig_name=list(leads),
                d_signal=np.rint(mv * gain).astype(np.int16), fmt=["16"] * len(leads),
                adc_gain=[gain] * len(leads), baseline=[0] * len(leads), write_dir=str((source_dir / rel).parent))
    return rel


@pytest.fixture
def ecg_source(tmp_path):
    '''Records: subject 1 at 10:00 (in window) and 3 days later (out), subject 3 one invalid record.'''
    src = tmp_path / "ecg"
    rows = [(1, 101, "2180-01-01 10:00:00", write_ecg(src, 1, 101, 0.5)),
            (1, 102, "2180-01-04 10:00:00", write_ecg(src, 1, 102, 1.0)),
            (3, 301, "2180-01-01 09:00:00", write_ecg(src, 3, 301, 2.0, n_samples=100)),
            (9, 901, "2180-01-01 09:00:00", write_ecg(src, 9, 901, 3.0))]   # subject not in the cohort
    pd.DataFrame(rows, columns=["subject_id", "study_id", "ecg_time", "path"]).assign(
        file_name=lambda d: d["study_id"]).to_csv(src / "record_list.csv", index=False)
    return src


@pytest.fixture
def cohort_store(tmp_path):
    path = str(tmp_path / "cohort.zarr")
    writer = ZarrWriter(path, "subject_id")
    writer.write_dataframe(pd.DataFrame({"subject_id": [1, 2, 3], "hadm_id": [11, 21, 31],
                                         "admittime": ["2180-01-01 08:00:00", "", "2180-01-01 08:00:00"]}), "admission")
    writer.write_index(np.array([1, 2, 3]))
    return path


def ecg_config(src, **extra):
    return {"adapter": "MIMICECGAdapter", "parameters": {"source_dir": str(src), "workers": 2}, **extra}


def test_ecg_adapter_reads_leads_in_mimic_order_and_rescales(tmp_path):
    rel = write_ecg(tmp_path, 1, 1, 0.5, gain=100, leads=LEADS[::-1])   # other gain and lead order
    adapter = MIMICECGAdapter(str(tmp_path))
    record = adapter.load(pd.Series({"path": rel}))
    assert record.shape == (5000, 12) and record.dtype == np.int16
    expected_mv = 0.5 + np.arange(12)[::-1]    # lead j of the file holds 0.5 + j
    np.testing.assert_array_equal(record[0], np.rint(expected_mv * MIMICECGAdapter.GAIN))


def test_select_records_window_closest_anchor_and_first():
    manifest = pd.DataFrame({"subject": [1, 1, 1], "record_id": [1, 2, 3], "path": ["a", "b", "c"],
                             "record_time": pd.to_datetime(["2180-01-01 10:00", "2180-01-02 07:00", "2180-01-09 00:00"])})
    anchors = pd.DataFrame({"subject": [1, 1], "hadm": [11, 12],
                            "anchor_time": pd.to_datetime(["2180-01-01 08:00", "2180-01-02 08:00"])})
    records = select_records(manifest, [1], anchors, hours=[-2, 24])
    assert records["record_id"].tolist() == [1, 2]
    assert records["hadm"].tolist() == [11, 12]                  # record 2 is in both windows, 12 is closer
    assert records["offset_h"].tolist() == [2.0, -1.0]
    assert select_records(manifest, [1], select="last")["record_id"].tolist() == [3]
    assert select_records(manifest, [1], anchors, [-2, 24], select="first")["record_id"].tolist() == [1]


def test_add_modality_writes_csr_index(ecg_source, cohort_store):
    window = {"anchor_group": "admission", "anchor_time": "admittime", "hours": [-12, 24]}
    add_modality(cohort_store, "subject_id", "ecg", ecg_config(ecg_source, window=window))

    ecg = ZarrLoader(cohort_store).store["ecg"]
    assert ecg["record_id"][:].tolist() == [101, 301]            # 102 out of window, 901 not in the cohort
    assert ecg["patient_ptr"][:].tolist() == [0, 1, 1, 2]        # subject 2: no record
    assert ecg["valid"][:].tolist() == [True, False]             # 301 has 100 samples instead of 5000
    assert ecg["record_offset_h"][:].tolist() == [2.0, 1.0]
    assert ecg["data"].shape == (2, 5000, 12)
    assert ecg.attrs["leads"] == LEADS and ecg.attrs["window"] == window
    assert ecg["record_time"][:][0] == np.datetime64("2180-01-01T10:00:00")


def test_multimodal_and_record_datasets_read_ecgs(ecg_source, cohort_store):
    torch = pytest.importorskip("torch")
    from modules.torch_loader import MultimodalDataset, RecordDataset
    add_modality(cohort_store, "subject_id", "ecg", ecg_config(ecg_source))   # no window: all records

    dataset = MultimodalDataset(cohort_store, "subject_id", ["admission"], modalities=["ecg"])
    first = dataset[0]
    assert first["ecg"].shape == (2, 5000, 12)
    assert torch.allclose(first["ecg"][0, 0], torch.arange(12) + 0.5)    # decoded to mV
    assert torch.isnan(first["ecg_offset_h"]).all()
    assert dataset[1]["ecg"].shape == (0, 5000, 12)
    assert dataset[2]["ecg"].shape == (0, 5000, 12)               # its only record is invalid
    assert pickle.loads(pickle.dumps(dataset))[0]["ecg"].shape == (2, 5000, 12)   # DataLoader workers

    records = RecordDataset(cohort_store, "ecg")
    assert len(records) == 2
    signal, subject, _ = records[1]
    assert subject == 1 and torch.allclose(signal[0], torch.arange(12) + 1.0)


def test_cohort_build_and_save_adds_modalities(ecg_source, tmp_path):
    class LocalCohort(TabularCohort):
        def download_file(self, file):
            return
    pd.DataFrame({"subject_id": [1, 3], "anchor_age": [60, 70]}).to_csv(tmp_path / "patients.csv", index=False)
    schema = {"name": "demo", "patient": {"file": "patients.csv", "columns": ["subject_id", "anchor_age"],
                                          "processor": {"name": "EHRDataFrameProcessor"}}}
    path = str(tmp_path / "out.zarr")
    LocalCohort(str(tmp_path), "subject_id", schema, modalities={"ecg": ecg_config(ecg_source, select="last")}
                ).build_and_save(path)
    assert ZarrLoader(path).store["ecg"]["record_id"][:].tolist() == [102, 301]
