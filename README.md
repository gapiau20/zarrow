<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="pictures/zarrow-logo-horizontal.svg">
    <img src="pictures/zarrow-logo-horizontal.svg" alt="Zarrow" width="420">
  </picture>
</p>

# Zarrow

Build **clinical cohorts** (MIMIC-IV, MIMIC-III…) from a YAML config, store them in **[Zarr](https://zarr.dev)** format, then use them with pandas, scikit-learn or PyTorch.

```
YAML config ──► PhysioNet download ──► chunked reading + filters ──► Zarr store ──► ML
```

## Installation

Python ≥ 3.10.

```bash
python -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt   # + requirements-dev.txt for pytest
```

PyTorch installs the CPU build. For GPU, uncomment the `--extra-index-url` line in `requirements.txt`.

### PhysioNet credentials (full MIMIC-IV)

The demos (`mimic-iv-demo`, `mimiciii-demo`) are public. For full MIMIC-IV, you need a **credentialed** PhysioNet account that has signed the project's data use agreement (DUA). Provide your credentials one of two ways (never in the repo):

```powershell
$env:PHYSIONET_USERNAME = "user"; $env:PHYSIONET_PASSWORD = "password"
```

or via a `~/.netrc` file (on Windows: `%USERPROFILE%\_netrc`):

```
machine physionet.org
login user
password password
```

Test access on a small file before launching a large cohort build:

```python
from modules.download import download_physionet_file
download_physionet_file("mimiciv", "3.1", "hosp/patients.csv.gz", "data/mimiciv-tmp")
```

PhysioNet only accepts authentication with a `Wget/<version>`-style User-Agent (otherwise it answers 403, even with valid credentials). [modules/download.py](modules/download.py) sends it automatically. A file already present in `tmp_dir` is not re-downloaded, so you can also drop the tables there yourself.

## Quickstart

```python
from modules.physionet_cohort import MIMICPatientCohort
from modules.cohort import get_schema_from_config
from modules.zarr_tools import ZarrLoader

schema = get_schema_from_config("config/mimiciv_demo.yaml")
cohort = MIMICPatientCohort("data/tmp", "subject_id", schema)
cohort.build_and_save("data/demo.zarr")                  # downloads, filters, writes

df = ZarrLoader("data/demo.zarr").load_group("admission", as_df=True)
```

Full example (28-day mortality after a myocardial infarction, using demographics and laboratory values, logistic regression and subgroup analysis):

```bash
python scripts/mortality28d.py     # config: config/mimic_iv_infarction.yaml
```

The script only rebuilds the cohort if `data/cohort.zarr` does not exist. Delete that folder to rebuild it.

Same model with the presenting ECG added (heart rate, RR variability, per-lead QRS amplitude and ST deviation of the median beat), compared with and without ECG on the same folds:

```bash
python scripts/mortality28d-ecg.py   # adds the ECGs to data/cohort.zarr if missing: ~26k ECGs, ~3 GB (open access)
```

## Defining a cohort

```yaml
dataset:
  name: mimiciv                 # PhysioNet project
  version: '3.1'                # project version
  diagnoses:                    # a "group" = a table = a Zarr group
    file: hosp/diagnoses_icd.csv.gz
    inclusion: true             # its filters define who is in the cohort
    read_options: {encoding: utf-8}   # optional, pandas kwargs
    columns: [subject_id, hadm_id, icd_code]
    processor: {name: EHRDataFrameProcessor}
    filters:
      - name: ICDFilter
        parameters: {diag_column: icd_code, icd_codes: ['I21*', '410*']}
```

Rules to know:
- **Each group contains the index column** (e.g. `subject_id`).
- **Filters apply before the `columns` selection**: they can therefore operate on a column that isn't kept.
- **`inclusion: true`** restricts *all* groups to the intersection of patients from the inclusion groups. Without this flag, a filter only selects rows within its own group (for example, which lab tests to keep).

| Filter | Parameters |
|---|---|
| `AgeFilter` | `age_column`, `age_min`, `age_max` (inclusive bounds) |
| `SexFilter` | `sex_column`, `sex` |
| `ICDFilter` | `diag_column`, `icd_codes` (the `*` suffix means "prefix") |
| `PatientFilter` | `patient_col`, `subject_ids` |
| `LabEventFilter` / `ProcedureFilter` / `MedicationFilter` / `CharteventFilter` | `<x>_column`, `<x>s` (list of values) |

**Adding a filter or a processor**: just subclass `BaseFilter` (`apply(df)` method) or `DatabaseProcessor` (`process(df)` method). The class is registered automatically and usable by name in the YAML.

## Zarr store produced

```
cohort.zarr/
├── subject_id          # index: sorted unique identifiers
├── admission/          # one group per group in the config
│   ├── subject_id      # one column = one array
│   └── admittime       # UTF-8 text; missing value = ''
├── labs/ ...
└── ecg/                # one group per modality, along a record axis
    ├── data            # (n_records, 5000, 12) int16, one record per chunk, 256 per shard
    ├── record_subject, record_id, record_time, record_hadm, record_offset_h, valid
    └── patient_ptr     # (n_patients + 1,): records of patient i are [ptr[i], ptr[i+1])
```

Tables are "long" (one row per record, not per patient): join on `subject_id` / `hadm_id`. For PyTorch, `MultimodalDataset(zarr_path, "subject_id", groups)` returns one item per patient, with numeric columns sorted by name (see `dataset.columns`). The number of rows varies from patient to patient, so a padding `collate_fn` is needed.

## Adding modalities (ECG)

Non-tabular modalities are declared in a top-level `modalities:` section of the config, next to `dataset:` (see [config/mimiciv_demo.yaml](config/mimiciv_demo.yaml)):

```yaml
modalities:
  ecg:
    adapter: MIMICECGAdapter
    parameters: {source_dir: data/mimic-iv-ecg, project: mimic-iv-ecg, version: '1.0'}
    window: {anchor_group: admission, anchor_time: admittime, hours: [-12, 24], restrict_to: diagnoses}
    select: all                   # all | first | last record per patient
```

```python
from modules.multimodal import get_modalities_from_config, add_modality
cohort = MIMICPatientCohort("data/tmp", "subject_id", schema, modalities=get_modalities_from_config(cfg))
# or add it to an existing store without rebuilding the tables:
add_modality("data/cohort.zarr", "subject_id", "ecg", get_modalities_from_config(cfg)["ecg"])
```

- Only the records of cohort patients within `hours` of an anchor time (here the admission time) are kept; a record in several windows is matched to the closest anchor. `restrict_to` keeps only the anchors whose `(subject_id, hadm_id)` appear in another group, e.g. the ICD-filtered admissions.
- MIMIC-IV-ECG is open access. Only the selected records are downloaded, file by file (about 3 files/s): for a large cohort, download the project archive into `source_dir` first, files already there are not downloaded again.
- ECGs are stored losslessly as int16 at the native gain (200 units/mV) and decoded to mV on read. Lead order is the one of the files: `I, II, III, aVR, aVF, aVL, V1…V6` (see the `leads` attribute). Missing samples come back as NaN: handle them before a model.
- **Adding a modality**: subclass `ModalityAdapter` (`manifest()`, `load(record)`, optionally `fetch()` and `attrs()`), it is registered automatically and usable by name in the YAML.

For PyTorch, `MultimodalDataset(zarr_path, "subject_id", groups, modalities=["ecg"])` adds each patient's ECGs (`(k, 5000, 12)`) and their offsets from the anchor (`ecg_offset_h`), reading only that patient's records from disk. `RecordDataset(zarr_path, "ecg")` gives one item per ECG.

## Code organization

| File | Role |
|---|---|
| [modules/cohort.py](modules/cohort.py) | `BaseCohort` / `TabularCohort`: pipeline, tabular reading with encoding detection |
| [modules/db_filters.py](modules/db_filters.py), [modules/db_processors.py](modules/db_processors.py) | Filters and processors, automatic registries |
| [modules/physionet_cohort.py](modules/physionet_cohort.py) | PhysioNet cohorts (`MIMICPatientCohort`) |
| [modules/download.py](modules/download.py) | Authenticated, streaming PhysioNet download with resume |
| [modules/zarr_tools.py](modules/zarr_tools.py) | `ZarrWriter` / `ZarrLoader` |
| [modules/multimodal.py](modules/multimodal.py) | Modalities: `ModalityAdapter`, `MIMICECGAdapter`, time windows, `add_modality` |
| [modules/torch_loader.py](modules/torch_loader.py) | `MultimodalDataset`, `RecordDataset` |
| [modules/utils.py](modules/utils.py) | Utilities for ICD codes |
| [modules/features.py](modules/features.py), [main.py](main.py) | Skeletons |
| [config/](config/) | Cohorts: `mimiciv_demo`, `mimiciii_demo`, `mimic_iv_infarction` |
| `data/` | Input data (git-ignored) |

## Tests

```bash
python -m pytest -q      # 51 tests
```

## Open points

- **To validate clinically**: the infarction codes (`I21*`, `I22*`, `410*`) and the laboratory `itemid`s of `mimic_iv_infarction.yaml` (check against `d_labitems`). High-Sensitivity CRP (itemid `51652`) was removed from that config: it has only ~40 measurements in the whole MIMIC-IV AMI cohort, i.e. it is essentially never ordered on the index admission, so it added no signal and mostly produced missing-value features (see `AUDIT_zarrow.md`).
- **Existing Zarr stores**: those written before the filtering fix (`inclusion` flag) contain all patients. They need to be rebuilt.
- **Temporary files**: `remove_tmp_dir()` is not called automatically.
- **Dates**: they are stored as text and must be converted back on read.
- **Index**: `ZarrLoader.load_index()` reads the first array found at the root.

## Branches and roadmap

`master` = working code, `dev` = development (see [contributing.md](contributing.md)). Next steps: laboratory time series, chest X-rays, CT (DICOM), waveforms, tabular MIMIC-III, then IMPROVE, infarction registries and VitalDB.

## Reference

Johnson, A.E.W. et al. *MIMIC-IV, a freely accessible electronic health record dataset.* Sci Data 10, 1 (2023). https://doi.org/10.1038/s41597-022-01899-x

License: see [LICENSE](LICENSE).
