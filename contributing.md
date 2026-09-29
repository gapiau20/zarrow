# Branches
- Working code is put on _master_
- Development is put on branch _dev_ 
# Roadmap
## MIMIC-IV
- [x] MIMIC-IV cohort creation
- [ ] Add lab values time series
- [x] MIMIC-IV cohort 28-day MI mortality with a toy example (now uses demographics + lab values, see `scripts/mortality28d.py`)
- [ ] Add ECG modality
- [ ] Add Waveforms modality
- [ ] Extend to MIMIC-III (Tabular data only)

## IMPROVE
## MI registries
## VitalDB

# Known issues / before you touch the code

From a static audit of the repo (see `AUDIT_zarrow.md` at the repo root — git-ignored, regenerate it with `/paper-repo-audit` if you need the full report). Check these before relying on results for a paper/report, and any of them makes a good first contribution:

- 🟠 `scripts/mortality28d.py`: the `mortality_28d` label doesn't guard against `dod < admittime` (a recorded death date earlier than the admission). Harmless on the current MIMIC-IV run, but add `& ((df["dod"] - df["admittime"]).dt.days >= 0)` before reusing this label logic on another, less clean dataset.
- 🟡 `modules/db_processors.py::EHRDataFrameProcessor.process` / `modules/cohort.py::read_tabular_file`: raw CSVs are always parsed with every column (no `usecols`) and only sliced down to the schema's `columns` afterwards. Wastes memory/time on wide tables (`labevents.csv.gz`) and is the direct cause of `DtypeWarning`s on columns that aren't even requested (e.g. `order_provider_id`) when building a cohort.
- 🟡 `modules/cohort.py::BaseCohort.__init__`: no check that `zarr_index_name` (e.g. `subject_id`) is actually listed in a group's `columns`. A misconfigured YAML fails late and unclearly (`KeyError` during filtering/intersection) instead of at construction time.
- 🟡 `modules/cohort.py::BaseCohort.build_cohort`: `remove_tmp_dir()` is commented out — temporary PhysioNet downloads are never cleaned up. Either re-enable it or expose a `cleanup=True/False` parameter.
- 🟡 `modules/torch_loader.py::MultimodalDataset`: `select_dtypes("number")` doesn't exclude `index_name`, so `subject_id` ends up as a feature column in the returned tensors (asserted by `tests/test_torch_loader.py`, so intentional today, but a foot-gun for anyone plugging the dataset straight into a model without checking `dataset.columns`).
- `inclusion: true` groups (see README) restrict the cohort at **patient** granularity, not `(subject_id, hadm_id)`. A per-admission filter (e.g. `ICDFilter` on diagnoses) keeps *all* admissions of a matching patient, not just the matching one — `scripts/mortality28d.py` re-applies the finer join manually (merge on `subject_id` + `hadm_id`). Any new script built on an event-level `inclusion` filter needs to do the same.

# Data quality notes

- `labevents.csv.gz`'s `value` column is free text and can be masked (`"___"`) by MIMIC-IV's de-identification; always aggregate `valuenum` for numeric lab features, not `value` (see `config/mimic_iv_infarction.yaml`).
- Before adding a lab `itemid` to a cohort config, check its actual coverage on the target cohort's index admissions — a rare test can end up ~0% populated (this happened with High-Sensitivity CRP / itemid `51652` for the AMI cohort; it was removed from `mimic_iv_infarction.yaml`).