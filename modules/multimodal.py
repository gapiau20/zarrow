'''
Utilities to read the modality and store it into the zarr database

A modality (ECG, chest X-ray, CT...) is stored along a record axis, not a patient axis:
a patient can have 0..n records. Layout of a modality group in the cohort store:
    <name>/
        data              (R, *record_shape)  one record per chunk, chunks grouped in shards
        record_subject    (R,)                index value (e.g. subject_id) of each record
        record_id         (R,)                source identifier (e.g. ECG study_id)
        record_time       (R,)                acquisition time, datetime64[s]
        record_hadm       (R,)                matched anchor admission (-1 without time window)
        record_offset_h   (R,)                hours from the anchor time (NaN without time window)
        valid             (R,)                False if the record could not be read (data left at 0)
        patient_ptr       (N+1,)              CSR offsets: records of the i-th index value are [ptr[i], ptr[i+1])
Records are sorted by (position in the root index, time), so the records of a patient are contiguous.

Example usage:
    add_modality("data/demo.zarr", "subject_id", "ecg", {
        "adapter": "MIMICECGAdapter",
        "parameters": {"source_dir": "data/mimic-iv-ecg-demo", "project": "mimic-iv-ecg-demo", "version": "0.1"},
        "window": {"anchor_group": "admission", "anchor_time": "admittime", "hours": [-12, 24]},
    })
'''
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import yaml
import zarr
from tqdm import tqdm

from modules.download import download_physionet_file
from modules.zarr_tools import ZarrLoader

MODALITY_REGISTRY = {}
MANIFEST_COLUMNS = ["subject", "record_id", "record_time", "path"]

class ModalityAdapter:
    '''
    Read one modality from its source files.
    Subclasses are registered automatically and usable by name in the YAML (modalities.<name>.adapter).
    '''
    record_shape:tuple = ()
    dtype:str = "float32"
    records_per_shard:int = 256

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        MODALITY_REGISTRY[cls.__name__] = cls

    def __init__(self, source_dir:str, workers:int=8):
        self.source_dir = source_dir
        self.workers = workers

    def manifest(self)->pd.DataFrame:
        '''One row per available record, with the MANIFEST_COLUMNS (path relative to source_dir).'''
        raise NotImplementedError('Implement in daughter class.')

    def fetch(self, records:pd.DataFrame):
        '''Make the files of the selected records available under source_dir (default: nothing to do).'''

    def load(self, record)->np.ndarray:
        '''Read one manifest row into an array of shape record_shape and type dtype.'''
        raise NotImplementedError('Implement in daughter class.')

    def attrs(self)->dict:
        '''Metadata needed to interpret the stored arrays (units, scale, channel names...).'''
        return {}

class MIMICECGAdapter(ModalityAdapter):
    '''
    12-lead, 10 s diagnostic ECGs of MIMIC-IV-ECG (https://physionet.org/content/mimic-iv-ecg/).
    Stored as int16 at the native gain of the files (200 ADC units per mV), so the storage is lossless.
    '''
    LEADS = ["I", "II", "III", "aVR", "aVF", "aVL", "V1", "V2", "V3", "V4", "V5", "V6"]  # order of the files
    FS = 500
    GAIN = 200                  # ADC units per mV
    MISSING = -32768            # WFDB invalid sample for 16-bit formats
    record_shape = (5000, 12)
    dtype = "int16"

    def __init__(self, source_dir:str, project:str="mimic-iv-ecg", version:str="1.0", workers:int=8):
        super().__init__(source_dir, workers)
        self.project = project
        self.version = str(version)

    def _download(self, file:str):
        if not os.path.exists(os.path.join(self.source_dir, file)):
            download_physionet_file(self.project, self.version, file, self.source_dir)

    def manifest(self)->pd.DataFrame:
        self._download("record_list.csv")
        df = pd.read_csv(os.path.join(self.source_dir, "record_list.csv"))
        return pd.DataFrame({"subject": df["subject_id"], "record_id": df["study_id"],
                             "record_time": pd.to_datetime(df["ecg_time"]), "path": df["path"]})

    def fetch(self, records:pd.DataFrame):
        files = [p + ext for p in records["path"] for ext in (".hea", ".dat")]
        with ThreadPoolExecutor(self.workers) as ex:
            list(tqdm(ex.map(self._download, files), total=len(files), desc="Fetching ECG files", leave=False))

    def load(self, record)->np.ndarray:
        import wfdb
        rec = wfdb.rdrecord(os.path.join(self.source_dir, record.path), physical=False)
        if rec.fs != self.FS or rec.sig_len != self.record_shape[0]:
            raise ValueError(f"{record.path}: expected {self.record_shape[0]} samples at {self.FS} Hz, "
                             f"got {rec.sig_len} at {rec.fs} Hz")
        order = [rec.sig_name.index(lead) for lead in self.LEADS]   # fails loudly on a missing lead
        digital = rec.d_signal[:, order]
        mv = (digital - np.asarray(rec.baseline)[order]) / np.asarray(rec.adc_gain)[order]
        out = np.clip(np.rint(mv * self.GAIN), -32767, 32767).astype(np.int16)
        out[digital == self.MISSING] = self.MISSING
        return out

    def attrs(self)->dict:
        return {"leads": self.LEADS, "fs": self.FS, "units": "mV", "scale": 1 / self.GAIN,
                "missing_value": self.MISSING, "source": f"{self.project}/{self.version}"}

def register_modality_adapters():
    return MODALITY_REGISTRY

def get_modalities_from_config(config_path)->dict:
    '''The optional top-level 'modalities' section of a cohort config (next to 'dataset').'''
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config.get('modalities') or {}

def load_anchors(loader:ZarrLoader, index_name:str, anchor_group:str, anchor_time:str,
                 restrict_to:str=None)->pd.DataFrame:
    '''
    Anchor times (subject, hadm, anchor_time) read from a tabular group of the store, e.g. admission/admittime.
    restrict_to: keep only the anchors whose (index, hadm_id) appear in this other group,
    e.g. the ICD-filtered diagnoses, to anchor on the matching admissions only.
    '''
    df = loader.load_group(anchor_group, as_df=True)
    anchors = pd.DataFrame({"subject": df[index_name],
                            "hadm": df["hadm_id"] if "hadm_id" in df else -1,
                            "anchor_time": pd.to_datetime(df[anchor_time], errors="coerce")})  # '' -> NaT
    if restrict_to:
        keep = loader.load_group(restrict_to, as_df=True)[[index_name, "hadm_id"]].drop_duplicates()
        keep.columns = ["subject", "hadm"]
        anchors = anchors.merge(keep, on=["subject", "hadm"])
    return anchors.dropna(subset=["anchor_time"]).drop_duplicates()

def select_records(manifest:pd.DataFrame, ids, anchors:pd.DataFrame=None, hours=None, select:str="all")->pd.DataFrame:
    '''
    Keep the records of the cohort ids, then, with anchors, those within [hours[0], hours[1]] of an anchor time.
    A record within the window of several anchors is matched to the closest one.
    select: all | first | last record of each patient.
    '''
    records = manifest[manifest["subject"].isin(ids)]
    if anchors is None:
        records = records.assign(hadm=-1, offset_h=np.nan)
    else:
        records = records.merge(anchors, on="subject")
        records["offset_h"] = (records["record_time"] - records["anchor_time"]).dt.total_seconds() / 3600
        records = records[records["offset_h"].between(hours[0], hours[1])]
        records = (records.assign(dist=records["offset_h"].abs()).sort_values("dist")
                   .drop_duplicates("record_id").drop(columns=["dist", "anchor_time"]))
    records = records.sort_values(["subject", "record_time"])
    if select == "first":
        records = records.groupby("subject").head(1)
    elif select == "last":
        records = records.groupby("subject").tail(1)
    elif select != "all":
        raise ValueError(f"Unknown select '{select}' (expected all, first or last).")
    return records.reset_index(drop=True)

def write_modality(zarr_path:str, index_name:str, name:str, adapter:ModalityAdapter, records:pd.DataFrame,
                   attrs:dict=None):
    '''Write the selected records of a modality as <name>/ in an existing cohort store (replaced if present).'''
    root = zarr.open_group(zarr_path, mode="r+")
    index = root[index_name][:]
    # sort the records by position in the root index, so each patient's records are contiguous
    pos = pd.Index(index).get_indexer(records["subject"])
    if (pos < 0).any():
        raise ValueError(f"{name}: some records belong to subjects absent from the root index.")
    order = np.lexsort((records["record_time"].to_numpy(), pos))
    records, pos = records.iloc[order].reset_index(drop=True), pos[order]

    if name in root:
        del root[name]
    group = root.create_group(name)
    group.attrs.update({**(attrs or {}), **adapter.attrs(), "record_shape": list(adapter.record_shape)})
    n, shape = len(records), tuple(adapter.record_shape)
    per_shard = max(1, min(adapter.records_per_shard, n))
    data = group.create_array("data", shape=(n, *shape), dtype=adapter.dtype, chunks=(1, *shape),
                              shards=(per_shard, *shape), fill_value=0)

    valid = np.zeros(n, dtype=bool)
    def load(i):
        try:
            return adapter.load(records.iloc[i])
        except Exception as err:
            print(f"Skipping record {records['record_id'].iloc[i]}: {err}")
            return None
    # one write per shard: a partial shard write would read-modify-write it
    with ThreadPoolExecutor(adapter.workers) as ex:
        for start in tqdm(range(0, n, per_shard), desc=f"Writing {name}", leave=False):
            stop = min(start + per_shard, n)
            block = np.zeros((stop - start, *shape), dtype=adapter.dtype)
            for i, arr in enumerate(ex.map(load, range(start, stop))):
                if arr is not None:
                    block[i], valid[start + i] = arr, True
            data[start:stop] = block

    group.create_array("record_subject", data=records["subject"].to_numpy())
    group.create_array("record_id", data=records["record_id"].to_numpy())
    group.create_array("record_time", data=records["record_time"].to_numpy().astype("datetime64[s]"))
    group.create_array("record_hadm", data=records["hadm"].to_numpy(dtype=np.int64))
    group.create_array("record_offset_h", data=records["offset_h"].to_numpy(dtype=np.float32))
    group.create_array("valid", data=valid)
    group.create_array("patient_ptr", data=np.searchsorted(pos, np.arange(len(index) + 1)).astype(np.int64))
    print(f"{name}: {valid.sum()}/{n} records written for {len(np.unique(pos))}/{len(index)} subjects.")

def add_modality(zarr_path:str, index_name:str, name:str, config:dict):
    '''
    Add a modality to an existing cohort store, from its config:
        adapter: name of a registered ModalityAdapter
        parameters: keyword arguments of the adapter (source_dir...)
        window: optional {anchor_group, anchor_time, hours: [start, end], restrict_to}
        select: all (default) | first | last
    '''
    adapter_cls = MODALITY_REGISTRY.get(config.get("adapter"))
    if adapter_cls is None:
        raise ValueError(f"Modality adapter {config.get('adapter')} not found in the registered adapters.")
    adapter = adapter_cls(**config.get("parameters", {}))
    loader = ZarrLoader(zarr_path)
    ids = loader.store[index_name][:]

    window = dict(config.get("window") or {})
    hours = window.pop("hours", None)
    anchors = load_anchors(loader, index_name, **window) if window else None
    if anchors is not None and hours is None:
        raise ValueError(f"{name}: a window needs 'hours: [start, end]'.")

    records = select_records(adapter.manifest(), ids, anchors, hours, config.get("select", "all"))
    adapter.fetch(records)
    write_modality(zarr_path, index_name, name, adapter, records,
                   attrs={"adapter": config["adapter"], "parameters": config.get("parameters", {}),
                          "window": config.get("window"), "select": config.get("select", "all")})

def decode_records(data:np.ndarray, attrs)->np.ndarray:
    '''Stored records -> float32 in physical units (attrs 'scale'), missing samples -> NaN.'''
    out = data.astype(np.float32)
    missing = attrs.get("missing_value")
    if missing is not None:
        out[data == missing] = np.nan
    return out * attrs.get("scale", 1.0)
