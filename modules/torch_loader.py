'''
Dataset and dataloader utilities for pytorch.
'''
import numpy as np
import torch
import zarr
from torch.utils.data import Dataset

from modules.multimodal import decode_records
from modules.zarr_tools import ZarrLoader

class _LazyModalities:
    '''Open the modality arrays on first access, so that each DataLoader worker opens its own.'''
    def _modality_data(self, name):
        if getattr(self, "_arrays", None) is None:
            store = zarr.open_group(self.zarr_path, mode="r")
            self._arrays = {m: store[m]["data"] for m in self.modality_names}
        return self._arrays[name]

    def __getstate__(self):
        return {**self.__dict__, "_arrays": None}

class MultimodalDataset(_LazyModalities, Dataset):
    '''
    One item per subject of the zarr index.
    Returns, for each requested group, the numeric columns of that subject's rows as a float tensor.
    For each requested modality, returns the subject's valid records as a float tensor (k, *record_shape)
    in physical units (NaN = missing sample), and their offsets from the anchor time under "<modality>_offset_h".
    Only the records of the requested subject are read from disk.
    Items have a variable number of rows/records: use a padding collate_fn with a DataLoader.
    Zarr does not keep the column order, so columns are sorted by name (see self.columns).
    '''
    def __init__(self, zarr_path:str, index_name:str, groups:list[str], modalities:list[str]=()):
        loader = ZarrLoader(zarr_path)
        self.zarr_path = zarr_path
        self.ids = loader.store[index_name][:]
        self.columns = {}
        self.tables = {}
        for g in groups:
            df = loader.load_group(g, as_df=True)
            self.columns[g] = sorted(df.select_dtypes("number").columns)
            self.tables[g] = dict(tuple(df[self.columns[g]].groupby(df[index_name])))
        # small per-record arrays in memory, the records themselves stay on disk
        self.modality_names = list(modalities)
        self.modalities = {}
        for m in modalities:
            group = loader.store[m]
            self.modalities[m] = {"ptr": group["patient_ptr"][:], "valid": group["valid"][:],
                                  "offset_h": group["record_offset_h"][:], "attrs": dict(group.attrs)}
        self._arrays = None

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        sid = self.ids[idx]
        out = {}
        for g, rows_by_id in self.tables.items():
            rows = rows_by_id.get(sid)
            # contiguous copy: column reordering can yield negative strides, which torch rejects
            values = np.ascontiguousarray(rows.to_numpy(dtype=np.float32)) if rows is not None \
                else np.empty((0, len(self.columns[g])), dtype=np.float32)
            out[g] = torch.tensor(values, dtype=torch.float32)
        for m, meta in self.modalities.items():
            start, stop = meta["ptr"][idx], meta["ptr"][idx + 1]
            keep = meta["valid"][start:stop]
            records = decode_records(self._modality_data(m)[start:stop], meta["attrs"])[keep]
            out[m] = torch.from_numpy(np.ascontiguousarray(records))
            out[f"{m}_offset_h"] = torch.from_numpy(meta["offset_h"][start:stop][keep])
        return out

class RecordDataset(_LazyModalities, Dataset):
    '''
    One item per valid record of a modality (e.g. per ECG, for pretraining or record-level labels):
    (record as float tensor in physical units, subject id, offset from the anchor time in hours).
    '''
    def __init__(self, zarr_path:str, modality:str):
        group = ZarrLoader(zarr_path).store[modality]
        self.zarr_path = zarr_path
        self.modality_names = [modality]
        self.attrs = dict(group.attrs)
        self.rows = np.flatnonzero(group["valid"][:])
        self.subjects = group["record_subject"][:]
        self.offset_h = group["record_offset_h"][:]
        self._arrays = None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        record = decode_records(self._modality_data(self.modality_names[0])[row], self.attrs)
        return torch.from_numpy(record), int(self.subjects[row]), float(self.offset_h[row])
