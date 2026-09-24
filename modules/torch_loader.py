'''
Dataset and dataloader utilities for pytorch.
'''
import numpy as np
import torch
from torch.utils.data import Dataset

from modules.zarr_tools import ZarrLoader

class MultimodalDataset(Dataset):
    '''
    One item per subject of the zarr index.
    Returns, for each requested group, the numeric columns of that subject's rows as a float tensor.
    Items have a variable number of rows: use a padding collate_fn with a DataLoader.
    '''
    def __init__(self, zarr_path:str, index_name:str, groups:list[str]):
        loader = ZarrLoader(zarr_path)
        self.ids = loader.store[index_name][:]
        self.tables = {g: dict(tuple(loader.load_group(g, as_df=True).groupby(index_name))) for g in groups}

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        sid = self.ids[idx]
        out = {}
        for g, rows_by_id in self.tables.items():
            rows = rows_by_id.get(sid)
            values = rows.select_dtypes("number").to_numpy() if rows is not None else np.empty((0, 0))
            out[g] = torch.tensor(values, dtype=torch.float32)
        return out
