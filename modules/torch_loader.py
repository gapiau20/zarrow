'''
Dataset and dataloader utilities for pytorch.
'''
import zarr
import torch
from torch.utils.data import Dataset

class MultimodalDataset(Dataset):
    def __init__(self, zarr_path):
        self.data = zarr.load(zarr_path)
        self.keys = list(self.data.keys())

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        item = self.data[str(self.keys[idx])]
        # return tabular, ecg, cxr, text as torch tensors
        return {k: torch.tensor(v) for k,v in item.items()}