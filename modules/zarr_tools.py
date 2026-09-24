import pandas as pd
import numpy as np
import zarr

class ZarrWriter:
    def __init__(self, zarr_path:str,index_id:str):
        self.store = zarr.open(zarr_path, mode='w')
        self.index_id=index_id
    def write_dataframe(self, df, group_name):
        group = self.store.require_group(group_name)
        chunks = (max(1, min(len(df), 1024)),)
        for col in df.columns:
            data = df[col].to_numpy(copy=True)

            # strings -> variable-length UTF-8 (Zarr v3 spec), missing values -> ''
            if data.dtype == object:
                data = pd.Series(data).fillna("").astype(str).to_numpy(dtype=object)
                arr = group.create_array(name=col, shape=data.shape, dtype=str, chunks=chunks, overwrite=True)
                arr[:] = data
                continue

            # int64/float64 -> int32/float32, only when values fit
            if data.dtype.kind == "i" and (len(data) == 0 or (
                    data.min() >= np.iinfo(np.int32).min and data.max() <= np.iinfo(np.int32).max)):
                data = data.astype("int32")
            elif data.dtype.kind == "f":
                data = data.astype("float32")

            group.create_array(name=col, data=data, chunks=chunks, overwrite=True)

    def write_index(self, subject_ids:np.array):
        self.store.create_array(
            name=self.index_id,
            data=subject_ids,
            chunks=(max(1, min(len(subject_ids), 1024)),),
            overwrite=True
        )
        
class ZarrLoader:
    def __init__(self, zarr_path:str):
        self.store = zarr.open(zarr_path, mode='r')

    def load_group(self, group_name:str, as_df:bool=False):
        """
        Load a group from zarr store.
        - convert bytes-> str
        - return a Dataframe or Dict of np arrays
        """
        group = self.store[group_name]
        data_dict = {}

        for col in group.array_keys():
            arr = group[col][:]
            if arr.dtype.kind == "S":      # stores written by the previous version (utf-8 bytes)
                arr = np.char.decode(arr, "utf-8")
            elif arr.dtype.kind == "T":    # numpy StringDType -> object for pandas
                arr = arr.astype(object)
            data_dict[col] = arr

        if as_df:
            return pd.DataFrame(data_dict)
        return data_dict

    def load_index(self):
        """
        load index subject_id
        """
        keys = list(self.store.array_keys())
        arr = self.store[keys[0]][:] if keys else np.array([])
        return arr