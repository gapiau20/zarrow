import pandas as pd
import numpy as np
import zarr

class ZarrWriter:
    def __init__(self, zarr_path:str,index_id:str):
        self.store = zarr.open(zarr_path, mode='w')
        self.index_id=index_id
    def write_dataframe(self, df, group_name):
        group = self.store.require_group(group_name)
        for col in df.columns:
            data = df[col].to_numpy(copy=True)

            # strings -> bytes (stable Zarr V3)
            if data.dtype == object:
                data = data.astype("S")  # string -> bytes

            # int64/float64 -> int32/float32
            elif data.dtype.kind in ["i"]:  # integer
                data = data.astype("int32")
            elif data.dtype.kind in ["f"]:  # float
                data = data.astype("float32")

            # delete if already existing
            if col in group:
                del group[col]

            # create_array with chunks
            group.create_array(
                name=col,
                data=data,
                chunks=(min(len(data), 1024),),
                overwrite=True
        )

    def write_index(self, subject_ids:np.array):
        self.store.create_array(
        name=self.index_id,
        data=subject_ids,
        chunks=(min(len(subject_ids), 1024),),
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
            # bytes -> str
            if arr.dtype.kind == "S":
                arr = arr.astype(str)
            data_dict[col] = arr

        if as_df:
            return pd.DataFrame(data_dict)
        return data_dict

    def load_index(self):
        """
        load index subject_id
        """
        arr = self.store[self.store.array_keys()[0]][:] if len(self.store.array_keys())>0 else np.array([])
        return arr