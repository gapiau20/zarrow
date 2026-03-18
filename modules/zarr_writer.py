import zarr
import numpy as np

class ZarrWriter:
    def __init__(self, zarr_path:str,index_id:str):
        self.store = zarr.open(zarr_path, mode='w')
        self.index_id=index_id
    def write_dataframe(self, df, group_name):
        group = self.store.create_group(group_name)

        for col in df.columns:
            data = df[col].to_numpy()

            # string
            if data.dtype == object:
                data = data.astype("U")

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