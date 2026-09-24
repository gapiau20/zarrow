import codecs
import gzip
import yaml
import shutil
import pandas as pd

import os
from modules.zarr_tools import ZarrWriter,ZarrLoader

from modules.db_filters import register_filters
from modules.db_processors import DatabaseProcessor, register_database_processors

def get_columns_from_dataframe(df:pd.DataFrame):
    '''Build a schema dict from a dataframe containing the necessary information to build the cohort.'''
    return [col for col in df.columns]

def detect_encoding(filepath:str, encodings=("utf-8", "cp1252", "latin1"), block_size:int=1<<20)->str:
    '''Return the first encoding able to decode the whole file (streamed, gzip aware).'''
    opener = gzip.open if filepath.lower().endswith('.gz') else open
    for enc in encodings:
        decoder = codecs.getincrementaldecoder(enc)()
        try:
            with opener(filepath, 'rb') as f:
                while block := f.read(block_size):
                    decoder.decode(block)
                decoder.decode(b'', final=True)
            return enc
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not decode file {filepath} with tried encodings.")

def read_tabular_file(filepath:str,**kwargs):
    """Read a file using pandas based on its extension.
    With chunksize, always returns an iterable of DataFrames (like pd.read_csv)."""
    filepath = str(filepath)
    name = filepath.lower()
    ext = os.path.splitext(name[:-3] if name.endswith('.gz') else name)[-1]

    if ext == ".csv":
        # pass encoding explicitly (e.g. from the YAML read_options) to skip the detection pass on large files
        kwargs.setdefault("encoding", detect_encoding(filepath))
        return pd.read_csv(filepath, **kwargs)

    chunksize = kwargs.pop("chunksize", None)
    if ext in [".xls", ".xlsx"]:
        df = pd.read_excel(filepath, **kwargs)
    elif ext == ".parquet":
        df = pd.read_parquet(filepath, **kwargs)
    elif ext == ".json":
        df = pd.read_json(filepath, **kwargs)
    else:
        raise ValueError(f"Unsupported file format: {ext}")
    return [df] if chunksize else df

def get_schema_from_tabular_file(file_path)->dict:
    schema={}
    #add the filename key
    schema['file']=file_path
    #add the columns key
    schema['columns']=get_columns_from_dataframe(read_tabular_file(file_path))
    #add filters
    schema['filters']=[]
    schema['processor']=None
    return schema

def build_default_schema_from_tabular_file(input_file,output_yaml):
    schema=get_schema_from_tabular_file(input_file)
    with open(output_yaml, 'w') as f :
        yaml.safe_dump(schema,f)
    return
def get_schema_from_config(config_path):
    '''
    Get the schema for cohort building from a yaml config file.
    The config file should have the following structure:
    dataset:
        name: name of the dataset (e.g. mimic-iv-demo)
        group_key1:
            file: path to the csv file to process for this group (relative to the tmp_dir)
            columns:
                - subject_id
                - etc. #columns to read from the csv file
            processor:
                name: EHRDataFrameProcessor #name of the processor class to use for this group (should be a subclass of DatabaseProcessor and should be registered in the processor registry)
            filters:
                - name: name of the filter class (e.g. AgeFilter, SexFilter, etc.)
                  parameters: parameters to initialize the filter class (e.g. age_threshold: 18)
        group_key2:
            etc.
    '''
    import yaml
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config['dataset']



class BaseCohort:
    def __init__(self,tmp_dir:str,zarr_index_name:str,schema={},chunk_size:int=100_000):
        '''
        Base class for building a cohort and saving it into a zarr dataset.
        tmp_dir: temporary directory to store intermediate files during cohort building
        zarr_index: name of the column to use as index in the zarr dataset (e.g. subject_id)
        filters: list of filters to apply on the patient table to select the cohort
        should be a dict with keys as group where the filters apply and values as a list of filter objects 
        (e.g. PatientFilter, AgeFilter, etc.)
        e.g. filters={'patient': [AgeFilter(age_min=50), 'sex': SexFilter(sex='M')]}.
        '''
        self.tmp_dir=tmp_dir
        self.schema=schema
        self.db=self.schema['name']
        self.zarr_index=zarr_index_name
        self.chunk_size=chunk_size

        #create processors for each group in the schema and add the filters to the procesors
        self.processors={}
        #get the processor and filter mappings
        filter_map=register_filters()
        processor_map=register_database_processors()

        #create the processor for each group in the schema
        for group_key, group_info in self.schema.items():
            if not isinstance(group_info, dict):  # dataset metadata (name, version...), not a group
                continue
            #get the processor class from the schema
            processor_cls_name=(group_info.get('processor') or {}).get('name')
            if processor_cls_name is None:
                raise ValueError(f'Processor class not specified for group {group_key} in the schema.')
            processor_cls=processor_map.get(processor_cls_name, None)
            if processor_cls is None:
                raise ValueError(f'Processor class {processor_cls_name} not found in the registered processors.')
            #get the filters for the group from the schema            
            filters=[]
            for filter_info in group_info.get('filters', []):
                filter_cls_name=filter_info.get('name', None)
                if filter_cls_name is None:
                    raise ValueError(f'Filter class not specified for group {group_key} in the schema.')
                filter_cls=filter_map.get(filter_cls_name, None)
                if filter_cls is None:
                    raise ValueError(f'Filter class {filter_cls_name} not found in the registered filters.')
                filter_params=filter_info.get('parameters', {})
                filters.append(filter_cls(**filter_params))
            
            #initialize the processor with the filters
            self.processors[group_key]=processor_cls(zarr_index=self.zarr_index, columns=group_info.get('columns'), filters=filters)

        # groups whose filters define who belongs to the cohort (intersection of their subjects)
        self.inclusion_groups=[k for k, v in self.schema.items() if isinstance(v, dict) and v.get('inclusion', False)]
        return
            
    def remove_tmp_dir(self):
        '''
        Clean temporary dir that was used to build the cohort
        '''
        shutil.rmtree(self.tmp_dir)

    
    def download_file(self,file:str):
        raise NotImplementedError("This method should be implemented in the subclass to download the necessary files to build the cohort.")
    
    def build_cohort(self)->pd.DataFrame:
        #first download the csv_files that can serve to filter the cohort
        cohort={}
        #for each group in the schema, build the corresponding cohort and apply the filters
        for p in self.processors.keys():
            print(f'Building cohort for group {p}...')
            #load the necessary table to build the cohort for the group and apply the filters defined in the processor
            dataset_csv_file=self.schema[p]['file']
            self.download_file(dataset_csv_file)
            print(f'Filters to apply: {self.processors[p].filters}')
            group=self.process_chunks(os.path.join(self.tmp_dir, dataset_csv_file), self.processors[p],
                                      **self.schema[p].get('read_options', {}))
            cohort[p]=group
        # self.remove_tmp_dir() #commented out for now for debugging purposes, but should be uncommented in production to avoid filling up the disk with temporary files TODO
        if self.inclusion_groups:
            ids=set.intersection(*(set(cohort[g][self.zarr_index]) for g in self.inclusion_groups))
            cohort={g: df[df[self.zarr_index].isin(ids)] for g, df in cohort.items()}
        return cohort
    
    def build_and_save(self,zarr_path:str):
        '''
        Build the cohort and save it into a zarr dataset    
        '''
        #cohort build
        cohort = self.build_cohort()
        #save the cohort into a zarr dataset
        writer = ZarrWriter(zarr_path, self.zarr_index)
        for k in cohort.keys():
            writer.write_dataframe(cohort[k], k)
        ids = pd.concat([df[self.zarr_index] for df in cohort.values()]).drop_duplicates().sort_values()
        writer.write_index(ids.to_numpy())

    def load(self, groups:list[str],zarr_path:str):
        if not os.path.exists(zarr_path):
            print(f'Building cohort at: {zarr_path}')
            self.build_and_save(zarr_path)
        print(f'Loading cohort from {zarr_path}')
        loader=ZarrLoader(zarr_path)
        return {g:loader.load_group(g,as_df=False) for g in groups}

class TabularCohort(BaseCohort):
    def process_chunks(self,file_path:str,processor:DatabaseProcessor,**read_options)->pd.DataFrame:
        '''
        Generic function to process a csv file in chunks and apply the processor to each chunk.
        file_path: path to the csv file
        processor: processor to apply to each chunk (should be a subclass of DatabaseProcessor)
        read_options: extra keyword arguments for the reader (e.g. encoding, usecols)
        '''
        cohort = []
        for chunk in read_tabular_file(file_path,chunksize=self.chunk_size,**read_options):
            processed_chunk, zarr_index = processor.process(chunk)
            if not processed_chunk.empty:
                cohort.append(processed_chunk)
        if not cohort:
            return pd.DataFrame(columns=processor.columns)
        cohort=pd.concat(cohort,axis=0)
        return cohort
    