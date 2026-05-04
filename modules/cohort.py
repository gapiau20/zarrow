import yaml
import shutil
import pandas as pd

from modules.db_filters import register_filters
from modules.db_processors import EHRDatabaseBaseProcessor, register_database_processors

def get_columns_from_dataframe(df:pd.DataFrame):
    '''Build a schema dict from a dataframe containing the necessary information to build the cohort.'''
    return [col for col in df.columns]
def get_schema_from_csv(file_path):
    schema={}
    #add the filename key
    schema['file']=file_path
    #add the columns key
    schema['columns']=get_columns_from_dataframe(pd.read_csv(file_path, nrows=0))
    #add filters
    schema['filters']=[]
    return schema
def get_schema_from_config(config_path):
    '''
    Get the schema for cohort building from a yaml config file.
    The config file should have the following structure:
    dataset:
        name: name of the dataset (e.g. mimic-iv-demo)
        patient:
            file: path to the patient csv file (relative to the tmp_dir)
            filters:
                - name: name of the filter class (e.g. AgeFilter, SexFilter, etc.)
                  parameters: parameters to initialize the filter class (e.g. age_threshold: 18)
        admission:
            file: path to the admission csv file (relative to the tmp_dir)
            filters:
                - name: name of the filter class (e.g. AgeFilter, SexFilter, etc.)
                  parameters: parameters to initialize the filter class (e.g. age_threshold: 18)
        diagnoses:
            file: path to the diagnoses csv file (relative to the tmp_dir)
            filters:
                - name: name of the filter class (e.g. ICDFilter, etc.)
                  parameters: parameters to initialize the filter class (e.g. icd_codes: ['I21*'])
        labevents:
            file: path to the labevents csv file (relative to the tmp_dir)
            filters:
                - name: name of the filter class (e.g. LabEventFilter, etc.)
                  parameters: parameters to initialize the filter class (e.g. itemids: [50811, 50907, etc.])
    '''
    import yaml
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config['dataset']
class PatientProcessor(EHRDatabaseBaseProcessor):
    pass
class BaseCohort:
    def __init__(self,tmp_dir:str,zarr_index_name:str,schema={}):
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

        #create processors for each group in the schema and add the filters to the procesors
        self.processors={}
        #get the processor and filter mappings
        filter_map=register_filters()
        processor_map=register_database_processors()

        #create the processor for each group in the schema
        for group_key, group_info in self.schema.items():
            if group_key in ['name']:
                continue
            #get the processor class from the schema
            processor_cls_name=group_info.get('processor', None).get('name', None)
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
        return
            
    def remove_tmp_dir(self):
        '''
        Clean temporary dir that was used to build the cohort
        '''
        shutil.rmtree(self.tmp_dir)
    
    def build_cohort(self):
        '''
        Build the cohort and return a dict of dataframes corresponding to the different groups (e.g. patient, admission, diagnoses, etc.)
        The keys of the dict should correspond to the keys in the filters dict.
        '''
        raise NotImplementedError('Implement in daughter class.')