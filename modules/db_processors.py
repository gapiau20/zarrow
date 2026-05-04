import os

import pandas as pd
from .db_filters import BaseFilter

class EHRDatabaseBaseProcessor:
    '''
    Base class for database processors
    '''
    def __init__(self, zarr_index:str,columns:list[str]=None, filters:list[BaseFilter]=[]):
        self.zarr_index=zarr_index
        self.columns=columns
        self.filters=filters

    def add_filters(self, filters:list[BaseFilter]):
        '''
        Add filters to the processor
        '''
        self.filters.extend(filters)

    def process(self, df:pd.DataFrame) ->tuple[pd.DataFrame, str]:
        '''
        Process the dataframe and return the processed dataframe
        '''
        raise NotImplementedError('Implement in daughter class.')

class EHRDataFrameProcessor(EHRDatabaseBaseProcessor):
    '''
    Processor for EHR dataframes, to be used in the cohort building process.
    '''
    def __init__(self, zarr_index:str, columns:list[str]=None, filters:list[BaseFilter]=[]):
        super().__init__(zarr_index, columns, filters)

    def process(self, df:pd.DataFrame) ->tuple[pd.DataFrame, str]:
        '''
        Process the dataframe and return the processed dataframe
        '''
        #default implementation is to return the dataframe with its specified columns and the zarr index column, after applying the filters
        for filter in self.filters:
            df=filter.apply(df)

        return df[self.columns], self.zarr_index
    
class PatientPreprocessor(EHRDatabaseBaseProcessor):
    pass

def register_database_processors():
    processor_map = {}
    for name, obj in globals().items():
        if isinstance(obj, type) and issubclass(obj, EHRDatabaseBaseProcessor):
            processor_map[name] = obj
    return processor_map
