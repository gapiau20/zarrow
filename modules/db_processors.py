import os

import pandas as pd
from .db_filters import BaseFilter

PROCESSOR_REGISTRY = {}
class DatabaseProcessor:
    '''
    Base class for database processors
    '''
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls is not DatabaseProcessor:
            PROCESSOR_REGISTRY[cls.__name__] = cls

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

class EHRDataFrameProcessor(DatabaseProcessor):
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
    
class PatientPreprocessor(DatabaseProcessor):
    pass

def register_database_processors():
    return PROCESSOR_REGISTRY
