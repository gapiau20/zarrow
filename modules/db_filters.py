'''
Module with filter functions to apply to a dataFrame

Example usage: 
    filter=PatientFilter(..)
    df=pd.DataFrame(...)
    filtered_df=filter.apply(df)
'''
import pandas as pd
######filtering functions#########
class BaseFilter: 
    '''
    Condition filter to apply on pd.dataframe
    for cohort extraction
    '''
    def apply(self,df:pd.DataFrame)->pd.DataFrame:
        raise NotImplementedError('Implement in daughter class.')

class PatientFilter(BaseFilter):
    '''
    Filter based only on a list of predefined patients
    '''
    def __init__(self,patient_col ,subject_ids):
        self.patient_col=patient_col
        self.patient_ids = set(subject_ids)

    def apply(self, df):
        return df[df[self.patient_col].isin(self.patient_ids)]

    
class AgeFilter(BaseFilter):
    '''
    Filter by age min and max
    '''
    def __init__(self,age_column='anchor_age',age_min=None,age_max=None):
        super().__init__()
        self.age_column=age_column
        self.age_min=age_min
        self.age_max=age_max
    def apply(self, df:pd.DataFrame):
        if self.age_min is not None:
            df=df[df[self.age_column]>=self.age_min]
        if self.age_max is not None:
            df=df[df[self.age_column]<=self.age_max]
        return df

class ICDFilter(BaseFilter):
    '''
    Filter by sex
    '''
    def __init__(self,diag_column, icd_codes):
        self.diag_column=diag_column
        self.icd_codes = set(icd_codes)

    def apply(self, df):
        return df[df[self.diag_column].isin(self.icd_codes)]

class SexFilter(BaseFilter):
    def __init__(self,sex_column,sex):
        super().__init__()
        self.sex_column=sex_column
        self.sex=sex
    def apply(self, df):
        return df[df[self.sex_column]==self.sex]
