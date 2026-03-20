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
#####Patient filters######
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

class SexFilter(BaseFilter):
    def __init__(self,sex_column,sex):
        super().__init__()
        self.sex_column=sex_column
        self.sex=sex
    def apply(self, df):
        return df[df[self.sex_column]==self.sex]
    
###########Diagnostic filters###########
class ICDFilter(BaseFilter):
    '''
    Filter by ICD codes in a given column
    '''
    def __init__(self,diag_column, icd_codes):
        self.diag_column=diag_column
        self.icd_codes = set( str(icd_code) for icd_code in icd_codes)

    def apply(self, df):
        mask = pd.Series(False, index=df.index)
        for code in self.icd_codes:
            if code.endswith("*"):  # wildcard
                prefix = code[:-1]   # enlever le '*'
                mask |= df[self.diag_column].astype(str).str.startswith(prefix)
            else:
                mask |= df[self.diag_column].astype(str) == code
        return df[mask]

##########EVent filters###########
class EventFilter(BaseFilter):
    '''
    Filter by event in a given column
    '''
    def __init__(self,event_column, events):
        self.event_column=event_column
        self.events = set(events)

    def apply(self, df):
        return df[df[self.event_column].isin(self.events)]
    
class LabEventFilter(EventFilter):
    '''
    Filter by lab events in a given column
    '''
    def __init__(self,lab_event_column, lab_events):
        super().__init__(lab_event_column, lab_events)

class ProcedureFilter(EventFilter):
    '''
    Filter by procedure events in a given column
    '''
    def __init__(self,procedure_event_column, procedure_events):
        super().__init__(procedure_event_column, procedure_events)
class MedicationFilter(EventFilter):
    '''
    Filter by medication events in a given column
    '''
    def __init__(self,medication_event_column, medication_events):
        super().__init__(medication_event_column, medication_events)
        
class CharteventFilter(EventFilter):
    '''
    Filter by chartevents in a given column
    '''
    def __init__(self,chartevent_column, chartevents):
        super().__init__(chartevent_column, chartevents)