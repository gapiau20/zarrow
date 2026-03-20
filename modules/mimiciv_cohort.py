'''
Utilities to build cohorts into a zarr database.
'''
from wfdb.io import dl_files
import os
import shutil
from pathlib import Path
import pandas as pd

# relative imports
from .zarr_tools import ZarrWriter
from .utils import normalize_icd_code


class BaseCohort:
    def __init__(self,tmp_dir,zarr_index,filters={}):
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
        self.filters=filters
        self.zarr_index=zarr_index
    def remove_tmp_dir(self):
        '''
        Clean temporary dir that was used to build the cohort
        '''
        shutil.rmtree(self.tmp_dir)
    
    def apply_filters(self,df:pd.DataFrame,filters:list)->pd.DataFrame:
        '''
        Apply a list of filters to a dataframe
        filters: list of filter objects to apply on the dataframe
        '''
        for f in filters:
            df=f.apply(df)
        return df


    def build_cohort(self,filters:dict)->pd.DataFrame:
        '''
        Generate the patient cohort for which to extract data then
        '''
        raise NotImplementedError('Implement in a subclass')
       
    def build_patients(self,filters:dict):
        '''
        Build the patient table for the cohort, with demographic information and anchor age
        '''
        raise NotImplementedError('Implement in a subclass')
    def build_admissions(self,filters:dict):
        '''
        Build the admission table for the cohort
        '''
        raise NotImplementedError('Implement in a subclass')



class MIMICIVPatientCohort(BaseCohort):
    '''
    Include patients based on criteria from patient characteristics only.
    Include all hospitals admissions for the selected patients, but no other modalities (e.g. lab values) for now.

    ```
    +----------------+
    |Patient Table ()|
    +----------------+
         |
         |--->criteria based on patient ID, Age, Gender, Anchor Year
         |
         v
  
    ```

    ```
    cohort.zarr/
    │
    |-- patient/
    │     |-- subject_id      (N,)
    │     |-- anchor_age      (N,)
    |     |--anchor_year     (N,)
    |     |-- anchor_year_group(N,)
    |     |-- dod              (N,)
    │     |-- gender          (N,)
    |-- demographics/
    │     |-- subject_id      (N,)
    │     |-- insurance       (N,)
    │     |-- language        (N,)
    │     |-- marital_status  (N,)
    │     |-- race             (N,)
    │
    |-- admission/
    │     |-- hadm_id         (N,)
    │     |-- subject_id      (N,)
    │     |-- admittime       (N,)
    │
    +-- index/
        |--subject_id      (N,)
    ```
    
    '''
    def __init__(self,db,
                 patients_file,
                 admission_file,
                 diagnoses_file, 
                 lab_file,
                 tmp_dir,zarr_index,chunk_size=64,filters={},group_name='patient',index_id='index/subject_id'): 
        #temporary dir for processing
        super().__init__(tmp_dir,zarr_index,filters)
        #attributes
        self.db=db
        self.patients_file=Path(patients_file)
        self.admission_file=Path(admission_file)
        self.diagnoses_file=Path(diagnoses_file)
        self.lab_file=Path(lab_file)
        #for chunks
        self.chunk_size=chunk_size
        #for zarr saving
        self.group_name=group_name
        self.index_id=index_id

    def download_files(self): 
        '''Download the necessary csv files to build the clinical cohort'''
        os.makedirs(self.tmp_dir,exist_ok=True)
        dl_files(self.db,self.tmp_dir,[self.patients_file,
                                       self.admission_file,
                                       self.diagnoses_file,
                                       self.lab_file],keep_subdirs=True)

    def build_patients(self):
        group_key='patient'
        patient_file = os.path.join(self.tmp_dir, self.patients_file)

        # 2. patient table
        patient_cohort = []
        patient_filters=self.filters.get(group_key,[])
        if len(patient_filters) > 0:
            for chunk in pd.read_csv(patient_file, chunksize=self.chunk_size):
                chunk=self.apply_filters(chunk,patient_filters)
                if not chunk.empty:
                    patient_cohort.append(chunk)
        #otherwise just keep all patients
        else:
            for chunk in pd.read_csv(patient_file,chunksize=self.chunk_size):
                if not chunk.empty: 
                    patient_cohort.append(chunk)
        patient_cohort=pd.concat(patient_cohort,axis=0)
        return patient_cohort,group_key
    
    def _build_from_csv(self,file_path, id_col, ids, group_key, use_cols=None,extra_processing=None):
        '''
        Generic function to build a cohort from a csv file based on a list of ids to keep and filters to apply.
        file_path: path to the csv file
        id_col: name of the column containing the ids to filter on (e.g. subject_id, hadm_id, etc.)
        ids: list of ids to keep
        group_key: key in the filters dict corresponding to the group (e.g. 'patient', 'admission', etc.)
        use_cols: list of columns to read from the csv file (if None, read all columns)
        extra_processing: function to apply on the dataframe after filtering (e.g. for additional processing such as normalizing icd codes, etc.)
        '''
        cohort = []
        filters=self.filters.get(group_key,[])
        for chunk in pd.read_csv(file_path,chunksize=self.chunk_size,usecols=use_cols):
            chunk=chunk[chunk[id_col].isin(ids)]
            if len(filters)>0:
                chunk=self.apply_filters(chunk,filters)
            if extra_processing is not None:
                chunk=extra_processing(chunk)
            if not chunk.empty:
                cohort.append(chunk)
        cohort=pd.concat(cohort,axis=0)
        return cohort, group_key
    def build_admissions(self, patient_ids,id_col='subject_id',group_key='admission',use_cols=None):
        return self._build_from_csv(os.path.join(self.tmp_dir, self.admission_file),
                                   id_col=id_col,
                                   ids=patient_ids,
                                   group_key=group_key,
                                   use_cols=use_cols)
    
    def build_diagnoses(self, hadm_ids ,id_col='hadm_id',group_key='diagnoses',use_cols=None): 
        def _normalize_icd_code(df):
            df['icd_code']=df['icd_code'].apply(normalize_icd_code)
            return df
        extra_processing=_normalize_icd_code
        return self._build_from_csv(os.path.join(self.tmp_dir, self.diagnoses_file),
                                   id_col=id_col,
                                   ids=hadm_ids,
                                   group_key=group_key,
                                   use_cols=use_cols,
                                   extra_processing=extra_processing)
    def build_labevents(self, hadm_ids ,id_col='hadm_id',group_key='labevents',use_cols=None):
        def max_value(uom):
            if uom in ['mg/dL', 'mg/dl']:
                return 30
            elif uom in ['µmol/L', 'umol/L']:
                return 2655
            else:
                return 1e6  # default max value for unknown units
   
        def clean_labevents(df):
            # nettoyage des valeurs aberrantes
            df = df[pd.to_numeric(df['valuenum'], errors='coerce').notnull()]
            df = df[df['valuenum'] >= 0]

            # filtrage max selon unité (ex: mg/dL ou µmol/L pour creatinine)
            
            df = df[df['valuenum'] <= df['valueuom'].apply(max_value)]
            return df
    
        return self._build_from_csv(os.path.join(self.tmp_dir, self.lab_file),
                                   id_col=id_col,
                                   ids=hadm_ids,
                                   group_key=group_key,
                                   use_cols=use_cols,
                                   extra_processing=clean_labevents)   
    def build_chartevents(self,hadm_ids):
        #TODO: add chartevents for the selected admissions (e.g. vital signs, etc.)
        pass
    def build_procedures(self,hadm_ids):
        #TODO: add procedure events (e.g. ventilation, vasopressors, etc.)
        pass
    def build_icustays(self,hadm_ids):
        #TODO: add icu stays for the selected admissions
        pass
    ################full cohort building and saving into zarr################

    def build_cohort(self)->pd.DataFrame:
        #first download the csv_files that can serve to filter the cohort
        self.download_files()
        # 1.  build the patient table with demographic information and anchor age
        patient_cohort,patient_cohort_key=self.build_patients()
        # 2. build the admission table for the selected patients
        patient_ids=patient_cohort['subject_id'].unique()
        admissions_cohort,admissions_cohort_key=self.build_admissions(patient_ids,id_col='subject_id',
                                                                      group_key='admission',
                                                                      use_cols=None)
        # 3. build demographics and anchor age for the patient cohort
        demographics_key='demographics'
        demographics_cols=['subject_id','insurance', 'language', 'marital_status', 'race']
        demographics=admissions_cohort[demographics_cols].drop_duplicates(subset=['subject_id'])
        #remove the demographics from the admission columns
        admissions_cohort=admissions_cohort.drop(columns=demographics_cols[1:]) 

        # 4. build diagnoses table for the selected admissions if needed (e.g. if we want to filter the cohort based on diagnoses)
        hadm_ids=admissions_cohort['hadm_id'].unique()
        diagnoses_cohort,diagnoses_cohort_key=self.build_diagnoses(hadm_ids,id_col='hadm_id',group_key='diagnoses',use_cols=None)
        # 5. build labevents table for the selected admissions if needed (e.g. if we want to filter the cohort based on lab values)
        labevents_cohort,labevents_cohort_key=self.build_labevents(hadm_ids,id_col='hadm_id',group_key='labevents',use_cols=None)
        #clear the temporary dir once the cohort has been selected
        self.remove_tmp_dir()
        return {patient_cohort_key: patient_cohort, 
                admissions_cohort_key: admissions_cohort, 
                demographics_key: demographics,
                diagnoses_cohort_key: diagnoses_cohort,
                labevents_cohort_key: labevents_cohort}
    
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
        writer.write_index(cohort['patient'][self.zarr_index].to_numpy())