'''
Utilities to build cohorts into a zarr database.
'''
from importlib.resources import files

from wfdb.io import dl_files
import os
import shutil
from pathlib import Path
import pandas as pd

from modules.db_processors import EHRDatabaseBaseProcessor

# relative imports
from .zarr_tools import ZarrWriter
from .utils import normalize_icd_code
from .cohort import BaseCohort, get_schema_from_config


class MIMICIVPatientCohort(BaseCohort):
    '''
    Include patients based on criteria from patient characteristics only.
    Include all hospitals admissions for the selected patients, but no other modalities (e.g. lab values) for now.

 

    '''
    def __init__(self, tmp_dir, zarr_index_name, schema={},chunk_size=64):
        super().__init__(tmp_dir, zarr_index_name, schema)
        self.chunk_size=chunk_size
        

    def download_files(self): 
        '''Download the necessary csv files to build the clinical cohort'''
        os.makedirs(self.tmp_dir,exist_ok=True)
        #sanity check, if all files already there, skip downloading
        self.files_to_dl=[f['file'] for f in self.schema.values() if isinstance(f, dict) and 'file' in f]
        full_paths = [os.path.join(self.tmp_dir, f) for f in self.files_to_dl]
        for p in full_paths:
            print("Checking:", p, "->", os.path.exists(p))
        if all(os.path.exists(p) for p in full_paths):
            print("Files already present. Skipping download.")
            return
        print("Downloading files from PhysioNet...")
        dl_files(self.db,self.tmp_dir,self.files_to_dl,keep_subdirs=True)

   
  
    
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
        # patient_cohort,patient_cohort_key=self.build_patients()
        # # 2. build the admission table for the selected patients
        # patient_ids=patient_cohort['subject_id'].unique()
        # admissions_cohort,admissions_cohort_key=self.build_admissions(patient_ids,id_col='subject_id',
        #                                                               group_key='admission',
        #                                                               use_cols=None)
        # # 3. build demographics and anchor age for the patient cohort
        # demographics_key='demographics'
        # demographics_cols=['subject_id','insurance', 'language', 'marital_status', 'race']
        # demographics=admissions_cohort[demographics_cols].drop_duplicates(subset=['subject_id'])
        # #remove the demographics from the admission columns
        # admissions_cohort=admissions_cohort.drop(columns=demographics_cols[1:]) 

        # # 4. build diagnoses table for the selected admissions if needed (e.g. if we want to filter the cohort based on diagnoses)
        # hadm_ids=admissions_cohort['hadm_id'].unique()
        # diagnoses_cohort,diagnoses_cohort_key=self.build_diagnoses(hadm_ids,id_col='hadm_id',group_key='diagnoses',use_cols=None)
        # # 5. build labevents table for the selected admissions if needed (e.g. if we want to filter the cohort based on lab values)
        # labevents_cohort,labevents_cohort_key=self.build_labevents(hadm_ids,id_col='hadm_id',group_key='labevents',use_cols=None)

        # #clear the temporary dir once the cohort has been selected
        # # self.remove_tmp_dir() #commented out for now for debugging purposes, but should be uncommented in production to avoid filling up the disk with temporary files TODO
        # return {patient_cohort_key: patient_cohort, 
        #         admissions_cohort_key: admissions_cohort, 
        #         demographics_key: demographics,
        #         diagnoses_cohort_key: diagnoses_cohort,
        #         labevents_cohort_key: labevents_cohort}
    
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


