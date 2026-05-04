'''
Utilities to build cohorts into a zarr database.
'''
from importlib.resources import files

from wfdb.io import dl_files
import os
import shutil
from pathlib import Path
import pandas as pd

from modules.db_processors import DatabaseProcessor

# relative imports
from .utils import normalize_icd_code
from .cohort import BaseCohort, get_schema_from_config


class MIMICPatientCohort(BaseCohort):
    '''
    Include patients based on criteria from patient characteristics only.
    Include all hospitals admissions for the selected patients, but no other modalities (e.g. lab values) for now.

 

    '''
    def __init__(self, tmp_dir, zarr_index_name, schema={},chunk_size=64):
        super().__init__(tmp_dir, zarr_index_name, schema)
        self.chunk_size=chunk_size
        

    def download_file(self,file): 
        '''Download the necessary csv files to build the clinical cohort'''
        os.makedirs(self.tmp_dir,exist_ok=True)
        #sanity check, if all files already there, skip downloading
        full_paths = [file]
        for p in full_paths:
            print("Checking:", p, "->", os.path.exists(p))
        if all(os.path.exists(p) for p in full_paths):
            print("Files already present. Skipping download.")
            return
        print("Downloading files from PhysioNet...")
        dl_files(self.db,self.tmp_dir,[file],keep_subdirs=True)

   
  
    
    # def _build_from_csv(self,file_path, id_col, ids, group_key, use_cols=None,extra_processing=None):
    #     '''
    #     Generic function to build a cohort from a csv file based on a list of ids to keep and filters to apply.
    #     file_path: path to the csv file
    #     id_col: name of the column containing the ids to filter on (e.g. subject_id, hadm_id, etc.)
    #     ids: list of ids to keep
    #     group_key: key in the filters dict corresponding to the group (e.g. 'patient', 'admission', etc.)
    #     use_cols: list of columns to read from the csv file (if None, read all columns)
    #     extra_processing: function to apply on the dataframe after filtering (e.g. for additional processing such as normalizing icd codes, etc.)
    #     '''
    #     cohort = []
    #     filters=self.filters.get(group_key,[])
    #     for chunk in pd.read_csv(file_path,chunksize=self.chunk_size,usecols=use_cols):
    #         chunk=chunk[chunk[id_col].isin(ids)]
    #         if len(filters)>0:
    #             chunk=self.apply_filters(chunk,filters)
    #         if extra_processing is not None:
    #             chunk=extra_processing(chunk)
    #         if not chunk.empty:
    #             cohort.append(chunk)
    #     cohort=pd.concat(cohort,axis=0)
    #     return cohort, group_key
    
    # def build_admissions(self, patient_ids,id_col='subject_id',group_key='admission',use_cols=None):
    #     return self._build_from_csv(os.path.join(self.tmp_dir, self.admission_file),
    #                                id_col=id_col,
    #                                ids=patient_ids,
    #                                group_key=group_key,
    #                                use_cols=use_cols)
    
    # def build_diagnoses(self, hadm_ids ,id_col='hadm_id',group_key='diagnoses',use_cols=None): 
    #     def _normalize_icd_code(df):
    #         df['icd_code']=df['icd_code'].apply(normalize_icd_code)
    #         return df
    #     extra_processing=_normalize_icd_code
    #     return self._build_from_csv(os.path.join(self.tmp_dir, self.diagnoses_file),
    #                                id_col=id_col,
    #                                ids=hadm_ids,
    #                                group_key=group_key,
    #                                use_cols=use_cols,
    #                                extra_processing=extra_processing)
    # def build_labevents(self, hadm_ids ,id_col='hadm_id',group_key='labevents',use_cols=None):
    #     def max_value(uom):
    #         if uom in ['mg/dL', 'mg/dl']:
    #             return 30
    #         elif uom in ['µmol/L', 'umol/L']:
    #             return 2655
    #         else:
    #             return 1e6  # default max value for unknown units
   
    #     def clean_labevents(df):
    #         # nettoyage des valeurs aberrantes
    #         df = df[pd.to_numeric(df['valuenum'], errors='coerce').notnull()]
    #         df = df[df['valuenum'] >= 0]

    #         # filtrage max selon unité (ex: mg/dL ou µmol/L pour creatinine)
            
    #         df = df[df['valuenum'] <= df['valueuom'].apply(max_value)]
    #         return df
    
    #     return self._build_from_csv(os.path.join(self.tmp_dir, self.lab_file),
    #                                id_col=id_col,
    #                                ids=hadm_ids,
    #                                group_key=group_key,
    #                                use_cols=use_cols,
    #                                extra_processing=clean_labevents)   
    # def build_chartevents(self,hadm_ids):
    #     #TODO: add chartevents for the selected admissions (e.g. vital signs, etc.)
    #     pass
    # def build_procedures(self,hadm_ids):
    #     #TODO: add procedure events (e.g. ventilation, vasopressors, etc.)
    #     pass
    # def build_icustays(self,hadm_ids):
    #     #TODO: add icu stays for the selected admissions
    #     pass
    ################full cohort building and saving into zarr################
    
    

    
    


