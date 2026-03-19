'''
Utilities to build cohorts into a zarr database.
'''
from wfdb.io import dl_files
import os
import shutil
from pathlib import Path
import pandas as pd

from .zarr_tools import ZarrWriter


class BaseCohort:
    def __init__(self,tmp_dir,zarr_index,filters=[]):
        self.tmp_dir=tmp_dir
        self.filters=filters
        self.zarr_index=zarr_index
    def remove_tmp_dir(self):
        '''
        Clean temporary dir that was used to build the cohort
        '''
        shutil.rmtree(self.tmp_dir)

    def build_cohort(self):
        '''
        Generate the patient cohort for which to extract data then
        '''
        raise NotImplementedError('Implement in a subclass')
       
    def build_patients(self):
        '''
        Build the patient table for the cohort, with demographic information and anchor age
        '''
        raise NotImplementedError('Implement in a subclass')
    def build_admissions(self):
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
    +----------------------+      +----------------+
    |Filtered patient table|      | Admission table|
    +----------------------+      +----------------+
         |                                 |
         |<--- Merge on subject_id---------+
    ```

    ```
    cohort.zarr/
    │
    ├── patient/
    │     ├── subject_id      (N,)
    │     ├── anchor_age      (N,)
    │     ├── gender          (N,)
    │
    ├── admission/
    │     ├── hadm_id         (N,)
    │     ├── subject_id      (N,)
    │     ├── admittime       (N,)
    │
    └── index/
        ├── subject_id      (N,)
    ```
    
    '''
    def __init__(self,db,patients_file,admission_file, tmp_dir,zarr_index,chunk_size=64,filters=[],group_name='patient',index_id='index/subject_id'): 
        #temporary dir for processing
        super().__init__(tmp_dir,zarr_index,filters)
        #attributes
        self.db=db
        self.patients_file=Path(patients_file)
        self.admission_file=Path(admission_file)

        #for chunks
        self.chunk_size=chunk_size
        #for zarr saving
        self.group_name=group_name
        self.index_id=index_id

    def download_files(self): 
        '''Download the necessary csv files to build the clinical cohort'''
        os.makedirs(self.tmp_dir,exist_ok=True)
        dl_files(self.db,self.tmp_dir,[self.patients_file,self.admission_file],keep_subdirs=True)

    def build_patients(self):
        patient_file = os.path.join(self.tmp_dir, self.patients_file)
        # 2. patient table
        patient_cohort = []
        if len(self.filters) > 0:
            for chunk in pd.read_csv(patient_file, chunksize=self.chunk_size):
                for f in self.filters:
                    chunk=f.apply(chunk)
                if not chunk.empty:
                    patient_cohort.append(chunk)
        #otherwise just keep all patients
        else:
            for chunk in pd.read_csv(patient_file,chunksize=self.chunk_size):
                if not chunk.empty: 
                    patient_cohort.append(chunk)
        patient_cohort=pd.concat(patient_cohort,axis=0)
        return patient_cohort
    
    def build_admissions(self,patient_ids):
        admission_file = os.path.join(self.tmp_dir, self.admission_file)
        admissions_cohort=[]
        for chunk in pd.read_csv(admission_file,chunksize=self.chunk_size):
            chunk=chunk[chunk['subject_id'].isin(patient_ids)]
            if not chunk.empty:
                admissions_cohort.append(chunk)
        admissions_cohort=pd.concat(admissions_cohort,axis=0)
        return admissions_cohort
    
    def build_cohort(self)->pd.DataFrame:
        #first download the csv_files that can serve to filter the cohort
        self.download_files()
        # 1.  build the patient table with demographic information and anchor age
        patient_cohort=self.build_patients()
        # 2. build the admission table for the selected patients
        patient_ids=patient_cohort['subject_id'].unique()
        admissions_cohort=self.build_admissions(patient_ids)
        # 3. build demographics and anchor age for the patient cohort
        demographics_cols=['subject_id','insurance', 'language', 'marital_status', 'race']
        demographics=admissions_cohort[demographics_cols].drop_duplicates(subset=['subject_id'])
        #remove the demographics from the admission columns
        admissions_cohort=admissions_cohort.drop(columns=demographics_cols[1:]) 
        #clear the temporary dir once the cohort has been selected
        self.remove_tmp_dir()
        return {'patient':patient_cohort,'admission':admissions_cohort,'demographics':demographics}
    def clean_types(self,cohort:pd.DataFrame):
        return 
    def build_and_save(self,zarr_path:str):
        '''
        Build the cohort and save it into a zarr dataset    
        '''
        #cohort build
        cohort = self.build_cohort()

        # TODO (2) : Sauvegarder les données démographiques dans le groupe patient du Zarr
        # TODO (1) : Sauvegarder la table admission dans le groupe admission du Zarr
        # TODO (3) : Sauvegarder les lab values filtrés dans le groupe lab du Zarr

        #save the cohort into a zarr dataset
        writer = ZarrWriter(zarr_path, self.zarr_index)
        for k in cohort.keys():
            writer.write_dataframe(cohort[k], k)
        writer.write_index(cohort['patient'][self.zarr_index].to_numpy())