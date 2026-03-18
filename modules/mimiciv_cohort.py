'''
Utilities to build cohorts into a zarr database.
'''
from wfdb.io import dl_files
import os
import shutil
from pathlib import Path
import pandas as pd

from .zarr_writer import ZarrWriter


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
       
    


class MIMICIVPatientCohort(BaseCohort):
    '''
    Include patients based on criteria from patient characteristics only

    ```
    +----------------+
    |Patient Table ()|
    +----------------+
         |
         |--->criteria based on patient ID, Age, Gender, Anchor Year
         |
         v
    +----------------------+
    |Filtered patient table|
    +----------------------+    
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

    def build_cohort(self)->pd.DataFrame:
        #first download the csv_files that can serve to filter the cohort
        self.download_files()

        patient_file=os.path.join(self.tmp_dir,self.patients_file)

        #read the csv by chunks
        patient_cohort=[]
        #if there is a filtering function
        if len(self.filters)>0:
            for chunk in pd.read_csv(patient_file,chunksize=self.chunk_size):
                # filter the patients: TODO
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



        #clear the temporary dir once the cohort has been selected
        self.remove_tmp_dir()
        return patient_cohort
    
    def build_and_save(self,zarr_path:str):
        '''
        Build the cohort and save it into a zarr dataset    
        '''
        cohort=self.build_cohort()
        writer=ZarrWriter(zarr_path,self.index_id)
        #features
        writer.write_dataframe(cohort,self.group_name)
        #dataframe
        writer.write_index(cohort[self.zarr_index].to_numpy())