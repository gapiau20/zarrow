'''
Utilities to build cohorts into a zarr database.
'''
from wfdb.io import dl_files
import os
import shutil
from pathlib import Path
import pandas as pd
import zarr


class ClinicalCohort:
    def __init__(self,tmp_dir,filters=[]):
        self.tmp_dir=tmp_dir
        self.filters=filters
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

class MIMICIVPatientCohort(ClinicalCohort):
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
    
    '''
    def __init__(self,db,patients_file,admission_file, tmp_dir,chunk_size=64,filters=[]): 
        #temporary dir for processing
        super().__init__(tmp_dir,filters)
        #attributes
        self.db=db
        self.patients_file=Path(patients_file)
        self.admission_file=Path(admission_file)

        #for chunks
        self.chunk_size=chunk_size

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
                if not chunk.empty:
                    patient_cohort.append(chunk)
        #otherwise just keep all patients
        else:
            for chunk in pd.read_csv(patient_file,chunksize=self.chunk_size):
                # filter the patients: TODO
                for f in self.filters:
                    chunk=f.apply(chunk)
                if not chunk.empty: 
                    patient_cohort.append(chunk)

        patient_cohort=pd.concat(patient_cohort,axis=0)



        #clear the temporary dir once the cohort has been selected
        self.remove_tmp_dir()
        return patient_cohort