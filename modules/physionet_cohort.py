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
from .cohort import TabularCohort, get_schema_from_config


class PhysioNetPatientCohort(TabularCohort):
    '''
    Include patients based on criteria from patient characteristics only.
    Include all hospitals admissions for the selected patients, but no other modalities (e.g. lab values) for now.

 

    ''' 

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

class MIMICPatientCohort(PhysioNetPatientCohort):
    pass
    
    

    
    


