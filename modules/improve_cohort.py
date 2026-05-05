'''Cohort for IMPROVE study'''
'''
Utilities to build cohorts into a zarr database.
'''
from importlib.resources import files

from wfdb.io import dl_files
import os
import shutil
from pathlib import Path
import pandas as pd
import os

from modules.db_processors import DatabaseProcessor

# relative imports
from .utils import normalize_icd_code
from .cohort import TabularCohort, get_schema_from_config

class IMPROVECohort(TabularCohort):
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
        if os.path.exists(file) and not os.path.exists(os.path.join(self.tmp,file)):
            print(f'Copy file {file} into {self.tmp}')
            shutil.copy(file,os.path.join(self.tmp,file))
        raise ValueError(f'{file} not found. ')

class IMPROVETAVICohort(IMPROVECohort):
    pass