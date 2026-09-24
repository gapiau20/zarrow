'''
Utilities to build cohorts into a zarr database.
'''
from wfdb.io import dl_files
import os

# relative imports
from .cohort import TabularCohort, get_schema_from_config  # get_schema_from_config re-exported for scripts


class PhysioNetPatientCohort(TabularCohort):
    '''
    Include patients based on criteria from patient characteristics only.
    Include all hospitals admissions for the selected patients, but no other modalities (e.g. lab values) for now.
    ''' 

    def download_file(self,file): 
        '''Download the necessary csv files to build the clinical cohort'''
        os.makedirs(self.tmp_dir,exist_ok=True)
        #skip downloading if the file is already where process_chunks will read it
        if os.path.exists(os.path.join(self.tmp_dir, file)):
            print(f"{file} already present in {self.tmp_dir}. Skipping download.")
            return
        print("Downloading files from PhysioNet...")
        dl_files(self.db,self.tmp_dir,[file],keep_subdirs=True)

class MIMICPatientCohort(PhysioNetPatientCohort):
    pass
    
    

    
    


