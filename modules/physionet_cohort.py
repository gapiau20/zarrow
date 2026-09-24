'''
Utilities to build cohorts into a zarr database.
'''
import os

# relative imports
from .cohort import TabularCohort, get_schema_from_config  # get_schema_from_config re-exported for scripts
from .download import download_physionet_file


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
        version=self.schema.get('version')
        if version is None:
            raise ValueError(f"Missing 'version' for PhysioNet project {self.db} in the config (e.g. version: '3.1').")
        print(f"Downloading {file} from PhysioNet ({self.db}/{version})...")
        download_physionet_file(self.db, str(version), file, self.tmp_dir)

class MIMICPatientCohort(PhysioNetPatientCohort):
    pass
    
    

    
    


