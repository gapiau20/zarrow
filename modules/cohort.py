'''
Utilities to build cohorts into a zarr database.
'''

import pandas as pd
import zarr

def select_cohort(admissions, patients, save_path="data/cohort.zarr"):
    cohort = admissions[admissions['admission_type']=='EMERGENCY']
    cohort = cohort.merge(patients, on='subject_id', how='inner')
    cohort.to_zarr(save_path, mode='w')
    return cohort