import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.improve_cohort import IMPROVETAVICohort
from modules.cohort import get_schema_from_config
schema=get_schema_from_config('config\\tavi_improve.yaml')
cohort=IMPROVETAVICohort('data\\tmp','Fallnummer',schema)
cohort.build_and_save('data\\improve')