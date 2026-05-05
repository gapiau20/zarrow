import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.cohort import get_schema_from_tabular_file, build_default_schema_from_tabular_file
schema=get_schema_from_tabular_file('data\\improve\\output_AK.xlsx')
print(schema)
build_default_schema_from_tabular_file('data\\improve\\output_AK.xlsx','config\\tavi_improve.yaml')
