'''Generic utility functions for the project.'''
####utils
def find_icd_version(code:str)->int:
    """
    Guess the ICD version (9 or 10) of a code from its format.
    Ambiguous for ICD-9 E/V codes: prefer the dataset's icd_version column when available.
    """
    code = str(code).strip().upper()
    if not code:
        raise ValueError("Empty ICD code.")
    return 9 if code[0].isdigit() else 10
def normalize_icd_code(code):
    '''
    Normalize ICD codes to a standard format (e.g. remove dots, leading zeros, etc.)
    '''
    version=find_icd_version(code)
    code = str(code).strip()
    if version == 9 and code.isdigit() and len(code) > 3:
        code = code[:3] + '.' + code[3:]
    return code