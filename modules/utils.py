'''Generic utility functions for the project.'''
####utils
def find_icd_version(code:str)->str:
    """
    Find the ICD version (9 or 10) of a given code based on its format.
    """
    code = str(code).strip().upper()
    if code[0].isdigit():  # commence par un chiffre
        return 9
    elif code[0].isalpha():  # commence par une lettre
        return 10
    else:
        return 10  # par défaut, on suppose ICD-10
def normalize_icd_code(code):
    '''
    Normalize ICD codes to a standard format (e.g. remove dots, leading zeros, etc.)
    '''
    version=find_icd_version(code)
    code = str(code).strip()
    if version == 9 and code.isdigit() and len(code) > 3:
        code = code[:3] + '.' + code[3:]
    return code