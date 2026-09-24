import pandas as pd

from modules.utils import find_icd_version, normalize_icd_code


def test_find_icd_version_numeric():
    assert find_icd_version("123") == 9
    assert find_icd_version("00123") == 9


def test_find_icd_version_alpha():
    assert find_icd_version("E123") == 10
    assert find_icd_version("a123") == 10


def test_normalize_icd_code_icd9():
    assert normalize_icd_code("12345") == "123.45"
    assert normalize_icd_code(" 12345 ") == "123.45"


def test_normalize_icd_code_icd10():
    assert normalize_icd_code("E123") == "E123"


def test_find_icd_version_empty_code():
    import pytest
    with pytest.raises(ValueError):
        find_icd_version("  ")
