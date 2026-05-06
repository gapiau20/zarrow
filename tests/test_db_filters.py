import pandas as pd

from modules.db_filters import (
    PatientFilter,
    AgeFilter,
    SexFilter,
    ICDFilter,
    EventFilter,
    LabEventFilter,
    ProcedureFilter,
    MedicationFilter,
    CharteventFilter,
    register_filters,
)


def test_patient_filter_applies_patient_list():
    df = pd.DataFrame({"subject_id": [1, 2, 3]})
    filt = PatientFilter("subject_id", [1, 3])
    result = filt.apply(df)
    assert result["subject_id"].tolist() == [1, 3]


def test_age_filter_respects_bounds():
    df = pd.DataFrame({"anchor_age": [10, 20, 30]})
    result = AgeFilter(age_min=15, age_max=25).apply(df)
    assert result["anchor_age"].tolist() == [20]


def test_sex_filter_matches_value():
    df = pd.DataFrame({"sex": ["M", "F", "M"]})
    result = SexFilter("sex", "M").apply(df)
    assert result["sex"].tolist() == ["M", "M"]


def test_icd_filter_supports_exact_and_wildcard():
    df = pd.DataFrame({"diag_code": ["A123", "A124", "B100", "A12"]})
    exact = ICDFilter("diag_code", ["A123"]).apply(df)
    assert exact["diag_code"].tolist() == ["A123"]

    wildcard = ICDFilter("diag_code", ["A12*"]).apply(df)
    assert wildcard["diag_code"].tolist() == ["A123", "A124", "A12"]


def test_event_filters_return_subset():
    df = pd.DataFrame({"event": ["x", "y", "z"]})
    expected = ["x", "y"]
    assert EventFilter("event", ["x", "y"]).apply(df)["event"].tolist() == expected
    assert LabEventFilter("event", ["x"]).apply(df)["event"].tolist() == ["x"]
    assert ProcedureFilter("event", ["y"]).apply(df)["event"].tolist() == ["y"]
    assert MedicationFilter("event", ["z"]).apply(df)["event"].tolist() == ["z"]
    assert CharteventFilter("event", ["x"]).apply(df)["event"].tolist() == ["x"]


def test_register_filters_includes_known_filter_classes():
    registered = register_filters()
    assert "PatientFilter" in registered
    assert "AgeFilter" in registered
    assert "SexFilter" in registered
    assert "ICDFilter" in registered
    assert "EventFilter" in registered
