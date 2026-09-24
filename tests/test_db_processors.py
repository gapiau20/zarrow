import pandas as pd

from modules.db_processors import DatabaseProcessor, EHRDataFrameProcessor, register_database_processors
from modules.db_filters import SexFilter


def test_database_processor_add_filters():
    processor = DatabaseProcessor("subject_id")
    processor.add_filters([SexFilter("sex", "M")])
    assert len(processor.filters) == 1


def test_ehr_dataframe_processor_process_filters_and_columns():
    df = pd.DataFrame({"subject_id": [1, 2, 3], "sex": ["M", "F", "M"], "value": [10, 20, 30]})
    processor = EHRDataFrameProcessor("subject_id", columns=["subject_id", "value"], filters=[SexFilter("sex", "M")])
    processed, index = processor.process(df)
    assert index == "subject_id"
    assert processed["value"].tolist() == [10, 30]
    assert processed.columns.tolist() == ["subject_id", "value"]


def test_filters_default_is_not_shared_between_instances():
    EHRDataFrameProcessor("subject_id").add_filters([SexFilter("sex", "M")])
    assert EHRDataFrameProcessor("subject_id").filters == []


def test_ehr_dataframe_processor_without_columns_keeps_all():
    df = pd.DataFrame({"subject_id": [1, 2], "value": [10, 20]})
    processed, _ = EHRDataFrameProcessor("subject_id").process(df)
    assert processed.columns.tolist() == ["subject_id", "value"]


def test_register_database_processors_returns_processor_types():
    registry = register_database_processors()
    assert "DatabaseProcessor" not in registry  # abstract base is not registered
    assert "EHRDataFrameProcessor" in registry
