import pandas as pd
import pytest

from modules.features import FeatureExtractor, first_in_window


def test_first_in_window_keeps_first_value_per_item_within_window():
    anchors = pd.DataFrame({"subject_id": [1, 2], "anchor_time": pd.to_datetime(["2020-01-01 12:00", "2020-01-05 08:00"])})
    events = pd.DataFrame({
        "subject_id": [1, 1, 1, 1, 1, 2, 3],
        "charttime": ["2020-01-01 10:00",   # -2 h, first glucose
                      "2020-01-01 09:00",   # -3 h but missing value: skipped
                      "2020-01-01 14:00",   # +2 h, later glucose: dropped
                      "2019-12-31 20:00",   # -16 h, outside the window
                      "2020-01-02 11:00",   # +23 h, first hemoglobin
                      "2020-01-05 07:00",   # subject 2, -1 h
                      "2020-01-01 12:00"],  # subject 3 has no anchor
        "itemid": [50931, 50931, 50931, 50811, 50811, 50931, 50931],
        "valuenum": [110.0, None, 150.0, 9.0, 13.5, 95.0, 80.0],
    })
    wide = first_in_window(events, anchors, "subject_id", "charttime", "itemid", "valuenum", (-12, 24))
    assert wide.loc[1, 50931] == 110.0
    assert wide.loc[1, 50811] == 13.5
    assert wide.loc[2, 50931] == 95.0
    assert pd.isna(wide.loc[2, 50811])
    assert 3 not in wide.index


def test_feature_extractor_behaves_as_base_class(tmp_path):
    extractor = FeatureExtractor(["subject1"], str(tmp_path / "features"))
    with pytest.raises(NotImplementedError):
        extractor.to_zarr()
    assert extractor.filter_by_event({}) is None
    assert extractor.filter_by_event_time_window({}, 30) is None
    assert extractor.filter_by_first({}) is None
