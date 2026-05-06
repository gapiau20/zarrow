import pytest

from modules.features import FeatureExtractor


def test_feature_extractor_behaves_as_base_class(tmp_path):
    extractor = FeatureExtractor(["subject1"], str(tmp_path / "features"))
    with pytest.raises(NotImplementedError):
        extractor.to_zarr()
    assert extractor.filter_by_event({}) is None
    assert extractor.filter_by_event_time_window({}, 30) is None
    assert extractor.filter_by_first({}) is None
