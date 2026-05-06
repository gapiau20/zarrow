import modules.download as download


def test_download_module_imports():
    assert download is not None
    assert download.__doc__ is not None
