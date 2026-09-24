import pytest

import modules.download as download


class FakeResponse:
    def __init__(self, status_code, body=b""):
        self.status_code = status_code
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def iter_content(self, chunk_size):
        for i in range(0, len(self.body), chunk_size):
            yield self.body[i:i + chunk_size]

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def fake_get(responses, calls):
    def _get(url, **kwargs):
        calls.append((url, kwargs))
        return responses.pop(0)
    return _get


def test_download_module_imports():
    assert download is not None
    assert download.__doc__ is not None


def test_physionet_auth_from_env(monkeypatch):
    monkeypatch.setenv("PHYSIONET_USERNAME", "user")
    monkeypatch.setenv("PHYSIONET_PASSWORD", "secret")
    assert download.physionet_auth() == ("user", "secret")


def test_physionet_auth_falls_back_to_netrc(monkeypatch):
    monkeypatch.delenv("PHYSIONET_USERNAME", raising=False)
    monkeypatch.delenv("PHYSIONET_PASSWORD", raising=False)
    assert download.physionet_auth() is None  # requests then reads ~/.netrc


def test_download_physionet_file_streams_to_disk(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(download.requests, "get", fake_get([FakeResponse(200, b"a,b\n1,2\n")], calls))
    dst = download.download_physionet_file("mimiciv", "3.1", "hosp/patients.csv.gz", str(tmp_path), chunk_size=3)

    assert calls[0][0] == "https://physionet.org/files/mimiciv/3.1/hosp/patients.csv.gz"
    assert calls[0][1]["stream"] is True
    assert (tmp_path / "hosp" / "patients.csv.gz").read_bytes() == b"a,b\n1,2\n"
    assert dst == str(tmp_path / "hosp" / "patients.csv.gz")
    assert not (tmp_path / "hosp" / "patients.csv.gz.part").exists()


def test_download_physionet_file_resumes_partial_download(tmp_path, monkeypatch):
    part = tmp_path / "hosp" / "patients.csv.gz.part"
    part.parent.mkdir()
    part.write_bytes(b"a,b\n")
    calls = []
    monkeypatch.setattr(download.requests, "get", fake_get([FakeResponse(206, b"1,2\n")], calls))
    download.download_physionet_file("mimiciv", "3.1", "hosp/patients.csv.gz", str(tmp_path))

    assert calls[0][1]["headers"] == {"Range": "bytes=4-"}
    assert (tmp_path / "hosp" / "patients.csv.gz").read_bytes() == b"a,b\n1,2\n"


def test_download_physionet_file_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(download.requests, "get", fake_get([FakeResponse(403)], []))
    with pytest.raises(PermissionError):
        download.download_physionet_file("mimiciv", "3.1", "hosp/patients.csv.gz", str(tmp_path))
    assert not (tmp_path / "hosp" / "patients.csv.gz").exists()
