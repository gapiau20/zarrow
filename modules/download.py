'''
Utils to download the files from the physionet databases.

Credentialed projects (e.g. mimiciv) need a PhysioNet account that signed the data use agreement.
Credentials are read, in order, from:
    - the PHYSIONET_USERNAME / PHYSIONET_PASSWORD environment variables
    - the user's ~/.netrc (Windows: %USERPROFILE%\\_netrc), entry "machine physionet.org"
      (picked up automatically by requests when no explicit auth is given)
'''
import os

import requests

PN_FILES_URL = "https://physionet.org/files/"

def physionet_auth():
    '''Credentials from PHYSIONET_USERNAME/PHYSIONET_PASSWORD, else None (requests then falls back to ~/.netrc).'''
    user, pwd = os.environ.get("PHYSIONET_USERNAME"), os.environ.get("PHYSIONET_PASSWORD")
    return (user, pwd) if user and pwd else None

def download_physionet_file(db:str, version:str, file:str, dl_dir:str, chunk_size:int=1<<20, timeout:int=60)->str:
    '''
    Stream one file of a (possibly credentialed) PhysioNet project into dl_dir/file.
    db: project slug (e.g. mimiciv, mimic-iv-demo)
    version: project version (e.g. 3.1)
    file: path relative to the project root (e.g. hosp/patients.csv.gz)
    The file is written to dl_dir/file.part and renamed once complete,
    so an interrupted download is resumed (HTTP Range) and never mistaken for a complete file.
    '''
    url = f"{PN_FILES_URL}{db}/{version}/{file}"
    dst = os.path.normpath(os.path.join(dl_dir, file))
    part = dst + ".part"
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)

    done = os.path.getsize(part) if os.path.exists(part) else 0
    headers = {"Range": f"bytes={done}-"} if done else {}
    with requests.get(url, auth=physionet_auth(), headers=headers, stream=True, timeout=timeout) as r:
        if r.status_code in (401, 403):
            raise PermissionError(
                f"PhysioNet refused {url} (HTTP {r.status_code}): check your credentials "
                "(PHYSIONET_USERNAME/PHYSIONET_PASSWORD or ~/.netrc) and that you signed the project's data use agreement.")
        if r.status_code != 416:  # 416: .part already holds the whole file
            r.raise_for_status()
            # 206: server honoured the Range header, append; 200: full content, restart
            with open(part, "ab" if r.status_code == 206 else "wb") as f:
                for block in r.iter_content(chunk_size):
                    f.write(block)
    os.replace(part, dst)
    return dst
