"""Mini benchmark Zarr vs HDF5 vs Parquet vs memmap. Run from the zarrow root: python scripts/bench_formats.py"""
import os, sys, time, shutil, tempfile, numpy as np, pandas as pd, zarr, h5py
from pathlib import Path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
rng, tmp = np.random.default_rng(0), Path(tempfile.mkdtemp())

def tm(f, n=3): s = time.perf_counter(); [f() for _ in range(n)]; return (time.perf_counter() - s) / n
def mb(p): return sum(f.stat().st_size for f in p.rglob('*') if f.is_file()) / 1e6 if p.is_dir() else p.stat().st_size / 1e6

# ---- Part 1: a real table of the cohort (labs), as loaded in mortality28d.py ----
try:
    from modules.zarr_tools import ZarrLoader
    labs = pd.DataFrame(ZarrLoader("data/cohort.zarr").load_group("labs"))
except Exception:                                   # fallback: synthetic table
    labs = pd.DataFrame({"subject_id": rng.integers(0, 5000, 2_000_000), "itemid": rng.integers(50000, 52000, 2_000_000),
                         "valuenum": rng.normal(size=2_000_000)})
num = labs.select_dtypes("number"); col = num.columns[-1]       # numeric columns only
P, Z, H = tmp / "t.parquet", tmp / "t.zarr", tmp / "t.h5"

def w_zarr():
    for c in num: z = zarr.open(str(Z / c), mode="w", shape=num[c].shape, dtype=num[c].dtype); z[:] = num[c].to_numpy()
def w_h5():
    with h5py.File(H, "w") as f:
        for c in num: f.create_dataset(c, data=num[c].to_numpy(), chunks=True, compression="gzip")
def r_zarr(cols): return {c: zarr.open(str(Z / c), mode="r")[:] for c in cols}
def r_h5(cols):
    with h5py.File(H, "r") as f: return {c: f[c][:] for c in cols}

rows = []
for name, path, w, ra, r1 in [
    ("Parquet", P, lambda: num.to_parquet(P), lambda: pd.read_parquet(P), lambda: pd.read_parquet(P, columns=[col])),
    ("Zarr", Z, w_zarr, lambda: r_zarr(num), lambda: r_zarr([col])),
    ("HDF5", H, w_h5, lambda: r_h5(num), lambda: r_h5([col]))]:
    rows.append([name, tm(w, 1), mb(path), tm(ra), tm(r1)])
print(f"\nPart 1 - labs table {num.shape}")
print(pd.DataFrame(rows, columns=["format", "write s", "size MB", "read all s", "read 1 col s"]).round(3).to_string(index=False))

# ---- Part 2: ECG-like array (N records x 12 leads x T samples), random access of single records ----
N, L, T = 2000, 12, 1000
ecg = rng.normal(size=(N, L, T)).astype("float32"); idx = rng.integers(0, N, 500)
np.save(tmp / "e.npy", ecg)
ze = zarr.open(str(tmp / "e.zarr"), mode="w", shape=ecg.shape, chunks=(1, L, T), dtype="float32"); ze[:] = ecg
with h5py.File(tmp / "e.h5", "w") as f: f.create_dataset("x", data=ecg, chunks=(1, L, T))
arrs = {"memmap": ("e.npy", np.load(tmp / "e.npy", mmap_mode="r")), "Zarr": ("e.zarr", zarr.open(str(tmp / "e.zarr"), mode="r")),
        "HDF5": ("e.h5", h5py.File(tmp / "e.h5", "r")["x"])}
rows = [[n, mb(tmp / f), tm(lambda: [np.array(a[i]) for i in idx])] for n, (f, a) in arrs.items()]
print(f"\nPart 2 - ECG array {ecg.shape}, 500 random records (warm cache)")
print(pd.DataFrame(rows, columns=["format", "size MB", "500 reads s"]).round(3).to_string(index=False))

# ---- Part 3: same array, batch access (what a DataLoader can do): 8 contiguous batches of 64 records ----
B = 64
starts = range(0, 512, B)
z3 = zarr.open(str(tmp / "e3.zarr"), mode="w", shape=ecg.shape, chunks=(B, L, T), dtype="float32"); z3[:] = ecg
with h5py.File(tmp / "e3.h5", "w") as f: f.create_dataset("x", data=ecg, chunks=(B, L, T))
arrs3 = {"memmap": ("e.npy", arrs["memmap"][1]), "Zarr (chunks=64)": ("e3.zarr", z3),
         "HDF5 (chunks=64)": ("e3.h5", h5py.File(tmp / "e3.h5", "r")["x"])}
rows = []
for n, (f, a) in arrs3.items():
    t = tm(lambda: [np.array(a[s:s + B]) for s in starts]); rows.append([n, mb(tmp / f), t, 512 / t])
print(f"\nPart 3 - same array, 8 contiguous batches of {B} records (512 records)")
print(pd.DataFrame(rows, columns=["format", "size MB", "8 batches s", "records/s"]).round(3).to_string(index=False))

shutil.rmtree(tmp, ignore_errors=True)
