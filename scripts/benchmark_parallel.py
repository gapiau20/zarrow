"""Where Zarr should win: parallel writes of disjoint, chunk-aligned blocks. Run: python scripts/bench_parallel.py"""
import time, tempfile, numpy as np, zarr, h5py
from multiprocessing import Pool
from pathlib import Path
N, L, T, B = 8192, 12, 1000, 64                      # B = chunk size along records

def block(b): return np.random.default_rng(b).normal(size=(B, L, T)).astype("float32")
def w_zarr(a): path, b = a; z = zarr.open(path, mode="r+"); z[b:b + B] = block(b)   # disjoint, chunk-aligned

if __name__ == "__main__":
    tmp, starts, rows = Path(tempfile.mkdtemp()), list(range(0, N, B)), []
    t = time.perf_counter()                           # HDF5: one writer (parallel needs an MPI build)
    with h5py.File(tmp / "a.h5", "w") as f:
        d = f.create_dataset("x", shape=(N, L, T), dtype="float32", chunks=(B, L, T), compression="gzip")
        for b in starts: d[b:b + B] = block(b)
    rows.append(("HDF5 gzip, 1 process", time.perf_counter() - t))
    t = time.perf_counter()                           # HDF5 without compression, one writer
    with h5py.File(tmp / "b.h5", "w") as f:
        d = f.create_dataset("x", shape=(N, L, T), dtype="float32", chunks=(B, L, T))
        for b in starts: d[b:b + B] = block(b)
    rows.append(("HDF5 raw, 1 process", time.perf_counter() - t))
    
    p = str(tmp / "serial.zarr"); zarr.create_array(p, shape=(N, L, T), chunks=(B, L, T), dtype="float32", overwrite=True, compressors=None)
    t = time.perf_counter()
    for b in starts: w_zarr((p, b))                     # same code, no process pool
    rows.append(("Zarr raw, serial (no pool)", time.perf_counter() - t))
    for nw, raw in [(1, False), (4, False), (1, True), (4, True)]:   # Zarr: default codec vs no compression
        p = str(tmp / f"z{nw}{raw}.zarr"); kw = {"compressors": None} if raw else {}
        zarr.create_array(p, shape=(N, L, T), chunks=(B, L, T), dtype="float32", overwrite=True, **kw)
        t = time.perf_counter()
        with Pool(nw) as pool: pool.map(w_zarr, [(p, b) for b in starts])
        rows.append((f"Zarr {'raw' if raw else 'default'}, {nw} worker(s)", time.perf_counter() - t))
        assert np.allclose(zarr.open(p, mode="r")[:B], block(0))          # sanity check
    for n, s in rows: print(f"{n:<22}{s:8.2f} s")