from __future__ import annotations

import argparse
import statistics
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import zarr
from zarr.codecs import BloscCodec, BytesCodec, ShardingCodec
from zarr.storage import LocalStore

from zarr_daos import DAOSStore

import os
import zarr.storage._local as _zlocal

from zarr_daos.libdaos_store import DAOSArrayStore

def _pil4dfs_safe_delete_dir(path, prefix=""):
    if not os.path.exists(path):
        return
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            os.unlink(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))
    os.rmdir(path)

_zlocal._delete_dir = _pil4dfs_safe_delete_dir

@dataclass
class BenchResult:
    name: str
    total_mib: float
    write_s: float
    read_s: float
    extra: str = ""

    @property
    def write_mib_s(self) -> float:
        return self.total_mib / self.write_s if self.write_s > 0 else 0.0

    @property
    def read_mib_s(self) -> float:
        return self.total_mib / self.read_s if self.read_s > 0 else 0.0


def _time_fn(fn, repeats: int = 3) -> float:
    """Run callable `repeats` times and return the median wall-clock duration in seconds."""
    times: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)

def bench_chunk_sweep(
    pool: str,
    cont: str,
    concurrency: int,
    dfuse_dir: Path | None = None,
    posix_cont: str | None = None,
) -> list[BenchResult]:
    """Benchmark 1: Sweep chunk sizes (64 KiB, 1 MiB, 8 MiB) on a 256 MiB float32 array."""
    zarr.config.set({"async.concurrency": concurrency})
    results: list[BenchResult] = []

    if isinstance(dfuse_dir, str):
        dfuse_dir = Path(dfuse_dir)

    # 256 MiB float32 array: (64, 1024, 1024)
    shape = (64, 1024, 1024)
    data = np.random.default_rng(42).standard_normal(shape, dtype=np.float32)
    total_mib = data.nbytes / (1024**2)

    chunk_configs = [
        ("64 KiB chunks (4096 objs)", (1, 128, 128)),
        ("1 MiB chunks (256 objs)", (4, 256, 256)),
        ("8 MiB chunks (32 objs)", (8, 512, 512)),
    ]

    for label, chunks in chunk_configs:
        print(f"  [1/3] Chunk sweep: {label}...", flush=True)
        store = DAOSStore(pool=pool, cont=cont, prefix=f"bench_sweep_{uuid.uuid4().hex}")

        def _write() -> None:
            arr = zarr.create_array(
                store=store,
                shape=shape,
                chunks=chunks,
                dtype="float32",
                compressors=[BloscCodec(cname="lz4", clevel=1, shuffle="bitshuffle")],
                overwrite=True,
            )
            arr[:] = data

        def _read() -> None:
            arr = zarr.open_array(store=store, mode="r")
            _ = arr[:]

        print("    -> starting write...", flush=True)
        w_time = _time_fn(_write, repeats=2)
        print(f"    -> write done ({w_time:.2f}s), starting read...", flush=True)
        r_time = _time_fn(_read, repeats=3)
        results.append(BenchResult(f"DAOSStore | {label}", total_mib, w_time, r_time))

    if posix_cont is not None:
        for label, chunks in chunk_configs:
             print(f"  [1/3] Chunk sweep (libdaos Array): {label}...", flush=True)
             array_store = DAOSArrayStore(
                 pool=pool,
                 cont=posix_cont,
                 prefix=f"bench_arr_{uuid.uuid4().hex}",
             )
             try:
                 def _write_arr() -> None:
                     arr = zarr.create_array(
                         store=array_store,
                         shape=shape,
                         chunks=chunks,
                         dtype="float32",
                         compressors=[BloscCodec(cname="lz4", clevel=1, shuffle="bitshuffle")],
                         overwrite=True,
                     )
                     arr[:] = data

                 def _read_arr() -> None:
                     arr = zarr.open_array(store=array_store, mode="r")
                     _ = arr[:]

                 print("    -> starting write...", flush=True)
                 w_time = _time_fn(_write_arr, repeats=2)
                 print(f"    -> write done ({w_time:.2f}s), starting read...", flush=True)
                 r_time = _time_fn(_read_arr, repeats=3)
                 results.append(BenchResult(f"DAOSArrayStore | {label}", total_mib, w_time, r_time))
             finally:
                 array_store.close()

    # Optional baseline against dfuse LocalStore at 1 MiB chunks
    if dfuse_dir is not None:
        local_path = dfuse_dir / f"bench_dfuse_{uuid.uuid4().hex}"
        local_store = LocalStore(local_path)

        def _write_dfuse() -> None:
            arr = zarr.create_array(
                store=local_store,
                shape=shape,
                chunks=(4, 256, 256),
                dtype="float32",
                compressors=[BloscCodec(cname="lz4", clevel=1, shuffle="bitshuffle")],
                overwrite=True,
            )
            arr[:] = data

        def _read_dfuse() -> None:
            arr = zarr.open_array(store=local_store, mode="r")
            _ = arr[:]

        w_time = _time_fn(_write_dfuse, repeats=2)
        r_time = _time_fn(_read_dfuse, repeats=3)
        results.append(BenchResult("dfuse LocalStore | 1 MiB chunks", total_mib, w_time, r_time))

    return results

def _make_backends(pool: str, cont: str, tag: str, dfuse_path: Path | None):
    out = [("DAOSStore", DAOSStore(pool=pool, cont=cont, prefix=f"{tag}_{uuid.uuid4().hex}"))]
    if dfuse_path is not None:
        out.append(("DFuse", zarr.storage.LocalStore(dfuse_path / f"{tag}_{uuid.uuid4().hex}")))
    return out


def bench_sharding_and_subvolume(pool: str, cont: str, concurrency: int, dfuse_path: Path | None = None) -> list[BenchResult]:
    zarr.config.set({"async.concurrency": concurrency})
    shape = (128, 1024, 1024)
    data = np.random.default_rng(7).standard_normal(shape, dtype=np.float32)
    total_mib = data.nbytes / (1024**2)
    results = []
    for label, store in _make_backends(pool, cont, "bench_shard", dfuse_path):
        def _write(st=store):
            arr = zarr.create_array(store=st, shape=shape, shards=(64, 256, 256), chunks=(16, 64, 64),
                                    dtype="float32", compressors=[BloscCodec(cname="zstd", clevel=3, shuffle="bitshuffle")], overwrite=True)
            arr[:] = data
        print(f"  [2/3] {label} | Sharded (16 MiB / 256 KiB)...", flush=True)
        w = _time_fn(_write, repeats=2)
        r = _time_fn(lambda st=store: zarr.open_array(store=st, mode="r")[:], repeats=3)
        arr = zarr.open_array(store=store, mode="r")
        roi_ms = _time_fn(lambda: arr[32:64, 200:456, 200:456], repeats=5) * 1000.0
        results.append(BenchResult(f"{label} | Sharded (16 MiB / 256 KiB)", total_mib, w, r, f"ROI slice latency: {roi_ms:.1f} ms"))
    return results


def bench_domain_access_patterns(pool: str, cont: str, concurrency: int, dfuse_path: Path | None = None) -> list[BenchResult]:
    zarr.config.set({"async.concurrency": concurrency})
    results: list[BenchResult] = []
    cube = np.random.default_rng(11).standard_normal((512, 512, 512), dtype=np.float32)
    for label, st in _make_backends(pool, cont, "bench_radio", dfuse_path):
        print(f"  [3/3] {label} | Radio Cube...", flush=True)
        w = _time_fn(lambda st=st: zarr.create_array(store=st, shape=(512, 512, 512), chunks=(64, 256, 256),
                     dtype="float32", compressors=[BloscCodec(cname="zstd", clevel=3, shuffle="bitshuffle")], overwrite=True).__setitem__(slice(None), cube), repeats=1)
        arr = zarr.open_array(store=st, mode="r")
        ch = _time_fn(lambda: arr[256, :, :], repeats=5) * 1000.0
        sp = _time_fn(lambda: arr[:, 256, 256], repeats=5) * 1000.0
        results.append(BenchResult(f"{label} | Radio Cube (512^3, 16 MiB)", cube.nbytes / (1024**2), w, ch / 1000.0, f"2D channel: {ch:.1f} ms | 1D spectrum: {sp:.1f} ms"))

    mri = np.random.default_rng(19).standard_normal((256, 256, 256), dtype=np.float32)
    for label, st in _make_backends(pool, cont, "bench_mri", dfuse_path):
        print(f"  [3/3] {label} | 3D MRI...", flush=True)
        w = _time_fn(lambda st=st: zarr.create_array(store=st, shape=(256, 256, 256), chunks=(64, 64, 64),
                     dtype="float32", compressors=[BloscCodec(cname="zstd", clevel=3, shuffle="shuffle")], overwrite=True).__setitem__(slice(None), mri), repeats=1)
        arr = zarr.open_array(store=st, mode="r")
        ax = _time_fn(lambda: arr[128, :, :], repeats=5) * 1000.0
        co = _time_fn(lambda: arr[:, 128, :], repeats=5) * 1000.0
        sa = _time_fn(lambda: arr[:, :, 128], repeats=5) * 1000.0
        results.append(BenchResult(f"{label} | 3D MRI (256^3, 64^3)", mri.nbytes / (1024**2), w, ax / 1000.0, f"Axial: {ax:.1f} ms | Coronal: {co:.1f} ms | Sagittal: {sa:.1f} ms"))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark DAOSStore on a live DAOS cluster")
    parser.add_argument("--pool", required=True, help="DAOS pool label or UUID")
    parser.add_argument("--cont", required=True, help="DAOS container label or UUID (--type=PYTHON)")
    parser.add_argument("--concurrency", type=int, default=64, help="Zarr async.concurrency setting")
    parser.add_argument("--dfuse-path", type=Path, default=None, help="Optional dfuse mount path for comparison")
    parser.add_argument("--posix-cont", default=None, help="POSIX container label for DAOSArrayStore")
    args = parser.parse_args()

    print(f"\nRunning DAOS-Zarr Benchmarks | Pool: {args.pool} | Cont: {args.cont} | async.concurrency={args.concurrency}\n")

    all_results: list[BenchResult] = []
    all_results.extend(bench_chunk_sweep(args.pool, args.cont, args.concurrency, args.dfuse_path, posix_cont=args.posix_cont))
    all_results.extend(bench_sharding_and_subvolume(args.pool, args.cont, args.concurrency, args.dfuse_path))
    all_results.extend(bench_domain_access_patterns(args.pool, args.cont, args.concurrency, args.dfuse_path))

    header = f"{'Benchmark':<42} | {'Size':>8} | {'Write (MiB/s)':>13} | {'Read (MiB/s)':>12} | Notes"
    print(header)
    print("-" * len(header) + "-" * 30)
    for r in all_results:
        print(
            f"{r.name:<42} | {r.total_mib:6.1f} M | {r.write_mib_s:13.1f} | {r.read_mib_s:12.1f} | {r.extra}"
        )


if __name__ == "__main__":
    main()
