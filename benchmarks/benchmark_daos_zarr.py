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
) -> list[BenchResult]:
    """Benchmark 1: Sweep chunk sizes (64 KiB, 1 MiB, 8 MiB) on a 256 MiB float32 array."""
    zarr.config.set({"async.concurrency": concurrency})
    results: list[BenchResult] = []

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
        store = DAOSStore(pool=pool, cont=cont, dict_name=f"bench_sweep_{uuid.uuid4().hex}")

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

        w_time = _time_fn(_write, repeats=2)
        r_time = _time_fn(_read, repeats=3)
        results.append(BenchResult(f"DAOSStore | {label}", total_mib, w_time, r_time))

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


def bench_sharding_and_subvolume(pool: str, cont: str, concurrency: int) -> BenchResult:
    """Benchmark 2: ShardingCodec (16 MiB shards, 256 KiB inner chunks) + random ROI read."""
    zarr.config.set({"async.concurrency": concurrency})
    shape = (128, 1024, 1024)  # 512 MiB float32
    data = np.random.default_rng(7).standard_normal(shape, dtype=np.float32)
    total_mib = data.nbytes / (1024**2)

    shard_codec = ShardingCodec(
        chunk_shape=(16, 64, 64),  # 256 KiB inner chunks
        codecs=[BytesCodec(), BloscCodec(cname="zstd", clevel=3, shuffle="bitshuffle")],
    )
    store = DAOSStore(pool=pool, cont=cont, dict_name=f"bench_shard_{uuid.uuid4().hex}")

    def _write() -> None:
        arr = zarr.create_array(
            store=store,
            shape=shape,
            chunks=(64, 256, 256),  # 16 MiB outer shards (32 shards total)
            dtype="float32",
            codecs=[shard_codec],
            overwrite=True,
        )
        arr[:] = data

    def _read_full() -> None:
        arr = zarr.open_array(store=store, mode="r")
        _ = arr[:]

    w_time = _time_fn(_write, repeats=2)
    r_time = _time_fn(_read_full, repeats=3)

    # Measure random sub-volume extraction latency (exercises get_partial_values + bget)
    arr = zarr.open_array(store=store, mode="r")
    roi_ms = _time_fn(lambda: arr[32:64, 200:456, 200:456], repeats=5) * 1000.0

    return BenchResult(
        name="DAOSStore | Sharded (16 MiB / 256 KiB)",
        total_mib=total_mib,
        write_s=w_time,
        read_s=r_time,
        extra=f"ROI slice latency: {roi_ms:.1f} ms",
    )


def bench_domain_access_patterns(pool: str, cont: str, concurrency: int) -> list[BenchResult]:
    """Benchmark 3: Radio spectral cube (spatial vs. spectral) & MRI orthogonal slicing."""
    zarr.config.set({"async.concurrency": concurrency})
    results: list[BenchResult] = []

    # --- A. Radio Astronomy Spectral Cube: (freq=512, dec=512, ra=512) = 512 MiB ---
    cube_shape = (512, 512, 512)
    cube_chunks = (64, 256, 256)  # 16 MiB balanced blocks
    cube_data = np.random.default_rng(11).standard_normal(cube_shape, dtype=np.float32)

    radio_store = DAOSStore(pool=pool, cont=cont, dict_name=f"bench_radio_{uuid.uuid4().hex}")
    w_radio = _time_fn(
        lambda: zarr.create_array(
            store=radio_store,
            shape=cube_shape,
            chunks=cube_chunks,
            dtype="float32",
            compressors=[BloscCodec(cname="zstd", clevel=3, shuffle="bitshuffle")],
            overwrite=True,
        ).__setitem__(slice(None), cube_data),
        repeats=1,
    )

    radio_arr = zarr.open_array(store=radio_store, mode="r")
    # 1. Single 2D frequency channel map (512x512 = 1 MiB)
    channel_ms = _time_fn(lambda: radio_arr[256, :, :], repeats=5) * 1000.0
    # 2. 1D spectrum across all 512 channels at single pixel (touches 8 chunks)
    spectrum_ms = _time_fn(lambda: radio_arr[:, 256, 256], repeats=5) * 1000.0

    results.append(
        BenchResult(
            name="Domain | Radio Cube (512^3, 16 MiB chunks)",
            total_mib=cube_data.nbytes / (1024**2),
            write_s=w_radio,
            read_s=channel_ms / 1000.0,
            extra=f"2D channel: {channel_ms:.1f} ms | 1D spectrum: {spectrum_ms:.1f} ms",
        )
    )

    # --- B. High-Res 3D MRI Volume: (z=256, y=256, x=256) = 64 MiB ---
    mri_shape = (256, 256, 256)
    mri_chunks = (64, 64, 64)  # 1 MiB isotropic cubes
    mri_data = np.random.default_rng(19).standard_normal(mri_shape, dtype=np.float32)

    mri_store = DAOSStore(pool=pool, cont=cont, dict_name=f"bench_mri_{uuid.uuid4().hex}")
    w_mri = _time_fn(
        lambda: zarr.create_array(
            store=mri_store,
            shape=mri_shape,
            chunks=mri_chunks,
            dtype="float32",
            compressors=[BloscCodec(cname="zstd", clevel=3, shuffle="shuffle")],
            overwrite=True,
        ).__setitem__(slice(None), mri_data),
        repeats=1,
    )

    mri_arr = zarr.open_array(store=mri_store, mode="r")
    axial_ms = _time_fn(lambda: mri_arr[128, :, :], repeats=5) * 1000.0
    coronal_ms = _time_fn(lambda: mri_arr[:, 128, :], repeats=5) * 1000.0
    sagittal_ms = _time_fn(lambda: mri_arr[:, :, 128], repeats=5) * 1000.0

    results.append(
        BenchResult(
            name="Domain | 3D MRI (256^3, 64^3 isotropic)",
            total_mib=mri_data.nbytes / (1024**2),
            write_s=w_mri,
            read_s=axial_ms / 1000.0,
            extra=f"Axial: {axial_ms:.1f} ms | Coronal: {coronal_ms:.1f} ms | Sagittal: {sagittal_ms:.1f} ms",
        )
    )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark DAOSStore on a live DAOS cluster")
    parser.add_argument("--pool", required=True, help="DAOS pool label or UUID")
    parser.add_argument("--cont", required=True, help="DAOS container label or UUID (--type=PYTHON)")
    parser.add_argument("--concurrency", type=int, default=64, help="Zarr async.concurrency setting")
    parser.add_argument("--dfuse-path", type=Path, default=None, help="Optional dfuse mount path for comparison")
    args = parser.parse_args()

    print(f"\nRunning DAOS-Zarr Benchmarks | Pool: {args.pool} | Cont: {args.cont} | async.concurrency={args.concurrency}\n")

    all_results: list[BenchResult] = []
    all_results.extend(bench_chunk_sweep(args.pool, args.cont, args.concurrency, args.dfuse_path))
    all_results.append(bench_sharding_and_subvolume(args.pool, args.cont, args.concurrency))
    all_results.extend(bench_domain_access_patterns(args.pool, args.cont, args.concurrency))

    header = f"{'Benchmark':<42} | {'Size':>8} | {'Write (MiB/s)':>13} | {'Read (MiB/s)':>12} | Notes"
    print(header)
    print("-" * len(header) + "-" * 30)
    for r in all_results:
        print(
            f"{r.name:<42} | {r.total_mib:6.1f} M | {r.write_mib_s:13.1f} | {r.read_mib_s:12.1f} | {r.extra}"
        )


if __name__ == "__main__":
    main()
