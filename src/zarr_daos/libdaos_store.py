from __future__ import annotations

import asyncio
import atexit
from concurrent.futures import ThreadPoolExecutor
import ctypes
import json
import math
from typing import AsyncGenerator, Iterable

import numpy as np
from zarr.abc.store import ByteRequest, OffsetByteRequest, RangeByteRequest, Store, SuffixByteRequest
from zarr.core.buffer import Buffer, BufferPrototype, default_buffer_prototype

from . import libdaos_ffi as ffi

_STRIPE_THRESHOLD = 256 * 1024  # Use OC_SX multi-target striping at >= 256 KiB
_DER_EXIST = -1004
_DER_NONEXIST = -1005

def _buf_to_mv(buf: Buffer) -> memoryview:
    try:
        arr = buf.as_array_like()
        mv = memoryview(arr)
        return mv if mv.format == "B" else mv.cast("B")
    except Exception:
        return memoryview(buf.to_bytes())

def _is_chunk_key(key: str) -> bool:
    leaf = key.rsplit("/", 1)[-1]
    if leaf in {"zarr.json", ".zarray", ".zgroup", ".zattrs", ".zmetadata"}:
        return False
    return "/c/" in f"/{key}" or leaf.replace(".", "").replace("_", "").isdigit()


def _infer_chunk_bytes_from_zarr_json(raw: bytes | memoryview) -> int | None:
    try:
        meta = json.loads(bytes(raw))
        chunk_shape = meta["chunk_grid"]["configuration"]["chunk_shape"]
        dtype_str = str(meta["data_type"])
        itemsize = (
            8
            if any(x in dtype_str for x in ("64", "complex64"))
            else (2 if "16" in dtype_str else (1 if "8" in dtype_str and "128" not in dtype_str else 4))
        )
        if "complex128" in dtype_str:
            itemsize = 16
        return math.prod(int(d) for d in chunk_shape) * itemsize
    except Exception:
        return None


def _mv_to_void_ptr(mv: memoryview) -> tuple[ctypes.c_void_p, object]:
    """Return a zero-copy void* pointer and keepalive object for any readable memoryview (even readonly)."""
    arr = np.frombuffer(mv, dtype=np.uint8)
    return arr.ctypes.data_as(ctypes.c_void_p), arr


class _GlobalDAOSContext:
    """Process-wide single-thread DAOS fabric context (prevents multi-thread OFI hangs)."""

    def __init__(self) -> None:
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="libdaos_ofi")
        self.initialized = False
        self.pool_name: str | None = None
        self.cont_name: str | None = None
        self.poh = ffi.daos_handle_t(ffi.DAOS_HDL_INVAL)
        self.coh = ffi.daos_handle_t(ffi.DAOS_HDL_INVAL)
        self.eqh = ffi.daos_handle_t(ffi.DAOS_HDL_INVAL)
        self.oc_s1: int = 0
        self.oc_sx: int = 0
        atexit.register(self._shutdown)

    def ensure_open(self, pool: str, cont: str) -> None:
        def _open() -> None:
            if not self.initialized:
                ffi.check_rc(ffi.libdaos.daos_init(), "daos_init")
                self.oc_s1 = ffi.resolve_oclass("S1")
                self.oc_sx = ffi.resolve_oclass("SX")
                ffi.check_rc(ffi.libdaos.daos_eq_create(ctypes.byref(self.eqh)), "daos_eq_create")
                self.initialized = True

            if self.pool_name == pool and self.cont_name == cont and self.coh.value != ffi.DAOS_HDL_INVAL:
                return

            if self.coh.value != ffi.DAOS_HDL_INVAL:
                ffi.libdaos.daos_cont_close(self.coh, None)
                self.coh = ffi.daos_handle_t(ffi.DAOS_HDL_INVAL)
            if self.poh.value != ffi.DAOS_HDL_INVAL and self.pool_name != pool:
                ffi.libdaos.daos_pool_disconnect(self.poh, None)
                self.poh = ffi.daos_handle_t(ffi.DAOS_HDL_INVAL)

            if self.poh.value == ffi.DAOS_HDL_INVAL:
                ffi.check_rc(
                    ffi.libdaos.daos_pool_connect(
                        pool.encode("utf-8"), None, ffi.DAOS_PC_RW, ctypes.byref(self.poh), None, None
                    ),
                    "daos_pool_connect",
                )
                self.pool_name = pool

            ffi.check_rc(
                ffi.libdaos.daos_cont_open(
                    self.poh, cont.encode("utf-8"), ffi.DAOS_COO_RW, ctypes.byref(self.coh), None, None
                ),
                "daos_cont_open",
            )
            self.cont_name = cont

        self.executor.submit(_open).result()

    def _shutdown(self) -> None:
        if not self.initialized:
            return

        def _fini() -> None:
            if self.coh.value != ffi.DAOS_HDL_INVAL:
                ffi.libdaos.daos_cont_close(self.coh, None)
            if self.poh.value != ffi.DAOS_HDL_INVAL:
                ffi.libdaos.daos_pool_disconnect(self.poh, None)
            if self.eqh.value != ffi.DAOS_HDL_INVAL:
                ffi.libdaos.daos_eq_destroy(self.eqh, 0)
            ffi.libdaos.daos_fini()

        try:
            self.executor.submit(_fini).result(timeout=5)
        except Exception:
            pass
        self.executor.shutdown(wait=False)


_CTX = _GlobalDAOSContext()


class DAOSArrayStore(Store):
    """Zero-copy Zarr v3 Store backed by libdaos (daos_kv + pipelined daos_array)."""

    supports_writes: bool = True
    supports_deletes: bool = True
    supports_partial_writes: bool = False
    supports_listing: bool = True

    def __init__(
        self,
        pool: str,
        cont: str,
        prefix: str = "",
        read_only: bool = False,
        stripe_size: int = 1048576,
    ) -> None:
        super().__init__(read_only=read_only)
        self.pool = pool
        self.cont = cont
        self.prefix = prefix.strip("/")
        self.stripe_size = stripe_size
        self._kv_oh = ffi.daos_handle_t(ffi.DAOS_HDL_INVAL)
        self._chunk_bytes_hint: int | None = None
        _CTX.ensure_open(pool, cont)
        _CTX.executor.submit(self._open_kv_sync).result()

    async def _open(self) -> None:
        await super()._open()
        if self._kv_oh.value == ffi.DAOS_HDL_INVAL:
            _CTX.ensure_open(self.pool, self.cont)
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(_CTX.executor, self._open_kv_sync)

    def _full_key(self, key: str) -> str:
        clean = key.lstrip("/")
        return f"{self.prefix}/{clean}" if self.prefix else clean

    def _open_kv_sync(self) -> None:
        o_flags = ffi.DAOS_OO_RO if self.read_only else ffi.DAOS_OO_RW
        kv_oid = ffi.make_oid(_CTX.coh, f"zarr_root_kv:{self.prefix}", ffi.DAOS_OT_KV_HASHED, _CTX.oc_s1)
        ffi.check_rc(
            ffi.libdaos.daos_kv_open(_CTX.coh, kv_oid, o_flags, ctypes.byref(self._kv_oh), None),
            "daos_kv_open",
        )

    def close(self) -> None:
        super().close()
        self._is_open = False
        if self._kv_oh.value != ffi.DAOS_HDL_INVAL:
            kv = self._kv_oh
            self._kv_oh = ffi.daos_handle_t(ffi.DAOS_HDL_INVAL)
            _CTX.executor.submit(lambda: ffi.libdaos.daos_kv_close(kv, None)).result()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DAOSArrayStore):
            return NotImplemented
        return (
            self.pool == other.pool
            and self.cont == other.cont
            and self.prefix == other.prefix
            and self.read_only == other.read_only
        )

    def _select_oclass(self, nbytes: int | None = None) -> int:
        sz = nbytes if nbytes is not None else (self._chunk_bytes_hint or 0)
        return _CTX.oc_sx if sz >= _STRIPE_THRESHOLD else _CTX.oc_s1

    def _array_oid(self, fkey: str, nbytes: int | None = None) -> ffi.daos_obj_id_t:
        return ffi.make_oid(_CTX.coh, f"zarr_chunk:{fkey}", ffi.DAOS_OT_ARRAY_BYTE, self._select_oclass(nbytes))

    # --- KV metadata helpers ---

    def _kv_get_sync(self, fkey: str) -> bytearray | None:
        b_key = fkey.encode("utf-8")
        sz = ffi.daos_size_t(ffi.DAOS_REC_ANY)
        rc = ffi.libdaos.daos_kv_get(self._kv_oh, ffi.DAOS_TX_NONE, 0, b_key, ctypes.byref(sz), None, None)
        if rc == _DER_NONEXIST:
            return None
        ffi.check_rc(rc, f"kv_get_size({fkey})")
        if sz.value == 0:
            return bytearray(0)

        out = bytearray(sz.value)
        c_buf = (ctypes.c_char * sz.value).from_buffer(out)
        ffi.check_rc(
            ffi.libdaos.daos_kv_get(self._kv_oh, ffi.DAOS_TX_NONE, 0, b_key, ctypes.byref(sz), c_buf, None),
            f"kv_get({fkey})",
        )
        return out

    def _kv_put_sync(self, fkey: str, mv: memoryview) -> None:
        if fkey.endswith("zarr.json"):
            inferred = _infer_chunk_bytes_from_zarr_json(mv)
            if inferred is not None:
                self._chunk_bytes_hint = inferred

        b_key = fkey.encode("utf-8")
        nbytes = len(mv)
        if nbytes == 0:
            ffi.check_rc(
                ffi.libdaos.daos_kv_put(self._kv_oh, ffi.DAOS_TX_NONE, 0, b_key, 0, None, None),
                f"kv_put({fkey})",
            )
            return
        ptr, keep = _mv_to_void_ptr(mv)
        ffi.check_rc(
            ffi.libdaos.daos_kv_put(self._kv_oh, ffi.DAOS_TX_NONE, 0, b_key, nbytes, ptr, None),
            f"kv_put({fkey})",
        )
        del keep

    # --- Pipelined Batch Array Write / Read ---

    def _open_or_create_array(self, fkey: str, nbytes: int) -> ffi.daos_handle_t:
        oid = self._array_oid(fkey, nbytes)
        aoh = ffi.daos_handle_t()
        rc = ffi.libdaos.daos_array_create(
            _CTX.coh, oid, ffi.DAOS_TX_NONE, 1, self.stripe_size, ctypes.byref(aoh), None
        )
        if rc == _DER_EXIST:
            cell_sz, chunk_sz = ffi.daos_size_t(), ffi.daos_size_t()
            rc = ffi.libdaos.daos_array_open(
                _CTX.coh, oid, ffi.DAOS_TX_NONE, ffi.DAOS_OO_RW,
                ctypes.byref(cell_sz), ctypes.byref(chunk_sz), ctypes.byref(aoh), None,
            )
            ffi.check_rc(rc, f"array_open_rw({fkey})")
            # Truncate existing array to exact target size before writing
            ffi.check_rc(
                ffi.libdaos.daos_array_set_size(aoh, ffi.DAOS_TX_NONE, nbytes, None),
                f"array_set_size({fkey})",
            )
        else:
            ffi.check_rc(rc, f"array_create({fkey})")
        return aoh

    def _write_arrays_batch_sync(self, chunk_items: list[tuple[str, memoryview]]) -> None:
        if not chunk_items:
            return

        non_empty: list[tuple[str, memoryview, ffi.daos_handle_t]] = []
        for fkey, mv in chunk_items:
            nbytes = len(mv)
            aoh = self._open_or_create_array(fkey, nbytes)
            if nbytes == 0:
                # 0-byte array is already created with size 0; never post a 0-len async I/O
                ffi.libdaos.daos_array_close(aoh, None)
            else:
                non_empty.append((fkey, mv, aoh))

        m = len(non_empty)
        if m > 0:
            events = (ffi.daos_event_t * m)()
            keepalive: list[object] = []

            for i, (fkey, mv, aoh) in enumerate(non_empty):
                nbytes = len(mv)
                ffi.check_rc(ffi.libdaos.daos_event_init(ctypes.byref(events[i]), _CTX.eqh, None), "event_init")
                ptr, arr_keep = _mv_to_void_ptr(mv)
                iov = ffi.d_iov_t(iov_buf=ptr, iov_buf_len=nbytes, iov_len=nbytes)
                sgl = ffi.d_sg_list_t(sg_nr=1, sg_nr_out=0, sg_iovs=ctypes.pointer(iov))
                rg = ffi.daos_range_t(rg_len=nbytes, rg_idx=0)
                iod = ffi.daos_array_iod_t(arr_nr=1, arr_rgs=ctypes.pointer(rg), arr_nr_short_read=0)
                keepalive.extend((arr_keep, iov, sgl, rg, iod))

                ffi.check_rc(
                    ffi.libdaos.daos_array_write(
                        aoh, ffi.DAOS_TX_NONE, ctypes.byref(iod), ctypes.byref(sgl), ctypes.byref(events[i])
                    ),
                    f"array_write_async({fkey})",
                )

            ev_ptrs = (ctypes.POINTER(ffi.daos_event_t) * m)()
            completed = 0
            try:
                while completed < m:
                    rc = ffi.libdaos.daos_eq_poll(_CTX.eqh, 1, ffi.DAOS_EQ_WAIT, m - completed, ev_ptrs)
                    if rc < 0:
                        ffi.check_rc(rc, "daos_eq_poll(write)")
                    for j in range(rc):
                        ffi.check_rc(ev_ptrs[j].contents.ev_error, "array_write_completion")
                    completed += rc
            finally:
                for i, (_, _, aoh) in enumerate(non_empty):
                    ffi.libdaos.daos_event_fini(ctypes.byref(events[i]))
                    ffi.libdaos.daos_array_close(aoh, None)

        # Record chunk keys in KV index so StoreTests list/list_prefix/list_dir see all keys
        for fkey, _ in chunk_items:
            self._kv_put_sync(fkey, memoryview(b"\x01"))

    def _read_arrays_batch_sync(
        self, chunk_reqs: list[tuple[str, ByteRequest | None]]
    ) -> list[bytearray | None]:
        if not chunk_reqs:
            return []

        results: list[bytearray | None] = [None] * len(chunk_reqs)
        active: list[tuple[int, ffi.daos_handle_t, bytearray, int, int]] = []

        for idx, (fkey, br) in enumerate(chunk_reqs):
            oid = self._array_oid(fkey)
            aoh = ffi.daos_handle_t()
            cell_sz, chunk_sz = ffi.daos_size_t(), ffi.daos_size_t()
            rc = ffi.libdaos.daos_array_open(
                _CTX.coh, oid, ffi.DAOS_TX_NONE, ffi.DAOS_OO_RO,
                ctypes.byref(cell_sz), ctypes.byref(chunk_sz), ctypes.byref(aoh), None,
            )
            if rc == _DER_NONEXIST:
                continue
            ffi.check_rc(rc, f"array_open_ro({fkey})")

            total_sz = ffi.daos_size_t()
            ffi.check_rc(
                ffi.libdaos.daos_array_get_size(aoh, ffi.DAOS_TX_NONE, ctypes.byref(total_sz), None),
                f"array_get_size({fkey})",
            )
            size = total_sz.value

            if isinstance(br, RangeByteRequest):
                offset = br.start
                length = max(0, min(br.end, size) - offset)
            elif isinstance(br, OffsetByteRequest):
                offset = br.offset
                length = max(0, size - offset)
            elif isinstance(br, SuffixByteRequest):
                length = min(br.suffix, size)
                offset = size - length
            else:
                offset, length = 0, size

            out = bytearray(length)
            results[idx] = out
            if length == 0:
                ffi.libdaos.daos_array_close(aoh, None)
                continue

            active.append((idx, aoh, out, offset, length))

        m = len(active)
        if m == 0:
            return results

        events = (ffi.daos_event_t * m)()
        keepalive: list[object] = []
        for i, (_, aoh, out, offset, length) in enumerate(active):
            ffi.check_rc(ffi.libdaos.daos_event_init(ctypes.byref(events[i]), _CTX.eqh, None), "event_init")
            c_buf = (ctypes.c_char * length).from_buffer(out)
            iov = ffi.d_iov_t(iov_buf=ctypes.cast(c_buf, ctypes.c_void_p), iov_buf_len=length, iov_len=0)
            sgl = ffi.d_sg_list_t(sg_nr=1, sg_nr_out=0, sg_iovs=ctypes.pointer(iov))
            rg = ffi.daos_range_t(rg_len=length, rg_idx=offset)
            iod = ffi.daos_array_iod_t(arr_nr=1, arr_rgs=ctypes.pointer(rg), arr_nr_short_read=0)
            keepalive.extend((c_buf, iov, sgl, rg, iod))

            ffi.check_rc(
                ffi.libdaos.daos_array_read(
                    aoh, ffi.DAOS_TX_NONE, ctypes.byref(iod), ctypes.byref(sgl), ctypes.byref(events[i])
                ),
                "array_read_async",
            )

        ev_ptrs = (ctypes.POINTER(ffi.daos_event_t) * m)()
        completed = 0
        try:
            while completed < m:
                rc = ffi.libdaos.daos_eq_poll(_CTX.eqh, 1, ffi.DAOS_EQ_WAIT, m - completed, ev_ptrs)
                if rc < 0:
                    ffi.check_rc(rc, "daos_eq_poll(read)")
                for j in range(rc):
                    ffi.check_rc(ev_ptrs[j].contents.ev_error, "array_read_completion")
                completed += rc
        finally:
            for i, (_, aoh, _, _, _) in enumerate(active):
                ffi.libdaos.daos_event_fini(ctypes.byref(events[i]))
                ffi.libdaos.daos_array_close(aoh, None)

        return results

    # --- Zarr v3 Store Async API ---

    async def get(
        self,
        key: str,
        prototype: BufferPrototype = default_buffer_prototype,
        byte_range: ByteRequest | None = None,
    ) -> Buffer | None:
        res = await self.get_partial_values(prototype, [(key, byte_range)])
        return res[0]

    async def get_partial_values(
        self,
        prototype: BufferPrototype,
        key_ranges: Iterable[tuple[str, ByteRequest | None]],
    ) -> list[Buffer | None]:
        self._check_open()
        loop = asyncio.get_running_loop()
        reqs = list(key_ranges)

        def _op() -> list[bytearray | None]:
            out: list[bytearray | None] = [None] * len(reqs)
            chunk_positions: list[int] = []
            chunk_args: list[tuple[str, ByteRequest | None]] = []

            for i, (key, br) in enumerate(reqs):
                fkey = self._full_key(key)
                if _is_chunk_key(key):
                    chunk_positions.append(i)
                    chunk_args.append((fkey, br))
                else:
                    raw = self._kv_get_sync(fkey)
                    if raw is not None and key.endswith("zarr.json"):
                        inferred = _infer_chunk_bytes_from_zarr_json(raw)
                        if inferred is not None:
                            self._chunk_bytes_hint = inferred
                    if raw is None or br is None:
                        out[i] = raw
                    elif isinstance(br, RangeByteRequest):
                        out[i] = raw[br.start : br.end]
                    elif isinstance(br, OffsetByteRequest):
                        out[i] = raw[br.offset :]
                    elif isinstance(br, SuffixByteRequest):
                        out[i] = raw[-br.suffix :]

            if chunk_args:
                chunk_vals = self._read_arrays_batch_sync(chunk_args)
                for pos, val in zip(chunk_positions, chunk_vals):
                    out[pos] = val
            return out

        raw_list = await loop.run_in_executor(_CTX.executor, _op)
        return [prototype.buffer.from_bytes(item) if item is not None else None for item in raw_list]

    async def set(self, key: str, value: Buffer) -> None:
        await self.set_many([(key, value)])

    async def set_many(self, values: Iterable[tuple[str, Buffer]]) -> None:
        self._check_open()
        self._check_writable()
        items = [(key, self._full_key(key), _buf_to_mv(buf)) for key, buf in values]

        def _op() -> None:
            chunk_items: list[tuple[str, memoryview]] = []
            for key, fkey, mv in items:
                if _is_chunk_key(key):
                    chunk_items.append((fkey, mv))
                else:
                    self._kv_put_sync(fkey, mv)
            if chunk_items:
                self._write_arrays_batch_sync(chunk_items)

        await loop.run_in_executor(_CTX.executor, _op)

    async def exists(self, key: str) -> bool:
        fkey = self._full_key(key)
        await self._ensure_open()
        loop = asyncio.get_running_loop()

        def _op() -> bool:
            if _is_chunk_key(key):
                oid = self._array_oid(fkey)
                aoh = ffi.daos_handle_t()
                cell_sz, chunk_sz = ffi.daos_size_t(), ffi.daos_size_t()
                rc = ffi.libdaos.daos_array_open(
                    _CTX.coh, oid, ffi.DAOS_TX_NONE, ffi.DAOS_OO_RO,
                    ctypes.byref(cell_sz), ctypes.byref(chunk_sz), ctypes.byref(aoh), None,
                )
                if rc == 0:
                    ffi.libdaos.daos_array_close(aoh, None)
                    return True
                return False
            sz = ffi.daos_size_t(ffi.DAOS_REC_ANY)
            return ffi.libdaos.daos_kv_get(
                self._kv_oh, ffi.DAOS_TX_NONE, 0, fkey.encode("utf-8"), ctypes.byref(sz), None, None
            ) == 0

        return await loop.run_in_executor(_CTX.executor, _op)

    async def delete(self, key: str) -> None:
        self._check_writable()
        fkey = self._full_key(key)
        await self._ensure_open()
        loop = asyncio.get_running_loop()

        def _op() -> None:
            if _is_chunk_key(key):
                oid = self._array_oid(fkey)
                aoh = ffi.daos_handle_t()
                cell_sz, chunk_sz = ffi.daos_size_t(), ffi.daos_size_t()
                rc = ffi.libdaos.daos_array_open(
                    _CTX.coh, oid, ffi.DAOS_TX_NONE, ffi.DAOS_OO_RW,
                    ctypes.byref(cell_sz), ctypes.byref(chunk_sz), ctypes.byref(aoh), None,
                )
                if rc == 0:
                    ffi.libdaos.daos_array_destroy(aoh, ffi.DAOS_TX_NONE, None)
                    ffi.libdaos.daos_array_close(aoh, None)
            ffi.libdaos.daos_kv_remove(self._kv_oh, ffi.DAOS_TX_NONE, 0, fkey.encode("utf-8"), None)

        await loop.run_in_executor(_CTX.executor, _op)

    def _list_keys_sync(self) -> list[str]:
        keys: list[str] = []
        anchor = ffi.daos_anchor_t()
        buf = ctypes.create_string_buffer(65536)
        kds = (ffi.daos_key_desc_t * 256)()

        while not ffi.anchor_is_eof(anchor):
            nr = ctypes.c_uint32(256)
            if nr.value == 0: break
            iov = ffi.d_iov_t(iov_buf=ctypes.cast(buf, ctypes.c_void_p), iov_buf_len=65536, iov_len=0)
            sgl = ffi.d_sg_list_t(sg_nr=1, sg_nr_out=0, sg_iovs=ctypes.pointer(iov))
            ffi.check_rc(
                ffi.libdaos.daos_kv_list(
                    self._kv_oh, ffi.DAOS_TX_NONE, ctypes.byref(nr), kds, ctypes.byref(sgl), ctypes.byref(anchor), None
                ),
                "daos_kv_list",
            )
            offset = 0
            raw_bytes = buf.raw
            for i in range(nr.value):
                klen = kds[i].kd_key_len
                k_str = raw_bytes[offset : offset + klen].decode("utf-8")
                offset += klen
                if self.prefix:
                    if k_str.startswith(f"{self.prefix}/"):
                        keys.append(k_str[len(self.prefix) + 1 :])
                else:
                    keys.append(k_str)
        return keys

    async def list(self) -> AsyncGenerator[str, None]:
        await self._ensure_open()
        loop = asyncio.get_running_loop()
        keys = await loop.run_in_executor(_CTX.executor, self._list_keys_sync)
        for k in keys:
            yield k

    async def list_prefix(self, prefix: str) -> AsyncGenerator[str, None]:
        clean = prefix.lstrip("/")
        async for k in self.list():
            if k.startswith(clean):
                yield k

    async def list_dir(self, prefix: str) -> AsyncGenerator[str, None]:
        clean = prefix.strip("/")
        pfx = f"{clean}/" if clean else ""
        seen: set[str] = set()
        async for k in self.list():
            if k.startswith(pfx):
                rest = k[len(pfx) :]
                entry = rest.split("/", 1)[0]
                if entry and entry not in seen:
                    seen.add(entry)
                    yield entry
