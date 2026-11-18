from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterable
from typing import Any

import pydaos
from zarr.abc.store import ByteRequest, Store
from zarr.core.buffer import Buffer, BufferPrototype


class DAOSStore(Store):
    """Zarr-Python 3 Store backed by a PyDAOS Key-Value dictionary (DDict)."""

    supports_writes: bool = True
    supports_deletes: bool = True
    supports_partial_writes: bool = False
    supports_listing: bool = True

    def __init__(
        self,
        pool: str,
        cont: str,
        dict_name: str = "zarr_root",
        *,
        read_only: bool = False,
        create_dict: bool = True,
    ) -> None:
        super().__init__(read_only=read_only)
        self.pool = pool
        self.cont = cont
        self.dict_name = dict_name
        self._create_dict = create_dict
        self._dcont: pydaos.DCont | None = None
        self._ddict: pydaos.DDict | None = None

    async def _open(self) -> None:
        if self._ddict is not None:
            return

        def _connect() -> tuple[pydaos.DCont, pydaos.DDict]:
            dcont = pydaos.DCont(self.pool, self.cont)
            try:
                ddict = dcont.get(self.dict_name)
            except KeyError:
                if not self._create_dict or self.read_only:
                    raise
                ddict = dcont.dict(self.dict_name)
            return dcont, ddict

        self._dcont, self._ddict = await asyncio.to_thread(_connect)

    def _ensure_open(self) -> pydaos.DDict:
        if self._ddict is None:
            raise RuntimeError("DAOSStore is not open. Call 'await store.open()' first.")
        return self._ddict

    async def get(
        self,
        key: str,
        prototype: BufferPrototype,
        byte_range: ByteRequest | None = None,
    ) -> Buffer | None:
        ddict = self._ensure_open()

        def _read() -> bytes | None:
            try:
                return ddict[key]
            except KeyError:
                return None

        raw = await asyncio.to_thread(_read)
        if raw is None:
            return None

        if byte_range is not None:
            raw = self._slice_bytes(raw, byte_range)

        return prototype.buffer.from_bytes(raw)

    async def get_partial_values(
        self,
        prototype: BufferPrototype,
        key_ranges: Iterable[tuple[str, ByteRequest | None]],
    ) -> list[Buffer | None]:
        ddict = self._ensure_open()
        requests = list(key_ranges)
        if not requests:
            return []

        unique_keys = list(dict.fromkeys(key for key, _ in requests))

        def _bulk_read() -> dict[str, bytes | None]:
            if hasattr(ddict, "bget"):
                return ddict.bget(unique_keys)
            out: dict[str, bytes | None] = {}
            for k in unique_keys:
                try:
                    out[k] = ddict[k]
                except KeyError:
                    out[k] = None
            return out

        fetched = await asyncio.to_thread(_bulk_read)

        results: list[Buffer | None] = []
        for key, byte_range in requests:
            raw = fetched.get(key)
            if raw is None:
                results.append(None)
                continue
            if byte_range is not None:
                raw = self._slice_bytes(raw, byte_range)
            results.append(prototype.buffer.from_bytes(raw))
        return results

    async def set(self, key: str, value: Buffer) -> None:
        self._check_writable()
        ddict = self._ensure_open()
        payload = value.to_bytes()

        def _write() -> None:
            ddict[key] = payload

        await asyncio.to_thread(_write)

    async def set_if_not_exists(self, key: str, value: Buffer) -> None:
        self._check_writable()
        ddict = self._ensure_open()
        payload = value.to_bytes()

        def _write_if_missing() -> None:
            if key not in ddict:
                ddict[key] = payload

        await asyncio.to_thread(_write_if_missing)

    async def delete(self, key: str) -> None:
        self._check_writable()
        ddict = self._ensure_open()

        def _del() -> None:
            try:
                del ddict[key]
            except KeyError:
                pass

        await asyncio.to_thread(_del)

    async def delete_dir(self, prefix: str) -> None:
        self._check_writable()
        ddict = self._ensure_open()
        normalized = prefix.rstrip("/")
        if normalized:
            normalized += "/"

        def _del_prefix() -> None:
            to_delete = [k for k in ddict.keys() if k == prefix or k.startswith(normalized)]
            if not to_delete:
                return
            if hasattr(ddict, "bput"):
                # PyDAOS bput removes keys mapped to None
                ddict.bput({k: None for k in to_delete})
            else:
                for k in to_delete:
                    try:
                        del ddict[k]
                    except KeyError:
                        pass

        await asyncio.to_thread(_del_prefix)

    async def exists(self, key: str) -> bool:
        ddict = self._ensure_open()
        return await asyncio.to_thread(lambda: key in ddict)

    async def getsize(self, key: str) -> int:
        ddict = self._ensure_open()

        def _size() -> int:
            try:
                return len(ddict[key])
            except KeyError as exc:
                raise FileNotFoundError(key) from exc

        return await asyncio.to_thread(_size)

    async def list(self) -> AsyncIterator[str]:
        ddict = self._ensure_open()
        keys = await asyncio.to_thread(lambda: list(ddict.keys()))
        for key in keys:
            yield key

    async def list_prefix(self, prefix: str) -> AsyncIterator[str]:
        async for key in self.list():
            if key.startswith(prefix):
                yield key

    async def list_dir(self, prefix: str) -> AsyncIterator[str]:
        normalized = prefix.rstrip("/")
        if normalized:
            normalized += "/"
        seen: set[str] = set()

        async for key in self.list_prefix(normalized):
            remainder = key[len(normalized) :]
            if not remainder:
                continue
            head = remainder.split("/", 1)[0]
            if head not in seen:
                seen.add(head)
                yield head

    def close(self) -> None:
        self._ddict = None
        self._dcont = None
        super().close()

    @staticmethod
    def _slice_bytes(data: bytes, byte_range: ByteRequest) -> bytes:
        if hasattr(byte_range, "start") and hasattr(byte_range, "end"):
            return data[byte_range.start : byte_range.end]
        if hasattr(byte_range, "offset"):
            return data[byte_range.offset :]
        if hasattr(byte_range, "suffix"):
            return data[-byte_range.suffix :]
        return data

    def __repr__(self) -> str:
        return (
            f"DAOSStore(pool={self.pool!r}, cont={self.cont!r}, "
            f"dict_name={self.dict_name!r}, read_only={self.read_only!r})"
        )

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, DAOSStore)
            and self.pool == other.pool
            and self.cont == other.cont
            and self.dict_name == other.dict_name
            and self.read_only == other.read_only
        )

    def __getstate__(self) -> dict[str, Any]:
        return {
            "pool": self.pool,
            "cont": self.cont,
            "dict_name": self.dict_name,
            "read_only": self.read_only,
            "create_dict": self._create_dict,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        super().__init__(read_only=state["read_only"])
        self.pool = state["pool"]
        self.cont = state["cont"]
        self.dict_name = state["dict_name"]
        self._create_dict = state["create_dict"]
        self._dcont = None
        self._ddict = None
