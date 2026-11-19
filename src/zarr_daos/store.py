from __future__ import annotations

import threading
import asyncio
from collections.abc import AsyncIterator, Iterable
from typing import Any

try:
    import pydaos
except ImportError:
    pydaos = None  # type: ignore[assignment]

from zarr.abc.store import ByteRequest, Store
from zarr.core.buffer import Buffer, BufferPrototype

# 1-byte header prevents PyDAOS from treating 0-byte payloads (b"") as missing/deleted keys
_PAYLOAD_PREFIX = b"\x01"


def _encode_payload(data: bytes) -> bytes:
    return _PAYLOAD_PREFIX + data


def _decode_payload(raw: bytes) -> bytes:
    if raw.startswith(_PAYLOAD_PREFIX):
        return raw[1:]
    return raw


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
        lock: threading.Lock | None = None,
    ) -> None:
        super().__init__(read_only=read_only)
        self.pool = pool
        self.cont = cont
        self.dict_name = dict_name
        self._create_dict = create_dict
        self._dcont: Any = None
        self._ddict: Any = None
        self._lock = lock if lock is not None else threading.Lock()

    @staticmethod
    def _not_found_errors() -> tuple[type[Exception], ...]:
        errs: list[type[Exception]] = [KeyError]
        if pydaos is not None:
            for attr in ("DObjNotFound", "PyDError"):
                exc = getattr(pydaos, attr, None)
                if isinstance(exc, type) and issubclass(exc, Exception):
                    errs.append(exc)
        return tuple(errs)

    async def _open(self) -> None:
        if pydaos is None:
            raise ImportError(
                "The 'pydaos' module is not available. Load your DAOS environment module "
                "or add PyDAOS to PYTHONPATH before opening a DAOSStore."
            )
        await super()._open()
        not_found = self._not_found_errors()

        def _connect() -> tuple[Any, Any]:
            dcont = pydaos.DCont(self.pool, self.cont)
            try:
                ddict = dcont.get(self.dict_name)
            except not_found:
                if self.read_only or not self._create_dict:
                    ddict = None
                else:
                    ddict = dcont.dict(self.dict_name)
            if ddict is not None and hasattr(ddict, "value_size"):
                ddict.value_size = 32 * 1024 * 1024  # Prevent 1 MiB bget truncation
            return dcont, ddict

    async def _get_ddict(self) -> Any:
        if not self._is_open:
            await self._open()
        if self._ddict is None and self._dcont is not None:
            not_found = self._not_found_errors()

            def _retry_get() -> Any:
                try:
                    return self._dcont.get(self.dict_name)
                except not_found:
                    return None

            self._ddict = await asyncio.to_thread(_retry_get)
        return self._ddict

    async def get(
        self,
        key: str,
        prototype: BufferPrototype,
        byte_range: ByteRequest | None = None,
    ) -> Buffer | None:
        if byte_range is not None:
            self._validate_byte_range(byte_range)

        ddict = await self._get_ddict()
        if ddict is None:
            return None

        def _read() -> bytes | None:
            try:
                return ddict[key]
            except KeyError:
                return None

        raw = await asyncio.to_thread(_read)
        if raw is None:
            return None

        data = _decode_payload(raw)
        if byte_range is not None:
            data = self._slice_bytes(data, byte_range)

        return prototype.buffer.from_bytes(data)

    async def get_partial_values(
        self,
        prototype: BufferPrototype,
        key_ranges: Iterable[tuple[str, ByteRequest | None]],
    ) -> list[Buffer | None]:
        requests = list(key_ranges)
        if not requests:
            return []

        for _, byte_range in requests:
            if byte_range is not None:
                self._validate_byte_range(byte_range)

        ddict = await self._get_ddict()
        if ddict is None:
            return [None] * len(requests)

        unique_keys = list(dict.fromkeys(key for key, _ in requests))


        def _bulk_read() -> dict[str, bytes | None]:
            with self._lock:
                if hasattr(ddict, "bget"):
                    query_dict: dict[str, bytes | None] = {k: None for k in unique_keys}
                    res = ddict.bget(query_dict)
                    return res if res is not None else query_dict
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
            data = _decode_payload(raw)
            if byte_range is not None:
                data = self._slice_bytes(data, byte_range)
            results.append(prototype.buffer.from_bytes(data))

        return results

    def _bulk_read() -> dict[str, bytes | None]:
            with self._lock:
                if hasattr(ddict, "bget"):
                    query_dict: dict[str, bytes | None] = {k: None for k in unique_keys}
                    res = ddict.bget(query_dict)
                    return res if res is not None else query_dict
                out: dict[str, bytes | None] = {}
                for k in unique_keys:
                    try:
                        out[k] = ddict[k]
                    except KeyError:
                        out[k] = None
                return out

    async def _set_many(self, values: Iterable[tuple[str, Buffer]]) -> None:
        self._check_writable()
        items = list(values)
        if not items:
            return

        ddict = await self._get_ddict()
        if ddict is None:
            def _create_now() -> Any:
                with self._lock:
                    if self._dcont is None:
                        self._dcont = pydaos.DCont(self.pool, self.cont)
                    self._ddict = self._dcont.dict(self.dict_name)
                    if hasattr(self._ddict, "value_size"):
                        self._ddict.value_size = 32 * 1024 * 1024
                    return self._ddict

            ddict = await asyncio.to_thread(_create_now)

        # Pre-encode payloads outside the DAOS lock
        encoded_batch = {key: _encode_payload(buf.to_bytes()) for key, buf in items}

        def _bulk_write() -> None:
            with self._lock:
                if hasattr(ddict, "bput"):
                    # Batch in chunks of 512 keys to keep RPC payload sizes bounded
                    keys = list(encoded_batch.keys())
                    for i in range(0, len(keys), 512):
                        sub = {k: encoded_batch[k] for k in keys[i : i + 512]}
                        ddict.bput(sub)
                else:
                    for k, v in encoded_batch.items():
                        ddict[k] = v

        await asyncio.to_thread(_bulk_write)
 
    async def set(self, key: str, value: Buffer) -> None:
        self._check_writable()
        ddict = await self._get_ddict()
        if ddict is None:
            def _create_now() -> Any:
                with self._lock:
                    if self._dcont is None:
                        self._dcont = pydaos.DCont(self.pool, self.cont)
                    self._ddict = self._dcont.dict(self.dict_name)
                    if hasattr(self._ddict, "value_size"):
                        self._ddict.value_size = 32 * 1024 * 1024
                    return self._ddict

            ddict = await asyncio.to_thread(_create_now)

        payload = _encode_payload(value.to_bytes())

        def _write() -> None:
            with self._lock:
                ddict[key] = payload

        await asyncio.to_thread(_write)

    async def set_if_not_exists(self, key: str, value: Buffer) -> None:
        self._check_writable()
        ddict = await self._get_ddict()
        assert ddict is not None
        payload = _encode_payload(value.to_bytes())

        def _write_if_missing() -> None:
            if key not in ddict:
                ddict[key] = payload

        await asyncio.to_thread(_write_if_missing)

    async def delete(self, key: str) -> None:
        self._check_writable()
        ddict = await self._get_ddict()
        if ddict is None:
            return

        def _del() -> None:
            try:
                del ddict[key]
            except KeyError:
                pass

        await asyncio.to_thread(_del)

    async def delete_dir(self, prefix: str) -> None:
        self._check_writable()
        ddict = await self._get_ddict()
        if ddict is None:
            return

        normalized = prefix.rstrip("/")
        if normalized:
            normalized += "/"

        def _del_prefix() -> None:
            # PyDAOS DDict implements __iter__ over keys (not .keys())
            to_delete = [
                k for k in list(ddict) if prefix == "" or k == prefix or k.startswith(normalized)
            ]
            if not to_delete:
                return
            if hasattr(ddict, "bput"):
                ddict.bput({k: None for k in to_delete})
            else:
                for k in to_delete:
                    try:
                        del ddict[k]
                    except KeyError:
                        pass

        await asyncio.to_thread(_del_prefix)

    async def exists(self, key: str) -> bool:
        ddict = await self._get_ddict()
        if ddict is None:
            return False
        return await asyncio.to_thread(lambda: key in ddict)

    async def getsize(self, key: str) -> int:
        ddict = await self._get_ddict()
        if ddict is None:
            raise FileNotFoundError(key)

        def _size() -> int:
            try:
                return len(_decode_payload(ddict[key]))
            except KeyError as exc:
                raise FileNotFoundError(key) from exc

        return await asyncio.to_thread(_size)

    async def list(self) -> AsyncIterator[str]:
        ddict = await self._get_ddict()
        if ddict is None:
            return
        # PyDAOS DDict yields keys directly when iterated
        keys = await asyncio.to_thread(lambda: list(ddict))
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
    def _validate_byte_range(byte_range: ByteRequest) -> None:
        if not (
            (hasattr(byte_range, "start") and hasattr(byte_range, "end"))
            or hasattr(byte_range, "offset")
            or hasattr(byte_range, "suffix")
        ):
            raise ValueError(f"Unexpected byte_range, got {byte_range!r}")

    @classmethod
    def _slice_bytes(cls, data: bytes, byte_range: ByteRequest) -> bytes:
        cls._validate_byte_range(byte_range)
        if hasattr(byte_range, "start") and hasattr(byte_range, "end"):
            return data[byte_range.start : byte_range.end]
        if hasattr(byte_range, "offset"):
            return data[byte_range.offset :]
        if hasattr(byte_range, "suffix"):
            return data[-byte_range.suffix :]
        raise ValueError(f"Unexpected byte_range, got {byte_range!r}")

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
        self._lock = threading.Lock()
