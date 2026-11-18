from __future__ import annotations

import asyncio
import os
import sys
import types
import uuid
from typing import Any

import numpy as np
import pytest
import zarr
from zarr.codecs import BloscCodec, BytesCodec, ShardingCodec
from zarr.core.buffer import Buffer, cpu
from zarr.testing.store import StoreTests
from zarr_daos.adapter import store_from_daos_url

if os.environ.get("USE_MOCK_DAOS") == "1" and "pydaos" not in sys.modules:
    _CONTAINERS: dict[tuple[str, str], dict[str, dict[str, bytes]]] = {}

    class _MockDDict(dict[str, bytes]):
        def bget(self, keys: list[str]) -> dict[str, bytes | None]:
            return {k: self.get(k) for k in keys}

        def bput(self, mapping: dict[str, bytes | None]) -> None:
            for k, v in mapping.items():
                if v is None:
                    self.pop(k, None)
                else:
                    self[k] = v

    class _MockDCont:
        def __init__(self, pool: str, cont: str) -> None:
            self._store = _CONTAINERS.setdefault((pool, cont), {})

        def get(self, name: str) -> _MockDDict:
            if name not in self._store:
                raise KeyError(name)
            return self._store[name]  # type: ignore[return-value]

        def dict(self, name: str) -> _MockDDict:
            if name in self._store:
                raise KeyError(name)
            d = _MockDDict()
            self._store[name] = d
            return d

    mock_mod = types.ModuleType("pydaos")
    mock_mod.DCont = _MockDCont  # type: ignore[attr-defined]
    mock_mod.DDict = _MockDDict  # type: ignore[attr-defined]
    sys.modules["pydaos"] = mock_mod

import pydaos  # noqa: E402
from zarr_daos.store import DAOSStore  # noqa: E402

DAOS_POOL = os.environ.get("DAOS_TEST_POOL", "test_pool")
DAOS_CONT = os.environ.get("DAOS_TEST_CONT", "test_cont")


def _daos_available() -> bool:
    try:
        pydaos.DCont(DAOS_POOL, DAOS_CONT)
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _daos_available(),
    reason=f"DAOS pool '{DAOS_POOL}' / container '{DAOS_CONT}' not reachable",
)


class TestDAOSStoreCompliance(StoreTests[DAOSStore, cpu.Buffer]):
    store_cls = DAOSStore
    buffer_cls = cpu.Buffer

    @pytest.fixture
    def store_kwargs(self) -> dict[str, Any]:
        return {
            "pool": DAOS_POOL,
            "cont": DAOS_CONT,
            "dict_name": f"zarr_test_{uuid.uuid4().hex}",
        }

    @pytest.fixture
    def open_kwargs(self, store_kwargs: dict[str, Any]
    uri = f"daos://{store.pool}/{store.cont}/{store.dict_name}"
    resolved_store, sub_path = store_from_daos_url(uri, read_only=True)
    assert sub_path == ""
    reopened = zarr.open_array(store=resolved_store, mode="r")
    np.testing.assert_array_equal(reopened[16:48, 16:48], data[16:48, 16:48])) -> dict[str, Any]:
        return store_kwargs

    async def get(self, store: DAOSStore, key: str) -> Buffer:
        def _raw_get() -> bytes:
            return pydaos.DCont(store.pool, store.cont).get(store.dict_name)[key]

        raw = await asyncio.to_thread(_raw_get)
        return self.buffer_cls.from_bytes(raw)

    async def set(self, store: DAOSStore, key: str, value: Buffer) -> None:
        payload = value.to_bytes()

        def _raw_set() -> None:
            dcont = pydaos.DCont(store.pool, store.cont)
            try:
                ddict = dcont.get(store.dict_name)
            except KeyError:
                ddict = dcont.dict(store.dict_name)
            ddict[key] = payload

        await asyncio.to_thread(_raw_set)


def test_sharding_and_url_adapter() -> None:
    store = DAOSStore(
        pool=DAOS_POOL,
        cont=DAOS_CONT,
        dict_name=f"zarr_shard_{uuid.uuid4().hex}",
    )
    shard_codec = ShardingCodec(
        chunk_shape=(16, 16),
        codecs=[BytesCodec(), BloscCodec(cname="lz4", clevel=1)],
    )
    data = np.arange(64 * 64, dtype="int32").reshape(64, 64)

    arr = zarr.create_array(
        store=store,
        shape=(64, 64),
        chunks=(32, 32),
        dtype="int32",
        codecs=[shard_codec],
    )
    arr[:] = data

    uri = f"daos://{store.pool}/{store.cont}/{store.dict_name}"
    resolved_store, sub_path = store_from_daos_url(uri, read_only=True)
    assert sub_path == ""
    reopened = zarr.open_array(store=resolved_store, mode="r")
    np.testing.assert_array_equal(reopened[16:48, 16:48], data[16:48, 16:48])
