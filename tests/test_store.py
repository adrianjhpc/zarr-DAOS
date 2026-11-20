from __future__ import annotations

import os
import uuid
from typing import Any

import numpy as np
import pytest
import zarr
from zarr.codecs import BloscCodec
from zarr.core.buffer import Buffer, cpu
from zarr.testing.store import StoreTests

from zarr_daos.adapter import store_from_daos_url
from zarr_daos.libdaos_store import DAOSArrayStore, _CTX

DAOS_POOL = os.environ.get("DAOS_TEST_POOL", "test_pool")
DAOS_CONT = os.environ.get("DAOS_TEST_CONT", "test_cont")


def _daos_available() -> bool:
    try:
        _CTX.ensure_open(DAOS_POOL, DAOS_CONT)
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _daos_available(),
    reason=f"DAOS pool '{DAOS_POOL}' / container '{DAOS_CONT}' not reachable",
)


class TestDAOSArrayStoreCompliance(StoreTests[DAOSArrayStore, cpu.Buffer]):
    store_cls = DAOSArrayStore
    buffer_cls = cpu.Buffer

    @pytest.fixture
    def store_kwargs(self) -> dict[str, Any]:
        return {
            "pool": DAOS_POOL,
            "cont": DAOS_CONT,
            "prefix": f"zarr_test_{uuid.uuid4().hex}",
        }

    @pytest.fixture
    def open_kwargs(self, store_kwargs: dict[str, Any]) -> dict[str, Any]:
        return store_kwargs

    async def get(self, store: DAOSArrayStore, key: str) -> Buffer:
        res = await store.get(key)
        assert res is not None
        return res

    async def set(self, store: DAOSArrayStore, key: str, value: Buffer) -> None:
        await store.set(key, value)


def test_sharding_and_url_adapter() -> None:
    prefix = f"zarr_shard_{uuid.uuid4().hex}"
    store = DAOSArrayStore(
        pool=DAOS_POOL,
        cont=DAOS_CONT,
        prefix=prefix,
    )
    data = np.arange(64 * 64, dtype="int32").reshape(64, 64)

    arr = zarr.create_array(
        store=store,
        shape=(64, 64),
        shards=(32, 32),
        chunks=(16, 16),
        dtype="int32",
        compressors=[BloscCodec(cname="lz4", clevel=1)],
    )
    arr[:] = data

    uri = f"daos://{store.pool}/{store.cont}/{store.prefix}"
    resolved_store, sub_path = store_from_daos_url(uri, read_only=True)
    assert sub_path == ""
    reopened = zarr.open_array(store=resolved_store, mode="r")
    np.testing.assert_array_equal(reopened[16:48, 16:48], data[16:48, 16:48])
    store.close()
    resolved_store.close()
