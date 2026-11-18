from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

from zarr_daos.store import DAOSStore


def store_from_daos_url(
    url: str,
    *,
    read_only: bool = False,
    storage_options: dict[str, Any] | None = None,
) -> tuple[DAOSStore, str]:
    """Parse 'daos://<pool>/<cont>/<dict_name>[/<subpath>]' into a (DAOSStore, residual_path) tuple."""
    parsed = urlparse(url)
    if parsed.scheme != "daos":
        raise ValueError(f"Expected 'daos://' scheme, got '{parsed.scheme}://'.")

    pool = parsed.netloc
    parts = [p for p in parsed.path.strip("/").split("/") if p]
    if not pool or len(parts) < 2:
        raise ValueError(
            f"Invalid DAOS URL '{url}'. Expected 'daos://<pool>/<cont>/<dict_name>[/<subpath>]'."
        )

    cont, dict_name = parts[0], parts[1]
    residual_path = "/".join(parts[2:])
    opts = dict(storage_options or {})

    store = DAOSStore(
        pool=pool,
        cont=cont,
        dict_name=dict_name,
        read_only=read_only,
        **opts,
    )
    return store, residual_path
