from urllib.parse import urlparse
from zarr_daos.libdaos_store import DAOSArrayStore


def store_from_daos_url(url: str, read_only: bool = False) -> DAOSArrayStore:
    """Parse daos://pool/cont[/prefix] into a DAOSArrayStore."""
    parsed = urlparse(url)
    if parsed.scheme != "daos":
        raise ValueError(f"Expected daos:// URL, got: {url}")
    pool = parsed.netloc
    parts = parsed.path.lstrip("/").split("/", 1)
    cont = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""
    if not pool or not cont:
        raise ValueError(f"Invalid DAOS URL (missing pool or container): {url}")
    return DAOSArrayStore(pool=pool, cont=cont, prefix=prefix, read_only=read_only)
