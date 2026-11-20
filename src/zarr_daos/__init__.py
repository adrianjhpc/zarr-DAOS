from zarr_daos.adapter import store_from_daos_url
from zarr_daos.libdaos_store import DAOSArrayStore

DAOSStore = DAOSArrayStore

__all__ = ["DAOSArrayStore", "DAOSStore", "store_from_daos_url"]
