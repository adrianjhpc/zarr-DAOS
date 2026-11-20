import ctypes
import ctypes.util
import hashlib
from typing import Iterator

_lib_path = ctypes.util.find_library("daos") or "/usr/lib64/libdaos.so"
libdaos = ctypes.CDLL(_lib_path, use_errno=True)

# Handle & scalar typedefs
daos_handle_t = ctypes.c_uint64
daos_size_t = ctypes.c_uint64
daos_off_t = ctypes.c_uint64
daos_oclass_id_t = ctypes.c_uint32
daos_oclass_hints_t = ctypes.c_uint16

DAOS_HDL_INVAL = 0
DAOS_TX_NONE = 0
DAOS_REC_ANY = ctypes.c_uint64(-1).value

# Pool / Container / Object access flags
DAOS_PC_RO = 1 << 0
DAOS_PC_RW = 1 << 1
DAOS_COO_RO = 1 << 0
DAOS_COO_RW = 1 << 1
DAOS_OO_RO = 1 << 1
DAOS_OO_RW = 1 << 2

# Object types (from /usr/include/daos_obj.h)
DAOS_OT_KV_HASHED = 8
DAOS_OT_ARRAY_BYTE = 13

DAOS_EQ_NOWAIT = 0
DAOS_EQ_WAIT = -1


class _daos_event_private(ctypes.Structure):
    _fields_ = [("space", ctypes.c_uint64 * 20)]


class daos_event_t(ctypes.Structure):
    _fields_ = [
        ("ev_error", ctypes.c_int),
        ("ev_private", _daos_event_private),
        ("ev_debug", ctypes.c_uint64),
    ]

libdaos.daos_eq_create.argtypes = [ctypes.POINTER(daos_handle_t)]
libdaos.daos_eq_create.restype = ctypes.c_int

libdaos.daos_eq_destroy.argtypes = [daos_handle_t, ctypes.c_int]
libdaos.daos_eq_destroy.restype = ctypes.c_int

libdaos.daos_event_init.argtypes = [
    ctypes.POINTER(daos_event_t),
    daos_handle_t,
    ctypes.POINTER(daos_event_t),
]
libdaos.daos_event_init.restype = ctypes.c_int

libdaos.daos_event_fini.argtypes = [ctypes.POINTER(daos_event_t)]
libdaos.daos_event_fini.restype = ctypes.c_int

libdaos.daos_eq_poll.argtypes = [
    daos_handle_t,
    ctypes.c_int,
    ctypes.c_int64,
    ctypes.c_uint32,
    ctypes.POINTER(ctypes.POINTER(daos_event_t)),
]
libdaos.daos_eq_poll.restype = ctypes.c_int

class daos_obj_id_t(ctypes.Structure):
    _fields_ = [("lo", ctypes.c_uint64), ("hi", ctypes.c_uint64)]


class d_iov_t(ctypes.Structure):
    _fields_ = [
        ("iov_buf", ctypes.c_void_p),
        ("iov_buf_len", ctypes.c_size_t),
        ("iov_len", ctypes.c_size_t),
    ]


class d_sg_list_t(ctypes.Structure):
    _fields_ = [
        ("sg_nr", ctypes.c_uint32),
        ("sg_nr_out", ctypes.c_uint32),
        ("sg_iovs", ctypes.POINTER(d_iov_t)),
    ]


class daos_range_t(ctypes.Structure):
    _fields_ = [("rg_len", daos_size_t), ("rg_idx", daos_off_t)]


class daos_array_iod_t(ctypes.Structure):
    _fields_ = [
        ("arr_nr", daos_size_t),
        ("arr_rgs", ctypes.POINTER(daos_range_t)),
        ("arr_nr_short_read", daos_size_t),
    ]


class daos_key_desc_t(ctypes.Structure):
    _fields_ = [
        ("kd_key_len", daos_size_t),
        ("kd_val_type", ctypes.c_uint32),
        ("kd_csum_type", ctypes.c_uint16),
        ("kd_csum_len", ctypes.c_uint16),
    ]


class daos_anchor_t(ctypes.Structure):
    _fields_ = [
        ("da_type", ctypes.c_uint16),
        ("da_shard", ctypes.c_uint16),
        ("da_flags", ctypes.c_uint32),
        ("da_sub_anchors", ctypes.c_uint64),
        ("da_buf", ctypes.c_uint8 * 112),
    ]


# Function prototypes
libdaos.daos_init.argtypes = []
libdaos.daos_init.restype = ctypes.c_int

libdaos.daos_fini.argtypes = []
libdaos.daos_fini.restype = ctypes.c_int

libdaos.daos_pool_connect.argtypes = [
    ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint,
    ctypes.POINTER(daos_handle_t), ctypes.c_void_p, ctypes.c_void_p,
]
libdaos.daos_pool_connect.restype = ctypes.c_int

libdaos.daos_pool_disconnect.argtypes = [daos_handle_t, ctypes.c_void_p]
libdaos.daos_pool_disconnect.restype = ctypes.c_int

libdaos.daos_cont_open.argtypes = [
    daos_handle_t, ctypes.c_char_p, ctypes.c_uint,
    ctypes.POINTER(daos_handle_t), ctypes.c_void_p, ctypes.c_void_p,
]
libdaos.daos_cont_open.restype = ctypes.c_int

libdaos.daos_cont_close.argtypes = [daos_handle_t, ctypes.c_void_p]
libdaos.daos_cont_close.restype = ctypes.c_int

libdaos.daos_oclass_name2id.argtypes = [ctypes.c_char_p]
libdaos.daos_oclass_name2id.restype = daos_oclass_id_t

libdaos.daos_obj_generate_oid.argtypes = [
    daos_handle_t, ctypes.POINTER(daos_obj_id_t),
    ctypes.c_int, daos_oclass_id_t, daos_oclass_hints_t, ctypes.c_uint32,
]
libdaos.daos_obj_generate_oid.restype = ctypes.c_int

# KV API
libdaos.daos_kv_open.argtypes = [
    daos_handle_t, daos_obj_id_t, ctypes.c_uint,
    ctypes.POINTER(daos_handle_t), ctypes.c_void_p,
]
libdaos.daos_kv_open.restype = ctypes.c_int

libdaos.daos_kv_close.argtypes = [daos_handle_t, ctypes.c_void_p]
libdaos.daos_kv_close.restype = ctypes.c_int

libdaos.daos_kv_put.argtypes = [
    daos_handle_t, daos_handle_t, ctypes.c_uint64,
    ctypes.c_char_p, daos_size_t, ctypes.c_void_p, ctypes.c_void_p,
]
libdaos.daos_kv_put.restype = ctypes.c_int

libdaos.daos_kv_get.argtypes = [
    daos_handle_t, daos_handle_t, ctypes.c_uint64,
    ctypes.c_char_p, ctypes.POINTER(daos_size_t), ctypes.c_void_p, ctypes.c_void_p,
]
libdaos.daos_kv_get.restype = ctypes.c_int

libdaos.daos_kv_remove.argtypes = [
    daos_handle_t, daos_handle_t, ctypes.c_uint64, ctypes.c_char_p, ctypes.c_void_p,
]
libdaos.daos_kv_remove.restype = ctypes.c_int

libdaos.daos_kv_list.argtypes = [
    daos_handle_t, daos_handle_t, ctypes.POINTER(ctypes.c_uint32),
    ctypes.POINTER(daos_key_desc_t), ctypes.POINTER(d_sg_list_t),
    ctypes.POINTER(daos_anchor_t), ctypes.c_void_p,
]
libdaos.daos_kv_list.restype = ctypes.c_int

# Array API
libdaos.daos_array_create.argtypes = [
    daos_handle_t, daos_obj_id_t, daos_handle_t,
    daos_size_t, daos_size_t, ctypes.POINTER(daos_handle_t), ctypes.c_void_p,
]
libdaos.daos_array_create.restype = ctypes.c_int

libdaos.daos_array_open.argtypes = [
    daos_handle_t, daos_obj_id_t, daos_handle_t, ctypes.c_uint,
    ctypes.POINTER(daos_size_t), ctypes.POINTER(daos_size_t),
    ctypes.POINTER(daos_handle_t), ctypes.c_void_p,
]
libdaos.daos_array_open.restype = ctypes.c_int

libdaos.daos_array_write.argtypes = [
    daos_handle_t, daos_handle_t,
    ctypes.POINTER(daos_array_iod_t), ctypes.POINTER(d_sg_list_t), ctypes.c_void_p,
]
libdaos.daos_array_write.restype = ctypes.c_int

libdaos.daos_array_read.argtypes = [
    daos_handle_t, daos_handle_t,
    ctypes.POINTER(daos_array_iod_t), ctypes.POINTER(d_sg_list_t), ctypes.c_void_p,
]
libdaos.daos_array_read.restype = ctypes.c_int

libdaos.daos_array_get_size.argtypes = [
    daos_handle_t, daos_handle_t, ctypes.POINTER(daos_size_t), ctypes.c_void_p,
]
libdaos.daos_array_get_size.restype = ctypes.c_int

libdaos.daos_array_destroy.argtypes = [daos_handle_t, daos_handle_t, ctypes.c_void_p]
libdaos.daos_array_destroy.restype = ctypes.c_int

libdaos.daos_array_close.argtypes = [daos_handle_t, ctypes.c_void_p]
libdaos.daos_array_close.restype = ctypes.c_int


def check_rc(rc: int, context: str) -> None:
    if rc != 0:
        raise RuntimeError(f"libdaos error in {context}: rc={rc}")


def resolve_oclass(name: str) -> int:
    cid = libdaos.daos_oclass_name2id(name.encode("utf-8"))
    if cid == 0:
        raise ValueError(f"Unknown DAOS object class: {name}")
    return cid


def make_oid(coh: daos_handle_t, seed: str, otype: int, oclass: int) -> daos_obj_id_t:
    digest = hashlib.blake2b(seed.encode("utf-8"), digest_size=16).digest()
    lo = int.from_bytes(digest[:8], "little")
    hi = int.from_bytes(digest[8:], "little") & 0xFFFFFFFF
    oid = daos_obj_id_t(lo=lo, hi=hi)
    check_rc(libdaos.daos_obj_generate_oid(coh, ctypes.byref(oid), otype, oclass, 0, 0), f"generate_oid({seed})")
    return oid


def anchor_is_eof(anchor: daos_anchor_t) -> bool:
    # DAOS_ANCHOR_TYPE_EOF == 2
    return anchor.da_type == 2
