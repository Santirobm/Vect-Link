"""Low-level ctypes bindings and shared library loader for libvectlink."""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import platform
import sys
from pathlib import Path
from typing import List, Optional

VTLK_HEADER_SIZE = 64
VTLK_PROTOCOL_VERSION = 0x0001
VTLK_LAYER_FINAL = 0xFFFF
VTLK_RING_CTRL_SIZE = 192
VTLK_SHM_VERSION = 1

VTLK_OK = 0
VTLK_ERR_NULL_PTR = -1
VTLK_ERR_BUFFER_TOO_SMALL = -2
VTLK_ERR_INVALID_MAGIC = -3
VTLK_ERR_INVALID_HEADER_SIZE = -4
VTLK_ERR_UNSUPPORTED_VERSION = -5
VTLK_ERR_CRC_MISMATCH = -6
VTLK_ERR_INVALID_PARAM = -7
VTLK_ERR_FULL = -8
VTLK_ERR_EMPTY = -9
VTLK_ERR_SHM = -10
VTLK_ERR_BUSY = -11


class VectLinkError(Exception):
    """Base exception for Vect-Link C API errors."""

    def __init__(self, code: int, message: str = "") -> None:
        self.code = int(code)
        super().__init__(message or f"Vect-Link error ({self.code})")


class BufferTooSmallError(VectLinkError):
    pass


class CrcMismatchError(VectLinkError):
    pass


class InvalidParamError(VectLinkError):
    pass


class RingFullError(VectLinkError):
    pass


class RingEmptyError(VectLinkError):
    pass


class ShmError(VectLinkError):
    pass


class RingBusyError(VectLinkError):
    pass


_ERROR_MAP = {
    VTLK_ERR_BUFFER_TOO_SMALL: (BufferTooSmallError, "Buffer too small"),
    VTLK_ERR_CRC_MISMATCH: (CrcMismatchError, "CRC32C checksum mismatch"),
    VTLK_ERR_INVALID_PARAM: (InvalidParamError, "Invalid parameter or argument"),
    VTLK_ERR_FULL: (RingFullError, "Ring buffer is full"),
    VTLK_ERR_EMPTY: (RingEmptyError, "Ring buffer is empty"),
    VTLK_ERR_SHM: (ShmError, "Shared memory operation failed"),
    VTLK_ERR_BUSY: (RingBusyError, "Ring slot is already in flight (busy)"),
}


def check_status(code: int, context: str = "") -> None:
    """Raise a typed VectLinkError if code != VTLK_OK."""
    if code == VTLK_OK:
        return
    exc_cls, default_msg = _ERROR_MAP.get(code, (VectLinkError, "Vect-Link error"))
    msg = f"{context}: {default_msg} (code={code})" if context else f"{default_msg} (code={code})"
    raise exc_cls(code, msg)


class VtlkHeader(ctypes.Structure):
    _fields_ = [
        ("magic", ctypes.c_uint8 * 4),
        ("header_bytes", ctypes.c_uint16),
        ("version", ctypes.c_uint16),
        ("flags", ctypes.c_uint32),
        ("model_family_hash", ctypes.c_uint64),
        ("d_model", ctypes.c_uint32),
        ("n_tokens", ctypes.c_uint32),
        ("quant_scheme", ctypes.c_uint8),
        ("layout", ctypes.c_uint8),
        ("codec", ctypes.c_uint8),
        ("reserved0", ctypes.c_uint8),
        ("payload_bytes", ctypes.c_uint32),
        ("raw_bytes", ctypes.c_uint32),
        ("sequence", ctypes.c_uint64),
        ("payload_crc32c", ctypes.c_uint32),
        ("layer_index", ctypes.c_uint16),
        ("pos_offset", ctypes.c_uint32),
        ("reserved1", ctypes.c_uint16),
        ("header_crc32c", ctypes.c_uint32),
    ]


class VtlkRing(ctypes.Structure):
    _fields_ = [
        ("base", ctypes.POINTER(ctypes.c_uint8)),
        ("total_bytes", ctypes.c_size_t),
        ("capacity", ctypes.c_uint32),
        ("slot_size", ctypes.c_uint32),
        ("max_payload_bytes", ctypes.c_uint32),
        ("cached_read_seq", ctypes.c_uint64),
        ("cached_write_seq", ctypes.c_uint64),
        ("write_in_flight", ctypes.c_int),
        ("write_slot_index", ctypes.c_uint32),
        ("write_seq_pending", ctypes.c_uint64),
        ("read_in_flight", ctypes.c_int),
        ("read_slot_index", ctypes.c_uint32),
        ("read_seq_pending", ctypes.c_uint64),
    ]


class VtlkShm(ctypes.Structure):
    _fields_ = [
        ("addr", ctypes.c_void_p),
        ("size", ctypes.c_size_t),
        ("name", ctypes.c_char * 256),
        ("is_owner", ctypes.c_int),
        ("handle" if sys.platform == "win32" else "fd",
         ctypes.c_void_p if sys.platform == "win32" else ctypes.c_int),
    ]


class VtlkWriteSlot(ctypes.Structure):
    _fields_ = [
        ("payload", ctypes.c_void_p),
        ("capacity", ctypes.c_size_t),
        ("slot_index", ctypes.c_uint32),
    ]


class VtlkReadSlot(ctypes.Structure):
    _fields_ = [
        ("header", VtlkHeader),
        ("payload", ctypes.c_void_p),
        ("bytes", ctypes.c_size_t),
        ("slot_index", ctypes.c_uint32),
    ]


def _candidate_lib_names() -> tuple[str, ...]:
    if sys.platform == "win32":
        return ("vectlink.dll", "libvectlink.dll")
    if sys.platform == "darwin":
        return ("libvectlink.dylib",)
    return ("libvectlink.so",)


def find_vectlink_library() -> str:
    names = _candidate_lib_names()
    searched: List[str] = []

    env_path = os.environ.get("VTLK_LIB_PATH")
    if env_path:
        p = Path(env_path).expanduser()
        if p.is_dir():
            for name in names:
                cand = p / name
                searched.append(str(cand))
                if cand.is_file():
                    return str(cand)
        else:
            searched.append(str(p))
            if p.is_file():
                return str(p)
        raise FileNotFoundError(
            f"VTLK_LIB_PATH={env_path!r} did not resolve to a valid library on "
            f"{sys.platform} ({platform.machine()}). Checked: {searched}"
        )

    pkg_dir = Path(__file__).resolve().parent
    repo_root = pkg_dir.parent.parent
    search_dirs = [
        pkg_dir,
        pkg_dir.parent,
        repo_root / "build",
        repo_root / "build" / "Release",
        repo_root / "build" / "Debug",
        repo_root / "build-llama",
        repo_root / "build-shared",
        repo_root / "out" / "build",
    ]
    for d in search_dirs:
        for name in names:
            cand = d / name
            searched.append(str(cand))
            if cand.is_file():
                return str(cand)

    sys_found = ctypes.util.find_library("vectlink")
    if sys_found:
        return sys_found

    searched_str = "\n  - ".join(searched)
    raise FileNotFoundError(
        f"Could not locate Vect-Link shared library ({', '.join(names)}) for "
        f"platform={sys.platform!r}, arch={platform.machine()!r}.\n"
        f"Searched paths:\n  - {searched_str}\n"
        f"Build it with: cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && "
        f"cmake --build build --target vectlink_shared\n"
        f"Or set VTLK_LIB_PATH=/path/to/{names[0]}"
    )


def _bind_library(path: str) -> ctypes.CDLL:
    lib = ctypes.CDLL(path)

    # Protocol
    lib.vtlk_header_init_defaults.argtypes = [ctypes.POINTER(VtlkHeader)]
    lib.vtlk_header_init_defaults.restype = None

    lib.vtlk_header_encode.argtypes = [
        ctypes.POINTER(VtlkHeader),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_size_t,
    ]
    lib.vtlk_header_encode.restype = ctypes.c_int

    lib.vtlk_header_decode.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_size_t,
        ctypes.POINTER(VtlkHeader),
    ]
    lib.vtlk_header_decode.restype = ctypes.c_int

    # Checksum
    lib.vtlk_crc32c.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    lib.vtlk_crc32c.restype = ctypes.c_uint32

    # Codec
    lib.vtlk_codec_encoded_bytes.argtypes = [
        ctypes.c_int,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_size_t),
    ]
    lib.vtlk_codec_encoded_bytes.restype = ctypes.c_int

    lib.vtlk_encode_tensor.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_size_t),
    ]
    lib.vtlk_encode_tensor.restype = ctypes.c_int

    lib.vtlk_decode_tensor.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_size_t,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
    ]
    lib.vtlk_decode_tensor.restype = ctypes.c_int

    lib.vtlk_cosine_similarity.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
    ]
    lib.vtlk_cosine_similarity.restype = ctypes.c_double

    lib.vtlk_max_abs_error.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
    ]
    lib.vtlk_max_abs_error.restype = ctypes.c_float

    # Ring & Slot Loan API
    lib.vtlk_ring_required_bytes.argtypes = [
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_uint32),
    ]
    lib.vtlk_ring_required_bytes.restype = ctypes.c_size_t

    lib.vtlk_ring_init.argtypes = [
        ctypes.POINTER(VtlkRing),
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_uint32,
        ctypes.c_uint32,
    ]
    lib.vtlk_ring_init.restype = ctypes.c_int

    lib.vtlk_ring_attach.argtypes = [
        ctypes.POINTER(VtlkRing),
        ctypes.c_void_p,
        ctypes.c_size_t,
    ]
    lib.vtlk_ring_attach.restype = ctypes.c_int

    lib.vtlk_ring_acquire_write.argtypes = [
        ctypes.POINTER(VtlkRing),
        ctypes.POINTER(VtlkWriteSlot),
    ]
    lib.vtlk_ring_acquire_write.restype = ctypes.c_int

    lib.vtlk_ring_commit_write.argtypes = [
        ctypes.POINTER(VtlkRing),
        ctypes.POINTER(VtlkWriteSlot),
        ctypes.POINTER(VtlkHeader),
        ctypes.c_size_t,
    ]
    lib.vtlk_ring_commit_write.restype = ctypes.c_int

    lib.vtlk_ring_abort_write.argtypes = [
        ctypes.POINTER(VtlkRing),
        ctypes.POINTER(VtlkWriteSlot),
    ]
    lib.vtlk_ring_abort_write.restype = ctypes.c_int

    lib.vtlk_ring_acquire_read.argtypes = [
        ctypes.POINTER(VtlkRing),
        ctypes.POINTER(VtlkReadSlot),
    ]
    lib.vtlk_ring_acquire_read.restype = ctypes.c_int

    lib.vtlk_ring_release_read.argtypes = [
        ctypes.POINTER(VtlkRing),
        ctypes.POINTER(VtlkReadSlot),
    ]
    lib.vtlk_ring_release_read.restype = ctypes.c_int

    lib.vtlk_ring_abort_read.argtypes = [
        ctypes.POINTER(VtlkRing),
        ctypes.POINTER(VtlkReadSlot),
    ]
    lib.vtlk_ring_abort_read.restype = ctypes.c_int

    lib.vtlk_ring_close.argtypes = [ctypes.POINTER(VtlkRing)]
    lib.vtlk_ring_close.restype = ctypes.c_int

    # OS Shared Memory
    lib.vtlk_shm_create.argtypes = [
        ctypes.POINTER(VtlkShm),
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    lib.vtlk_shm_create.restype = ctypes.c_int

    lib.vtlk_shm_open.argtypes = [
        ctypes.POINTER(VtlkShm),
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    lib.vtlk_shm_open.restype = ctypes.c_int

    lib.vtlk_shm_close.argtypes = [ctypes.POINTER(VtlkShm)]
    lib.vtlk_shm_close.restype = None

    lib.vtlk_shm_unlink.argtypes = [ctypes.c_char_p]
    lib.vtlk_shm_unlink.restype = ctypes.c_int

    return lib


_LIB: Optional[ctypes.CDLL] = None


def get_lib() -> ctypes.CDLL:
    global _LIB
    if _LIB is None:
        _LIB = _bind_library(find_vectlink_library())
    return _LIB
