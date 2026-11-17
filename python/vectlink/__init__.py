"""Zero-copy Python bindings for Vect-Link shared-memory IPC and tensor codecs."""

from ._ffi import (
    BufferTooSmallError,
    CrcMismatchError,
    InvalidParamError,
    RingBusyError,
    RingEmptyError,
    RingFullError,
    ShmError,
    VectLinkError,
)
from .kv import (
    Branch,
    KVArena,
    KVArenaStats,
    KVBlockView,
    KVDtype,
    KVIdentity,
    Snapshot,
)
from .ring import PacketHeader, QuantScheme, ReadSlot, VectLinkRing, WriteSlot

__all__ = [
    "Branch",
    "BufferTooSmallError",
    "CrcMismatchError",
    "InvalidParamError",
    "KVArena",
    "KVArenaStats",
    "KVBlockView",
    "KVDtype",
    "KVIdentity",
    "PacketHeader",
    "QuantScheme",
    "ReadSlot",
    "RingBusyError",
    "RingEmptyError",
    "RingFullError",
    "ShmError",
    "Snapshot",
    "VectLinkError",
    "VectLinkRing",
    "WriteSlot",
]
