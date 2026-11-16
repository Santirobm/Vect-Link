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
from .ring import PacketHeader, QuantScheme, ReadSlot, VectLinkRing, WriteSlot

__all__ = [
    "BufferTooSmallError",
    "CrcMismatchError",
    "InvalidParamError",
    "PacketHeader",
    "QuantScheme",
    "ReadSlot",
    "RingBusyError",
    "RingEmptyError",
    "RingFullError",
    "ShmError",
    "VectLinkError",
    "VectLinkRing",
    "WriteSlot",
]
