"""Pythonic zero-copy ring buffer, WriteSlot, and ReadSlot wrappers."""

from __future__ import annotations

import ctypes
import struct
import weakref
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, List, Optional

from ._ffi import (
    VTLK_RING_CTRL_SIZE,
    BufferTooSmallError,
    ShmError,
    VtlkHeader,
    VtlkRing,
    VtlkReadSlot,
    VtlkShm,
    VtlkWriteSlot,
    check_status,
    get_lib,
)


class QuantScheme(IntEnum):
    FP16 = 0
    INT4_BLOCK = 1
    INT4 = 1
    RVQ10 = 2
    RAW = 3


@dataclass(frozen=True)
class PacketHeader:
    n_tokens: int
    d_model: int
    quant_scheme: int
    model_family_hash: int
    flags: int
    pos_offset: int
    payload_bytes: int
    raw_bytes: int
    sequence: int
    payload_crc32c: int
    layer_index: int

    @classmethod
    def from_c(cls, hdr: VtlkHeader) -> "PacketHeader":
        return cls(
            n_tokens=int(hdr.n_tokens),
            d_model=int(hdr.d_model),
            quant_scheme=int(hdr.quant_scheme),
            model_family_hash=int(hdr.model_family_hash),
            flags=int(hdr.flags),
            pos_offset=int(hdr.pos_offset),
            payload_bytes=int(hdr.payload_bytes),
            raw_bytes=int(hdr.raw_bytes),
            sequence=int(hdr.sequence),
            payload_crc32c=int(hdr.payload_crc32c),
            layer_index=int(hdr.layer_index),
        )


def _as_float32_ptr(buf: Any, total_elems: int) -> ctypes.POINTER(ctypes.c_float):
    mv = memoryview(buf)
    if not mv.c_contiguous:
        raise ValueError("Tensor buffer must be C-contiguous")
    if mv.format != "f" or mv.itemsize != 4:
        raise TypeError(f"Expected float32 ('f') buffer, got format={mv.format!r}")
    if (mv.nbytes // 4) < total_elems:
        raise ValueError(
            f"Float32 buffer too small: has {mv.nbytes // 4} elements, need {total_elems}"
        )
    if not mv.readonly:
        arr = (ctypes.c_float * total_elems).from_buffer(mv)
    else:
        arr = (ctypes.c_float * total_elems).from_buffer_copy(mv)
    return ctypes.cast(arr, ctypes.POINTER(ctypes.c_float))


class WriteSlot:
    """Zero-copy mutable view of a loaned SHM ring write slot."""

    def __init__(self, ring: "VectLinkRing", slot: VtlkWriteSlot) -> None:
        self._ring = ring
        self._slot = slot
        self._lib = get_lib()
        cap = int(slot.capacity)
        if cap > 0:
            raw_arr = (ctypes.c_uint8 * cap).from_address(int(slot.payload))
            self._raw_ref: Optional[weakref.ref[Any]] = weakref.ref(raw_arr)
            self._mv: Optional[memoryview] = memoryview(raw_arr).cast("B")
            del raw_arr
        else:
            self._raw_ref = None
            self._mv = memoryview(bytearray())
        self._released = False
        self._borrowed_views: List[weakref.ref[Any]] = []
        self._pending_bytes: Optional[int] = None
        self._pending_n_tokens: int = 1
        self._pending_d_model: int = 1
        self._pending_quant: int = int(QuantScheme.RAW)
        self._pending_model_hash: int = 0
        self._pending_raw_bytes: Optional[int] = None

    @property
    def capacity(self) -> int:
        return int(self._slot.capacity)

    @property
    def buffer(self) -> memoryview:
        if self._released or self._mv is None:
            raise ValueError("WriteSlot has already been committed or aborted")
        return self._mv

    def write_bytes(self, data: bytes | bytearray | memoryview) -> int:
        buf = self.buffer
        src = memoryview(data)
        n = src.nbytes
        if n > len(buf):
            raise BufferTooSmallError(
                -2, f"Payload ({n} B) exceeds slot capacity ({len(buf)} B)"
            )
        buf[:n] = src.cast("B")
        self._pending_bytes = n
        self._pending_n_tokens = 1
        self._pending_d_model = 1
        self._pending_quant = int(QuantScheme.RAW)
        self._pending_raw_bytes = n
        return n

    def encode_tensor(
        self,
        float_buffer: Any,
        n_tokens: int,
        d_model: int,
        quant: int = QuantScheme.INT4_BLOCK,
        model_hash: int = 0,
    ) -> int:
        if self._released:
            raise ValueError("WriteSlot has already been committed or aborted")
        total_elems = int(n_tokens) * int(d_model)
        src_ptr = _as_float32_ptr(float_buffer, total_elems)
        dst_ptr = ctypes.cast(int(self._slot.payload), ctypes.POINTER(ctypes.c_uint8))
        written = ctypes.c_size_t(0)
        rc = self._lib.vtlk_encode_tensor(
            src_ptr,
            int(n_tokens),
            int(d_model),
            int(quant),
            dst_ptr,
            int(self._slot.capacity),
            ctypes.byref(written),
        )
        check_status(rc, "vtlk_encode_tensor")
        self._pending_bytes = int(written.value)
        self._pending_n_tokens = int(n_tokens)
        self._pending_d_model = int(d_model)
        self._pending_quant = int(quant)
        self._pending_model_hash = int(model_hash)
        self._pending_raw_bytes = total_elems * 4
        return int(written.value)

    def commit(
        self,
        payload_bytes: Optional[int] = None,
        *,
        n_tokens: Optional[int] = None,
        d_model: Optional[int] = None,
        quant: Optional[int] = None,
        model_hash: Optional[int] = None,
        flags: int = 0,
        pos_offset: int = 0,
        sequence: int = 0,
    ) -> None:
        if self._released:
            raise ValueError("WriteSlot has already been committed or aborted")

        nbytes = self._pending_bytes if payload_bytes is None else int(payload_bytes)
        if nbytes is None:
            raise ValueError("payload_bytes must be specified or set via write_bytes/encode_tensor")

        hdr = VtlkHeader()
        self._lib.vtlk_header_init_defaults(ctypes.byref(hdr))
        hdr.n_tokens = self._pending_n_tokens if n_tokens is None else int(n_tokens)
        hdr.d_model = self._pending_d_model if d_model is None else int(d_model)
        hdr.quant_scheme = (self._pending_quant if quant is None else int(quant)) & 0xFF
        hdr.model_family_hash = (
            self._pending_model_hash if model_hash is None else int(model_hash)
        )
        hdr.flags = int(flags)
        hdr.pos_offset = int(pos_offset)
        hdr.sequence = int(sequence)
        hdr.raw_bytes = (
            nbytes if self._pending_raw_bytes is None else int(self._pending_raw_bytes)
        )

        self._release_mv()
        rc = self._lib.vtlk_ring_commit_write(
            ctypes.byref(self._ring._ring),
            ctypes.byref(self._slot),
            ctypes.byref(hdr),
            nbytes,
        )
        check_status(rc, "vtlk_ring_commit_write")

    def abort(self) -> None:
        if self._released:
            return
        self._release_mv()
        rc = self._lib.vtlk_ring_abort_write(
            ctypes.byref(self._ring._ring),
            ctypes.byref(self._slot),
        )
        check_status(rc, "vtlk_ring_abort_write")

    def _release_mv(self) -> None:
        if self._mv is not None:
            self._mv.release()
            self._mv = None
        if self._raw_ref is not None and self._raw_ref() is not None:
            raise BufferError(
                "Cannot release WriteSlot while a slice or NumPy/ctypes view of slot.buffer is still alive"
            )
        self._released = True

    def __enter__(self) -> "WriteSlot":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._released:
            return
        if exc_type is not None or self._pending_bytes is None:
            self.abort()
        else:
            self.commit()


class ReadSlot:
    """Zero-copy read-only view of a loaned SHM ring read slot."""

    def __init__(self, ring: "VectLinkRing", slot: VtlkReadSlot) -> None:
        self._ring = ring
        self._slot = slot
        self._lib = get_lib()
        self._header = PacketHeader.from_c(slot.header)
        nbytes = int(slot.bytes)
        if nbytes > 0:
            raw_arr = (ctypes.c_uint8 * nbytes).from_address(int(slot.payload))
            self._raw_ref: Optional[weakref.ref[Any]] = weakref.ref(raw_arr)
            self._mv: Optional[memoryview] = memoryview(raw_arr).cast("B").toreadonly()
            del raw_arr
        else:
            self._raw_ref = None
            self._mv = memoryview(b"")
        self._released = False

    @property
    def header(self) -> PacketHeader:
        return self._header

    @property
    def bytes(self) -> int:
        return int(self._slot.bytes)

    @property
    def buffer(self) -> memoryview:
        if self._released or self._mv is None:
            raise ValueError("ReadSlot has already been released")
        return self._mv

    def decode_tensor(self, out_float_buffer: Any = None) -> Any:
        if self._released:
            raise ValueError("ReadSlot has already been released")
        n_tok = self._header.n_tokens
        d_mod = self._header.d_model
        total_elems = n_tok * d_mod

        allocated = out_float_buffer is None
        if allocated:
            backing = bytearray(total_elems * 4)
            out_mv = memoryview(backing).cast("f")
        else:
            out_mv = memoryview(out_float_buffer)
            if out_mv.readonly:
                raise ValueError("out_float_buffer must be writable")

        dst_ptr = _as_float32_ptr(out_mv, total_elems)
        src_ptr = ctypes.cast(int(self._slot.payload), ctypes.POINTER(ctypes.c_uint8))
        rc = self._lib.vtlk_decode_tensor(
            src_ptr,
            int(self._slot.bytes),
            n_tok,
            d_mod,
            int(self._header.quant_scheme),
            dst_ptr,
            total_elems,
        )
        check_status(rc, "vtlk_decode_tensor")
        return out_mv if allocated else out_float_buffer

    def to_numpy(self, dtype: Any = None, shape: Any = None, *, copy: bool = True) -> Any:
        """Return slot payload as a NumPy ndarray.

        By default (copy=True), returns an independent copy safe to use after
        exiting the ReadSlot context. With copy=False, returns a zero-copy
        view tied to the ReadSlot lifetime (must be deleted before release()).
        """
        import numpy as np

        if dtype is None:
            dtype = np.float32
        arr = np.frombuffer(self.buffer, dtype=dtype)
        if shape is not None:
            arr = arr.reshape(shape)
        return arr.copy() if copy else arr

    def _release_mv(self) -> None:
        if self._mv is not None:
            self._mv.release()
            self._mv = None
        if self._raw_ref is not None and self._raw_ref() is not None:
            raise BufferError(
                "Cannot release ReadSlot while a slice or NumPy/ctypes view of slot.buffer is still alive"
            )
        self._released = True

    def release(self) -> None:
        if self._released:
            return
        self._release_mv()
        rc = self._lib.vtlk_ring_release_read(
            ctypes.byref(self._ring._ring),
            ctypes.byref(self._slot),
        )
        check_status(rc, "vtlk_ring_release_read")

    def abort(self) -> None:
        if self._released:
            return
        self._release_mv()
        rc = self._lib.vtlk_ring_abort_read(
            ctypes.byref(self._ring._ring),
            ctypes.byref(self._slot),
        )
        check_status(rc, "vtlk_ring_abort_read")

    def __enter__(self) -> "ReadSlot":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.release()


class VectLinkRing:
    """Lock-free SPSC shared-memory ring buffer with zero-copy slot loans."""

    def __init__(self, name: str, shm: VtlkShm, ring: VtlkRing, is_owner: bool) -> None:
        self.name = name
        self._shm = shm
        self._ring = ring
        self._is_owner = is_owner
        self._lib = get_lib()
        self._closed = False

    @property
    def capacity(self) -> int:
        return int(self._ring.capacity)

    @property
    def max_payload_bytes(self) -> int:
        return int(self._ring.max_payload_bytes)

    @classmethod
    def create(
        cls,
        name: str,
        slot_capacity: int,
        slot_count: int = 8,
    ) -> "VectLinkRing":
        lib = get_lib()
        req = lib.vtlk_ring_required_bytes(int(slot_count), int(slot_capacity), None)
        if req == 0:
            raise ValueError(
                f"Invalid ring parameters: slot_count={slot_count} (must be power of 2), "
                f"slot_capacity={slot_capacity}"
            )
        name_b = name.encode("utf-8")
        lib.vtlk_shm_unlink(name_b)

        shm = VtlkShm()
        rc = lib.vtlk_shm_create(ctypes.byref(shm), name_b, req)
        check_status(rc, f"vtlk_shm_create({name!r})")

        ring = VtlkRing()
        rc = lib.vtlk_ring_init(
            ctypes.byref(ring),
            shm.addr,
            shm.size,
            int(slot_count),
            int(slot_capacity),
        )
        if rc != 0:
            lib.vtlk_shm_close(ctypes.byref(shm))
            lib.vtlk_shm_unlink(name_b)
            check_status(rc, "vtlk_ring_init")

        return cls(name, shm, ring, is_owner=True)

    @classmethod
    def open(cls, name: str) -> "VectLinkRing":
        lib = get_lib()
        name_b = name.encode("utf-8")

        probe = VtlkShm()
        rc = lib.vtlk_shm_open(ctypes.byref(probe), name_b, VTLK_RING_CTRL_SIZE)
        check_status(rc, f"vtlk_shm_open({name!r}, ctrl)")

        try:
            ctrl_bytes = (ctypes.c_uint8 * VTLK_RING_CTRL_SIZE).from_address(int(probe.addr))
            cap, _, max_pay = struct.unpack_from("<III", bytes(ctrl_bytes), 12)
        finally:
            lib.vtlk_shm_close(ctypes.byref(probe))

        full_bytes = lib.vtlk_ring_required_bytes(cap, max_pay, None)
        if full_bytes == 0:
            raise ShmError(-7, f"Invalid ring metadata in SHM {name!r}: cap={cap}, max_pay={max_pay}")

        shm = VtlkShm()
        rc = lib.vtlk_shm_open(ctypes.byref(shm), name_b, full_bytes)
        check_status(rc, f"vtlk_shm_open({name!r}, full)")

        ring = VtlkRing()
        rc = lib.vtlk_ring_attach(ctypes.byref(ring), shm.addr, shm.size)
        if rc != 0:
            lib.vtlk_shm_close(ctypes.byref(shm))
            check_status(rc, "vtlk_ring_attach")

        return cls(name, shm, ring, is_owner=False)

    def acquire_write(self) -> WriteSlot:
        if self._closed:
            raise ValueError("VectLinkRing is closed")
        slot = VtlkWriteSlot()
        rc = self._lib.vtlk_ring_acquire_write(ctypes.byref(self._ring), ctypes.byref(slot))
        check_status(rc, "vtlk_ring_acquire_write")
        return WriteSlot(self, slot)

    def acquire_read(self) -> ReadSlot:
        if self._closed:
            raise ValueError("VectLinkRing is closed")
        slot = VtlkReadSlot()
        rc = self._lib.vtlk_ring_acquire_read(ctypes.byref(self._ring), ctypes.byref(slot))
        check_status(rc, "vtlk_ring_acquire_read")
        return ReadSlot(self, slot)

    def unlink(self) -> None:
        self._lib.vtlk_shm_unlink(self.name.encode("utf-8"))

    def close(self) -> None:
        if self._closed:
            return
        rc = self._lib.vtlk_ring_close(ctypes.byref(self._ring))
        check_status(rc, "vtlk_ring_close")
        self._closed = True
        self._lib.vtlk_shm_close(ctypes.byref(self._shm))

    def __del__(self) -> None:
        try:
            if not getattr(self, "_closed", True):
                self.close()
        except Exception:
            pass

    def __enter__(self) -> "VectLinkRing":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
        if self._is_owner:
            self.unlink()
