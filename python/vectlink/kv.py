"""Pythonic zero-copy CoW KV-cache arena, Snapshot, Fork, and Restore wrappers."""

from __future__ import annotations

import ctypes
import weakref
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from ._ffi import (
    VTLK_ERR_EMPTY,
    VTLK_KV_DTYPE_BF16,
    VTLK_KV_DTYPE_FP16,
    VTLK_KV_DTYPE_FP32,
    VTLK_KV_DTYPE_Q4_0,
    VTLK_KV_DTYPE_Q8_0,
    VTLK_KV_DTYPE_RAW_LLAMA,
    VTLK_KV_MAX_BRANCH_BLOCKS,
    VtlkKvArena,
    VtlkKvArenaStats,
    VtlkKvBlockHeader,
    VtlkKvBranch,
    VtlkKvIdentity,
    check_status,
    get_lib,
)


class KVDtype(IntEnum):
    """KV-cache tensor element dtype stored in SHM blocks."""

    FP16 = VTLK_KV_DTYPE_FP16
    BF16 = VTLK_KV_DTYPE_BF16
    Q8_0 = VTLK_KV_DTYPE_Q8_0
    Q4_0 = VTLK_KV_DTYPE_Q4_0
    FP32 = VTLK_KV_DTYPE_FP32
    RAW_LLAMA = VTLK_KV_DTYPE_RAW_LLAMA


@dataclass(frozen=True)
class KVIdentity:
    """Content-addressed and config-validated identity of a KV block or branch."""

    token_prefix_hash: int = 0
    model_arch_hash: int = 0
    rope_config_hash: int = 0
    lora_adapter_hash: int = 0
    pos_offset: int = 0
    token_count: int = 0
    kv_dtype: Union[KVDtype, int] = KVDtype.FP16
    layer_count: int = 0
    head_dim: int = 0
    kv_head_count: int = 0

    def to_c(self) -> VtlkKvIdentity:
        return VtlkKvIdentity(
            token_prefix_hash=int(self.token_prefix_hash),
            model_arch_hash=int(self.model_arch_hash),
            rope_config_hash=int(self.rope_config_hash),
            lora_adapter_hash=int(self.lora_adapter_hash),
            pos_offset=int(self.pos_offset),
            token_count=int(self.token_count),
            kv_dtype=int(self.kv_dtype),
            layer_count=int(self.layer_count),
            head_dim=int(self.head_dim),
            kv_head_count=int(self.kv_head_count),
        )

    @classmethod
    def from_c(cls, c_id: VtlkKvIdentity) -> "KVIdentity":
        return cls(
            token_prefix_hash=int(c_id.token_prefix_hash),
            model_arch_hash=int(c_id.model_arch_hash),
            rope_config_hash=int(c_id.rope_config_hash),
            lora_adapter_hash=int(c_id.lora_adapter_hash),
            pos_offset=int(c_id.pos_offset),
            token_count=int(c_id.token_count),
            kv_dtype=KVDtype(int(c_id.kv_dtype))
            if int(c_id.kv_dtype) in KVDtype._value2member_map_
            else int(c_id.kv_dtype),
            layer_count=int(c_id.layer_count),
            head_dim=int(c_id.head_dim),
            kv_head_count=int(c_id.kv_head_count),
        )


@dataclass(frozen=True)
class KVArenaStats:
    """Atomic telemetry snapshot of a KVArena."""

    num_blocks: int
    active_blocks: int
    cached_blocks: int
    free_blocks: int
    hash_cap: int
    block_size_bytes: int
    total_bytes: int
    cache_hits: int
    cache_misses: int
    evictions: int

    @classmethod
    def from_c(cls, st: VtlkKvArenaStats) -> "KVArenaStats":
        return cls(
            num_blocks=int(st.num_blocks),
            active_blocks=int(st.active_blocks),
            cached_blocks=int(st.cached_blocks),
            free_blocks=int(st.free_blocks),
            hash_cap=int(st.hash_cap),
            block_size_bytes=int(st.block_size_bytes),
            total_bytes=int(st.total_bytes),
            cache_hits=int(st.cache_hits),
            cache_misses=int(st.cache_misses),
            evictions=int(st.evictions),
        )


@dataclass(frozen=True)
class Snapshot:
    """Immutable CoW snapshot metadata registered in SHM."""

    snapshot_id: int
    parent_snapshot_id: int
    block_count: int
    total_tokens: int
    block_ids: Tuple[int, ...]
    identity: KVIdentity


class KVBlockView:
    """Zero-copy memoryview wrapper over a single SHM KV block.

    Guards against dangling slices / NumPy views on release() or __exit__().
    """

    def __init__(
        self,
        arena: "KVArena",
        block_id: int,
        addr: int,
        size: int,
        *,
        readonly: bool = False,
        release_on_exit: bool = False,
    ) -> None:
        self._arena = arena
        self._block_id = int(block_id)
        self._size = int(size)
        self._readonly = readonly
        self._release_on_exit = release_on_exit
        self._lib = get_lib()
        self._released = False

        if self._size > 0:
            raw_arr = (ctypes.c_uint8 * self._size).from_address(int(addr))
            self._raw_ref: Optional[weakref.ref[Any]] = weakref.ref(raw_arr)
            mv = memoryview(raw_arr).cast("B")
            self._mv: Optional[memoryview] = mv.toreadonly() if readonly else mv
            del raw_arr
        else:
            self._raw_ref = None
            self._mv = memoryview(b"") if readonly else memoryview(bytearray())

    @property
    def block_id(self) -> int:
        return self._block_id

    @property
    def size(self) -> int:
        return self._size

    @property
    def buffer(self) -> memoryview:
        if self._released or self._mv is None:
            raise ValueError("KVBlockView has already been released")
        return self._mv

    def write_bytes(self, data: Union[bytes, bytearray, memoryview], offset: int = 0) -> int:
        if self._readonly:
            raise TypeError("Cannot write to a read-only KVBlockView")
        buf = self.buffer
        src = memoryview(data).cast("B")
        n = src.nbytes
        if offset < 0 or offset + n > len(buf):
            raise ValueError(
                f"Write range [{offset}, {offset + n}) exceeds block size ({len(buf)} B)"
            )
        buf[offset : offset + n] = src
        return n

    def seal(self, identity: KVIdentity) -> None:
        if self._released:
            raise ValueError("KVBlockView has already been released")
        c_id = identity.to_c()
        rc = self._lib.vtlk_kv_block_seal(
            ctypes.byref(self._arena._arena),
            self._block_id,
            ctypes.byref(c_id),
        )
        check_status(rc, "vtlk_kv_block_seal")

    def to_numpy(self, dtype: Any = None, shape: Any = None, *, copy: bool = True) -> Any:
        import numpy as np

        if dtype is None:
            dtype = np.uint8
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
                "Cannot release KVBlockView while a slice or NumPy/ctypes view of block.buffer is still alive"
            )
        self._released = True

    def release(self) -> None:
        if self._released:
            return
        self._release_mv()
        if self._release_on_exit:
            rc = self._lib.vtlk_kv_block_release(
                ctypes.byref(self._arena._arena),
                self._block_id,
            )
            check_status(rc, "vtlk_kv_block_release")

    def __enter__(self) -> "KVBlockView":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.release()


class Branch:
    """Context-managed CoW branch referencing retained SHM KV blocks."""

    def __init__(
        self,
        arena: "KVArena",
        c_branch: VtlkKvBranch,
        *,
        block_ptrs: Optional[List[int]] = None,
    ) -> None:
        self._arena = arena
        self._branch = c_branch
        self._lib = get_lib()
        self._released = False
        self._block_ptrs: Optional[List[int]] = block_ptrs
        self._views: Dict[int, Tuple[memoryview, weakref.ref[Any]]] = {}

    @property
    def parent_snapshot_id(self) -> int:
        return int(self._branch.parent_snapshot_id)

    @property
    def block_count(self) -> int:
        return int(self._branch.block_count)

    @property
    def total_tokens(self) -> int:
        return int(self._branch.total_tokens)

    @property
    def block_ids(self) -> Tuple[int, ...]:
        n = int(self._branch.block_count)
        return tuple(int(self._branch.block_ids[i]) for i in range(n))

    @property
    def identity(self) -> KVIdentity:
        return KVIdentity(
            model_arch_hash=int(self._branch.model_arch_hash),
            rope_config_hash=int(self._branch.rope_config_hash),
            lora_adapter_hash=int(self._branch.lora_adapter_hash),
            token_count=int(self._branch.total_tokens),
            kv_dtype=KVDtype(int(self._branch.kv_dtype))
            if int(self._branch.kv_dtype) in KVDtype._value2member_map_
            else int(self._branch.kv_dtype),
            layer_count=int(self._branch.layer_count),
            head_dim=int(self._branch.head_dim),
            kv_head_count=int(self._branch.kv_head_count),
        )

    @property
    def released(self) -> bool:
        return self._released

    def append_block(self, block_id: int) -> None:
        if self._released:
            raise ValueError("Branch has already been released")
        rc = self._lib.vtlk_kv_branch_append_block(
            ctypes.byref(self._arena._arena),
            ctypes.byref(self._branch),
            int(block_id),
        )
        check_status(rc, "vtlk_kv_branch_append_block")
        self._block_ptrs = None

    def block_view(self, index: int) -> memoryview:
        """Return a zero-copy read-only memoryview for block at `index` in this branch."""
        if self._released:
            raise ValueError("Branch has already been released")
        n = int(self._branch.block_count)
        if index < 0:
            index += n
        if index < 0 or index >= n:
            raise IndexError(f"Block index {index} out of range [0, {n})")

        if index in self._views:
            mv, _ = self._views[index]
            return mv

        block_id = int(self._branch.block_ids[index])
        if self._block_ptrs is not None and index < len(self._block_ptrs):
            addr = int(self._block_ptrs[index])
        else:
            addr = self._lib.vtlk_kv_block_data(ctypes.byref(self._arena._arena), block_id)
        if not addr:
            raise ValueError(f"Null SHM pointer for block_id={block_id}")

        size = int(self._arena.block_size_bytes)
        raw_arr = (ctypes.c_uint8 * size).from_address(int(addr))
        raw_ref = weakref.ref(raw_arr)
        mv = memoryview(raw_arr).cast("B").toreadonly()
        del raw_arr
        self._views[index] = (mv, raw_ref)
        return mv

    def block_views(self) -> List[memoryview]:
        """Return zero-copy read-only memoryviews for all blocks in this branch."""
        return [self.block_view(i) for i in range(self.block_count)]

    def create_snapshot(self) -> Snapshot:
        if self._released:
            raise ValueError("Branch has already been released")
        return self._arena.create_snapshot(self)

    def _release_views(self) -> None:
        dangling = False
        for idx in list(self._views.keys()):
            mv, raw_ref = self._views.pop(idx)
            mv.release()
            if raw_ref() is not None:
                dangling = True
        if dangling:
            raise BufferError(
                "Cannot release Branch while a slice or NumPy/ctypes view of a block memoryview is still alive"
            )

    def release(self) -> None:
        if self._released:
            return
        self._release_views()
        rc = self._lib.vtlk_kv_branch_release(
            ctypes.byref(self._arena._arena),
            ctypes.byref(self._branch),
        )
        check_status(rc, "vtlk_kv_branch_release")
        self._released = True

    def __enter__(self) -> "Branch":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.release()

    def __del__(self) -> None:
        try:
            if not getattr(self, "_released", True):
                self._views.clear()
                self._lib.vtlk_kv_branch_release(
                    ctypes.byref(self._arena._arena),
                    ctypes.byref(self._branch),
                )
                self._released = True
        except Exception:
            pass


class KVArena:
    """Shared-memory paged KV-cache arena with content-addressed deduplication and CoW forks."""

    def __init__(self, name: str, arena: VtlkKvArena, is_owner: bool) -> None:
        self.name = name
        self._arena = arena
        self._is_owner = is_owner
        self._lib = get_lib()
        self._closed = False

    @property
    def num_blocks(self) -> int:
        return int(self._arena.num_blocks)

    @property
    def block_size_bytes(self) -> int:
        return int(self._arena.block_size_bytes)

    @property
    def total_bytes(self) -> int:
        return int(self._arena.total_bytes)

    @staticmethod
    def required_bytes(num_blocks: int, block_size_bytes: int) -> int:
        lib = get_lib()
        return int(lib.vtlk_kv_arena_required_bytes(int(num_blocks), int(block_size_bytes)))

    @staticmethod
    def hash_tokens(token_ids: Sequence[int], prev_hash: int = 0) -> int:
        lib = get_lib()
        n = len(token_ids)
        if n == 0:
            return int(lib.vtlk_kv_hash_tokens(int(prev_hash), None, 0))
        arr = (ctypes.c_uint32 * n)(*(int(t) & 0xFFFFFFFF for t in token_ids))
        return int(lib.vtlk_kv_hash_tokens(int(prev_hash), arr, n))

    @staticmethod
    def hash_bytes(data: Union[bytes, bytearray, memoryview]) -> int:
        lib = get_lib()
        mv = memoryview(data).cast("B")
        if mv.nbytes == 0:
            return int(lib.vtlk_kv_hash_bytes(None, 0))
        arr = (ctypes.c_uint8 * mv.nbytes).from_buffer_copy(mv)
        return int(lib.vtlk_kv_hash_bytes(ctypes.cast(arr, ctypes.c_void_p), mv.nbytes))

    @classmethod
    def create(cls, shm_name: str, num_blocks: int, block_size_bytes: int) -> "KVArena":
        lib = get_lib()
        name_b = shm_name.encode("utf-8")
        lib.vtlk_shm_unlink(name_b)
        arena = VtlkKvArena()
        rc = lib.vtlk_kv_arena_create(
            ctypes.byref(arena),
            name_b,
            int(num_blocks),
            int(block_size_bytes),
        )
        check_status(rc, f"vtlk_kv_arena_create({shm_name!r})")
        return cls(shm_name, arena, is_owner=True)

    @classmethod
    def open(cls, shm_name: str) -> "KVArena":
        lib = get_lib()
        arena = VtlkKvArena()
        rc = lib.vtlk_kv_arena_open(ctypes.byref(arena), shm_name.encode("utf-8"))
        check_status(rc, f"vtlk_kv_arena_open({shm_name!r})")
        return cls(shm_name, arena, is_owner=False)

    def stats(self) -> KVArenaStats:
        if self._closed:
            raise ValueError("KVArena is closed")
        st = VtlkKvArenaStats()
        rc = self._lib.vtlk_kv_arena_stats(ctypes.byref(self._arena), ctypes.byref(st))
        check_status(rc, "vtlk_kv_arena_stats")
        return KVArenaStats.from_c(st)

    def alloc_block(self, *, release_on_exit: bool = True) -> KVBlockView:
        if self._closed:
            raise ValueError("KVArena is closed")
        block_id = ctypes.c_uint32(0)
        rc = self._lib.vtlk_kv_block_alloc(ctypes.byref(self._arena), ctypes.byref(block_id))
        check_status(rc, "vtlk_kv_block_alloc")
        bid = int(block_id.value)
        addr = self._lib.vtlk_kv_block_data(ctypes.byref(self._arena), bid)
        return KVBlockView(
            self,
            bid,
            int(addr),
            self.block_size_bytes,
            readonly=False,
            release_on_exit=release_on_exit,
        )

    def open_block(
        self,
        block_id: int,
        *,
        readonly: bool = True,
        release_on_exit: bool = False,
    ) -> KVBlockView:
        if self._closed:
            raise ValueError("KVArena is closed")
        addr = self._lib.vtlk_kv_block_data(ctypes.byref(self._arena), int(block_id))
        if not addr:
            raise ValueError(f"Invalid block_id={block_id}")
        return KVBlockView(
            self,
            int(block_id),
            int(addr),
            self.block_size_bytes,
            readonly=readonly,
            release_on_exit=release_on_exit,
        )

    def seal_block(self, block_id: int, identity: KVIdentity) -> None:
        if self._closed:
            raise ValueError("KVArena is closed")
        c_id = identity.to_c()
        rc = self._lib.vtlk_kv_block_seal(
            ctypes.byref(self._arena),
            int(block_id),
            ctypes.byref(c_id),
        )
        check_status(rc, "vtlk_kv_block_seal")

    def lookup_block(self, identity: KVIdentity) -> Optional[int]:
        if self._closed:
            raise ValueError("KVArena is closed")
        c_id = identity.to_c()
        out_id = ctypes.c_uint32(0)
        rc = self._lib.vtlk_kv_block_lookup(
            ctypes.byref(self._arena),
            ctypes.byref(c_id),
            ctypes.byref(out_id),
        )
        if rc == VTLK_ERR_EMPTY:
            return None
        check_status(rc, "vtlk_kv_block_lookup")
        return int(out_id.value)

    def retain_block(self, block_id: int) -> None:
        if self._closed:
            raise ValueError("KVArena is closed")
        rc = self._lib.vtlk_kv_block_retain(ctypes.byref(self._arena), int(block_id))
        check_status(rc, "vtlk_kv_block_retain")

    def release_block(self, block_id: int) -> None:
        if self._closed:
            raise ValueError("KVArena is closed")
        rc = self._lib.vtlk_kv_block_release(ctypes.byref(self._arena), int(block_id))
        check_status(rc, "vtlk_kv_block_release")

    def block_header(self, block_id: int) -> VtlkKvBlockHeader:
        if self._closed:
            raise ValueError("KVArena is closed")
        ptr = self._lib.vtlk_kv_block_header(ctypes.byref(self._arena), int(block_id))
        if not ptr:
            raise ValueError(f"Invalid block_id={block_id}")
        return ptr.contents

    def new_branch(self, identity: Optional[KVIdentity] = None) -> Branch:
        if self._closed:
            raise ValueError("KVArena is closed")
        c_branch = VtlkKvBranch()
        if identity is not None:
            c_branch.model_arch_hash = int(identity.model_arch_hash)
            c_branch.rope_config_hash = int(identity.rope_config_hash)
            c_branch.lora_adapter_hash = int(identity.lora_adapter_hash)
            c_branch.kv_dtype = int(identity.kv_dtype)
            c_branch.layer_count = int(identity.layer_count)
            c_branch.head_dim = int(identity.head_dim)
            c_branch.kv_head_count = int(identity.kv_head_count)
        return Branch(self, c_branch)

    def create_snapshot(self, branch: Branch) -> Snapshot:
        if self._closed:
            raise ValueError("KVArena is closed")
        snap_id = ctypes.c_uint64(0)
        rc = self._lib.vtlk_kv_snapshot_create(
            ctypes.byref(self._arena),
            ctypes.byref(branch._branch),
            ctypes.byref(snap_id),
        )
        check_status(rc, "vtlk_kv_snapshot_create")
        return Snapshot(
            snapshot_id=int(snap_id.value),
            parent_snapshot_id=branch.parent_snapshot_id,
            block_count=branch.block_count,
            total_tokens=branch.total_tokens,
            block_ids=branch.block_ids,
            identity=branch.identity,
        )

    def fork(self, snapshot: Union[int, Snapshot]) -> Branch:
        if self._closed:
            raise ValueError("KVArena is closed")
        snap_id = snapshot.snapshot_id if isinstance(snapshot, Snapshot) else int(snapshot)
        c_branch = VtlkKvBranch()
        rc = self._lib.vtlk_kv_fork(
            ctypes.byref(self._arena),
            snap_id,
            ctypes.byref(c_branch),
        )
        check_status(rc, "vtlk_kv_fork")
        return Branch(self, c_branch)

    def restore(
        self,
        snapshot: Union[int, Snapshot],
        expected_identity: Optional[KVIdentity] = None,
    ) -> Branch:
        if self._closed:
            raise ValueError("KVArena is closed")
        snap_id = snapshot.snapshot_id if isinstance(snapshot, Snapshot) else int(snapshot)
        c_branch = VtlkKvBranch()
        c_ptrs = (ctypes.c_void_p * VTLK_KV_MAX_BRANCH_BLOCKS)()
        c_id = expected_identity.to_c() if expected_identity is not None else None
        rc = self._lib.vtlk_kv_restore(
            ctypes.byref(self._arena),
            snap_id,
            ctypes.byref(c_id) if c_id is not None else None,
            ctypes.byref(c_branch),
            c_ptrs,
            VTLK_KV_MAX_BRANCH_BLOCKS,
        )
        check_status(rc, "vtlk_kv_restore")
        n = int(c_branch.block_count)
        ptrs = [int(c_ptrs[i]) for i in range(n)]
        return Branch(self, c_branch, block_ptrs=ptrs)

    def unlink(self) -> None:
        self._lib.vtlk_shm_unlink(self.name.encode("utf-8"))

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._lib.vtlk_kv_arena_close(ctypes.byref(self._arena))

    def __del__(self) -> None:
        try:
            if not getattr(self, "_closed", True):
                self.close()
        except Exception:
            pass

    def __enter__(self) -> "KVArena":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
        if self._is_owner:
            self.unlink()
