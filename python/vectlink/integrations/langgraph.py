"""LangGraph Tree-of-Thought, parallel hypothesis branching, and persistent L3 checkpointer."""

from __future__ import annotations

import base64
import ctypes
import hashlib
import json
import pickle
import shutil
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

from .._ffi import (
    VTLK_ERR_BUFFER_TOO_SMALL,
    VTLK_ERR_NOT_FOUND,
    VTLK_KV_DAG_FLAG_BRANCH,
    VTLK_KV_DAG_FLAG_CHECKPOINT,
    VTLK_KV_DAG_FLAG_WRITES,
    VTLK_KV_DAG_STR_MAX,
    VTLK_OK,
    VtlkKvCheckpointMeta,
    VtlkKvDag,
    check_status,
    get_lib,
)
from ..kv import Branch, KVArena, KVIdentity, Snapshot

try:
    from langgraph.checkpoint.base import (  # type: ignore[import-not-found]
        BaseCheckpointSaver,
        Checkpoint,
        CheckpointMetadata,
        CheckpointTuple,
    )

    HAS_LANGGRAPH = True
except ImportError:
    HAS_LANGGRAPH = False
    Checkpoint = Dict[str, Any]  # type: ignore[misc,assignment]
    CheckpointMetadata = Dict[str, Any]  # type: ignore[misc,assignment]

    @dataclass
    class CheckpointTuple:  # type: ignore[no-redef]
        config: Dict[str, Any]
        checkpoint: Dict[str, Any]
        metadata: Dict[str, Any]
        parent_config: Optional[Dict[str, Any]] = None
        pending_writes: Optional[List[Tuple[str, str, Any]]] = None

    class _FallbackSerde:
        def dumps_typed(self, obj: Any) -> Tuple[str, bytes]:
            try:
                return ("json", json.dumps(obj).encode("utf-8"))
            except TypeError:
                return ("pickle", pickle.dumps(obj))

        def loads_typed(self, data: Tuple[str, bytes]) -> Any:
            kind, raw = data
            if kind == "json":
                return json.loads(raw.decode("utf-8"))
            if kind == "pickle":
                return pickle.loads(raw)
            raise ValueError(f"Unsupported serialization type: {kind!r}")

    class BaseCheckpointSaver:  # type: ignore[no-redef]
        """Minimal fallback base class when langgraph is not installed."""

        serde: Any = _FallbackSerde()

        def __init__(self, *, serde: Any = None) -> None:
            self.serde = serde or self.serde

        def get_next_version(self, current: Any, channel: Any) -> Any:
            del channel
            if current is None:
                return 1
            return current + 1


def _encode_c_str(value: str) -> bytes:
    raw = value.encode("utf-8")
    if len(raw) >= VTLK_KV_DAG_STR_MAX:
        raw = raw[: VTLK_KV_DAG_STR_MAX - 1]
    return raw


def _decode_c_str(raw: bytes) -> str:
    return raw.decode("utf-8", errors="replace")


def _make_write_ckpt_id(ckpt_id: str, task_id: str) -> str:
    candidate = f"{ckpt_id}__w__{task_id}"
    if len(candidate.encode("utf-8")) < VTLK_KV_DAG_STR_MAX:
        return candidate
    digest = hashlib.sha256(f"{ckpt_id}:{task_id}".encode("utf-8")).hexdigest()[:24]
    return f"w_{digest}"


class _CDagHandle:
    """Thin Python wrapper over C99 vtlk_kv_dag_t."""

    def __init__(self, l3_dir: Union[str, Path]) -> None:
        self._lib = get_lib()
        self._dag = VtlkKvDag()
        self._l3_dir = str(Path(l3_dir))
        st = self._lib.vtlk_kv_dag_open(
            ctypes.byref(self._dag),
            self._l3_dir.encode("utf-8"),
        )
        check_status(st, "vtlk_kv_dag_open")

    @property
    def l3_dir(self) -> str:
        return self._l3_dir

    def close(self) -> None:
        if getattr(self, "_dag", None) is not None and self._dag.is_open:
            self._lib.vtlk_kv_dag_close(ctypes.byref(self._dag))

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def put(
        self,
        *,
        branch_name: str,
        thread_id: str,
        checkpoint_id: str,
        parent_checkpoint_id: str = "",
        snapshot_id: int = 0,
        flags: int = VTLK_KV_DAG_FLAG_CHECKPOINT,
        payload: bytes = b"",
    ) -> None:
        meta = VtlkKvCheckpointMeta()
        meta.branch_name = _encode_c_str(branch_name)
        meta.thread_id = _encode_c_str(thread_id)
        meta.checkpoint_id = _encode_c_str(checkpoint_id)
        meta.parent_checkpoint_id = _encode_c_str(parent_checkpoint_id)
        meta.snapshot_id = int(snapshot_id)
        meta.created_seq = 0
        meta.payload_bytes = len(payload)
        meta.payload_crc32c = 0
        meta.flags = int(flags)

        if payload:
            buf = (ctypes.c_uint8 * len(payload)).from_buffer_copy(payload)
            ptr = ctypes.cast(buf, ctypes.c_void_p)
        else:
            buf = None
            ptr = ctypes.c_void_p(0)

        st = self._lib.vtlk_kv_dag_put(
            ctypes.byref(self._dag),
            ctypes.byref(meta),
            ptr,
            len(payload),
        )
        check_status(st, "vtlk_kv_dag_put")

    def get(
        self,
        thread_id: str,
        checkpoint_id: Optional[str] = None,
    ) -> Optional[Tuple[VtlkKvCheckpointMeta, bytes]]:
        meta = VtlkKvCheckpointMeta()
        out_len = ctypes.c_size_t(0)
        tid_b = _encode_c_str(thread_id)
        cid_b = _encode_c_str(checkpoint_id) if checkpoint_id else b""

        st = self._lib.vtlk_kv_dag_get(
            ctypes.byref(self._dag),
            tid_b,
            cid_b,
            ctypes.byref(meta),
            None,
            0,
            ctypes.byref(out_len),
        )
        if st == VTLK_ERR_NOT_FOUND:
            return None
        if st == VTLK_OK and out_len.value == 0:
            return meta, b""
        if st != VTLK_ERR_BUFFER_TOO_SMALL:
            check_status(st, "vtlk_kv_dag_get(size)")

        buf = (ctypes.c_uint8 * out_len.value)()
        st = self._lib.vtlk_kv_dag_get(
            ctypes.byref(self._dag),
            tid_b,
            cid_b,
            ctypes.byref(meta),
            ctypes.cast(buf, ctypes.c_void_p),
            out_len.value,
            ctypes.byref(out_len),
        )
        if st == VTLK_ERR_NOT_FOUND:
            return None
        check_status(st, "vtlk_kv_dag_get(read)")
        return meta, bytes(buf[: out_len.value])

    def list(self, thread_id: Optional[str] = None) -> List[VtlkKvCheckpointMeta]:
        count = ctypes.c_uint32(0)
        tid_b = _encode_c_str(thread_id) if thread_id else None
        st = self._lib.vtlk_kv_dag_list(
            ctypes.byref(self._dag),
            tid_b,
            None,
            0,
            ctypes.byref(count),
        )
        check_status(st, "vtlk_kv_dag_list(count)")
        if count.value == 0:
            return []

        arr = (VtlkKvCheckpointMeta * count.value)()
        out_count = ctypes.c_uint32(0)
        st = self._lib.vtlk_kv_dag_list(
            ctypes.byref(self._dag),
            tid_b,
            arr,
            count.value,
            ctypes.byref(out_count),
        )
        check_status(st, "vtlk_kv_dag_list(read)")
        return [arr[i] for i in range(out_count.value)]


@dataclass(frozen=True)
class BranchRecord:
    """Lineage node linking a logical checkpoint/branch ID to a SHM KVArena snapshot."""

    checkpoint_id: str
    thread_id: str
    parent_checkpoint_id: Optional[str]
    snapshot_id: int
    total_tokens: int
    block_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at_ns: int = 0


class VectLinkBranchStore:
    """Git-for-Context CoW branch store backed by KVArena (SHM) + vtlk_kv_dag_t (L3 SSD).

    All DAG metadata and branch records are persisted in C/L3 via vtlk_kv_dag_t;
    no volatile Python dictionaries are used as source of truth.
    """

    def __init__(
        self,
        arena: KVArena,
        l3_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        self._arena = arena
        self._owns_temp_dir = l3_dir is None
        if l3_dir is None:
            self._temp_dir_path: Optional[str] = tempfile.mkdtemp(prefix="vtlk_dag_")
            resolved_dir: Union[str, Path] = self._temp_dir_path
        else:
            self._temp_dir_path = None
            resolved_dir = l3_dir
        self._dag = _CDagHandle(resolved_dir)

    @property
    def arena(self) -> KVArena:
        return self._arena

    @property
    def l3_dir(self) -> str:
        return self._dag.l3_dir

    def close(self) -> None:
        self._dag.close()
        if self._owns_temp_dir and self._temp_dir_path is not None:
            shutil.rmtree(self._temp_dir_path, ignore_errors=True)
            self._temp_dir_path = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    @staticmethod
    def _encode_record_payload(
        *,
        total_tokens: int,
        block_count: int,
        metadata: Optional[Dict[str, Any]],
        created_at_ns: int,
    ) -> bytes:
        doc = {
            "total_tokens": int(total_tokens),
            "block_count": int(block_count),
            "metadata": dict(metadata) if metadata is not None else {},
            "created_at_ns": int(created_at_ns),
        }
        return json.dumps(doc, default=str).encode("utf-8")

    @staticmethod
    def _decode_record(meta: VtlkKvCheckpointMeta, payload: bytes) -> BranchRecord:
        doc = json.loads(payload.decode("utf-8")) if payload else {}
        parent_str = _decode_c_str(meta.parent_checkpoint_id)
        raw_meta = doc.get("metadata", {})
        decoded_meta = dict(raw_meta) if isinstance(raw_meta, dict) else {}
        return BranchRecord(
            checkpoint_id=_decode_c_str(meta.checkpoint_id),
            thread_id=_decode_c_str(meta.thread_id),
            parent_checkpoint_id=parent_str if parent_str else None,
            snapshot_id=int(meta.snapshot_id),
            total_tokens=int(doc.get("total_tokens", 0)),
            block_count=int(doc.get("block_count", 0)),
            metadata=decoded_meta,
            created_at_ns=int(doc.get("created_at_ns", 0)),
        )

    def _find_meta_by_checkpoint_id(
        self, checkpoint_id: str
    ) -> Optional[VtlkKvCheckpointMeta]:
        for m in self._dag.list(None):
            if (m.flags & VTLK_KV_DAG_FLAG_WRITES) != 0:
                continue
            if _decode_c_str(m.checkpoint_id) == checkpoint_id:
                return m
        return None

    def _register_record(
        self,
        *,
        checkpoint_id: str,
        thread_id: str,
        parent_checkpoint_id: Optional[str],
        snap: Snapshot,
        metadata: Optional[Dict[str, Any]],
    ) -> BranchRecord:
        created_ns = time.time_ns()
        payload = self._encode_record_payload(
            total_tokens=snap.total_tokens,
            block_count=snap.block_count,
            metadata=metadata,
            created_at_ns=created_ns,
        )
        self._dag.put(
            branch_name="",
            thread_id=thread_id,
            checkpoint_id=checkpoint_id,
            parent_checkpoint_id=parent_checkpoint_id or "",
            snapshot_id=snap.snapshot_id,
            flags=VTLK_KV_DAG_FLAG_BRANCH,
            payload=payload,
        )
        return BranchRecord(
            checkpoint_id=checkpoint_id,
            thread_id=thread_id,
            parent_checkpoint_id=parent_checkpoint_id,
            snapshot_id=snap.snapshot_id,
            total_tokens=snap.total_tokens,
            block_count=snap.block_count,
            metadata=dict(metadata) if metadata is not None else {},
            created_at_ns=created_ns,
        )

    def commit_root(
        self,
        thread_id: str,
        branch: Branch,
        *,
        checkpoint_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Snapshot a root KV branch and persist its DAG record to L3."""
        cid = checkpoint_id or f"ckpt-{uuid.uuid4().hex[:12]}"
        if self._dag.get(thread_id, cid) is not None:
            raise ValueError(f"Duplicate checkpoint_id: {cid!r}")
        snap = self._arena.create_snapshot(branch)
        self._register_record(
            checkpoint_id=cid,
            thread_id=thread_id,
            parent_checkpoint_id=None,
            snap=snap,
            metadata=metadata,
        )
        return cid

    def commit_branch(
        self,
        parent_checkpoint_id: str,
        branch: Branch,
        *,
        thread_id: Optional[str] = None,
        checkpoint_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Snapshot a forked/extended KV branch as a child of `parent_checkpoint_id`."""
        parent = self.get_record(parent_checkpoint_id)
        cid = checkpoint_id or f"ckpt-{uuid.uuid4().hex[:12]}"
        target_thread = thread_id or parent.thread_id
        if self._dag.get(target_thread, cid) is not None:
            raise ValueError(f"Duplicate checkpoint_id: {cid!r}")
        snap = self._arena.create_snapshot(branch)
        self._register_record(
            checkpoint_id=cid,
            thread_id=target_thread,
            parent_checkpoint_id=parent_checkpoint_id,
            snap=snap,
            metadata=metadata,
        )
        return cid

    def fork_branch(self, checkpoint_id: str) -> Branch:
        """Return a zero-copy CoW Branch forked from `checkpoint_id`."""
        rec = self.get_record(checkpoint_id)
        return self._arena.fork(rec.snapshot_id)

    def restore_branch(
        self,
        checkpoint_id: str,
        expected_identity: Optional[KVIdentity] = None,
    ) -> Branch:
        """Validate model/RoPE/LoRA identity and restore a zero-copy Branch."""
        rec = self.get_record(checkpoint_id)
        return self._arena.restore(rec.snapshot_id, expected_identity)

    def get_record(self, checkpoint_id: str) -> BranchRecord:
        meta = self._find_meta_by_checkpoint_id(checkpoint_id)
        if meta is None:
            raise KeyError(f"Unknown checkpoint_id: {checkpoint_id!r}")
        res = self._dag.get(_decode_c_str(meta.thread_id), checkpoint_id)
        if res is None:
            raise KeyError(f"Unknown checkpoint_id: {checkpoint_id!r}")
        return self._decode_record(res[0], res[1])

    def get_snapshot_id(self, checkpoint_id: str) -> int:
        return self.get_record(checkpoint_id).snapshot_id

    def list_children(self, checkpoint_id: str) -> List[str]:
        if self._find_meta_by_checkpoint_id(checkpoint_id) is None:
            raise KeyError(f"Unknown checkpoint_id: {checkpoint_id!r}")
        metas = self._dag.list(None)
        children = [
            m
            for m in reversed(metas)
            if (m.flags & VTLK_KV_DAG_FLAG_WRITES) == 0
            and _decode_c_str(m.parent_checkpoint_id) == checkpoint_id
        ]
        return [_decode_c_str(m.checkpoint_id) for m in children]

    def get_lineage(self, checkpoint_id: str) -> List[BranchRecord]:
        """Return the branch lineage from root down to `checkpoint_id`."""
        chain: List[BranchRecord] = []
        cur: Optional[str] = checkpoint_id
        while cur is not None:
            rec = self.get_record(cur)
            chain.append(rec)
            cur = rec.parent_checkpoint_id
        chain.reverse()
        return chain

    def list_thread_branches(self, thread_id: str) -> List[BranchRecord]:
        records: List[BranchRecord] = []
        for m in reversed(self._dag.list(thread_id)):
            if (m.flags & VTLK_KV_DAG_FLAG_WRITES) != 0:
                continue
            res = self._dag.get(thread_id, _decode_c_str(m.checkpoint_id))
            if res is not None:
                records.append(self._decode_record(res[0], res[1]))
        return records

    def latest_checkpoint_id(self, thread_id: str) -> Optional[str]:
        for m in self._dag.list(thread_id):
            if (m.flags & VTLK_KV_DAG_FLAG_WRITES) == 0:
                return _decode_c_str(m.checkpoint_id)
        return None


class VectLinkCheckpointer(BaseCheckpointSaver):
    """LangGraph BaseCheckpointSaver backed by C99 vtlk_kv_dag_t (L3 SSD) + KVArena (SHM).

    Can be constructed either from an existing `VectLinkBranchStore`, or directly
    via `VectLinkCheckpointer(l3_dir=..., arena=...)` / `VectLinkCheckpointer(l3_dir=...)`.
    All state is persisted in C/L3 so a fresh process reconstructs everything from disk.
    """

    def __init__(
        self,
        branch_store: Optional[VectLinkBranchStore] = None,
        *,
        l3_dir: Optional[Union[str, Path]] = None,
        arena: Optional[KVArena] = None,
        serde: Any = None,
    ) -> None:
        super().__init__(serde=serde)
        if branch_store is not None:
            self.branch_store: Optional[VectLinkBranchStore] = branch_store
            self._dag = branch_store._dag
            self._owns_dag = False
            self._owns_store = False
        elif l3_dir is not None:
            if arena is not None:
                self.branch_store = VectLinkBranchStore(arena, l3_dir=l3_dir)
                self._dag = self.branch_store._dag
                self._owns_dag = False
                self._owns_store = True
            else:
                self.branch_store = None
                self._dag = _CDagHandle(l3_dir)
                self._owns_dag = True
                self._owns_store = False
        else:
            raise ValueError("Either branch_store or l3_dir must be provided")

    def close(self) -> None:
        if getattr(self, "_owns_store", False) and self.branch_store is not None:
            self.branch_store.close()
        elif getattr(self, "_owns_dag", False):
            self._dag.close()

    def __enter__(self) -> "VectLinkCheckpointer":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        del exc_type, exc, tb
        self.close()

    @staticmethod
    def _extract_cfg(config: Dict[str, Any]) -> Tuple[str, str, Optional[str]]:
        cfg = config.get("configurable", {})
        thread_id = str(cfg.get("thread_id", "default"))
        checkpoint_ns = str(cfg.get("checkpoint_ns", ""))
        checkpoint_id = cfg.get("checkpoint_id")
        return (
            thread_id,
            checkpoint_ns,
            str(checkpoint_id) if checkpoint_id is not None else None,
        )

    def _dumps_typed_b64(self, obj: Any) -> List[str]:
        kind, raw = self.serde.dumps_typed(obj)
        return [kind, base64.b64encode(raw).decode("ascii")]

    def _loads_typed_b64(self, pair: Sequence[str]) -> Any:
        kind, b64 = pair[0], pair[1]
        return self.serde.loads_typed((kind, base64.b64decode(b64.encode("ascii"))))

    def put(
        self,
        config: Dict[str, Any],
        checkpoint: Dict[str, Any],
        metadata: Dict[str, Any],
        new_versions: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        del new_versions
        thread_id, checkpoint_ns, parent_id = self._extract_cfg(config)
        ckpt_id = str(checkpoint.get("id") or f"ckpt-{uuid.uuid4().hex[:12]}")

        ckpt_copy = dict(checkpoint)
        ckpt_copy["id"] = ckpt_id
        meta_copy = dict(metadata)

        snapshot_id = 0
        total_tokens = 0
        block_count = 0

        kv_branch = meta_copy.pop("kv_branch", None)
        if isinstance(kv_branch, Branch):
            if self.branch_store is None:
                raise RuntimeError("kv_branch provided but no KVArena/branch_store attached")
            snap = self.branch_store.arena.create_snapshot(kv_branch)
            snapshot_id = snap.snapshot_id
            total_tokens = snap.total_tokens
            block_count = snap.block_count
            meta_copy["kv_snapshot_id"] = snapshot_id
        elif "kv_snapshot_id" in meta_copy:
            snapshot_id = int(meta_copy["kv_snapshot_id"])

        envelope = {
            "checkpoint_ns": checkpoint_ns,
            "checkpoint": self._dumps_typed_b64(ckpt_copy),
            "metadata": self._dumps_typed_b64(meta_copy),
            "parent_checkpoint_id": parent_id,
            "total_tokens": total_tokens,
            "block_count": block_count,
            "created_at_ns": time.time_ns(),
        }
        payload = json.dumps(envelope).encode("utf-8")

        self._dag.put(
            branch_name=checkpoint_ns,
            thread_id=thread_id,
            checkpoint_id=ckpt_id,
            parent_checkpoint_id=parent_id or "",
            snapshot_id=snapshot_id,
            flags=VTLK_KV_DAG_FLAG_CHECKPOINT,
            payload=payload,
        )

        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": ckpt_id,
            }
        }

    def _load_pending_writes(
        self, thread_id: str, checkpoint_ns: str, ckpt_id: str
    ) -> List[Tuple[str, str, Any]]:
        metas = self._dag.list(thread_id)
        write_metas = [
            m
            for m in reversed(metas)
            if (m.flags & VTLK_KV_DAG_FLAG_WRITES) != 0
            and _decode_c_str(m.parent_checkpoint_id) == ckpt_id
            and _decode_c_str(m.branch_name) == checkpoint_ns
        ]
        result: List[Tuple[str, str, Any]] = []
        for wm in write_metas:
            res = self._dag.get(thread_id, _decode_c_str(wm.checkpoint_id))
            if res is None:
                continue
            items = json.loads(res[1].decode("utf-8"))
            for item in items:
                tid, channel, encoded_val = item[0], item[1], item[2]
                result.append((tid, channel, self._loads_typed_b64(encoded_val)))
        return result

    def _tuple_from_raw(
        self,
        thread_id: str,
        meta: VtlkKvCheckpointMeta,
        payload: bytes,
    ) -> CheckpointTuple:
        doc = json.loads(payload.decode("utf-8"))
        checkpoint_ns = str(doc.get("checkpoint_ns", _decode_c_str(meta.branch_name)))
        ckpt_id = _decode_c_str(meta.checkpoint_id)
        parent_id = doc.get("parent_checkpoint_id")
        if not parent_id:
            p_c = _decode_c_str(meta.parent_checkpoint_id)
            parent_id = p_c if p_c else None

        ckpt = self._loads_typed_b64(doc["checkpoint"])
        metadata = self._loads_typed_b64(doc["metadata"])

        cfg: Dict[str, Any] = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": ckpt_id,
            }
        }
        parent_cfg: Optional[Dict[str, Any]] = None
        if parent_id is not None:
            parent_cfg = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": parent_id,
                }
            }

        writes = self._load_pending_writes(thread_id, checkpoint_ns, ckpt_id)
        return CheckpointTuple(
            config=cfg,
            checkpoint=ckpt,
            metadata=metadata,
            parent_config=parent_cfg,
            pending_writes=writes,
        )

    def get_tuple(self, config: Dict[str, Any]) -> Optional[CheckpointTuple]:
        thread_id, checkpoint_ns, ckpt_id = self._extract_cfg(config)
        if ckpt_id is None:
            metas = self._dag.list(thread_id)
            target_id: Optional[str] = None
            for m in metas:
                if (m.flags & VTLK_KV_DAG_FLAG_WRITES) != 0:
                    continue
                if _decode_c_str(m.branch_name) == checkpoint_ns:
                    target_id = _decode_c_str(m.checkpoint_id)
                    break
            if target_id is None:
                return None
            ckpt_id = target_id

        res = self._dag.get(thread_id, ckpt_id)
        if res is None:
            return None
        if (res[0].flags & VTLK_KV_DAG_FLAG_WRITES) != 0:
            return None
        return self._tuple_from_raw(thread_id, res[0], res[1])

    def get(self, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        tup = self.get_tuple(config)
        return tup.checkpoint if tup is not None else None

    def list(
        self,
        config: Optional[Dict[str, Any]],
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        if config is None:
            metas = [
                m
                for m in self._dag.list(None)
                if (m.flags & VTLK_KV_DAG_FLAG_WRITES) == 0
            ]
        else:
            thread_id, checkpoint_ns, _ = self._extract_cfg(config)
            metas = [
                m
                for m in self._dag.list(thread_id)
                if (m.flags & VTLK_KV_DAG_FLAG_WRITES) == 0
                and _decode_c_str(m.branch_name) == checkpoint_ns
            ]

        before_id: Optional[str] = None
        if before is not None:
            _, _, before_id = self._extract_cfg(before)

        count = 0
        past_before = before_id is None
        for m in metas:
            cid = _decode_c_str(m.checkpoint_id)
            if not past_before:
                if cid == before_id:
                    past_before = True
                continue
            tid = _decode_c_str(m.thread_id)
            res = self._dag.get(tid, cid)
            if res is None:
                continue
            tup = self._tuple_from_raw(tid, res[0], res[1])
            if filter:
                if not all(tup.metadata.get(k) == v for k, v in filter.items()):
                    continue
            yield tup
            count += 1
            if limit is not None and count >= limit:
                break

    def put_writes(
        self,
        config: Dict[str, Any],
        writes: Sequence[Tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        del task_path
        thread_id, checkpoint_ns, ckpt_id = self._extract_cfg(config)
        if ckpt_id is None:
            raise ValueError("checkpoint_id is required for put_writes")

        write_ckpt_id = _make_write_ckpt_id(ckpt_id, task_id)
        existing: List[Any] = []
        prev = self._dag.get(thread_id, write_ckpt_id)
        if prev is not None:
            existing = json.loads(prev[1].decode("utf-8"))

        for channel, value in writes:
            existing.append([task_id, channel, self._dumps_typed_b64(value)])

        payload = json.dumps(existing).encode("utf-8")
        self._dag.put(
            branch_name=checkpoint_ns,
            thread_id=thread_id,
            checkpoint_id=write_ckpt_id,
            parent_checkpoint_id=ckpt_id,
            snapshot_id=0,
            flags=VTLK_KV_DAG_FLAG_WRITES,
            payload=payload,
        )

    def fork_from_checkpoint(self, checkpoint_id: str) -> Branch:
        """Convenience helper to fork a CoW KV Branch for parallel hypothesis evaluation."""
        if self.branch_store is None:
            raise RuntimeError("No branch_store/KVArena attached to checkpointer")
        return self.branch_store.fork_branch(checkpoint_id)

    def get_next_version(self, current: Any, channel: Any) -> Any:
        del channel
        if current is None:
            return 1
        if isinstance(current, int):
            return current + 1
        return int(current) + 1

    async def aput(
        self,
        config: Dict[str, Any],
        checkpoint: Dict[str, Any],
        metadata: Dict[str, Any],
        new_versions: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return self.put(config, checkpoint, metadata, new_versions)

    async def aget_tuple(self, config: Dict[str, Any]) -> Optional[CheckpointTuple]:
        return self.get_tuple(config)

    async def alist(
        self,
        config: Optional[Dict[str, Any]],
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
    ) -> Any:
        for item in self.list(config, filter=filter, before=before, limit=limit):
            yield item

    async def aput_writes(
        self,
        config: Dict[str, Any],
        writes: Sequence[Tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        self.put_writes(config, writes, task_id, task_path=task_path)
