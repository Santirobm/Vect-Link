"""LangGraph Tree-of-Thought & parallel hypothesis branching backed by Vect-Link KVArena."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

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

    class BaseCheckpointSaver:  # type: ignore[no-redef]
        """Minimal fallback base class when langgraph is not installed."""

        def __init__(self, *, serde: Any = None) -> None:
            self.serde = serde


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
    """Git-for-Context CoW branch store backed by a Vect-Link KVArena.

    Enables Tree-of-Thought / parallel hypothesis branching where a shared
    system/prefix KV-cache is snapshotted once in SHM and forked with zero
    tensor copies across concurrent or sequential agent branches.
    """

    def __init__(self, arena: KVArena) -> None:
        self._arena = arena
        self._records: Dict[str, BranchRecord] = {}
        self._children: Dict[str, List[str]] = {}
        self._thread_heads: Dict[str, str] = {}

    @property
    def arena(self) -> KVArena:
        return self._arena

    def _register_record(
        self,
        *,
        checkpoint_id: str,
        thread_id: str,
        parent_checkpoint_id: Optional[str],
        snap: Snapshot,
        metadata: Optional[Dict[str, Any]],
    ) -> BranchRecord:
        rec = BranchRecord(
            checkpoint_id=checkpoint_id,
            thread_id=thread_id,
            parent_checkpoint_id=parent_checkpoint_id,
            snapshot_id=snap.snapshot_id,
            total_tokens=snap.total_tokens,
            block_count=snap.block_count,
            metadata=dict(metadata) if metadata is not None else {},
            created_at_ns=time.time_ns(),
        )
        self._records[checkpoint_id] = rec
        self._children.setdefault(checkpoint_id, [])
        if parent_checkpoint_id is not None:
            self._children.setdefault(parent_checkpoint_id, []).append(checkpoint_id)
        self._thread_heads[thread_id] = checkpoint_id
        return rec

    def commit_root(
        self,
        thread_id: str,
        branch: Branch,
        *,
        checkpoint_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Snapshot a root KV branch (e.g., prefilled system prompt) and return its checkpoint_id."""
        cid = checkpoint_id or f"ckpt-{uuid.uuid4().hex[:12]}"
        if cid in self._records:
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
        if cid in self._records:
            raise ValueError(f"Duplicate checkpoint_id: {cid!r}")
        snap = self._arena.create_snapshot(branch)
        self._register_record(
            checkpoint_id=cid,
            thread_id=thread_id or parent.thread_id,
            parent_checkpoint_id=parent_checkpoint_id,
            snap=snap,
            metadata=metadata,
        )
        return cid

    def fork_branch(self, checkpoint_id: str) -> Branch:
        """Return a zero-copy CoW Branch forked from `checkpoint_id` (context-manager ready)."""
        rec = self.get_record(checkpoint_id)
        return self._arena.fork(rec.snapshot_id)

    def restore_branch(
        self,
        checkpoint_id: str,
        expected_identity: Optional[KVIdentity] = None,
    ) -> Branch:
        """Validate model/RoPE/LoRA identity and restore a zero-copy Branch from `checkpoint_id`."""
        rec = self.get_record(checkpoint_id)
        return self._arena.restore(rec.snapshot_id, expected_identity)

    def get_record(self, checkpoint_id: str) -> BranchRecord:
        if checkpoint_id not in self._records:
            raise KeyError(f"Unknown checkpoint_id: {checkpoint_id!r}")
        return self._records[checkpoint_id]

    def get_snapshot_id(self, checkpoint_id: str) -> int:
        return self.get_record(checkpoint_id).snapshot_id

    def list_children(self, checkpoint_id: str) -> List[str]:
        if checkpoint_id not in self._records:
            raise KeyError(f"Unknown checkpoint_id: {checkpoint_id!r}")
        return list(self._children.get(checkpoint_id, []))

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
        return [r for r in self._records.values() if r.thread_id == thread_id]

    def latest_checkpoint_id(self, thread_id: str) -> Optional[str]:
        return self._thread_heads.get(thread_id)


class VectLinkCheckpointer(BaseCheckpointSaver):
    """LangGraph-compatible checkpointer backed by VectLinkBranchStore.

    Works both when `langgraph` is installed (subclassing BaseCheckpointSaver)
    and in standalone mode using the built-in fallback base class.
    """

    def __init__(self, branch_store: VectLinkBranchStore, *, serde: Any = None) -> None:
        super().__init__(serde=serde)
        self.branch_store = branch_store
        # (thread_id, checkpoint_ns) -> ordered list of checkpoint_ids
        self._history: Dict[Tuple[str, str], List[str]] = {}
        # (thread_id, checkpoint_ns, checkpoint_id) -> CheckpointTuple
        self._tuples: Dict[Tuple[str, str, str], CheckpointTuple] = {}
        # (thread_id, checkpoint_ns, checkpoint_id) -> list of pending writes
        self._writes: Dict[Tuple[str, str, str], List[Tuple[str, str, Any]]] = {}

    @staticmethod
    def _extract_cfg(config: Dict[str, Any]) -> Tuple[str, str, Optional[str]]:
        cfg = config.get("configurable", {})
        thread_id = str(cfg.get("thread_id", "default"))
        checkpoint_ns = str(cfg.get("checkpoint_ns", ""))
        checkpoint_id = cfg.get("checkpoint_id")
        return thread_id, checkpoint_ns, str(checkpoint_id) if checkpoint_id is not None else None

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

        # If caller attached a live Vect-Link Branch, snapshot it into KVArena
        kv_branch = meta_copy.pop("kv_branch", None)
        if isinstance(kv_branch, Branch):
            if parent_id is not None and parent_id in self.branch_store._records:
                self.branch_store.commit_branch(
                    parent_id,
                    kv_branch,
                    thread_id=thread_id,
                    checkpoint_id=ckpt_id,
                    metadata=meta_copy,
                )
            else:
                self.branch_store.commit_root(
                    thread_id,
                    kv_branch,
                    checkpoint_id=ckpt_id,
                    metadata=meta_copy,
                )
            meta_copy["kv_snapshot_id"] = self.branch_store.get_snapshot_id(ckpt_id)

        new_config: Dict[str, Any] = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": ckpt_id,
            }
        }
        parent_config: Optional[Dict[str, Any]] = None
        if parent_id is not None:
            parent_config = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": parent_id,
                }
            }

        key = (thread_id, checkpoint_ns, ckpt_id)
        tup = CheckpointTuple(
            config=new_config,
            checkpoint=ckpt_copy,
            metadata=meta_copy,
            parent_config=parent_config,
            pending_writes=[],
        )
        self._tuples[key] = tup
        self._history.setdefault((thread_id, checkpoint_ns), []).append(ckpt_id)
        return new_config

    def get_tuple(self, config: Dict[str, Any]) -> Optional[CheckpointTuple]:
        thread_id, checkpoint_ns, ckpt_id = self._extract_cfg(config)
        if ckpt_id is None:
            hist = self._history.get((thread_id, checkpoint_ns), [])
            if not hist:
                return None
            ckpt_id = hist[-1]
        key = (thread_id, checkpoint_ns, ckpt_id)
        tup = self._tuples.get(key)
        if tup is None:
            return None
        writes = list(self._writes.get(key, []))
        return CheckpointTuple(
            config=tup.config,
            checkpoint=tup.checkpoint,
            metadata=tup.metadata,
            parent_config=tup.parent_config,
            pending_writes=writes,
        )

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
            keys = list(self._tuples.keys())
        else:
            thread_id, checkpoint_ns, _ = self._extract_cfg(config)
            hist = self._history.get((thread_id, checkpoint_ns), [])
            keys = [(thread_id, checkpoint_ns, cid) for cid in reversed(hist)]

        before_id: Optional[str] = None
        if before is not None:
            _, _, before_id = self._extract_cfg(before)

        count = 0
        past_before = before_id is None
        for key in keys:
            if not past_before:
                if key[2] == before_id:
                    past_before = True
                continue
            tup = self.get_tuple(
                {
                    "configurable": {
                        "thread_id": key[0],
                        "checkpoint_ns": key[1],
                        "checkpoint_id": key[2],
                    }
                }
            )
            if tup is None:
                continue
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
        key = (thread_id, checkpoint_ns, ckpt_id)
        bucket = self._writes.setdefault(key, [])
        for channel, value in writes:
            bucket.append((task_id, channel, value))

    def fork_from_checkpoint(self, checkpoint_id: str) -> Branch:
        """Convenience helper to fork a CoW KV Branch for parallel hypothesis evaluation."""
        return self.branch_store.fork_branch(checkpoint_id)

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
