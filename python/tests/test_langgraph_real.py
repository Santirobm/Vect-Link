"""Real multi-process LangGraph integration tests (NO mocks).

Tests:
1. Real `langgraph.graph.StateGraph` with parallel 3-branch `Send` fan-out backed
   by Vect-Link KVArena CoW forks and VectLinkCheckpointer (L3 + SHM).
2. Process restart survival: run graph up to interrupt, terminate with os._exit(0),
   spawn fresh Python subprocess pointing to the same l3_dir + SHM, verify
   get_tuple()/list() and resume graph without recomputing earlier nodes.
3. Multi-process concurrency: 4 worker processes writing checkpoints and
   put_writes simultaneously to the same l3_dir without corruption.
"""

from __future__ import annotations

import json
import operator
import os
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Annotated, Any, Dict, List, TypedDict

PYTHON_ROOT = Path(__file__).resolve().parents[1]
try:
    import vectlink  # noqa: F401
except ImportError:
    if str(PYTHON_ROOT) not in sys.path:
        sys.path.insert(0, str(PYTHON_ROOT))

try:
    import pytest
    from langgraph.graph import END, START, StateGraph
    from langgraph.types import Send
except ImportError as exc:
    import unittest

    raise unittest.SkipTest(
        f"Real langgraph integration requires langgraph & pytest: {exc}"
    ) from exc

from vectlink.integrations.langgraph import VectLinkCheckpointer
from vectlink.kv import KVArena, KVIdentity


class BranchState(TypedDict):
    prompt_tokens: List[int]
    root_snapshot_id: int
    branch_results: Annotated[List[Dict[str, Any]], operator.add]
    node_calls: Annotated[List[str], operator.add]
    chosen_branch: int
    finalized: bool


class BranchTaskState(TypedDict):
    branch_idx: int
    root_snapshot_id: int


def _make_identity(prefix_hash: int, pos_offset: int, token_count: int) -> KVIdentity:
    return KVIdentity(
        token_prefix_hash=prefix_hash,
        model_arch_hash=0xA1B2C3D4,
        rope_config_hash=0x11223344,
        lora_adapter_hash=0x55667788,
        pos_offset=pos_offset,
        token_count=token_count,
        kv_dtype=0,
        layer_count=4,
        head_dim=64,
        kv_head_count=4,
    )


def _build_cow_graph(arena: KVArena, checkpointer: VectLinkCheckpointer, *, interrupt_before_finalize: bool = False):
    def prefill_node(state: BranchState) -> Dict[str, Any]:
        tokens = state["prompt_tokens"]
        h_root = KVArena.hash_tokens(tokens)
        id_root = _make_identity(h_root, pos_offset=0, token_count=len(tokens))
        with arena.alloc_block(release_on_exit=False) as blk:
            blk.write_bytes(b"SYSTEM_PREFILL_KV_TENSORS")
            blk.seal(id_root)
            bid = blk.block_id
        with arena.new_branch(id_root) as br:
            br.append_block(bid)
            arena.release_block(bid)
            snap = br.create_snapshot()
        return {
            "root_snapshot_id": snap.snapshot_id,
            "node_calls": ["prefill"],
        }

    def fan_out_branches(state: BranchState) -> List[Send]:
        return [
            Send(
                "eval_branch",
                {
                    "branch_idx": idx,
                    "root_snapshot_id": state["root_snapshot_id"],
                },
            )
            for idx in range(3)
        ]

    def eval_branch_node(state: BranchTaskState) -> Dict[str, Any]:
        idx = state["branch_idx"]
        root_snap_id = state["root_snapshot_id"]
        h_branch = KVArena.hash_tokens([9000 + idx], prev_hash=root_snap_id)
        id_branch = _make_identity(h_branch, pos_offset=4, token_count=1)
        with arena.alloc_block(release_on_exit=False) as blk:
            blk.write_bytes(f"COW_BRANCH_{idx}_KV".encode("utf-8"))
            blk.seal(id_branch)
            bid = blk.block_id

        with arena.fork(root_snap_id) as forked:
            forked.append_block(bid)
            arena.release_block(bid)
            child_snap = forked.create_snapshot()

        return {
            "branch_results": [
                {
                    "branch_idx": idx,
                    "snapshot_id": child_snap.snapshot_id,
                    "score": float(idx + 1) * 0.25,
                }
            ],
            "node_calls": [f"eval_branch_{idx}"],
        }

    def reduce_branches_node(state: BranchState) -> Dict[str, Any]:
        best = max(state["branch_results"], key=lambda r: r["score"])
        return {
            "chosen_branch": int(best["branch_idx"]),
            "node_calls": ["reduce_branches"],
        }

    def finalize_node(state: BranchState) -> Dict[str, Any]:
        best = max(state["branch_results"], key=lambda r: r["score"])
        with arena.restore(best["snapshot_id"]) as restored:
            views = restored.block_views()
            assert len(views) == 2
            assert bytes(views[0][:25]) == b"SYSTEM_PREFILL_KV_TENSORS"
            expected_tail = f"COW_BRANCH_{best['branch_idx']}_KV".encode("utf-8")
            assert bytes(views[1][: len(expected_tail)]) == expected_tail
        return {
            "finalized": True,
            "node_calls": ["finalize"],
        }

    builder = StateGraph(BranchState)
    builder.add_node("prefill", prefill_node)
    builder.add_node("eval_branch", eval_branch_node)
    builder.add_node("reduce_branches", reduce_branches_node)
    builder.add_node("finalize", finalize_node)

    builder.add_edge(START, "prefill")
    builder.add_conditional_edges("prefill", fan_out_branches, ["eval_branch"])
    builder.add_edge("eval_branch", "reduce_branches")
    builder.add_edge("reduce_branches", "finalize")
    builder.add_edge("finalize", END)

    interrupt_before = ["finalize"] if interrupt_before_finalize else None
    return builder.compile(checkpointer=checkpointer, interrupt_before=interrupt_before)


def test_real_langgraph_send_cow_fanout(tmp_path: Path) -> None:
    shm_name = f"vtlk_lg_real_{uuid.uuid4().hex[:8]}"
    l3_dir = tmp_path / "l3_dag"
    config = {"configurable": {"thread_id": "real-fanout-1", "checkpoint_ns": ""}}

    with KVArena.create(shm_name, num_blocks=32, block_size_bytes=512) as arena:
        with VectLinkCheckpointer(l3_dir=l3_dir, arena=arena) as checkpointer:
            graph = _build_cow_graph(arena, checkpointer)
            out = graph.invoke(
                {
                    "prompt_tokens": [11, 22, 33, 44],
                    "root_snapshot_id": 0,
                    "branch_results": [],
                    "node_calls": [],
                    "chosen_branch": -1,
                    "finalized": False,
                },
                config,
            )

            assert out["finalized"] is True
            assert out["chosen_branch"] == 2
            assert len(out["branch_results"]) == 3
            assert sorted(
                c for c in out["node_calls"] if c.startswith("eval_branch_")
            ) == ["eval_branch_0", "eval_branch_1", "eval_branch_2"]

            tup = checkpointer.get_tuple(config)
            assert tup is not None
            assert tup.checkpoint["channel_values"]["finalized"] is True
            history = list(checkpointer.list(config))
            assert len(history) >= 3


def test_process_restart_survival_without_recompute(tmp_path: Path) -> None:
    shm_name = f"vtlk_lg_survive_{uuid.uuid4().hex[:8]}"
    l3_dir = tmp_path / "l3_survive"
    thread_id = "survive-thread-1"
    repo_python = str(Path(__file__).resolve().parents[1])
    env = os.environ.copy()
    env["PYTHONPATH"] = repo_python

    # Hold SHM open in parent so Windows named mapping persists across child os._exit(0)
    with KVArena.create(shm_name, num_blocks=32, block_size_bytes=512):
        phase1_code = f"""
import os
from vectlink.kv import KVArena
from vectlink.integrations.langgraph import VectLinkCheckpointer
from tests.test_langgraph_real import _build_cow_graph

arena = KVArena.open({shm_name!r})
ckpt = VectLinkCheckpointer(l3_dir={str(l3_dir)!r}, arena=arena)
graph = _build_cow_graph(arena, ckpt, interrupt_before_finalize=True)
cfg = {{"configurable": {{"thread_id": {thread_id!r}, "checkpoint_ns": ""}}}}
out = graph.invoke(
    {{
        "prompt_tokens": [101, 102, 103, 104],
        "root_snapshot_id": 0,
        "branch_results": [],
        "node_calls": [],
        "chosen_branch": -1,
        "finalized": False,
    }},
    cfg,
)
assert out["finalized"] is False
assert out["chosen_branch"] == 2
assert len(out["branch_results"]) == 3
# Simulate abrupt process termination without Python cleanup
os._exit(0)
"""
        p1 = subprocess.run(
            [sys.executable, "-c", phase1_code],
            env=env,
            cwd=repo_python,
            capture_output=True,
            text=True,
            check=False,
        )
        assert p1.returncode == 0, f"Phase 1 failed:\n{p1.stdout}\n{p1.stderr}"

        # Fresh Python process reconstructs everything from L3 + SHM and resumes
        phase2_code = f"""
import json
from vectlink.kv import KVArena
from vectlink.integrations.langgraph import VectLinkCheckpointer
from tests.test_langgraph_real import _build_cow_graph

with KVArena.open({shm_name!r}) as arena:
    with VectLinkCheckpointer(l3_dir={str(l3_dir)!r}, arena=arena) as ckpt:
        cfg = {{"configurable": {{"thread_id": {thread_id!r}, "checkpoint_ns": ""}}}}
        tup = ckpt.get_tuple(cfg)
        assert tup is not None, "Expected persisted checkpoint after process restart"
        listed = list(ckpt.list(cfg))
        assert len(listed) >= 2

        graph = _build_cow_graph(arena, ckpt, interrupt_before_finalize=True)
        snap_state = graph.get_state(cfg)
        assert snap_state.next == ("finalize",), f"Unexpected next: {{snap_state.next}}"

        resumed = graph.invoke(None, cfg)
        assert resumed["finalized"] is True
        assert resumed["chosen_branch"] == 2
        # Verify prefill and eval_branch were NOT recomputed on resume
        calls = resumed["node_calls"]
        assert calls.count("prefill") == 1, calls
        assert calls.count("reduce_branches") == 1, calls
        assert calls.count("finalize") == 1, calls
        for idx in range(3):
            assert calls.count(f"eval_branch_{{idx}}") == 1, calls
        print(json.dumps({{"ok": True, "calls": calls}}))
"""
        p2 = subprocess.run(
            [sys.executable, "-c", phase2_code],
            env=env,
            cwd=repo_python,
            capture_output=True,
            text=True,
            check=False,
        )
        assert p2.returncode == 0, f"Phase 2 failed:\n{p2.stdout}\n{p2.stderr}"
        assert '"ok": true' in p2.stdout


def test_multiprocess_concurrent_checkpoint_writes(tmp_path: Path) -> None:
    l3_dir = tmp_path / "l3_concurrent"
    num_workers = 4
    ckpts_per_worker = 15
    repo_python = str(Path(__file__).resolve().parents[1])
    env = os.environ.copy()
    env["PYTHONPATH"] = repo_python

    worker_script = f"""
import sys
from vectlink.integrations.langgraph import VectLinkCheckpointer

wid = int(sys.argv[1])
n_items = int(sys.argv[2])
l3_dir = sys.argv[3]

with VectLinkCheckpointer(l3_dir=l3_dir) as ckpt:
    parent_cfg = {{"configurable": {{"thread_id": f"worker-{{wid}}", "checkpoint_ns": ""}}}}
    for step in range(n_items):
        cid = f"w{{wid}}-step-{{step}}"
        cfg = ckpt.put(
            parent_cfg,
            {{"id": cid, "channel_values": {{"worker": wid, "step": step}}}},
            {{"source": "mp_test", "step": step}},
        )
        ckpt.put_writes(
            cfg,
            [("metric", wid * 100 + step)],
            task_id=f"task-{{wid}}-{{step}}",
        )
        parent_cfg = cfg
"""
    procs = [
        subprocess.Popen(
            [
                sys.executable,
                "-c",
                worker_script,
                str(wid),
                str(ckpts_per_worker),
                str(l3_dir),
            ],
            env=env,
            cwd=repo_python,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for wid in range(num_workers)
    ]

    for p in procs:
        out, err = p.communicate(timeout=30)
        assert p.returncode == 0, f"Worker failed (rc={p.returncode}):\n{out}\n{err}"

    # Verify all 4 * 15 = 60 checkpoints + writes in a fresh checkpointer
    with VectLinkCheckpointer(l3_dir=l3_dir) as ckpt:
        all_items = list(ckpt.list(None))
        assert len(all_items) == num_workers * ckpts_per_worker

        for wid in range(num_workers):
            thread_items = list(
                ckpt.list({"configurable": {"thread_id": f"worker-{wid}", "checkpoint_ns": ""}})
            )
            assert len(thread_items) == ckpts_per_worker
            for step in range(ckpts_per_worker):
                cid = f"w{wid}-step-{step}"
                tup = ckpt.get_tuple(
                    {
                        "configurable": {
                            "thread_id": f"worker-{wid}",
                            "checkpoint_ns": "",
                            "checkpoint_id": cid,
                        }
                    }
                )
                assert tup is not None
                assert tup.checkpoint["channel_values"] == {"worker": wid, "step": step}
                assert tup.pending_writes == [(f"task-{wid}-{step}", "metric", wid * 100 + step)]
