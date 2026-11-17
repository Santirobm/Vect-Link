import gc
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vectlink import (  # noqa: E402
    Branch,
    InvalidParamError,
    KVArena,
    KVDtype,
    KVIdentity,
    Snapshot,
)
from vectlink.integrations import VectLinkBranchStore, VectLinkCheckpointer  # noqa: E402


def _make_identity(
    *,
    prefix_hash: int,
    pos_offset: int = 0,
    token_count: int = 16,
    model_arch_hash: int = 0x1111222233334444,
    rope_config_hash: int = 0xAAAABBBBCCCCDDDD,
    lora_adapter_hash: int = 0x0,
    kv_dtype: KVDtype = KVDtype.FP16,
    layer_count: int = 32,
    head_dim: int = 128,
    kv_head_count: int = 8,
) -> KVIdentity:
    return KVIdentity(
        token_prefix_hash=prefix_hash,
        model_arch_hash=model_arch_hash,
        rope_config_hash=rope_config_hash,
        lora_adapter_hash=lora_adapter_hash,
        pos_offset=pos_offset,
        token_count=token_count,
        kv_dtype=kv_dtype,
        layer_count=layer_count,
        head_dim=head_dim,
        kv_head_count=kv_head_count,
    )


class TestKvCowAndLangGraph(unittest.TestCase):
    def setUp(self) -> None:
        self.shm_name = f"/vtlk_kv_py_test_{os.getpid()}_{id(self)}"

    def test_hash_tokens_prefix_property_and_arena_stats(self) -> None:
        tokens_a = [101, 202, 303, 404]
        tokens_b = [505, 606, 707]
        h_a = KVArena.hash_tokens(tokens_a)
        h_ab_chained = KVArena.hash_tokens(tokens_b, prev_hash=h_a)
        h_ab_full = KVArena.hash_tokens(tokens_a + tokens_b)
        self.assertEqual(h_ab_chained, h_ab_full)
        self.assertNotEqual(KVArena.hash_bytes(b"rope_v1"), KVArena.hash_bytes(b"rope_v2"))

        req = KVArena.required_bytes(16, 4096)
        self.assertGreater(req, 16 * 4096)

        with KVArena.create(self.shm_name, num_blocks=16, block_size_bytes=4096) as arena:
            st = arena.stats()
            self.assertEqual(st.num_blocks, 16)
            self.assertEqual(st.free_blocks, 16)
            self.assertEqual(st.active_blocks, 0)
            self.assertEqual(st.block_size_bytes, 4096)

    def test_block_alloc_seal_lookup_and_cow_snapshot_fork_restore(self) -> None:
        with KVArena.create(self.shm_name, num_blocks=16, block_size_bytes=1024) as arena:
            h_sys = KVArena.hash_tokens([1, 2, 3, 4])
            id_sys = _make_identity(prefix_hash=h_sys, pos_offset=0, token_count=4)

            # Cache miss before sealing
            self.assertIsNone(arena.lookup_block(id_sys))

            with arena.alloc_block(release_on_exit=False) as blk:
                blk.write_bytes(b"SYS_KV_PREFIX_0123456789")
                blk.seal(id_sys)
                sys_block_id = blk.block_id

            # Cache hit after sealing
            found_id = arena.lookup_block(id_sys)
            self.assertEqual(found_id, sys_block_id)
            arena.release_block(found_id)

            # Build root branch and create snapshot
            with arena.new_branch(id_sys) as root_branch:
                root_branch.append_block(sys_block_id)
                arena.release_block(sys_block_id)  # branch now owns the ref
                snap = arena.create_snapshot(root_branch)

            self.assertIsInstance(snap, Snapshot)
            self.assertEqual(snap.block_count, 1)
            self.assertEqual(snap.total_tokens, 4)

            # Fork two independent child branches from snapshot
            h_child1 = KVArena.hash_tokens([10, 11], prev_hash=h_sys)
            id_child1 = _make_identity(prefix_hash=h_child1, pos_offset=4, token_count=2)

            with arena.alloc_block(release_on_exit=False) as b1:
                b1.write_bytes(b"CHILD_1_THOUGHT_KV")
                b1.seal(id_child1)
                c1_bid = b1.block_id

            with arena.fork(snap) as branch1:
                # Zero-copy read of parent block
                mv0 = branch1.block_view(0)
                self.assertTrue(mv0.readonly)
                self.assertEqual(bytes(mv0[:24]), b"SYS_KV_PREFIX_0123456789")
                branch1.append_block(c1_bid)
                arena.release_block(c1_bid)
                snap1 = branch1.create_snapshot()

            self.assertEqual(snap1.block_count, 2)
            self.assertEqual(snap1.total_tokens, 6)

            # Parent snapshot remains untouched (still 1 block, 4 tokens)
            with arena.restore(snap.snapshot_id, id_sys) as restored_root:
                self.assertEqual(restored_root.block_count, 1)
                self.assertEqual(restored_root.total_tokens, 4)
                self.assertEqual(
                    bytes(restored_root.block_view(0)[:24]),
                    b"SYS_KV_PREFIX_0123456789",
                )

            # Mismatched RoPE config must fail restore with InvalidParamError
            bad_rope = _make_identity(
                prefix_hash=h_sys,
                rope_config_hash=0xDEADBEEF,
            )
            with self.assertRaises(InvalidParamError):
                arena.restore(snap.snapshot_id, bad_rope)

    def test_dangling_view_raises_buffer_error_on_block_and_branch(self) -> None:
        with KVArena.create(self.shm_name, num_blocks=8, block_size_bytes=512) as arena:
            # 1. Dangling slice on KVBlockView
            blk = arena.alloc_block(release_on_exit=True)
            blk.write_bytes(b"ABCD" * 16)
            dangling_slice = blk.buffer[0:16]
            with self.assertRaises(BufferError):
                blk.release()
            del dangling_slice
            gc.collect()
            blk.release()

            # 2. Dangling slice on Branch
            h0 = KVArena.hash_tokens([7, 8, 9])
            id0 = _make_identity(prefix_hash=h0, pos_offset=0, token_count=3)
            with arena.alloc_block(release_on_exit=False) as b0:
                b0.write_bytes(b"BRANCH_ZERO_COPY_PAYLOAD")
                b0.seal(id0)
                bid = b0.block_id

            with arena.new_branch(id0) as br:
                br.append_block(bid)
                arena.release_block(bid)
                snap = br.create_snapshot()

            forked = arena.fork(snap)
            branch_slice = forked.block_view(0)[0:24]
            with self.assertRaises(BufferError):
                forked.release()
            del branch_slice
            gc.collect()
            forked.release()

    def test_dangling_numpy_view_raises_buffer_error(self) -> None:
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy not installed")

        with KVArena.create(self.shm_name, num_blocks=8, block_size_bytes=256) as arena:
            blk = arena.alloc_block(release_on_exit=True)
            arr = blk.to_numpy(dtype=np.uint8, copy=False)
            arr[:4] = [1, 2, 3, 4]
            with self.assertRaises(BufferError):
                blk.release()
            del arr
            gc.collect()
            blk.release()

    def test_langgraph_branch_store_and_checkpointer_tot(self) -> None:
        with KVArena.create(self.shm_name, num_blocks=32, block_size_bytes=512) as arena:
            store = VectLinkBranchStore(arena)
            checkpointer = VectLinkCheckpointer(store)

            # Prefill root system prompt
            h_sys = KVArena.hash_tokens([100, 200, 300, 400])
            id_sys = _make_identity(prefix_hash=h_sys, pos_offset=0, token_count=4)
            with arena.alloc_block(release_on_exit=False) as b_sys:
                b_sys.write_bytes(b"ROOT_SYSTEM_PROMPT_KV")
                b_sys.seal(id_sys)
                sys_bid = b_sys.block_id

            with arena.new_branch(id_sys) as root_br:
                root_br.append_block(sys_bid)
                arena.release_block(sys_bid)
                root_cfg = checkpointer.put(
                    {"configurable": {"thread_id": "tot-session-1", "checkpoint_ns": ""}},
                    {"id": "root-0", "channel_values": {"step": 0}},
                    {"kv_branch": root_br, "role": "system_prefill"},
                )

            root_ckpt_id = root_cfg["configurable"]["checkpoint_id"]
            self.assertEqual(root_ckpt_id, "root-0")

            # Fork 3 parallel Tree-of-Thought hypotheses from root-0
            child_ids = []
            for hyp_idx in range(3):
                h_hyp = KVArena.hash_tokens([1000 + hyp_idx], prev_hash=h_sys)
                id_hyp = _make_identity(prefix_hash=h_hyp, pos_offset=4, token_count=1)
                with arena.alloc_block(release_on_exit=False) as b_hyp:
                    b_hyp.write_bytes(f"HYPOTHESIS_{hyp_idx}_KV".encode("utf-8"))
                    b_hyp.seal(id_hyp)
                    hyp_bid = b_hyp.block_id

                with checkpointer.fork_from_checkpoint(root_ckpt_id) as hyp_br:
                    hyp_br.append_block(hyp_bid)
                    arena.release_block(hyp_bid)
                    child_cfg = checkpointer.put(
                        root_cfg,
                        {
                            "id": f"hyp-{hyp_idx}",
                            "channel_values": {"step": 1, "hypothesis": hyp_idx},
                        },
                        {"kv_branch": hyp_br, "hypothesis": hyp_idx},
                    )
                    child_ids.append(child_cfg["configurable"]["checkpoint_id"])

            self.assertEqual(store.list_children("root-0"), child_ids)
            lineage = store.get_lineage("hyp-2")
            self.assertEqual([r.checkpoint_id for r in lineage], ["root-0", "hyp-2"])
            self.assertEqual(lineage[1].total_tokens, 5)
            self.assertEqual(lineage[1].block_count, 2)

            # Verify zero-copy restore of each hypothesis independently
            for hyp_idx, cid in enumerate(child_ids):
                with store.restore_branch(cid, id_sys) as restored:
                    views = restored.block_views()
                    self.assertEqual(len(views), 2)
                    self.assertEqual(bytes(views[0][:21]), b"ROOT_SYSTEM_PROMPT_KV")
                    expected_tail = f"HYPOTHESIS_{hyp_idx}_KV".encode("utf-8")
                    self.assertEqual(bytes(views[1][: len(expected_tail)]), expected_tail)

            # Verify LangGraph checkpointer tuple, writes, and list()
            checkpointer.put_writes(
                {"configurable": {"thread_id": "tot-session-1", "checkpoint_id": "hyp-1"}},
                [("score", 0.92)],
                task_id="eval-task-1",
            )
            tup = checkpointer.get_tuple(
                {"configurable": {"thread_id": "tot-session-1", "checkpoint_id": "hyp-1"}}
            )
            self.assertIsNotNone(tup)
            assert tup is not None
            self.assertEqual(tup.checkpoint["channel_values"]["hypothesis"], 1)
            self.assertEqual(tup.pending_writes, [("eval-task-1", "score", 0.92)])

            listed = list(
                checkpointer.list(
                    {"configurable": {"thread_id": "tot-session-1"}},
                    filter={"hypothesis": 2},
                )
            )
            self.assertEqual(len(listed), 1)
            self.assertEqual(listed[0].checkpoint["id"], "hyp-2")


if __name__ == "__main__":
    unittest.main()
