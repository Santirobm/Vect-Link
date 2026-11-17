#ifndef _WIN32
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif
#endif

#include "bench_timer.h"
#include "vectlink_kv.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BENCH_CHECK(cond)                                                     \
    do {                                                                      \
        if (!(cond)) {                                                        \
            fprintf(stderr, "BENCH_CHECK failed at line %d: %s\n",            \
                    __LINE__, #cond);                                         \
            return 1;                                                         \
        }                                                                     \
    } while (0)

#define BENCH_BLOCKS       625u   /* 625 * 16 tokens = 10,000 tokens */
#define BENCH_TOKENS_BLK   16u
#define BENCH_BLOCK_SIZE   16384u /* 16 KB/block => 10.24 MB KV cache */
#define BENCH_ITERS        200u   /* <= VTLK_KV_MAX_SNAPSHOTS (256) */

static vtlk_kv_identity_t make_cfg(uint64_t prefix_hash, uint32_t pos_offset)
{
    vtlk_kv_identity_t id;
    memset(&id, 0, sizeof(id));
    id.token_prefix_hash = prefix_hash;
    id.model_arch_hash = vtlk_kv_hash_bytes("llama-3-8b", 10u);
    id.rope_config_hash = vtlk_kv_hash_bytes("rope-theta-500k", 15u);
    id.lora_adapter_hash = vtlk_kv_hash_bytes("base-none", 9u);
    id.pos_offset = pos_offset;
    id.token_count = BENCH_TOKENS_BLK;
    id.kv_dtype = VTLK_KV_DTYPE_FP16;
    id.layer_count = 32u;
    id.head_dim = 128u;
    id.kv_head_count = 8u;
    return id;
}

static void print_stats(const char *label,
                        const vtlk_bench_stats_t *st,
                        size_t payload_bytes,
                        size_t copied_bytes)
{
    printf("%-18s | min=%7.2f us | p50=%7.2f us | p95=%7.2f us | p99=%7.2f us | mean=%7.2f us | kv=%.2f MB | copied=%zu B\n",
           label,
           (double)st->min_ns / 1000.0,
           (double)st->p50_ns / 1000.0,
           (double)st->p95_ns / 1000.0,
           (double)st->p99_ns / 1000.0,
           st->mean_ns / 1000.0,
           (double)payload_bytes / (1024.0 * 1024.0),
           copied_bytes);
}

int main(void)
{
    const char *shm_name = "/vtlk_bench_cow_git";
    vtlk_kv_arena_t arena;
    vtlk_kv_branch_t base_branch;
    vtlk_kv_branch_t forked_branch;
    vtlk_kv_identity_t expected_cfg;
    const void *base_ptrs[BENCH_BLOCKS];
    const void *restored_ptrs[BENCH_BLOCKS];
    uint64_t snap_samples[BENCH_ITERS];
    uint64_t fork_samples[BENCH_ITERS];
    uint64_t restore_samples[BENCH_ITERS];
    uint64_t snap_ids[BENCH_ITERS];
    uint64_t prefix_hash = 0u;
    size_t total_kv_bytes = (size_t)BENCH_BLOCKS * (size_t)BENCH_BLOCK_SIZE;
    vtlk_bench_stats_t s_snap;
    vtlk_bench_stats_t s_fork;
    vtlk_bench_stats_t s_restore;
    uint32_t i;

    (void)vtlk_shm_unlink(shm_name);
    BENCH_CHECK(vtlk_kv_arena_create(&arena, shm_name, BENCH_BLOCKS + 16u, BENCH_BLOCK_SIZE) == VTLK_OK);

    memset(&base_branch, 0, sizeof(base_branch));
    expected_cfg = make_cfg(0u, 0u);

    for (i = 0u; i < BENCH_BLOCKS; ++i) {
        uint32_t bid = 0u;
        uint32_t tok = i + 1u;
        vtlk_kv_identity_t id;
        uint8_t *dst;

        prefix_hash = vtlk_kv_hash_tokens(prefix_hash, &tok, 1u);
        id = make_cfg(prefix_hash, i * BENCH_TOKENS_BLK);

        BENCH_CHECK(vtlk_kv_block_alloc(&arena, &bid) == VTLK_OK);
        dst = (uint8_t *)vtlk_kv_block_data(&arena, bid);
        memset(dst, (int)(i & 0xFFu), BENCH_BLOCK_SIZE);
        BENCH_CHECK(vtlk_kv_block_seal(&arena, bid, &id) == VTLK_OK);
        BENCH_CHECK(vtlk_kv_branch_append_block(&arena, &base_branch, bid) == VTLK_OK);
        BENCH_CHECK(vtlk_kv_block_release(&arena, bid) == VTLK_OK);
        base_ptrs[i] = dst;
    }

    /* Warmup */
    BENCH_CHECK(vtlk_kv_snapshot_create(&arena, &base_branch, &snap_ids[0]) == VTLK_OK);
    BENCH_CHECK(vtlk_kv_fork(&arena, snap_ids[0], &forked_branch) == VTLK_OK);
    BENCH_CHECK(vtlk_kv_restore(&arena, snap_ids[0], &expected_cfg, NULL, restored_ptrs, BENCH_BLOCKS) == VTLK_OK);
    BENCH_CHECK(vtlk_kv_branch_release(&arena, &forked_branch) == VTLK_OK);

    /* 1. Benchmark snapshot_create */
    for (i = 0u; i < BENCH_ITERS; ++i) {
        uint64_t t0 = vtlk_bench_now_ns();
        vtlk_status_t st = vtlk_kv_snapshot_create(&arena, &base_branch, &snap_ids[i]);
        uint64_t t1 = vtlk_bench_now_ns();
        BENCH_CHECK(st == VTLK_OK);
        snap_samples[i] = t1 - t0;
    }

    /* 2. Benchmark fork + verify 0 bytes of tensor payload copied */
    for (i = 0u; i < BENCH_ITERS; ++i) {
        uint64_t t0 = vtlk_bench_now_ns();
        vtlk_status_t st = vtlk_kv_fork(&arena, snap_ids[0], &forked_branch);
        uint64_t t1 = vtlk_bench_now_ns();
        BENCH_CHECK(st == VTLK_OK);
        fork_samples[i] = t1 - t0;

        if (i == 0u) {
            uint32_t b;
            for (b = 0u; b < BENCH_BLOCKS; ++b) {
                BENCH_CHECK(vtlk_kv_block_data(&arena, forked_branch.block_ids[b]) == base_ptrs[b]);
            }
        }
        BENCH_CHECK(vtlk_kv_branch_release(&arena, &forked_branch) == VTLK_OK);
    }

    /* 3. Benchmark restore (config validation + zero-copy pointer resolution) */
    for (i = 0u; i < BENCH_ITERS; ++i) {
        uint64_t t0 = vtlk_bench_now_ns();
        vtlk_status_t st = vtlk_kv_restore(&arena, snap_ids[0], &expected_cfg,
                                           NULL, restored_ptrs, BENCH_BLOCKS);
        uint64_t t1 = vtlk_bench_now_ns();
        BENCH_CHECK(st == VTLK_OK);
        restore_samples[i] = t1 - t0;
        if (i == 0u) {
            uint32_t b;
            for (b = 0u; b < BENCH_BLOCKS; ++b) {
                BENCH_CHECK(restored_ptrs[b] == base_ptrs[b]);
            }
        }
    }

    s_snap = vtlk_bench_compute_stats(snap_samples, BENCH_ITERS);
    s_fork = vtlk_bench_compute_stats(fork_samples, BENCH_ITERS);
    s_restore = vtlk_bench_compute_stats(restore_samples, BENCH_ITERS);

    printf("Vect-Link Git-for-Context CoW Benchmark (%u tokens, %u blocks x %u B = %.2f MiB, %u iters)\n",
           BENCH_BLOCKS * BENCH_TOKENS_BLK,
           BENCH_BLOCKS,
           BENCH_BLOCK_SIZE,
           (double)total_kv_bytes / (1024.0 * 1024.0),
           BENCH_ITERS);
    print_stats("snapshot_create", &s_snap, total_kv_bytes, 0u);
    print_stats("fork", &s_fork, total_kv_bytes, 0u);
    print_stats("restore", &s_restore, total_kv_bytes, 0u);

    BENCH_CHECK(vtlk_kv_branch_release(&arena, &base_branch) == VTLK_OK);
    vtlk_kv_arena_close(&arena);
    (void)vtlk_shm_unlink(shm_name);
    return 0;
}
