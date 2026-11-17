#ifndef _WIN32
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif
#endif

#include "vectlink_kv.h"
#include "vectlink_sync.h"
#include "vectlink_tier.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <direct.h>
#include <windows.h>
#else
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

/* --------------------------------------------------------------------------
 * Multi-process CoW configuration: 10,000 tokens = 625 blocks x 16 tokens
 * -------------------------------------------------------------------------- */
#define COW_TOKENS_PER_BLOCK  16u
#define COW_SHARED_BLOCKS     625u
#define COW_DIVERGENT_BLOCKS  16u
#define COW_BLOCK_SIZE        16384u
#define COW_ARENA_BLOCKS      720u
#define COW_NUM_WORKERS       3u

static vtlk_kv_identity_t make_base_identity(uint64_t prefix_hash,
                                             uint32_t pos_offset,
                                             uint32_t token_count)
{
    vtlk_kv_identity_t id;
    memset(&id, 0, sizeof(id));
    id.token_prefix_hash = prefix_hash;
    id.model_arch_hash = vtlk_kv_hash_bytes("llama-3-8b", 10u);
    id.rope_config_hash = vtlk_kv_hash_bytes("rope-theta-500k", 15u);
    id.lora_adapter_hash = vtlk_kv_hash_bytes("base-none", 9u);
    id.pos_offset = pos_offset;
    id.token_count = token_count;
    id.kv_dtype = VTLK_KV_DTYPE_FP16;
    id.layer_count = 32u;
    id.head_dim = 128u;
    id.kv_head_count = 8u;
    return id;
}

static void fill_kv_pattern(uint8_t *dst, size_t len, uint32_t tag, uint32_t idx)
{
    size_t i;
    for (i = 0u; i < len; ++i) {
        dst[i] = (uint8_t)(((tag * 131u) ^ (idx * 37u) ^ (uint32_t)(i * 17u)) & 0xFFu);
    }
}

static int verify_kv_pattern(const uint8_t *src, size_t len, uint32_t tag, uint32_t idx)
{
    size_t i;
    for (i = 0u; i < len; ++i) {
        uint8_t exp = (uint8_t)(((tag * 131u) ^ (idx * 37u) ^ (uint32_t)(i * 17u)) & 0xFFu);
        if (src[i] != exp) {
            return 0;
        }
    }
    return 1;
}

/* --------------------------------------------------------------------------
 * Unit Tests
 * -------------------------------------------------------------------------- */

static void test_kv_hash_chaining(void)
{
    uint32_t tokens_all[8] = { 101u, 202u, 303u, 404u, 505u, 606u, 707u, 808u };
    uint64_t h_all;
    uint64_t h_a;
    uint64_t h_ab;
    uint64_t hb1;
    uint64_t hb2;

    h_all = vtlk_kv_hash_tokens(0u, tokens_all, 8u);
    h_a = vtlk_kv_hash_tokens(0u, tokens_all, 3u);
    h_ab = vtlk_kv_hash_tokens(h_a, tokens_all + 3, 5u);
    assert(h_all == h_ab);
    assert(h_all != h_a);

    hb1 = vtlk_kv_hash_bytes("model-a", 7u);
    hb2 = vtlk_kv_hash_bytes("model-b", 7u);
    assert(hb1 != hb2);
    assert(hb1 == vtlk_kv_hash_bytes("model-a", 7u));
}

static void test_kv_arena_dedup_and_lru(void)
{
    const char *shm_name = "/vtlk_unit_kv_arena";
    vtlk_kv_arena_t arena;
    vtlk_kv_arena_stats_t stats;
    uint32_t block_ids[8];
    uint32_t extra_id = VTLK_KV_INVALID_BLOCK_ID;
    uint32_t lookup_id = VTLK_KV_INVALID_BLOCK_ID;
    vtlk_kv_identity_t id0;
    vtlk_kv_identity_t id_miss;
    vtlk_kv_block_header_t *hdr;
    void *ptr;
    uint32_t i;
    vtlk_status_t st;

    (void)vtlk_shm_unlink(shm_name);
    st = vtlk_kv_arena_create(&arena, shm_name, 8u, 4096u);
    assert(st == VTLK_OK);

    for (i = 0u; i < 8u; ++i) {
        uint32_t tok = 1000u + i;
        vtlk_kv_identity_t id = make_base_identity(
            vtlk_kv_hash_tokens(0u, &tok, 1u), i * 16u, 16u);
        st = vtlk_kv_block_alloc(&arena, &block_ids[i]);
        assert(st == VTLK_OK);
        ptr = vtlk_kv_block_data(&arena, block_ids[i]);
        assert(ptr != NULL);
        fill_kv_pattern((uint8_t *)ptr, 4096u, 7u, i);
        st = vtlk_kv_block_seal(&arena, block_ids[i], &id);
        assert(st == VTLK_OK);

        hdr = vtlk_kv_block_header(&arena, block_ids[i]);
        assert(hdr != NULL);
        assert((hdr->flags & VTLK_KV_FLAG_IMMUTABLE) != 0u);
        assert((hdr->flags & VTLK_KV_FLAG_EVICTABLE) != 0u);
        assert(hdr->refcount == 1u);
    }

    /* When all 8 blocks have refcount > 0, alloc must return VTLK_ERR_FULL. */
    st = vtlk_kv_block_alloc(&arena, &extra_id);
    assert(st == VTLK_ERR_FULL);

    /* Deduplication lookup on block 0. */
    {
        uint32_t tok0 = 1000u;
        id0 = make_base_identity(vtlk_kv_hash_tokens(0u, &tok0, 1u), 0u, 16u);
        st = vtlk_kv_block_lookup(&arena, &id0, &lookup_id);
        assert(st == VTLK_OK);
        assert(lookup_id == block_ids[0]);
        hdr = vtlk_kv_block_header(&arena, block_ids[0]);
        assert(hdr->refcount == 2u);
        assert(vtlk_kv_block_release(&arena, block_ids[0]) == VTLK_OK);

        id_miss = id0;
        id_miss.lora_adapter_hash = vtlk_kv_hash_bytes("lora-other", 10u);
        st = vtlk_kv_block_lookup(&arena, &id_miss, &lookup_id);
        assert(st == VTLK_ERR_EMPTY);
    }

    /* Release all 8 blocks so they become evictable (refcount == 0). */
    for (i = 0u; i < 8u; ++i) {
        assert(vtlk_kv_block_release(&arena, block_ids[i]) == VTLK_OK);
    }

    /* Touch block 0 via lookup+release so block 1 becomes older LRU. */
    assert(vtlk_kv_block_lookup(&arena, &id0, &lookup_id) == VTLK_OK);
    assert(vtlk_kv_block_release(&arena, lookup_id) == VTLK_OK);

    /* Allocating a new block should succeed via LRU eviction. */
    st = vtlk_kv_block_alloc(&arena, &extra_id);
    assert(st == VTLK_OK);
    assert(vtlk_kv_block_release(&arena, extra_id) == VTLK_OK);

    assert(vtlk_kv_arena_stats(&arena, &stats) == VTLK_OK);
    assert(stats.evictions >= 1u);
    assert(stats.cache_hits >= 2u);
    assert(stats.cache_misses >= 1u);

    vtlk_kv_arena_close(&arena);
    (void)vtlk_shm_unlink(shm_name);
}

static void test_kv_snapshot_fork_restore_mismatch(void)
{
    const char *shm_name = "/vtlk_unit_kv_cow";
    vtlk_kv_arena_t arena;
    vtlk_kv_branch_t base_branch;
    vtlk_kv_branch_t forked;
    vtlk_kv_branch_t restored;
    vtlk_kv_identity_t cfg;
    vtlk_kv_identity_t bad_cfg;
    const void *ptrs[4];
    uint64_t snap_id = 0u;
    uint32_t i;

    (void)vtlk_shm_unlink(shm_name);
    assert(vtlk_kv_arena_create(&arena, shm_name, 16u, 4096u) == VTLK_OK);

    memset(&base_branch, 0, sizeof(base_branch));
    cfg = make_base_identity(0u, 0u, 16u);

    for (i = 0u; i < 4u; ++i) {
        uint32_t bid = 0u;
        uint32_t tok = 500u + i;
        vtlk_kv_identity_t id = make_base_identity(
            vtlk_kv_hash_tokens(0u, &tok, 1u), i * 16u, 16u);
        assert(vtlk_kv_block_alloc(&arena, &bid) == VTLK_OK);
        fill_kv_pattern((uint8_t *)vtlk_kv_block_data(&arena, bid), 4096u, 11u, i);
        assert(vtlk_kv_block_seal(&arena, bid, &id) == VTLK_OK);
        assert(vtlk_kv_branch_append_block(&arena, &base_branch, bid) == VTLK_OK);
        assert(vtlk_kv_block_release(&arena, bid) == VTLK_OK);
    }

    assert(vtlk_kv_snapshot_create(&arena, &base_branch, &snap_id) == VTLK_OK);
    assert(snap_id != 0u);

    assert(vtlk_kv_fork(&arena, snap_id, &forked) == VTLK_OK);
    assert(forked.block_count == 4u);
    assert(forked.total_tokens == 64u);
    assert(forked.parent_snapshot_id == snap_id);

    assert(vtlk_kv_restore(&arena, snap_id, &cfg, &restored, ptrs, 4u) == VTLK_OK);
    assert(restored.block_count == 4u);
    for (i = 0u; i < 4u; ++i) {
        assert(ptrs[i] == vtlk_kv_block_data(&arena, base_branch.block_ids[i]));
        assert(verify_kv_pattern((const uint8_t *)ptrs[i], 4096u, 11u, i));
    }

    /* Config mismatch rejections */
    bad_cfg = cfg;
    bad_cfg.model_arch_hash ^= 1u;
    assert(vtlk_kv_restore(&arena, snap_id, &bad_cfg, NULL, NULL, 0u) == VTLK_ERR_INVALID_PARAM);

    bad_cfg = cfg;
    bad_cfg.rope_config_hash ^= 1u;
    assert(vtlk_kv_restore(&arena, snap_id, &bad_cfg, NULL, NULL, 0u) == VTLK_ERR_INVALID_PARAM);

    bad_cfg = cfg;
    bad_cfg.lora_adapter_hash ^= 1u;
    assert(vtlk_kv_restore(&arena, snap_id, &bad_cfg, NULL, NULL, 0u) == VTLK_ERR_INVALID_PARAM);

    bad_cfg = cfg;
    bad_cfg.kv_dtype = VTLK_KV_DTYPE_Q4_0;
    assert(vtlk_kv_restore(&arena, snap_id, &bad_cfg, NULL, NULL, 0u) == VTLK_ERR_INVALID_PARAM);

    bad_cfg = cfg;
    bad_cfg.head_dim = 64u;
    assert(vtlk_kv_restore(&arena, snap_id, &bad_cfg, NULL, NULL, 0u) == VTLK_ERR_INVALID_PARAM);

    assert(vtlk_kv_branch_release(&arena, &restored) == VTLK_OK);
    assert(vtlk_kv_branch_release(&arena, &forked) == VTLK_OK);
    assert(vtlk_kv_branch_release(&arena, &base_branch) == VTLK_OK);

    vtlk_kv_arena_close(&arena);
    (void)vtlk_shm_unlink(shm_name);
}

static void test_desc_ring_and_notify(void)
{
    size_t req = vtlk_desc_ring_required_bytes(4u);
    void *mem = malloc(req);
    vtlk_desc_ring_t ring_prod;
    vtlk_desc_ring_t ring_cons;
    vtlk_kv_descriptor_t d_in;
    vtlk_kv_descriptor_t d_out;
    volatile uint32_t word = 42u;
    uint32_t i;

    assert(req > 0u);
    assert(mem != NULL);

    assert(vtlk_desc_ring_init(&ring_prod, mem, req, 4u) == VTLK_OK);
    assert(vtlk_desc_ring_attach(&ring_cons, mem, req) == VTLK_OK);

    /* Non-blocking empty and bounded timeout empty */
    assert(vtlk_desc_ring_pop(&ring_cons, &d_out, 0u) == VTLK_ERR_EMPTY);
    assert(vtlk_desc_ring_pop(&ring_cons, &d_out, 5u) == VTLK_ERR_TIMEOUT);

    /* OS wait/wake primitives */
    assert(vtlk_notify_wait_u32(&word, 99u, 0u) == VTLK_OK);
    assert(vtlk_notify_wait_u32(&word, 42u, 5u) == VTLK_ERR_TIMEOUT);
    assert(vtlk_notify_wake_u32(&word, 1) == VTLK_OK);

    /* Push 4 descriptors (fill ring) */
    for (i = 0u; i < 4u; ++i) {
        memset(&d_in, 0, sizeof(d_in));
        d_in.snapshot_id = 100u + i;
        d_in.branch_id = 200u + i;
        d_in.sequence = i;
        d_in.pos_offset = i * 16u;
        d_in.block_count = 2u;
        d_in.block_ids[0] = i * 2u;
        d_in.block_ids[1] = i * 2u + 1u;
        assert(vtlk_desc_ring_push(&ring_prod, &d_in, 0u) == VTLK_OK);
    }

    /* Ring full */
    assert(vtlk_desc_ring_push(&ring_prod, &d_in, 0u) == VTLK_ERR_FULL);
    assert(vtlk_desc_ring_push(&ring_prod, &d_in, 5u) == VTLK_ERR_TIMEOUT);

    /* Pop and verify */
    for (i = 0u; i < 4u; ++i) {
        assert(vtlk_desc_ring_pop(&ring_cons, &d_out, 0u) == VTLK_OK);
        assert(d_out.snapshot_id == 100u + i);
        assert(d_out.branch_id == 200u + i);
        assert(d_out.sequence == i);
        assert(d_out.block_count == 2u);
        assert(d_out.block_ids[0] == i * 2u);
        assert(d_out.block_ids[1] == i * 2u + 1u);
    }

    free(mem);
}

static void make_temp_dir(const char *path)
{
#ifdef _WIN32
    (void)_mkdir(path);
#else
    (void)mkdir(path, 0755);
#endif
}

static void cleanup_tier_dir(const char *dir)
{
    char path[256];
    uint32_t i;
    snprintf(path, sizeof(path), "%s/vtlk_tier_index.bin", dir);
    remove(path);
    for (i = 0u; i < 8u; ++i) {
        snprintf(path, sizeof(path), "%s/vtlk_blk_%08u.bin", dir, i);
        remove(path);
        snprintf(path, sizeof(path), "%s/blk_%u.bin", dir, i);
        remove(path);
    }
#ifdef _WIN32
    (void)_rmdir(dir);
#else
    (void)rmdir(dir);
#endif
}

static void test_tier_l1_l2_l3_persistence(void)
{
    uint8_t l1_buf[2u * 512u];
    uint8_t blk[512u];
    const char *l2_path = "vtlk_test_tier_l2.bin";
    const char *l3_dir = "vtlk_test_tier_l3";
    vtlk_tier_config_t cfg;
    vtlk_tier_store_t store;
    vtlk_tier_stats_t stats;
    const void *ptr = NULL;
    uint32_t i;

    make_temp_dir(l3_dir);
    remove(l2_path);

    memset(&cfg, 0, sizeof(cfg));
    cfg.block_size = 512u;
    cfg.l1_capacity = 2u;
    cfg.l2_capacity = 2u;
    cfg.max_blocks = 8u;
    cfg.high_watermark = 2u;
    cfg.l1_mem = l1_buf;
    cfg.l2_path = l2_path;
    cfg.l3_dir = l3_dir;

    assert(vtlk_tier_init(&store, &cfg) == VTLK_OK);

    /* Insert 6 blocks: forces L1 -> L2 -> L3 evictions. */
    for (i = 0u; i < 6u; ++i) {
        fill_kv_pattern(blk, sizeof(blk), 23u, i);
        assert(vtlk_tier_put(&store, i, blk, sizeof(blk)) == VTLK_OK);
    }

    assert(vtlk_tier_stats(&store, &stats) == VTLK_OK);
    assert(stats.evictions_l1_to_l2 >= 4u);
    assert(stats.evictions_l2_to_l3 >= 2u);
    assert(stats.crc_errors == 0u);

    /* Acquire block 0 (evicted to L3) -> promotes back to L1 with CRC32C check. */
    assert(vtlk_tier_acquire(&store, 0u, &ptr) == VTLK_OK);
    assert(ptr != NULL);
    assert(verify_kv_pattern((const uint8_t *)ptr, sizeof(blk), 23u, 0u));
    assert(vtlk_tier_release(&store, 0u) == VTLK_OK);

    assert(vtlk_tier_stats(&store, &stats) == VTLK_OK);
    assert(stats.promotions_to_l1 >= 1u);
    assert(stats.crc_errors == 0u);

    assert(vtlk_tier_flush_index(&store) == VTLK_OK);
    vtlk_tier_close(&store);

    /* Re-open store and verify L3 persistence of block 1. */
    assert(vtlk_tier_init(&store, &cfg) == VTLK_OK);
    assert(vtlk_tier_acquire(&store, 1u, &ptr) == VTLK_OK);
    assert(ptr != NULL);
    assert(verify_kv_pattern((const uint8_t *)ptr, sizeof(blk), 23u, 1u));
    assert(vtlk_tier_release(&store, 1u) == VTLK_OK);
    vtlk_tier_close(&store);

    remove(l2_path);
    cleanup_tier_dir(l3_dir);
}

/* --------------------------------------------------------------------------
 * Multi-Process CoW Stress Test (10,000 tokens -> 1 snapshot -> 3 forks)
 * -------------------------------------------------------------------------- */

static uint32_t role_tag(const char *role)
{
    if (strcmp(role, "fork_a") == 0) {
        return 101u;
    }
    if (strcmp(role, "fork_b") == 0) {
        return 202u;
    }
    return 303u;
}

static int run_worker(const char *role, const char *shm_name, const char *snap_str)
{
    vtlk_kv_arena_t arena;
    vtlk_kv_branch_t branch;
    vtlk_kv_identity_t expected_cfg;
    const void *ptrs[COW_SHARED_BLOCKS];
    uint32_t div_ids[COW_DIVERGENT_BLOCKS];
    uint64_t snapshot_id = (uint64_t)strtoull(snap_str, NULL, 10);
    uint32_t rtag = role_tag(role);
    uint32_t i;
    vtlk_status_t st;

    st = vtlk_kv_arena_open(&arena, shm_name);
    if (st != VTLK_OK) {
        fprintf(stderr, "[%s] vtlk_kv_arena_open failed (%d)\n", role, (int)st);
        return 1;
    }

    st = vtlk_kv_fork(&arena, snapshot_id, &branch);
    if (st != VTLK_OK) {
        fprintf(stderr, "[%s] vtlk_kv_fork failed (%d)\n", role, (int)st);
        vtlk_kv_arena_close(&arena);
        return 1;
    }

    expected_cfg = make_base_identity(0u, 0u, COW_TOKENS_PER_BLOCK);
    st = vtlk_kv_restore(&arena, snapshot_id, &expected_cfg, NULL, ptrs, COW_SHARED_BLOCKS);
    if (st != VTLK_OK) {
        fprintf(stderr, "[%s] vtlk_kv_restore failed (%d)\n", role, (int)st);
        (void)vtlk_kv_branch_release(&arena, &branch);
        vtlk_kv_arena_close(&arena);
        return 1;
    }

    /* Verify bit-exactness of all 10,000 shared prefix tokens (625 blocks). */
    for (i = 0u; i < COW_SHARED_BLOCKS; ++i) {
        if (!verify_kv_pattern((const uint8_t *)ptrs[i], COW_BLOCK_SIZE, 1u, i)) {
            fprintf(stderr, "[%s] shared block %u corrupted!\n", role, i);
            (void)vtlk_kv_branch_release(&arena, &branch);
            vtlk_kv_arena_close(&arena);
            return 1;
        }
    }

    /* Append divergent blocks unique to this worker persona. */
    for (i = 0u; i < COW_DIVERGENT_BLOCKS; ++i) {
        uint32_t bid = 0u;
        uint32_t tok = rtag * 10000u + i;
        vtlk_kv_identity_t id = make_base_identity(
            vtlk_kv_hash_tokens(0u, &tok, 1u),
            (COW_SHARED_BLOCKS + i) * COW_TOKENS_PER_BLOCK,
            COW_TOKENS_PER_BLOCK);
        void *bdata;

        st = vtlk_kv_block_alloc(&arena, &bid);
        if (st != VTLK_OK) {
            fprintf(stderr, "[%s] alloc divergent %u failed (%d)\n", role, i, (int)st);
            (void)vtlk_kv_branch_release(&arena, &branch);
            vtlk_kv_arena_close(&arena);
            return 1;
        }

        bdata = vtlk_kv_block_data(&arena, bid);
        fill_kv_pattern((uint8_t *)bdata, COW_BLOCK_SIZE, rtag, i);

        st = vtlk_kv_block_seal(&arena, bid, &id);
        assert(st == VTLK_OK);
        st = vtlk_kv_branch_append_block(&arena, &branch, bid);
        assert(st == VTLK_OK);
        st = vtlk_kv_block_release(&arena, bid);
        assert(st == VTLK_OK);
        div_ids[i] = bid;
    }

    /* Verify own divergent blocks are bit-exact. */
    for (i = 0u; i < COW_DIVERGENT_BLOCKS; ++i) {
        const uint8_t *bdata = (const uint8_t *)vtlk_kv_block_data(&arena, div_ids[i]);
        assert(verify_kv_pattern(bdata, COW_BLOCK_SIZE, rtag, i));
    }

    assert(branch.block_count == COW_SHARED_BLOCKS + COW_DIVERGENT_BLOCKS);
    assert(branch.total_tokens == (COW_SHARED_BLOCKS + COW_DIVERGENT_BLOCKS) * COW_TOKENS_PER_BLOCK);

    assert(vtlk_kv_branch_release(&arena, &branch) == VTLK_OK);
    vtlk_kv_arena_close(&arena);
    return 0;
}

static int run_multiproc_cow_stress(const char *exe_path)
{
    static const char *roles[COW_NUM_WORKERS] = { "fork_a", "fork_b", "fork_c" };
    char shm_name[128];
    char snap_str[64];
    vtlk_kv_arena_t arena;
    vtlk_kv_branch_t base_branch;
    uint32_t shared_ids[COW_SHARED_BLOCKS];
    uint64_t snapshot_id = 0u;
    uint64_t prefix_hash = 0u;
    uint32_t i;
    vtlk_status_t st;

#ifdef _WIN32
    DWORD pid = GetCurrentProcessId();
    PROCESS_INFORMATION pis[COW_NUM_WORKERS];
#else
    pid_t pid = getpid();
    pid_t children[COW_NUM_WORKERS];
#endif

    snprintf(shm_name, sizeof(shm_name), "/vtlk_cow_stress_%lu", (unsigned long)pid);
    (void)vtlk_shm_unlink(shm_name);

    st = vtlk_kv_arena_create(&arena, shm_name, COW_ARENA_BLOCKS, COW_BLOCK_SIZE);
    assert(st == VTLK_OK);

    memset(&base_branch, 0, sizeof(base_branch));

    /* Write 10,000 tokens (625 blocks x 16 tokens), seal, and append to base branch. */
    for (i = 0u; i < COW_SHARED_BLOCKS; ++i) {
        uint32_t bid = 0u;
        uint32_t tok_chunk[COW_TOKENS_PER_BLOCK];
        uint32_t t;
        vtlk_kv_identity_t id;
        void *bdata;

        for (t = 0u; t < COW_TOKENS_PER_BLOCK; ++t) {
            tok_chunk[t] = i * COW_TOKENS_PER_BLOCK + t + 1u;
        }
        prefix_hash = vtlk_kv_hash_tokens(prefix_hash, tok_chunk, COW_TOKENS_PER_BLOCK);
        id = make_base_identity(prefix_hash, i * COW_TOKENS_PER_BLOCK, COW_TOKENS_PER_BLOCK);

        assert(vtlk_kv_block_alloc(&arena, &bid) == VTLK_OK);
        bdata = vtlk_kv_block_data(&arena, bid);
        fill_kv_pattern((uint8_t *)bdata, COW_BLOCK_SIZE, 1u, i);

        assert(vtlk_kv_block_seal(&arena, bid, &id) == VTLK_OK);
        assert(vtlk_kv_branch_append_block(&arena, &base_branch, bid) == VTLK_OK);
        assert(vtlk_kv_block_release(&arena, bid) == VTLK_OK);
        shared_ids[i] = bid;
    }

    assert(vtlk_kv_snapshot_create(&arena, &base_branch, &snapshot_id) == VTLK_OK);
    assert(snapshot_id != 0u);
    snprintf(snap_str, sizeof(snap_str), "%llu", (unsigned long long)snapshot_id);

    /* Spawn 3 concurrent diverging worker processes. */
    for (i = 0u; i < COW_NUM_WORKERS; ++i) {
#ifdef _WIN32
        char cmd_line[1024];
        STARTUPINFOA si;
        memset(&si, 0, sizeof(si));
        si.cb = sizeof(si);
        memset(&pis[i], 0, sizeof(pis[i]));
        snprintf(cmd_line, sizeof(cmd_line), "\"%s\" --worker %s %s %s",
                 exe_path, roles[i], shm_name, snap_str);
        if (!CreateProcessA(NULL, cmd_line, NULL, NULL, FALSE, 0u, NULL, NULL, &si, &pis[i])) {
            fprintf(stderr, "CreateProcessA failed for %s (%lu)\n",
                    roles[i], (unsigned long)GetLastError());
            vtlk_kv_arena_close(&arena);
            (void)vtlk_shm_unlink(shm_name);
            return 1;
        }
#else
        children[i] = fork();
        if (children[i] < 0) {
            fprintf(stderr, "fork failed for %s\n", roles[i]);
            vtlk_kv_arena_close(&arena);
            (void)vtlk_shm_unlink(shm_name);
            return 1;
        }
        if (children[i] == 0) {
            execl(exe_path, exe_path, "--worker", roles[i], shm_name, snap_str, (char *)NULL);
            _exit(127);
        }
#endif
    }

    /* Wait for all 3 children. */
    for (i = 0u; i < COW_NUM_WORKERS; ++i) {
#ifdef _WIN32
        DWORD exit_code = 1u;
        WaitForSingleObject(pis[i].hProcess, INFINITE);
        GetExitCodeProcess(pis[i].hProcess, &exit_code);
        CloseHandle(pis[i].hThread);
        CloseHandle(pis[i].hProcess);
        assert(exit_code == 0u);
#else
        int status = 0;
        assert(waitpid(children[i], &status, 0) >= 0);
        assert(WIFEXITED(status) && WEXITSTATUS(status) == 0);
#endif
    }

    /* Release base branch and verify shared blocks & divergent blocks. */
    assert(vtlk_kv_branch_release(&arena, &base_branch) == VTLK_OK);

    /* Release the snapshot pin on shared blocks to verify refcount == 0 on all blocks. */
    for (i = 0u; i < COW_SHARED_BLOCKS; ++i) {
        uint32_t bid = shared_ids[i];
        const uint8_t *bdata = (const uint8_t *)vtlk_kv_block_data(&arena, bid);
        vtlk_kv_block_header_t *hdr = vtlk_kv_block_header(&arena, bid);
        assert(hdr != NULL);
        /* Only the snapshot pin (refcount == 1) remains before releasing it. */
        assert(hdr->refcount == 1u);
        assert(verify_kv_pattern(bdata, COW_BLOCK_SIZE, 1u, i));
        assert(vtlk_kv_block_release(&arena, bid) == VTLK_OK);
    }

    for (i = 0u; i < COW_ARENA_BLOCKS; ++i) {
        vtlk_kv_block_header_t *hdr = vtlk_kv_block_header(&arena, i);
        assert(hdr != NULL);
        assert(hdr->refcount == 0u);
    }

    vtlk_kv_arena_close(&arena);
    (void)vtlk_shm_unlink(shm_name);
    printf("Multi-process CoW stress test passed (10000 tokens, 3 concurrent workers, refcount==0).\n");
    return 0;
}

int main(int argc, char **argv)
{
    if (argc == 5 && strcmp(argv[1], "--worker") == 0) {
        return run_worker(argv[2], argv[3], argv[4]);
    }

    test_kv_hash_chaining();
    test_kv_arena_dedup_and_lru();
    test_kv_snapshot_fork_restore_mismatch();
    test_desc_ring_and_notify();
    test_tier_l1_l2_l3_persistence();
    printf("All KV/Sync/Tier unit tests passed.\n");

    return run_multiproc_cow_stress(argv[0]);
}
