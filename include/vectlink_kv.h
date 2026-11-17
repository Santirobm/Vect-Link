#ifndef VECTLINK_KV_H
#define VECTLINK_KV_H

#include "vectlink/protocol.h"
#include "vectlink/transport.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define VTLK_KV_MAGIC            0x56544B56u /* "VTKV" */
#define VTLK_KV_FORMAT_VERSION   2u
#define VTLK_KV_MAX_BRANCH_BLOCKS 1024u
#define VTLK_KV_MAX_SNAPSHOTS    256u
#define VTLK_KV_INVALID_BLOCK_ID 0xFFFFFFFFu
#define VTLK_KV_HASH_EMPTY       0xFFFFFFFFu
#define VTLK_KV_HASH_TOMBSTONE   0xFFFFFFFEu
#define VTLK_KV_CACHE_LINE       64u

typedef enum vtlk_kv_dtype {
    VTLK_KV_DTYPE_FP16 = 0,
    VTLK_KV_DTYPE_BF16 = 1,
    VTLK_KV_DTYPE_Q8_0 = 2,
    VTLK_KV_DTYPE_Q4_0 = 3,
    VTLK_KV_DTYPE_FP32 = 4,
    VTLK_KV_DTYPE_RAW_LLAMA = 5
} vtlk_kv_dtype_t;

typedef enum vtlk_kv_block_flags {
    VTLK_KV_FLAG_NONE      = 0u,
    VTLK_KV_FLAG_IMMUTABLE = (1u << 0),
    VTLK_KV_FLAG_EVICTABLE = (1u << 1),
    VTLK_KV_FLAG_DIRTY     = (1u << 2),
    VTLK_KV_FLAG_TIER_L2   = (1u << 3),
    VTLK_KV_FLAG_TIER_L3   = (1u << 4)
} vtlk_kv_block_flags_t;

typedef enum vtlk_kv_snapshot_state {
    VTLK_KV_SNAP_FREE    = 0u,
    VTLK_KV_SNAP_WRITING = 1u,
    VTLK_KV_SNAP_READY   = 2u
} vtlk_kv_snapshot_state_t;

/**
 * @brief Content-addressed identity of a KV cache block.
 */
typedef struct vtlk_kv_identity {
    uint64_t token_prefix_hash;
    uint64_t model_arch_hash;
    uint64_t rope_config_hash;
    uint64_t lora_adapter_hash;
    uint32_t pos_offset;
    uint32_t token_count;
    uint32_t kv_dtype;
    uint32_t layer_count;
    uint32_t head_dim;
    uint32_t kv_head_count;
} vtlk_kv_identity_t;

/**
 * @brief 128-byte (2 x 64B cache lines) per-block metadata header in SHM.
 *
 * Bytes [0..79] hold immutable metadata written before seal.
 * Bytes [80..127] hold mutable atomic state updated concurrently.
 */
typedef struct vtlk_kv_block_header {
    /* Cache line 0 (bytes 0..63): immutable metadata */
    uint32_t magic;
    uint32_t storage_format_version;
    uint32_t kv_dtype;
    uint32_t token_count;
    uint64_t block_size_bytes;
    uint64_t token_prefix_hash;
    uint64_t model_arch_hash;
    uint64_t rope_config_hash;
    uint64_t lora_adapter_hash;
    uint32_t pos_offset;
    uint32_t payload_crc32c;

    /* Cache line 1 (bytes 64..127): remaining immutable + mutable atomic state */
    uint32_t layer_count;
    uint32_t head_dim;
    uint32_t kv_head_count;
    uint32_t reserved_imm;

    uint32_t refcount;
    uint32_t flags;
    uint64_t lru_stamp;
    uint32_t tier_state;
    uint32_t reserved_mut;
    uint64_t l2_offset;
    uint64_t l3_id;
    uint64_t reserved_end;
} vtlk_kv_block_header_t;

/**
 * @brief 256-byte SHM arena control block at offset 0 of the shared region.
 */
typedef struct vtlk_kv_shm_ctrl {
    uint32_t magic;
    uint32_t version;
    uint64_t total_bytes;
    uint32_t num_blocks;
    uint32_t hash_cap;
    uint64_t block_size_bytes;
    uint32_t max_snapshots;
    uint32_t reserved0;
    uint64_t hdr_offset;
    uint64_t freelist_next_offset;
    uint64_t hash_table_offset;
    uint64_t snapshots_offset;
    uint64_t block_headers_offset;
    uint64_t block_data_offset;
    uint64_t reserved1[5];

    /* Offset 128..255: atomic counters */
    uint64_t freelist_head;    /* [32-bit generation | 32-bit block_id] */
    uint64_t lru_clock;
    uint64_t next_snapshot_id;
    uint32_t active_blocks;
    uint32_t cached_blocks;
    uint64_t cache_hits;
    uint64_t cache_misses;
    uint64_t evictions;
    uint64_t reserved2[9];
} vtlk_kv_shm_ctrl_t;

/**
 * @brief Snapshot entry stored in the SHM snapshot registry.
 */
typedef struct vtlk_kv_snapshot {
    uint32_t state; /* vtlk_kv_snapshot_state_t */
    uint32_t block_count;
    uint64_t snapshot_id;
    uint64_t parent_snapshot_id;
    uint32_t total_tokens;
    uint32_t kv_dtype;
    uint64_t model_arch_hash;
    uint64_t rope_config_hash;
    uint64_t lora_adapter_hash;
    uint32_t layer_count;
    uint32_t head_dim;
    uint32_t kv_head_count;
    uint32_t reserved0;
    uint32_t block_ids[VTLK_KV_MAX_BRANCH_BLOCKS];
} vtlk_kv_snapshot_t;

/**
 * @brief Caller-owned CoW branch handle referencing retained KV blocks.
 */
typedef struct vtlk_kv_branch {
    uint64_t parent_snapshot_id;
    uint32_t block_count;
    uint32_t total_tokens;
    uint64_t model_arch_hash;
    uint64_t rope_config_hash;
    uint64_t lora_adapter_hash;
    uint32_t kv_dtype;
    uint32_t layer_count;
    uint32_t head_dim;
    uint32_t kv_head_count;
    uint32_t block_ids[VTLK_KV_MAX_BRANCH_BLOCKS];
} vtlk_kv_branch_t;

/**
 * @brief Arena runtime handle (process-local view over SHM).
 */
struct vtlk_tier_store;

typedef struct vtlk_kv_arena {
    vtlk_shm_t shm;
    uint8_t *base;
    size_t total_bytes;
    uint32_t num_blocks;
    uint64_t block_size_bytes;
    uint32_t hash_cap;
    uint32_t max_snapshots;
    int owns_shm;
    uint64_t freelist_next_offset;
    uint64_t hash_table_offset;
    uint64_t snapshots_offset;
    uint64_t block_headers_offset;
    uint64_t block_data_offset;
    struct vtlk_tier_store *tier;
} vtlk_kv_arena_t;

/**
 * @brief Arena telemetry snapshot.
 */
typedef struct vtlk_kv_arena_stats {
    uint32_t num_blocks;
    uint32_t active_blocks;
    uint32_t cached_blocks;
    uint32_t free_blocks;
    uint32_t hash_cap;
    uint32_t reserved0;
    uint64_t block_size_bytes;
    uint64_t total_bytes;
    uint64_t cache_hits;
    uint64_t cache_misses;
    uint64_t evictions;
} vtlk_kv_arena_stats_t;

/* --------------------------------------------------------------------------
 * Content-Addressed Hashing API
 * -------------------------------------------------------------------------- */

/**
 * @brief Computes a chained 64-bit token prefix hash.
 *
 * Preserves the prefix property:
 *   hash(hash(0, A, n), B, m) == hash(0, A||B, n+m).
 */
VTLK_API uint64_t vtlk_kv_hash_tokens(uint64_t prev_hash,
                                      const uint32_t *token_ids,
                                      size_t token_count);

/**
 * @brief Computes a 64-bit hash over arbitrary configuration or adapter bytes.
 */
VTLK_API uint64_t vtlk_kv_hash_bytes(const void *data, size_t len);

/* --------------------------------------------------------------------------
 * Arena Lifecycle API
 * -------------------------------------------------------------------------- */

/**
 * @brief Returns total SHM bytes required for @p num_blocks of @p block_size_bytes.
 *
 * Returns 0 on invalid parameters or integer overflow.
 */
VTLK_API size_t vtlk_kv_arena_required_bytes(uint32_t num_blocks,
                                             uint64_t block_size_bytes);

/**
 * @brief Creates and initializes a new SHM-backed KV arena.
 */
VTLK_API vtlk_status_t vtlk_kv_arena_create(vtlk_kv_arena_t *arena,
                                            const char *shm_name,
                                            uint32_t num_blocks,
                                            uint64_t block_size_bytes);

/**
 * @brief Opens and attaches to an existing SHM-backed KV arena.
 */
VTLK_API vtlk_status_t vtlk_kv_arena_open(vtlk_kv_arena_t *arena,
                                          const char *shm_name);

/**
 * @brief Closes a KV arena handle and unmaps its SHM region.
 */
VTLK_API void vtlk_kv_arena_close(vtlk_kv_arena_t *arena);

/**
 * @brief Reads current atomic arena statistics.
 */
VTLK_API vtlk_status_t vtlk_kv_arena_stats(const vtlk_kv_arena_t *arena,
                                           vtlk_kv_arena_stats_t *out_stats);

/**
 * @brief Attaches an optional hierarchical L1/L2/L3 tier store to the arena.
 *
 * When attached, LRU demotion spills unpinned (refcount == 0) blocks to L2/L3,
 * and vtlk_kv_restore() / vtlk_kv_block_retain() transparently promote them
 * back to L1 with CRC32C verification.
 */
VTLK_API vtlk_status_t vtlk_kv_arena_attach_tier(vtlk_kv_arena_t *arena,
                                                 struct vtlk_tier_store *tier);

/**
 * @brief Demotes up to @p max_demote unpinned (refcount == 0) L1 blocks to L2/L3.
 *
 * Returns the number of demoted blocks in @p out_demoted (optional).
 */
VTLK_API vtlk_status_t vtlk_kv_arena_demote_lru(vtlk_kv_arena_t *arena,
                                                uint32_t max_demote,
                                                uint32_t *out_demoted);

/* --------------------------------------------------------------------------
 * Block Allocator & Deduplication API
 * -------------------------------------------------------------------------- */

/**
 * @brief Allocates a mutable KV block (evicting an unused cached block if needed).
 *
 * Returns with refcount == 1 and VTLK_KV_FLAG_DIRTY set.
 */
VTLK_API vtlk_status_t vtlk_kv_block_alloc(vtlk_kv_arena_t *arena,
                                           uint32_t *out_block_id);

/**
 * @brief Seals a mutable block as immutable, computes CRC32C, and indexes it.
 *
 * Sets VTLK_KV_FLAG_IMMUTABLE | VTLK_KV_FLAG_EVICTABLE and inserts into the
 * SHM content-addressed hash table.
 */
VTLK_API vtlk_status_t vtlk_kv_block_seal(vtlk_kv_arena_t *arena,
                                          uint32_t block_id,
                                          const vtlk_kv_identity_t *identity);

/**
 * @brief Looks up a sealed block by full content-addressed identity.
 *
 * On cache hit, atomically retains the block (incrementing refcount) so it
 * cannot be evicted before the caller uses it, and returns VTLK_OK.
 * Returns VTLK_ERR_EMPTY on cache miss.
 */
VTLK_API vtlk_status_t vtlk_kv_block_lookup(vtlk_kv_arena_t *arena,
                                            const vtlk_kv_identity_t *identity,
                                            uint32_t *out_block_id);

/**
 * @brief Atomically increments a block's refcount (reviving cached refcount==0 blocks).
 *
 * Returns VTLK_ERR_EMPTY if the block has been freed or evicted.
 */
VTLK_API vtlk_status_t vtlk_kv_block_retain(vtlk_kv_arena_t *arena,
                                            uint32_t block_id);

/**
 * @brief Atomically decrements a block's refcount.
 *
 * Unsealed blocks reaching refcount == 0 are immediately returned to the free list.
 * Sealed blocks remain in the cache with refcount == 0 until evicted.
 */
VTLK_API vtlk_status_t vtlk_kv_block_release(vtlk_kv_arena_t *arena,
                                             uint32_t block_id);

/**
 * @brief Returns a direct pointer to a block's 64B-aligned tensor payload in SHM.
 */
VTLK_API void *vtlk_kv_block_data(vtlk_kv_arena_t *arena, uint32_t block_id);

/**
 * @brief Returns a pointer to a block's 128-byte header in SHM.
 */
VTLK_API vtlk_kv_block_header_t *vtlk_kv_block_header(vtlk_kv_arena_t *arena,
                                                      uint32_t block_id);

/* --------------------------------------------------------------------------
 * Git-for-Context CoW Snapshot / Fork / Restore API
 * -------------------------------------------------------------------------- */

/**
 * @brief Creates an immutable CoW snapshot of @p branch in the SHM registry.
 *
 * Retains all referenced blocks so they remain pinned while the snapshot exists.
 */
VTLK_API vtlk_status_t vtlk_kv_snapshot_create(vtlk_kv_arena_t *arena,
                                               const vtlk_kv_branch_t *branch,
                                               uint64_t *out_snapshot_id);

/**
 * @brief Forks a snapshot into a new branch (zero tensor copy).
 *
 * Copies only block_ids[] and atomically increments refcount on each shared block.
 */
VTLK_API vtlk_status_t vtlk_kv_fork(vtlk_kv_arena_t *arena,
                                    uint64_t snapshot_id,
                                    vtlk_kv_branch_t *out_branch);

/**
 * @brief Validates model/RoPE/LoRA/dtype/geometry config and restores a snapshot.
 *
 * Populates @p out_branch (if non-NULL, retaining blocks) and optionally fills
 * @p out_block_ptrs (if non-NULL, up to @p max_ptrs) with zero-copy SHM pointers.
 */
VTLK_API vtlk_status_t vtlk_kv_restore(vtlk_kv_arena_t *arena,
                                       uint64_t snapshot_id,
                                       const vtlk_kv_identity_t *expected_config,
                                       vtlk_kv_branch_t *out_branch,
                                       const void **out_block_ptrs,
                                       uint32_t max_ptrs);

/**
 * @brief Appends a sealed (or active) block to a branch and retains it.
 */
VTLK_API vtlk_status_t vtlk_kv_branch_append_block(vtlk_kv_arena_t *arena,
                                                   vtlk_kv_branch_t *branch,
                                                   uint32_t block_id);

/**
 * @brief Releases all block references held by @p branch and resets it.
 */
VTLK_API vtlk_status_t vtlk_kv_branch_release(vtlk_kv_arena_t *arena,
                                              vtlk_kv_branch_t *branch);

#ifdef __cplusplus
}
#endif

#endif /* VECTLINK_KV_H */
