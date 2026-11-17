#ifndef VECTLINK_TIER_H
#define VECTLINK_TIER_H

#include "vectlink/protocol.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define VTLK_TIER_PATH_MAX 512u

typedef enum vtlk_tier_id {
    VTLK_TIER_NONE = 0,
    VTLK_TIER_L1 = 1,
    VTLK_TIER_L2 = 2,
    VTLK_TIER_L3 = 3
} vtlk_tier_id_t;

typedef struct vtlk_tier_config {
    uint32_t block_size;
    uint32_t l1_capacity;
    uint32_t l2_capacity;
    uint32_t max_blocks;
    uint32_t high_watermark;
    void *l1_mem;
    const char *l2_path;
    const char *l3_dir;
} vtlk_tier_config_t;

typedef struct vtlk_tier_block_meta {
    uint32_t block_id;
    uint8_t tier;
    uint8_t valid;
    uint16_t reserved;
    uint32_t slot_index;
    uint32_t refcount;
    uint32_t crc32c;
    uint64_t lru_tick;
} vtlk_tier_block_meta_t;

typedef struct vtlk_tier_stats {
    uint32_t l1_used;
    uint32_t l2_used;
    uint32_t l3_used;
    uint64_t evictions_l1_to_l2;
    uint64_t evictions_l2_to_l3;
    uint64_t promotions_to_l1;
    uint64_t crc_errors;
} vtlk_tier_stats_t;

typedef struct vtlk_tier_store {
    uint32_t block_size;
    uint32_t l1_capacity;
    uint32_t l2_capacity;
    uint32_t max_blocks;
    uint32_t high_watermark;
    uint8_t *l1_mem;
    uint8_t *l2_mem;
    size_t l2_bytes;
    char l2_path[VTLK_TIER_PATH_MAX];
    char l3_dir[VTLK_TIER_PATH_MAX];
    int has_l2;
    int has_l3;
#ifdef _WIN32
    void *l2_file_handle;
    void *l2_map_handle;
#else
    int l2_fd;
#endif
    vtlk_tier_block_meta_t *blocks;
    uint8_t *l1_slot_used;
    uint8_t *l2_slot_used;
    uint64_t lru_clock;
    vtlk_tier_stats_t stats;
} vtlk_tier_store_t;

/**
 * @brief Initializes a hierarchical L1 (SHM) -> L2 (mmap) -> L3 (SSD) store.
 *
 * Allocates metadata tables once at init (zero malloc in hot paths),
 * maps the L2 backing file if configured, and loads any existing L3
 * persistent index (vtlk_tier_index.bin).
 */
VTLK_API vtlk_status_t vtlk_tier_init(vtlk_tier_store_t *store,
                                      const vtlk_tier_config_t *cfg);

/**
 * @brief Flushes the L3 index, unmaps L2, and frees store metadata.
 */
VTLK_API void vtlk_tier_close(vtlk_tier_store_t *store);

/**
 * @brief Inserts or updates a block in L1, computing its CRC32C.
 *
 * If L1 usage reaches high_watermark, evicts strict-LRU unpinned (refcount == 0)
 * blocks to L2 (and L2 -> L3 if L2 is full).
 */
VTLK_API vtlk_status_t vtlk_tier_put(vtlk_tier_store_t *store,
                                     uint32_t block_id,
                                     const void *data,
                                     size_t len);

/**
 * @brief Acquires a pinned L1 pointer to a block, promoting from L2/L3 if needed.
 *
 * Verifies CRC32C on promotion from L2 or L3 back to L1. Increments refcount
 * so the block cannot be evicted until vtlk_tier_release() is called.
 */
VTLK_API vtlk_status_t vtlk_tier_acquire(vtlk_tier_store_t *store,
                                         uint32_t block_id,
                                         const void **out_ptr);

/**
 * @brief Releases a pin on a block (decrements refcount) and updates LRU tick.
 */
VTLK_API vtlk_status_t vtlk_tier_release(vtlk_tier_store_t *store,
                                         uint32_t block_id);

/**
 * @brief Evicts up to @p target_evict unpinned L1 blocks in strict LRU order.
 */
VTLK_API vtlk_status_t vtlk_tier_evict_l1(vtlk_tier_store_t *store,
                                          uint32_t target_evict);

/**
 * @brief Persists the L3 block index to <l3_dir>/vtlk_tier_index.bin.
 */
VTLK_API vtlk_status_t vtlk_tier_flush_index(vtlk_tier_store_t *store);

/**
 * @brief Retrieves current tier occupancy and eviction/promotion counters.
 */
VTLK_API vtlk_status_t vtlk_tier_stats(const vtlk_tier_store_t *store,
                                       vtlk_tier_stats_t *out_stats);

#ifdef __cplusplus
}
#endif

#endif /* VECTLINK_TIER_H */
