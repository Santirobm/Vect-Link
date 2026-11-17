#include "vectlink_kv.h"
#include "vectlink/checksum.h"
#include "transport/atomic_compat.h"

#include <string.h>

typedef char vtlk_kv_sa_blk_hdr[(sizeof(vtlk_kv_block_header_t) == 128u) ? 1 : -1];
typedef char vtlk_kv_sa_shm_ctrl[(sizeof(vtlk_kv_shm_ctrl_t) == 256u) ? 1 : -1];

typedef struct vtlk_kv_layout {
    uint64_t total_bytes;
    uint32_t hash_cap;
    uint64_t aligned_block_size;
    uint64_t freelist_next_offset;
    uint64_t hash_table_offset;
    uint64_t snapshots_offset;
    uint64_t block_headers_offset;
    uint64_t block_data_offset;
} vtlk_kv_layout_t;

static uint64_t vtlk_kv_align_up_u64(uint64_t val, uint64_t align, int *overflow)
{
    uint64_t rem = val % align;
    if (rem == 0u) {
        return val;
    }
    if (val > UINT64_MAX - (align - rem)) {
        *overflow = 1;
        return 0u;
    }
    return val + (align - rem);
}

static uint64_t vtlk_kv_add_u64(uint64_t a, uint64_t b, int *overflow)
{
    if (a > UINT64_MAX - b) {
        *overflow = 1;
        return 0u;
    }
    return a + b;
}

static uint64_t vtlk_kv_mul_u64(uint64_t a, uint64_t b, int *overflow)
{
    if (a != 0u && b > UINT64_MAX / a) {
        *overflow = 1;
        return 0u;
    }
    return a * b;
}

static uint32_t vtlk_kv_next_pow2_u32(uint32_t v)
{
    uint32_t p = 16u;
    while (p < v && p <= (1u << 30)) {
        p <<= 1u;
    }
    if (p < v) {
        return 0u;
    }
    return p;
}

static int vtlk_kv_compute_layout(uint32_t num_blocks,
                                  uint64_t block_size_bytes,
                                  vtlk_kv_layout_t *out)
{
    int ov = 0;
    uint64_t off = (uint64_t)sizeof(vtlk_kv_shm_ctrl_t);
    uint32_t hash_cap;
    uint64_t aligned_blk;
    uint64_t sz;

    if (num_blocks == 0u || num_blocks > 0x3FFFFFFFu || block_size_bytes == 0u) {
        return 0;
    }

    hash_cap = vtlk_kv_next_pow2_u32(num_blocks * 2u);
    if (hash_cap == 0u) {
        return 0;
    }

    aligned_blk = vtlk_kv_align_up_u64(block_size_bytes, VTLK_KV_CACHE_LINE, &ov);

    out->hash_cap = hash_cap;
    out->aligned_block_size = aligned_blk;

    off = vtlk_kv_align_up_u64(off, VTLK_KV_CACHE_LINE, &ov);
    out->freelist_next_offset = off;
    sz = vtlk_kv_mul_u64((uint64_t)num_blocks, (uint64_t)sizeof(uint32_t), &ov);
    off = vtlk_kv_add_u64(off, sz, &ov);

    off = vtlk_kv_align_up_u64(off, VTLK_KV_CACHE_LINE, &ov);
    out->hash_table_offset = off;
    sz = vtlk_kv_mul_u64((uint64_t)hash_cap, (uint64_t)sizeof(uint32_t), &ov);
    off = vtlk_kv_add_u64(off, sz, &ov);

    off = vtlk_kv_align_up_u64(off, VTLK_KV_CACHE_LINE, &ov);
    out->snapshots_offset = off;
    sz = vtlk_kv_mul_u64((uint64_t)VTLK_KV_MAX_SNAPSHOTS,
                         (uint64_t)sizeof(vtlk_kv_snapshot_t), &ov);
    off = vtlk_kv_add_u64(off, sz, &ov);

    off = vtlk_kv_align_up_u64(off, VTLK_KV_CACHE_LINE, &ov);
    out->block_headers_offset = off;
    sz = vtlk_kv_mul_u64((uint64_t)num_blocks,
                         (uint64_t)sizeof(vtlk_kv_block_header_t), &ov);
    off = vtlk_kv_add_u64(off, sz, &ov);

    off = vtlk_kv_align_up_u64(off, VTLK_KV_CACHE_LINE, &ov);
    out->block_data_offset = off;
    sz = vtlk_kv_mul_u64((uint64_t)num_blocks, aligned_blk, &ov);
    off = vtlk_kv_add_u64(off, sz, &ov);

    if (ov || off > (uint64_t)SIZE_MAX) {
        return 0;
    }

    out->total_bytes = off;
    return 1;
}

static uint64_t vtlk_kv_identity_hash(const vtlk_kv_identity_t *id)
{
    uint64_t buf[7];
    buf[0] = id->token_prefix_hash;
    buf[1] = id->model_arch_hash;
    buf[2] = id->rope_config_hash;
    buf[3] = id->lora_adapter_hash;
    buf[4] = ((uint64_t)id->pos_offset << 32) | (uint64_t)id->token_count;
    buf[5] = ((uint64_t)id->kv_dtype << 32) | (uint64_t)id->layer_count;
    buf[6] = ((uint64_t)id->head_dim << 32) | (uint64_t)id->kv_head_count;
    return vtlk_kv_hash_bytes(buf, sizeof(buf));
}

static int vtlk_kv_identity_matches(const vtlk_kv_block_header_t *hdr,
                                    const vtlk_kv_identity_t *id)
{
    return hdr->token_prefix_hash == id->token_prefix_hash &&
           hdr->model_arch_hash == id->model_arch_hash &&
           hdr->rope_config_hash == id->rope_config_hash &&
           hdr->lora_adapter_hash == id->lora_adapter_hash &&
           hdr->pos_offset == id->pos_offset &&
           hdr->token_count == id->token_count &&
           hdr->kv_dtype == id->kv_dtype &&
           hdr->layer_count == id->layer_count &&
           hdr->head_dim == id->head_dim &&
           hdr->kv_head_count == id->kv_head_count;
}

static vtlk_kv_shm_ctrl_t *vtlk_kv_ctrl(const vtlk_kv_arena_t *arena)
{
    return (vtlk_kv_shm_ctrl_t *)(void *)arena->base;
}

static uint32_t *vtlk_kv_freelist_next(const vtlk_kv_arena_t *arena)
{
    return (uint32_t *)(void *)(arena->base + arena->freelist_next_offset);
}

static uint32_t *vtlk_kv_hash_table(const vtlk_kv_arena_t *arena)
{
    return (uint32_t *)(void *)(arena->base + arena->hash_table_offset);
}

static uint32_t vtlk_kv_freelist_pop(vtlk_kv_arena_t *arena)
{
    vtlk_kv_shm_ctrl_t *ctrl = vtlk_kv_ctrl(arena);
    uint32_t *next_arr = vtlk_kv_freelist_next(arena);
    uint64_t head = vtlk_atomic_load_acquire_u64(&ctrl->freelist_head);

    for (;;) {
        uint32_t id = (uint32_t)(head & 0xFFFFFFFFu);
        uint32_t gen = (uint32_t)(head >> 32);
        uint32_t next_id;
        uint64_t desired;

        if (id == VTLK_KV_INVALID_BLOCK_ID || id >= arena->num_blocks) {
            return VTLK_KV_INVALID_BLOCK_ID;
        }

        next_id = vtlk_atomic_load_acquire_u32(&next_arr[id]);
        desired = ((uint64_t)(gen + 1u) << 32) | (uint64_t)next_id;
        if (vtlk_atomic_cas_u64(&ctrl->freelist_head, &head, desired)) {
            return id;
        }
    }
}

static void vtlk_kv_freelist_push(vtlk_kv_arena_t *arena, uint32_t block_id)
{
    vtlk_kv_shm_ctrl_t *ctrl = vtlk_kv_ctrl(arena);
    uint32_t *next_arr = vtlk_kv_freelist_next(arena);
    uint64_t head = vtlk_atomic_load_acquire_u64(&ctrl->freelist_head);

    for (;;) {
        uint32_t cur_id = (uint32_t)(head & 0xFFFFFFFFu);
        uint32_t gen = (uint32_t)(head >> 32);
        uint64_t desired;

        vtlk_atomic_store_release_u32(&next_arr[block_id], cur_id);
        desired = ((uint64_t)(gen + 1u) << 32) | (uint64_t)block_id;
        if (vtlk_atomic_cas_u64(&ctrl->freelist_head, &head, desired)) {
            return;
        }
    }
}

static void vtlk_kv_hash_remove_block(vtlk_kv_arena_t *arena,
                                      uint32_t block_id,
                                      uint64_t identity_hash)
{
    uint32_t *table = vtlk_kv_hash_table(arena);
    uint32_t mask = arena->hash_cap - 1u;
    uint32_t start = (uint32_t)identity_hash & mask;
    uint32_t step;

    for (step = 0u; step < arena->hash_cap; ++step) {
        uint32_t idx = (start + step) & mask;
        uint32_t entry = vtlk_atomic_load_acquire_u32(&table[idx]);
        if (entry == VTLK_KV_HASH_EMPTY) {
            return;
        }
        if (entry == block_id) {
            if (vtlk_atomic_cas_u32(&table[idx], &entry, VTLK_KV_HASH_TOMBSTONE)) {
                return;
            }
        }
    }
}

static uint32_t vtlk_kv_evict_lru_block(vtlk_kv_arena_t *arena)
{
    vtlk_kv_shm_ctrl_t *ctrl = vtlk_kv_ctrl(arena);
    uint32_t attempt;

    for (attempt = 0u; attempt < 4u; ++attempt) {
        uint32_t best_id = VTLK_KV_INVALID_BLOCK_ID;
        uint64_t best_lru = UINT64_MAX;
        uint32_t i;

        for (i = 0u; i < arena->num_blocks; ++i) {
            vtlk_kv_block_header_t *hdr = vtlk_kv_block_header(arena, i);
            uint32_t flags = vtlk_atomic_load_acquire_u32(&hdr->flags);
            if ((flags & VTLK_KV_FLAG_EVICTABLE) != 0u &&
                vtlk_atomic_load_acquire_u32(&hdr->refcount) == 0u) {
                uint64_t stamp = vtlk_atomic_load_acquire_u64(&hdr->lru_stamp);
                if (best_id == VTLK_KV_INVALID_BLOCK_ID || stamp < best_lru) {
                    best_id = i;
                    best_lru = stamp;
                }
            }
        }

        if (best_id == VTLK_KV_INVALID_BLOCK_ID) {
            return VTLK_KV_INVALID_BLOCK_ID;
        }

        {
            vtlk_kv_block_header_t *hdr = vtlk_kv_block_header(arena, best_id);
            uint32_t flags = vtlk_atomic_load_acquire_u32(&hdr->flags);
            if ((flags & VTLK_KV_FLAG_EVICTABLE) == 0u) {
                continue;
            }
            if (vtlk_atomic_load_acquire_u32(&hdr->refcount) != 0u) {
                continue;
            }
            if (!vtlk_atomic_cas_u32(&hdr->flags, &flags, 0u)) {
                continue;
            }
            if (vtlk_atomic_load_acquire_u32(&hdr->refcount) != 0u) {
                vtlk_atomic_store_release_u32(&hdr->flags, flags);
                continue;
            }

            {
                vtlk_kv_identity_t id;
                id.token_prefix_hash = hdr->token_prefix_hash;
                id.model_arch_hash = hdr->model_arch_hash;
                id.rope_config_hash = hdr->rope_config_hash;
                id.lora_adapter_hash = hdr->lora_adapter_hash;
                id.pos_offset = hdr->pos_offset;
                id.token_count = hdr->token_count;
                id.kv_dtype = hdr->kv_dtype;
                id.layer_count = hdr->layer_count;
                id.head_dim = hdr->head_dim;
                id.kv_head_count = hdr->kv_head_count;
                vtlk_kv_hash_remove_block(arena, best_id,
                                          vtlk_kv_identity_hash(&id));
            }

            vtlk_atomic_fetch_sub_u32(&ctrl->cached_blocks, 1u);
            vtlk_atomic_fetch_add_u64(&ctrl->evictions, 1u);
            return best_id;
        }
    }

    return VTLK_KV_INVALID_BLOCK_ID;
}

size_t vtlk_kv_arena_required_bytes(uint32_t num_blocks,
                                    uint64_t block_size_bytes)
{
    vtlk_kv_layout_t layout;
    if (!vtlk_kv_compute_layout(num_blocks, block_size_bytes, &layout)) {
        return 0u;
    }
    return (size_t)layout.total_bytes;
}

vtlk_status_t vtlk_kv_arena_create(vtlk_kv_arena_t *arena,
                                   const char *shm_name,
                                   uint32_t num_blocks,
                                   uint64_t block_size_bytes)
{
    vtlk_kv_layout_t layout;
    vtlk_status_t st;
    vtlk_kv_shm_ctrl_t *ctrl;
    uint32_t *next_arr;
    uint32_t *table;
    uint32_t i;

    if (arena == NULL || shm_name == NULL) {
        return VTLK_ERR_NULL_PTR;
    }
    if (!vtlk_kv_compute_layout(num_blocks, block_size_bytes, &layout)) {
        return VTLK_ERR_INVALID_PARAM;
    }

    memset(arena, 0, sizeof(*arena));
    st = vtlk_shm_create(&arena->shm, shm_name, (size_t)layout.total_bytes);
    if (st != VTLK_OK) {
        return st;
    }

    arena->base = (uint8_t *)arena->shm.addr;
    arena->total_bytes = (size_t)layout.total_bytes;
    arena->num_blocks = num_blocks;
    arena->block_size_bytes = block_size_bytes;
    arena->hash_cap = layout.hash_cap;
    arena->max_snapshots = VTLK_KV_MAX_SNAPSHOTS;
    arena->owns_shm = 1;
    arena->freelist_next_offset = layout.freelist_next_offset;
    arena->hash_table_offset = layout.hash_table_offset;
    arena->snapshots_offset = layout.snapshots_offset;
    arena->block_headers_offset = layout.block_headers_offset;
    arena->block_data_offset = layout.block_data_offset;

    memset(arena->base, 0, arena->total_bytes);

    next_arr = vtlk_kv_freelist_next(arena);
    for (i = 0u; i < num_blocks; ++i) {
        next_arr[i] = (i + 1u < num_blocks) ? (i + 1u) : VTLK_KV_INVALID_BLOCK_ID;
    }

    table = vtlk_kv_hash_table(arena);
    for (i = 0u; i < layout.hash_cap; ++i) {
        table[i] = VTLK_KV_HASH_EMPTY;
    }

    ctrl = vtlk_kv_ctrl(arena);
    ctrl->version = VTLK_KV_FORMAT_VERSION;
    ctrl->total_bytes = layout.total_bytes;
    ctrl->num_blocks = num_blocks;
    ctrl->hash_cap = layout.hash_cap;
    ctrl->block_size_bytes = block_size_bytes;
    ctrl->max_snapshots = VTLK_KV_MAX_SNAPSHOTS;
    ctrl->hdr_offset = 0u;
    ctrl->freelist_next_offset = layout.freelist_next_offset;
    ctrl->hash_table_offset = layout.hash_table_offset;
    ctrl->snapshots_offset = layout.snapshots_offset;
    ctrl->block_headers_offset = layout.block_headers_offset;
    ctrl->block_data_offset = layout.block_data_offset;
    ctrl->freelist_head = 0ULL;
    vtlk_atomic_store_release_u32(&ctrl->magic, VTLK_KV_MAGIC);

    return VTLK_OK;
}

vtlk_status_t vtlk_kv_arena_open(vtlk_kv_arena_t *arena, const char *shm_name)
{
    vtlk_shm_t probe;
    vtlk_status_t st;
    const vtlk_kv_shm_ctrl_t *probe_ctrl;
    uint32_t magic;
    uint32_t version;
    uint64_t total_bytes;
    vtlk_kv_shm_ctrl_t *ctrl;

    if (arena == NULL || shm_name == NULL) {
        return VTLK_ERR_NULL_PTR;
    }

    memset(arena, 0, sizeof(*arena));
    memset(&probe, 0, sizeof(probe));

    st = vtlk_shm_open(&probe, shm_name, sizeof(vtlk_kv_shm_ctrl_t));
    if (st != VTLK_OK) {
        return st;
    }

    probe_ctrl = (const vtlk_kv_shm_ctrl_t *)probe.addr;
    magic = vtlk_atomic_load_acquire_u32(&probe_ctrl->magic);
    version = probe_ctrl->version;
    total_bytes = probe_ctrl->total_bytes;
    vtlk_shm_close(&probe);

    if (magic != VTLK_KV_MAGIC) {
        return VTLK_ERR_INVALID_MAGIC;
    }
    if (version != VTLK_KV_FORMAT_VERSION) {
        return VTLK_ERR_UNSUPPORTED_VERSION;
    }
    if (total_bytes < sizeof(vtlk_kv_shm_ctrl_t) ||
        total_bytes > (uint64_t)SIZE_MAX) {
        return VTLK_ERR_INVALID_HEADER_SIZE;
    }

    st = vtlk_shm_open(&arena->shm, shm_name, (size_t)total_bytes);
    if (st != VTLK_OK) {
        return st;
    }

    arena->base = (uint8_t *)arena->shm.addr;
    arena->total_bytes = (size_t)total_bytes;
    ctrl = vtlk_kv_ctrl(arena);
    arena->num_blocks = ctrl->num_blocks;
    arena->block_size_bytes = ctrl->block_size_bytes;
    arena->hash_cap = ctrl->hash_cap;
    arena->max_snapshots = ctrl->max_snapshots;
    arena->owns_shm = 0;
    arena->freelist_next_offset = ctrl->freelist_next_offset;
    arena->hash_table_offset = ctrl->hash_table_offset;
    arena->snapshots_offset = ctrl->snapshots_offset;
    arena->block_headers_offset = ctrl->block_headers_offset;
    arena->block_data_offset = ctrl->block_data_offset;

    return VTLK_OK;
}

void vtlk_kv_arena_close(vtlk_kv_arena_t *arena)
{
    if (arena == NULL) {
        return;
    }
    vtlk_shm_close(&arena->shm);
    memset(arena, 0, sizeof(*arena));
}

vtlk_status_t vtlk_kv_arena_stats(const vtlk_kv_arena_t *arena,
                                   vtlk_kv_arena_stats_t *out_stats)
{
    const vtlk_kv_shm_ctrl_t *ctrl;
    uint32_t active;

    if (arena == NULL || out_stats == NULL) {
        return VTLK_ERR_NULL_PTR;
    }
    if (arena->base == NULL) {
        return VTLK_ERR_INVALID_PARAM;
    }

    ctrl = vtlk_kv_ctrl(arena);
    active = vtlk_atomic_load_acquire_u32(&ctrl->active_blocks);

    memset(out_stats, 0, sizeof(*out_stats));
    out_stats->num_blocks = arena->num_blocks;
    out_stats->active_blocks = active;
    out_stats->cached_blocks = vtlk_atomic_load_acquire_u32(&ctrl->cached_blocks);
    out_stats->free_blocks = (active <= arena->num_blocks)
                                 ? (arena->num_blocks - active)
                                 : 0u;
    out_stats->hash_cap = arena->hash_cap;
    out_stats->block_size_bytes = arena->block_size_bytes;
    out_stats->total_bytes = (uint64_t)arena->total_bytes;
    out_stats->cache_hits = vtlk_atomic_load_acquire_u64(&ctrl->cache_hits);
    out_stats->cache_misses = vtlk_atomic_load_acquire_u64(&ctrl->cache_misses);
    out_stats->evictions = vtlk_atomic_load_acquire_u64(&ctrl->evictions);
    return VTLK_OK;
}

vtlk_kv_block_header_t *vtlk_kv_block_header(vtlk_kv_arena_t *arena,
                                             uint32_t block_id)
{
    vtlk_kv_block_header_t *hdrs;
    if (arena == NULL || arena->base == NULL || block_id >= arena->num_blocks) {
        return NULL;
    }
    hdrs = (vtlk_kv_block_header_t *)(void *)(arena->base +
                                              arena->block_headers_offset);
    return &hdrs[block_id];
}

void *vtlk_kv_block_data(vtlk_kv_arena_t *arena, uint32_t block_id)
{
    int ov = 0;
    uint64_t aligned_blk;
    if (arena == NULL || arena->base == NULL || block_id >= arena->num_blocks) {
        return NULL;
    }
    aligned_blk = vtlk_kv_align_up_u64(arena->block_size_bytes,
                                       VTLK_KV_CACHE_LINE, &ov);
    return (void *)(arena->base + arena->block_data_offset +
                    ((uint64_t)block_id * aligned_blk));
}

vtlk_status_t vtlk_kv_block_alloc(vtlk_kv_arena_t *arena,
                                  uint32_t *out_block_id)
{
    vtlk_kv_shm_ctrl_t *ctrl;
    uint32_t id;
    int from_freelist = 1;
    vtlk_kv_block_header_t *hdr;

    if (arena == NULL || out_block_id == NULL) {
        return VTLK_ERR_NULL_PTR;
    }
    if (arena->base == NULL) {
        return VTLK_ERR_INVALID_PARAM;
    }

    *out_block_id = VTLK_KV_INVALID_BLOCK_ID;
    ctrl = vtlk_kv_ctrl(arena);

    id = vtlk_kv_freelist_pop(arena);
    if (id == VTLK_KV_INVALID_BLOCK_ID) {
        id = vtlk_kv_evict_lru_block(arena);
        if (id == VTLK_KV_INVALID_BLOCK_ID) {
            return VTLK_ERR_FULL;
        }
        from_freelist = 0;
    }

    hdr = vtlk_kv_block_header(arena, id);
    memset(hdr, 0, sizeof(*hdr));
    hdr->magic = VTLK_KV_MAGIC;
    hdr->storage_format_version = VTLK_KV_FORMAT_VERSION;
    hdr->block_size_bytes = arena->block_size_bytes;
    vtlk_atomic_store_release_u32(&hdr->refcount, 1u);
    vtlk_atomic_store_release_u64(&hdr->lru_stamp,
                                  vtlk_atomic_fetch_add_u64(&ctrl->lru_clock, 1u) + 1u);
    vtlk_atomic_store_release_u32(&hdr->flags, VTLK_KV_FLAG_DIRTY);

    if (from_freelist) {
        vtlk_atomic_fetch_add_u32(&ctrl->active_blocks, 1u);
    }

    *out_block_id = id;
    return VTLK_OK;
}

vtlk_status_t vtlk_kv_block_seal(vtlk_kv_arena_t *arena,
                                 uint32_t block_id,
                                 const vtlk_kv_identity_t *identity)
{
    vtlk_kv_shm_ctrl_t *ctrl;
    vtlk_kv_block_header_t *hdr;
    uint32_t flags;
    const void *data;
    uint32_t *table;
    uint32_t mask;
    uint32_t start;
    uint32_t step;

    if (arena == NULL || identity == NULL) {
        return VTLK_ERR_NULL_PTR;
    }
    hdr = vtlk_kv_block_header(arena, block_id);
    if (hdr == NULL) {
        return VTLK_ERR_INVALID_PARAM;
    }

    flags = vtlk_atomic_load_acquire_u32(&hdr->flags);
    if (flags == 0u || (flags & VTLK_KV_FLAG_IMMUTABLE) != 0u) {
        return VTLK_ERR_BUSY;
    }
    if (vtlk_atomic_load_acquire_u32(&hdr->refcount) == 0u) {
        return VTLK_ERR_INVALID_PARAM;
    }

    data = vtlk_kv_block_data(arena, block_id);
    hdr->magic = VTLK_KV_MAGIC;
    hdr->storage_format_version = VTLK_KV_FORMAT_VERSION;
    hdr->kv_dtype = identity->kv_dtype;
    hdr->token_count = identity->token_count;
    hdr->block_size_bytes = arena->block_size_bytes;
    hdr->token_prefix_hash = identity->token_prefix_hash;
    hdr->model_arch_hash = identity->model_arch_hash;
    hdr->rope_config_hash = identity->rope_config_hash;
    hdr->lora_adapter_hash = identity->lora_adapter_hash;
    hdr->pos_offset = identity->pos_offset;
    hdr->layer_count = identity->layer_count;
    hdr->head_dim = identity->head_dim;
    hdr->kv_head_count = identity->kv_head_count;
    hdr->payload_crc32c = vtlk_crc32c(data, (size_t)arena->block_size_bytes);

    ctrl = vtlk_kv_ctrl(arena);
    vtlk_atomic_store_release_u64(&hdr->lru_stamp,
                                  vtlk_atomic_fetch_add_u64(&ctrl->lru_clock, 1u) + 1u);
    vtlk_atomic_store_release_u32(&hdr->flags,
                                  VTLK_KV_FLAG_IMMUTABLE | VTLK_KV_FLAG_EVICTABLE);

    table = vtlk_kv_hash_table(arena);
    mask = arena->hash_cap - 1u;
    start = (uint32_t)vtlk_kv_identity_hash(identity) & mask;

    for (step = 0u; step < arena->hash_cap; ++step) {
        uint32_t idx = (start + step) & mask;
        uint32_t entry = vtlk_atomic_load_acquire_u32(&table[idx]);
        if (entry == VTLK_KV_HASH_EMPTY || entry == VTLK_KV_HASH_TOMBSTONE) {
            if (vtlk_atomic_cas_u32(&table[idx], &entry, block_id)) {
                vtlk_atomic_fetch_add_u32(&ctrl->cached_blocks, 1u);
                return VTLK_OK;
            }
        }
    }

    return VTLK_ERR_FULL;
}

vtlk_status_t vtlk_kv_block_lookup(vtlk_kv_arena_t *arena,
                                   const vtlk_kv_identity_t *identity,
                                   uint32_t *out_block_id)
{
    vtlk_kv_shm_ctrl_t *ctrl;
    uint32_t *table;
    uint32_t mask;
    uint32_t start;
    uint32_t step;

    if (arena == NULL || identity == NULL || out_block_id == NULL) {
        return VTLK_ERR_NULL_PTR;
    }
    if (arena->base == NULL) {
        return VTLK_ERR_INVALID_PARAM;
    }

    *out_block_id = VTLK_KV_INVALID_BLOCK_ID;
    ctrl = vtlk_kv_ctrl(arena);
    table = vtlk_kv_hash_table(arena);
    mask = arena->hash_cap - 1u;
    start = (uint32_t)vtlk_kv_identity_hash(identity) & mask;

    for (step = 0u; step < arena->hash_cap; ++step) {
        uint32_t idx = (start + step) & mask;
        uint32_t entry = vtlk_atomic_load_acquire_u32(&table[idx]);
        vtlk_kv_block_header_t *hdr;
        uint32_t flags;

        if (entry == VTLK_KV_HASH_EMPTY) {
            break;
        }
        if (entry == VTLK_KV_HASH_TOMBSTONE || entry >= arena->num_blocks) {
            continue;
        }

        hdr = vtlk_kv_block_header(arena, entry);
        flags = vtlk_atomic_load_acquire_u32(&hdr->flags);
        if ((flags & VTLK_KV_FLAG_IMMUTABLE) != 0u &&
            vtlk_kv_identity_matches(hdr, identity)) {
            if (vtlk_kv_block_retain(arena, entry) == VTLK_OK) {
                vtlk_atomic_fetch_add_u64(&ctrl->cache_hits, 1u);
                *out_block_id = entry;
                return VTLK_OK;
            }
        }
    }

    vtlk_atomic_fetch_add_u64(&ctrl->cache_misses, 1u);
    return VTLK_ERR_EMPTY;
}

vtlk_status_t vtlk_kv_block_retain(vtlk_kv_arena_t *arena, uint32_t block_id)
{
    vtlk_kv_shm_ctrl_t *ctrl;
    vtlk_kv_block_header_t *hdr;

    if (arena == NULL) {
        return VTLK_ERR_NULL_PTR;
    }
    hdr = vtlk_kv_block_header(arena, block_id);
    if (hdr == NULL) {
        return VTLK_ERR_INVALID_PARAM;
    }

    ctrl = vtlk_kv_ctrl(arena);

    for (;;) {
        uint32_t flags = vtlk_atomic_load_acquire_u32(&hdr->flags);
        uint32_t rc;

        if (flags == 0u) {
            return VTLK_ERR_EMPTY;
        }

        rc = vtlk_atomic_load_acquire_u32(&hdr->refcount);
        if (vtlk_atomic_cas_u32(&hdr->refcount, &rc, rc + 1u)) {
            if (vtlk_atomic_load_acquire_u32(&hdr->flags) == 0u) {
                vtlk_atomic_fetch_sub_u32(&hdr->refcount, 1u);
                return VTLK_ERR_EMPTY;
            }
            vtlk_atomic_store_release_u64(
                &hdr->lru_stamp,
                vtlk_atomic_fetch_add_u64(&ctrl->lru_clock, 1u) + 1u);
            return VTLK_OK;
        }
    }
}

vtlk_status_t vtlk_kv_block_release(vtlk_kv_arena_t *arena, uint32_t block_id)
{
    vtlk_kv_shm_ctrl_t *ctrl;
    vtlk_kv_block_header_t *hdr;
    uint32_t prev;

    if (arena == NULL) {
        return VTLK_ERR_NULL_PTR;
    }
    hdr = vtlk_kv_block_header(arena, block_id);
    if (hdr == NULL) {
        return VTLK_ERR_INVALID_PARAM;
    }

    ctrl = vtlk_kv_ctrl(arena);
    prev = vtlk_atomic_fetch_sub_u32(&hdr->refcount, 1u);
    if (prev == 0u) {
        vtlk_atomic_fetch_add_u32(&hdr->refcount, 1u);
        return VTLK_ERR_INVALID_PARAM;
    }

    if (prev == 1u) {
        uint32_t flags = vtlk_atomic_load_acquire_u32(&hdr->flags);
        if ((flags & VTLK_KV_FLAG_IMMUTABLE) != 0u) {
            vtlk_atomic_store_release_u64(
                &hdr->lru_stamp,
                vtlk_atomic_fetch_add_u64(&ctrl->lru_clock, 1u) + 1u);
        } else {
            vtlk_atomic_store_release_u32(&hdr->flags, 0u);
            vtlk_atomic_fetch_sub_u32(&ctrl->active_blocks, 1u);
            vtlk_kv_freelist_push(arena, block_id);
        }
    }

    return VTLK_OK;
}
