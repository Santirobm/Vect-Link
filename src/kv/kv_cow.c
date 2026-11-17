#include "vectlink_kv.h"
#include "transport/atomic_compat.h"

#include <string.h>

static vtlk_kv_shm_ctrl_t *vtlk_kv_cow_ctrl(const vtlk_kv_arena_t *arena)
{
    return (vtlk_kv_shm_ctrl_t *)(void *)arena->base;
}

static vtlk_kv_snapshot_t *vtlk_kv_snapshots(const vtlk_kv_arena_t *arena)
{
    return (vtlk_kv_snapshot_t *)(void *)(arena->base + arena->snapshots_offset);
}

static const vtlk_kv_snapshot_t *vtlk_kv_find_snapshot(const vtlk_kv_arena_t *arena,
                                                       uint64_t snapshot_id)
{
    const vtlk_kv_snapshot_t *snaps = vtlk_kv_snapshots(arena);
    uint32_t i;

    for (i = 0u; i < arena->max_snapshots; ++i) {
        uint32_t st = vtlk_atomic_load_acquire_u32(&snaps[i].state);
        if (st == VTLK_KV_SNAP_READY && snaps[i].snapshot_id == snapshot_id) {
            return &snaps[i];
        }
    }

    return NULL;
}

static int vtlk_kv_config_matches_snap(const vtlk_kv_snapshot_t *snap,
                                       const vtlk_kv_identity_t *cfg)
{
    return snap->model_arch_hash == cfg->model_arch_hash &&
           snap->rope_config_hash == cfg->rope_config_hash &&
           snap->lora_adapter_hash == cfg->lora_adapter_hash &&
           snap->kv_dtype == cfg->kv_dtype &&
           snap->layer_count == cfg->layer_count &&
           snap->head_dim == cfg->head_dim &&
           snap->kv_head_count == cfg->kv_head_count;
}

static int vtlk_kv_config_matches_branch(const vtlk_kv_branch_t *branch,
                                         const vtlk_kv_block_header_t *hdr)
{
    return branch->model_arch_hash == hdr->model_arch_hash &&
           branch->rope_config_hash == hdr->rope_config_hash &&
           branch->lora_adapter_hash == hdr->lora_adapter_hash &&
           branch->kv_dtype == hdr->kv_dtype &&
           branch->layer_count == hdr->layer_count &&
           branch->head_dim == hdr->head_dim &&
           branch->kv_head_count == hdr->kv_head_count;
}

vtlk_status_t vtlk_kv_snapshot_create(vtlk_kv_arena_t *arena,
                                      const vtlk_kv_branch_t *branch,
                                      uint64_t *out_snapshot_id)
{
    vtlk_kv_shm_ctrl_t *ctrl;
    vtlk_kv_snapshot_t *snaps;
    uint32_t i;
    uint32_t claimed = UINT32_MAX;

    if (arena == NULL || branch == NULL || out_snapshot_id == NULL) {
        return VTLK_ERR_NULL_PTR;
    }
    if (arena->base == NULL || branch->block_count > VTLK_KV_MAX_BRANCH_BLOCKS) {
        return VTLK_ERR_INVALID_PARAM;
    }

    *out_snapshot_id = 0u;

    /* Retain every referenced block on behalf of the snapshot. */
    for (i = 0u; i < branch->block_count; ++i) {
        vtlk_status_t st = vtlk_kv_block_retain(arena, branch->block_ids[i]);
        if (st != VTLK_OK) {
            uint32_t j;
            for (j = 0u; j < i; ++j) {
                (void)vtlk_kv_block_release(arena, branch->block_ids[j]);
            }
            return st;
        }
    }

    snaps = vtlk_kv_snapshots(arena);
    for (i = 0u; i < arena->max_snapshots; ++i) {
        uint32_t expected = VTLK_KV_SNAP_FREE;
        if (vtlk_atomic_cas_u32(&snaps[i].state, &expected, VTLK_KV_SNAP_WRITING)) {
            claimed = i;
            break;
        }
    }

    if (claimed == UINT32_MAX) {
        for (i = 0u; i < branch->block_count; ++i) {
            (void)vtlk_kv_block_release(arena, branch->block_ids[i]);
        }
        return VTLK_ERR_FULL;
    }

    ctrl = vtlk_kv_cow_ctrl(arena);
    snaps[claimed].snapshot_id =
        vtlk_atomic_fetch_add_u64(&ctrl->next_snapshot_id, 1u) + 1u;
    snaps[claimed].parent_snapshot_id = branch->parent_snapshot_id;
    snaps[claimed].block_count = branch->block_count;
    snaps[claimed].total_tokens = branch->total_tokens;
    snaps[claimed].kv_dtype = branch->kv_dtype;
    snaps[claimed].model_arch_hash = branch->model_arch_hash;
    snaps[claimed].rope_config_hash = branch->rope_config_hash;
    snaps[claimed].lora_adapter_hash = branch->lora_adapter_hash;
    snaps[claimed].layer_count = branch->layer_count;
    snaps[claimed].head_dim = branch->head_dim;
    snaps[claimed].kv_head_count = branch->kv_head_count;
    snaps[claimed].reserved0 = 0u;

    if (branch->block_count > 0u) {
        memcpy(snaps[claimed].block_ids,
               branch->block_ids,
               (size_t)branch->block_count * sizeof(uint32_t));
    }

    *out_snapshot_id = snaps[claimed].snapshot_id;
    vtlk_atomic_store_release_u32(&snaps[claimed].state, VTLK_KV_SNAP_READY);
    return VTLK_OK;
}

vtlk_status_t vtlk_kv_fork(vtlk_kv_arena_t *arena,
                           uint64_t snapshot_id,
                           vtlk_kv_branch_t *out_branch)
{
    const vtlk_kv_snapshot_t *snap;
    uint32_t i;

    if (arena == NULL || out_branch == NULL) {
        return VTLK_ERR_NULL_PTR;
    }
    if (arena->base == NULL || snapshot_id == 0u) {
        return VTLK_ERR_INVALID_PARAM;
    }

    snap = vtlk_kv_find_snapshot(arena, snapshot_id);
    if (snap == NULL) {
        return VTLK_ERR_EMPTY;
    }

    memset(out_branch, 0, sizeof(*out_branch));
    out_branch->parent_snapshot_id = snapshot_id;
    out_branch->block_count = snap->block_count;
    out_branch->total_tokens = snap->total_tokens;
    out_branch->model_arch_hash = snap->model_arch_hash;
    out_branch->rope_config_hash = snap->rope_config_hash;
    out_branch->lora_adapter_hash = snap->lora_adapter_hash;
    out_branch->kv_dtype = snap->kv_dtype;
    out_branch->layer_count = snap->layer_count;
    out_branch->head_dim = snap->head_dim;
    out_branch->kv_head_count = snap->kv_head_count;

    if (snap->block_count > 0u) {
        memcpy(out_branch->block_ids,
               snap->block_ids,
               (size_t)snap->block_count * sizeof(uint32_t));
    }

    for (i = 0u; i < out_branch->block_count; ++i) {
        vtlk_status_t st = vtlk_kv_block_retain(arena, out_branch->block_ids[i]);
        if (st != VTLK_OK) {
            uint32_t j;
            for (j = 0u; j < i; ++j) {
                (void)vtlk_kv_block_release(arena, out_branch->block_ids[j]);
            }
            memset(out_branch, 0, sizeof(*out_branch));
            return st;
        }
    }

    return VTLK_OK;
}

vtlk_status_t vtlk_kv_restore(vtlk_kv_arena_t *arena,
                              uint64_t snapshot_id,
                              const vtlk_kv_identity_t *expected_config,
                              vtlk_kv_branch_t *out_branch,
                              const void **out_block_ptrs,
                              uint32_t max_ptrs)
{
    const vtlk_kv_snapshot_t *snap;
    uint32_t i;

    if (arena == NULL) {
        return VTLK_ERR_NULL_PTR;
    }
    if (arena->base == NULL || snapshot_id == 0u) {
        return VTLK_ERR_INVALID_PARAM;
    }

    snap = vtlk_kv_find_snapshot(arena, snapshot_id);
    if (snap == NULL) {
        return VTLK_ERR_EMPTY;
    }

    if (expected_config != NULL && !vtlk_kv_config_matches_snap(snap, expected_config)) {
        return VTLK_ERR_INVALID_PARAM;
    }

    if (out_block_ptrs != NULL && max_ptrs < snap->block_count) {
        return VTLK_ERR_BUFFER_TOO_SMALL;
    }

    if (out_branch != NULL) {
        vtlk_status_t st = vtlk_kv_fork(arena, snapshot_id, out_branch);
        if (st != VTLK_OK) {
            return st;
        }
    }

    if (out_block_ptrs != NULL) {
        for (i = 0u; i < snap->block_count; ++i) {
            out_block_ptrs[i] = vtlk_kv_block_data(arena, snap->block_ids[i]);
        }
    }

    return VTLK_OK;
}

vtlk_status_t vtlk_kv_branch_append_block(vtlk_kv_arena_t *arena,
                                          vtlk_kv_branch_t *branch,
                                          uint32_t block_id)
{
    vtlk_kv_block_header_t *hdr;
    vtlk_status_t st;

    if (arena == NULL || branch == NULL) {
        return VTLK_ERR_NULL_PTR;
    }
    if (arena->base == NULL) {
        return VTLK_ERR_INVALID_PARAM;
    }
    if (branch->block_count >= VTLK_KV_MAX_BRANCH_BLOCKS) {
        return VTLK_ERR_FULL;
    }

    hdr = vtlk_kv_block_header(arena, block_id);
    if (hdr == NULL || vtlk_atomic_load_acquire_u32(&hdr->flags) == 0u) {
        return VTLK_ERR_INVALID_PARAM;
    }

    if (branch->block_count == 0u) {
        branch->model_arch_hash = hdr->model_arch_hash;
        branch->rope_config_hash = hdr->rope_config_hash;
        branch->lora_adapter_hash = hdr->lora_adapter_hash;
        branch->kv_dtype = hdr->kv_dtype;
        branch->layer_count = hdr->layer_count;
        branch->head_dim = hdr->head_dim;
        branch->kv_head_count = hdr->kv_head_count;
    } else if (!vtlk_kv_config_matches_branch(branch, hdr)) {
        return VTLK_ERR_INVALID_PARAM;
    }

    st = vtlk_kv_block_retain(arena, block_id);
    if (st != VTLK_OK) {
        return st;
    }

    branch->block_ids[branch->block_count++] = block_id;
    branch->total_tokens += hdr->token_count;
    return VTLK_OK;
}

vtlk_status_t vtlk_kv_branch_release(vtlk_kv_arena_t *arena,
                                     vtlk_kv_branch_t *branch)
{
    uint32_t i;

    if (arena == NULL || branch == NULL) {
        return VTLK_ERR_NULL_PTR;
    }
    if (arena->base == NULL || branch->block_count > VTLK_KV_MAX_BRANCH_BLOCKS) {
        return VTLK_ERR_INVALID_PARAM;
    }

    for (i = 0u; i < branch->block_count; ++i) {
        (void)vtlk_kv_block_release(arena, branch->block_ids[i]);
    }

    branch->block_count = 0u;
    branch->total_tokens = 0u;
    return VTLK_OK;
}
