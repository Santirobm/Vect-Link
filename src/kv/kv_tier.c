#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

#include "vectlink_tier.h"
#include "vectlink/checksum.h"
#include "protocol/endian.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#define VTLK_IDX_MAGIC_LEN 8u
#define VTLK_IDX_VERSION   1u
#define VTLK_IDX_HDR_SIZE  24u
#define VTLK_IDX_ENT_SIZE  16u

static const uint8_t VTLK_IDX_MAGIC[VTLK_IDX_MAGIC_LEN] = {
    'V', 'T', 'L', 'K', '_', 'I', 'D', 'X'
};

static int vtlk_tier_format_path(char *dst,
                                 size_t cap,
                                 const char *dir,
                                 const char *file)
{
    int n = snprintf(dst, cap, "%s/%s", dir, file);
    if (n < 0 || (size_t)n >= cap) {
        return 0;
    }
    return 1;
}

static int vtlk_tier_format_block_path(char *dst,
                                       size_t cap,
                                       const char *dir,
                                       uint32_t block_id)
{
    int n = snprintf(dst, cap, "%s/blk_%u.bin", dir, (unsigned)block_id);
    if (n < 0 || (size_t)n >= cap) {
        return 0;
    }
    return 1;
}

static vtlk_status_t vtlk_tier_ensure_dir(const char *dir)
{
#ifdef _WIN32
    if (!CreateDirectoryA(dir, NULL)) {
        DWORD err = GetLastError();
        if (err != ERROR_ALREADY_EXISTS) {
            return VTLK_ERR_IO;
        }
    }
    return VTLK_OK;
#else
    if (mkdir(dir, 0700) != 0) {
        if (errno != EEXIST) {
            return VTLK_ERR_IO;
        }
    }
    return VTLK_OK;
#endif
}

static vtlk_status_t vtlk_tier_map_l2(vtlk_tier_store_t *store)
{
#ifdef _WIN32
    HANDLE h_file;
    HANDLE h_map;
    LARGE_INTEGER li_size;
    void *addr;

    h_file = CreateFileA(store->l2_path,
                         GENERIC_READ | GENERIC_WRITE,
                         FILE_SHARE_READ | FILE_SHARE_WRITE,
                         NULL,
                         OPEN_ALWAYS,
                         FILE_ATTRIBUTE_NORMAL,
                         NULL);
    if (h_file == INVALID_HANDLE_VALUE) {
        return VTLK_ERR_IO;
    }

    li_size.QuadPart = (LONGLONG)store->l2_bytes;
    if (!SetFilePointerEx(h_file, li_size, NULL, FILE_BEGIN) ||
        !SetEndOfFile(h_file)) {
        CloseHandle(h_file);
        return VTLK_ERR_IO;
    }

    h_map = CreateFileMappingA(h_file,
                               NULL,
                               PAGE_READWRITE,
                               (DWORD)((uint64_t)store->l2_bytes >> 32),
                               (DWORD)((uint64_t)store->l2_bytes & 0xFFFFFFFFu),
                               NULL);
    if (h_map == NULL) {
        CloseHandle(h_file);
        return VTLK_ERR_IO;
    }

    addr = MapViewOfFile(h_map, FILE_MAP_ALL_ACCESS, 0u, 0u, store->l2_bytes);
    if (addr == NULL) {
        CloseHandle(h_map);
        CloseHandle(h_file);
        return VTLK_ERR_IO;
    }

    store->l2_file_handle = (void *)h_file;
    store->l2_map_handle = (void *)h_map;
    store->l2_mem = (uint8_t *)addr;
    return VTLK_OK;
#else
    int fd;
    void *addr;

    fd = open(store->l2_path, O_CREAT | O_RDWR, 0600);
    if (fd < 0) {
        return VTLK_ERR_IO;
    }
    if (ftruncate(fd, (off_t)store->l2_bytes) != 0) {
        close(fd);
        return VTLK_ERR_IO;
    }

    addr = mmap(NULL, store->l2_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (addr == MAP_FAILED) {
        close(fd);
        return VTLK_ERR_IO;
    }

    store->l2_fd = fd;
    store->l2_mem = (uint8_t *)addr;
    return VTLK_OK;
#endif
}

static void vtlk_tier_unmap_l2(vtlk_tier_store_t *store)
{
#ifdef _WIN32
    if (store->l2_mem != NULL) {
        UnmapViewOfFile(store->l2_mem);
        store->l2_mem = NULL;
    }
    if (store->l2_map_handle != NULL) {
        CloseHandle((HANDLE)store->l2_map_handle);
        store->l2_map_handle = NULL;
    }
    if (store->l2_file_handle != NULL) {
        CloseHandle((HANDLE)store->l2_file_handle);
        store->l2_file_handle = NULL;
    }
#else
    if (store->l2_mem != NULL) {
        munmap(store->l2_mem, store->l2_bytes);
        store->l2_mem = NULL;
    }
    if (store->l2_fd >= 0) {
        close(store->l2_fd);
        store->l2_fd = -1;
    }
#endif
}

static vtlk_tier_block_meta_t *vtlk_tier_find_block(vtlk_tier_store_t *store,
                                                    uint32_t block_id)
{
    uint32_t i;
    for (i = 0u; i < store->max_blocks; ++i) {
        if (store->blocks[i].valid && store->blocks[i].block_id == block_id) {
            return &store->blocks[i];
        }
    }
    return NULL;
}

static vtlk_tier_block_meta_t *vtlk_tier_alloc_meta(vtlk_tier_store_t *store)
{
    uint32_t i;
    for (i = 0u; i < store->max_blocks; ++i) {
        if (!store->blocks[i].valid) {
            return &store->blocks[i];
        }
    }
    return NULL;
}

static int vtlk_tier_alloc_slot(uint8_t *used_arr, uint32_t cap, uint32_t *out_slot)
{
    uint32_t i;
    for (i = 0u; i < cap; ++i) {
        if (used_arr[i] == 0u) {
            used_arr[i] = 1u;
            *out_slot = i;
            return 1;
        }
    }
    return 0;
}

static vtlk_tier_block_meta_t *vtlk_tier_find_lru(vtlk_tier_store_t *store,
                                                  vtlk_tier_id_t tier)
{
    vtlk_tier_block_meta_t *best = NULL;
    uint32_t i;

    for (i = 0u; i < store->max_blocks; ++i) {
        vtlk_tier_block_meta_t *m = &store->blocks[i];
        if (!m->valid || m->tier != (uint8_t)tier || m->refcount != 0u) {
            continue;
        }
        if (best == NULL || m->lru_tick < best->lru_tick) {
            best = m;
        }
    }
    return best;
}

static vtlk_status_t vtlk_tier_write_l3_block(vtlk_tier_store_t *store,
                                              uint32_t block_id,
                                              const void *data)
{
    char path[VTLK_TIER_PATH_MAX];
    FILE *fp;
    size_t written;

    if (!store->has_l3) {
        return VTLK_ERR_IO;
    }
    if (!vtlk_tier_format_block_path(path, sizeof(path), store->l3_dir, block_id)) {
        return VTLK_ERR_INVALID_PARAM;
    }

    fp = fopen(path, "wb");
    if (fp == NULL) {
        return VTLK_ERR_IO;
    }
    written = fwrite(data, 1u, (size_t)store->block_size, fp);
    if (fflush(fp) != 0 || written != (size_t)store->block_size) {
        fclose(fp);
        return VTLK_ERR_IO;
    }
    fclose(fp);
    return VTLK_OK;
}

static vtlk_status_t vtlk_tier_read_l3_block(vtlk_tier_store_t *store,
                                             uint32_t block_id,
                                             void *dst)
{
    char path[VTLK_TIER_PATH_MAX];
    FILE *fp;
    size_t nread;

    if (!store->has_l3) {
        return VTLK_ERR_IO;
    }
    if (!vtlk_tier_format_block_path(path, sizeof(path), store->l3_dir, block_id)) {
        return VTLK_ERR_INVALID_PARAM;
    }

    fp = fopen(path, "rb");
    if (fp == NULL) {
        return VTLK_ERR_IO;
    }
    nread = fread(dst, 1u, (size_t)store->block_size, fp);
    fclose(fp);
    if (nread != (size_t)store->block_size) {
        return VTLK_ERR_IO;
    }
    return VTLK_OK;
}

static vtlk_status_t vtlk_tier_evict_l2_one(vtlk_tier_store_t *store)
{
    vtlk_tier_block_meta_t *victim;
    const void *src;
    vtlk_status_t st;

    if (!store->has_l3) {
        return VTLK_ERR_FULL;
    }

    victim = vtlk_tier_find_lru(store, VTLK_TIER_L2);
    if (victim == NULL) {
        return VTLK_ERR_BUSY;
    }

    src = store->l2_mem + ((size_t)victim->slot_index * (size_t)store->block_size);
    st = vtlk_tier_write_l3_block(store, victim->block_id, src);
    if (st != VTLK_OK) {
        return st;
    }

    store->l2_slot_used[victim->slot_index] = 0u;
    victim->tier = (uint8_t)VTLK_TIER_L3;
    victim->slot_index = 0u;
    store->stats.l2_used--;
    store->stats.l3_used++;
    store->stats.evictions_l2_to_l3++;

    return vtlk_tier_flush_index(store);
}

static vtlk_status_t vtlk_tier_evict_l1_one(vtlk_tier_store_t *store)
{
    vtlk_tier_block_meta_t *victim;
    const void *src;
    uint32_t l2_slot;
    vtlk_status_t st;

    victim = vtlk_tier_find_lru(store, VTLK_TIER_L1);
    if (victim == NULL) {
        return VTLK_ERR_BUSY;
    }

    src = store->l1_mem + ((size_t)victim->slot_index * (size_t)store->block_size);

    if (store->has_l2) {
        if (!vtlk_tier_alloc_slot(store->l2_slot_used, store->l2_capacity, &l2_slot)) {
            st = vtlk_tier_evict_l2_one(store);
            if (st != VTLK_OK) {
                return st;
            }
            if (!vtlk_tier_alloc_slot(store->l2_slot_used, store->l2_capacity, &l2_slot)) {
                return VTLK_ERR_FULL;
            }
        }
        memcpy(store->l2_mem + ((size_t)l2_slot * (size_t)store->block_size),
               src,
               (size_t)store->block_size);

        store->l1_slot_used[victim->slot_index] = 0u;
        victim->tier = (uint8_t)VTLK_TIER_L2;
        victim->slot_index = l2_slot;
        store->stats.l1_used--;
        store->stats.l2_used++;
        store->stats.evictions_l1_to_l2++;
        return VTLK_OK;
    }

    if (store->has_l3) {
        st = vtlk_tier_write_l3_block(store, victim->block_id, src);
        if (st != VTLK_OK) {
            return st;
        }
        store->l1_slot_used[victim->slot_index] = 0u;
        victim->tier = (uint8_t)VTLK_TIER_L3;
        victim->slot_index = 0u;
        store->stats.l1_used--;
        store->stats.l3_used++;
        store->stats.evictions_l1_to_l2++;
        return vtlk_tier_flush_index(store);
    }

    return VTLK_ERR_FULL;
}

static vtlk_status_t vtlk_tier_ensure_l1_slot(vtlk_tier_store_t *store,
                                             uint32_t *out_slot)
{
    vtlk_status_t st;

    if (vtlk_tier_alloc_slot(store->l1_slot_used, store->l1_capacity, out_slot)) {
        return VTLK_OK;
    }

    st = vtlk_tier_evict_l1_one(store);
    if (st != VTLK_OK) {
        return st;
    }

    if (!vtlk_tier_alloc_slot(store->l1_slot_used, store->l1_capacity, out_slot)) {
        return VTLK_ERR_FULL;
    }
    return VTLK_OK;
}

static vtlk_status_t vtlk_tier_load_index(vtlk_tier_store_t *store)
{
    char idx_path[VTLK_TIER_PATH_MAX];
    uint8_t hdr[VTLK_IDX_HDR_SIZE];
    uint8_t ent[VTLK_IDX_ENT_SIZE];
    FILE *fp;
    uint32_t version;
    uint32_t blk_sz;
    uint32_t count;
    uint32_t expected_crc;
    uint32_t running_crc;
    uint32_t i;

    if (!store->has_l3) {
        return VTLK_OK;
    }
    if (!vtlk_tier_format_path(idx_path,
                               sizeof(idx_path),
                               store->l3_dir,
                               "vtlk_tier_index.bin")) {
        return VTLK_ERR_INVALID_PARAM;
    }

    fp = fopen(idx_path, "rb");
    if (fp == NULL) {
        return VTLK_OK;
    }

    if (fread(hdr, 1u, VTLK_IDX_HDR_SIZE, fp) != VTLK_IDX_HDR_SIZE) {
        fclose(fp);
        return VTLK_ERR_IO;
    }

    if (memcmp(hdr, VTLK_IDX_MAGIC, VTLK_IDX_MAGIC_LEN) != 0) {
        fclose(fp);
        return VTLK_ERR_INVALID_MAGIC;
    }

    version = vtlk_read_u32_le(hdr + 8u);
    blk_sz = vtlk_read_u32_le(hdr + 12u);
    count = vtlk_read_u32_le(hdr + 16u);
    expected_crc = vtlk_read_u32_le(hdr + 20u);

    if (version != VTLK_IDX_VERSION ||
        blk_sz != store->block_size ||
        count > store->max_blocks) {
        fclose(fp);
        return VTLK_ERR_INVALID_PARAM;
    }

    running_crc = vtlk_crc32c(hdr, 20u);
    for (i = 0u; i < count; ++i) {
        vtlk_tier_block_meta_t *m = &store->blocks[i];
        if (fread(ent, 1u, VTLK_IDX_ENT_SIZE, fp) != VTLK_IDX_ENT_SIZE) {
            fclose(fp);
            return VTLK_ERR_IO;
        }
        running_crc = vtlk_crc32c_update(running_crc, ent, VTLK_IDX_ENT_SIZE);
        m->block_id = vtlk_read_u32_le(ent + 0u);
        m->crc32c = vtlk_read_u32_le(ent + 4u);
        m->lru_tick = vtlk_read_u64_le(ent + 8u);
        m->tier = (uint8_t)VTLK_TIER_L3;
        m->valid = 1u;
        m->refcount = 0u;
        m->slot_index = 0u;
        if (m->lru_tick > store->lru_clock) {
            store->lru_clock = m->lru_tick;
        }
    }
    fclose(fp);

    if (running_crc != expected_crc) {
        memset(store->blocks, 0, sizeof(vtlk_tier_block_meta_t) * store->max_blocks);
        store->stats.crc_errors++;
        return VTLK_ERR_CRC_MISMATCH;
    }

    store->stats.l3_used = count;
    return VTLK_OK;
}

vtlk_status_t vtlk_tier_flush_index(vtlk_tier_store_t *store)
{
    char idx_path[VTLK_TIER_PATH_MAX];
    char tmp_path[VTLK_TIER_PATH_MAX];
    uint8_t hdr[VTLK_IDX_HDR_SIZE];
    uint8_t ent[VTLK_IDX_ENT_SIZE];
    FILE *fp;
    uint32_t count = 0u;
    uint32_t crc;
    uint32_t i;

    if (store == NULL) {
        return VTLK_ERR_NULL_PTR;
    }
    if (!store->has_l3) {
        return VTLK_OK;
    }
    if (!vtlk_tier_format_path(idx_path,
                               sizeof(idx_path),
                               store->l3_dir,
                               "vtlk_tier_index.bin") ||
        !vtlk_tier_format_path(tmp_path,
                               sizeof(tmp_path),
                               store->l3_dir,
                               "vtlk_tier_index.tmp")) {
        return VTLK_ERR_INVALID_PARAM;
    }

    for (i = 0u; i < store->max_blocks; ++i) {
        if (store->blocks[i].valid && store->blocks[i].tier == (uint8_t)VTLK_TIER_L3) {
            count++;
        }
    }

    memcpy(hdr, VTLK_IDX_MAGIC, VTLK_IDX_MAGIC_LEN);
    vtlk_write_u32_le(hdr + 8u, VTLK_IDX_VERSION);
    vtlk_write_u32_le(hdr + 12u, store->block_size);
    vtlk_write_u32_le(hdr + 16u, count);

    crc = vtlk_crc32c(hdr, 20u);
    for (i = 0u; i < store->max_blocks; ++i) {
        const vtlk_tier_block_meta_t *m = &store->blocks[i];
        if (!m->valid || m->tier != (uint8_t)VTLK_TIER_L3) {
            continue;
        }
        vtlk_write_u32_le(ent + 0u, m->block_id);
        vtlk_write_u32_le(ent + 4u, m->crc32c);
        vtlk_write_u64_le(ent + 8u, m->lru_tick);
        crc = vtlk_crc32c_update(crc, ent, VTLK_IDX_ENT_SIZE);
    }
    vtlk_write_u32_le(hdr + 20u, crc);

    fp = fopen(tmp_path, "wb");
    if (fp == NULL) {
        return VTLK_ERR_IO;
    }
    if (fwrite(hdr, 1u, VTLK_IDX_HDR_SIZE, fp) != VTLK_IDX_HDR_SIZE) {
        fclose(fp);
        return VTLK_ERR_IO;
    }
    for (i = 0u; i < store->max_blocks; ++i) {
        const vtlk_tier_block_meta_t *m = &store->blocks[i];
        if (!m->valid || m->tier != (uint8_t)VTLK_TIER_L3) {
            continue;
        }
        vtlk_write_u32_le(ent + 0u, m->block_id);
        vtlk_write_u32_le(ent + 4u, m->crc32c);
        vtlk_write_u64_le(ent + 8u, m->lru_tick);
        if (fwrite(ent, 1u, VTLK_IDX_ENT_SIZE, fp) != VTLK_IDX_ENT_SIZE) {
            fclose(fp);
            return VTLK_ERR_IO;
        }
    }
    if (fflush(fp) != 0) {
        fclose(fp);
        return VTLK_ERR_IO;
    }
    fclose(fp);

#ifdef _WIN32
    if (!MoveFileExA(tmp_path, idx_path, MOVEFILE_REPLACE_EXISTING)) {
        return VTLK_ERR_IO;
    }
#else
    if (rename(tmp_path, idx_path) != 0) {
        return VTLK_ERR_IO;
    }
#endif

    return VTLK_OK;
}

vtlk_status_t vtlk_tier_init(vtlk_tier_store_t *store,
                             const vtlk_tier_config_t *cfg)
{
    uint32_t max_blocks;
    vtlk_status_t st;

    if (store == NULL || cfg == NULL || cfg->l1_mem == NULL) {
        return VTLK_ERR_NULL_PTR;
    }
    if (cfg->block_size == 0u || cfg->l1_capacity == 0u) {
        return VTLK_ERR_INVALID_PARAM;
    }

    memset(store, 0, sizeof(*store));
#ifndef _WIN32
    store->l2_fd = -1;
#endif

    store->block_size = cfg->block_size;
    store->l1_capacity = cfg->l1_capacity;
    store->l2_capacity = cfg->l2_capacity;
    store->high_watermark = (cfg->high_watermark == 0u || cfg->high_watermark > cfg->l1_capacity)
                                ? cfg->l1_capacity
                                : cfg->high_watermark;
    max_blocks = (cfg->max_blocks == 0u)
                     ? (cfg->l1_capacity + cfg->l2_capacity + 1024u)
                     : cfg->max_blocks;
    store->max_blocks = max_blocks;
    store->l1_mem = (uint8_t *)cfg->l1_mem;

    store->blocks = (vtlk_tier_block_meta_t *)calloc(
        (size_t)max_blocks, sizeof(vtlk_tier_block_meta_t));
    store->l1_slot_used = (uint8_t *)calloc((size_t)cfg->l1_capacity, 1u);
    if (store->blocks == NULL || store->l1_slot_used == NULL) {
        vtlk_tier_close(store);
        return VTLK_ERR_BUFFER_TOO_SMALL;
    }

    if (cfg->l2_path != NULL && cfg->l2_path[0] != '\0' && cfg->l2_capacity > 0u) {
        if (strlen(cfg->l2_path) >= VTLK_TIER_PATH_MAX) {
            vtlk_tier_close(store);
            return VTLK_ERR_INVALID_PARAM;
        }
        memcpy(store->l2_path, cfg->l2_path, strlen(cfg->l2_path) + 1u);
        store->l2_bytes = (size_t)cfg->l2_capacity * (size_t)cfg->block_size;
        store->l2_slot_used = (uint8_t *)calloc((size_t)cfg->l2_capacity, 1u);
        if (store->l2_slot_used == NULL) {
            vtlk_tier_close(store);
            return VTLK_ERR_BUFFER_TOO_SMALL;
        }
        st = vtlk_tier_map_l2(store);
        if (st != VTLK_OK) {
            vtlk_tier_close(store);
            return st;
        }
        store->has_l2 = 1;
    }

    if (cfg->l3_dir != NULL && cfg->l3_dir[0] != '\0') {
        if (strlen(cfg->l3_dir) >= VTLK_TIER_PATH_MAX) {
            vtlk_tier_close(store);
            return VTLK_ERR_INVALID_PARAM;
        }
        memcpy(store->l3_dir, cfg->l3_dir, strlen(cfg->l3_dir) + 1u);
        st = vtlk_tier_ensure_dir(store->l3_dir);
        if (st != VTLK_OK) {
            vtlk_tier_close(store);
            return st;
        }
        store->has_l3 = 1;
        st = vtlk_tier_load_index(store);
        if (st != VTLK_OK) {
            vtlk_tier_close(store);
            return st;
        }
    }

    return VTLK_OK;
}

void vtlk_tier_close(vtlk_tier_store_t *store)
{
    if (store == NULL) {
        return;
    }
    if (store->has_l3) {
        (void)vtlk_tier_flush_index(store);
    }
    vtlk_tier_unmap_l2(store);
    free(store->blocks);
    free(store->l1_slot_used);
    free(store->l2_slot_used);
    memset(store, 0, sizeof(*store));
#ifndef _WIN32
    store->l2_fd = -1;
#endif
}

vtlk_status_t vtlk_tier_put(vtlk_tier_store_t *store,
                            uint32_t block_id,
                            const void *data,
                            size_t len)
{
    vtlk_tier_block_meta_t *meta;
    uint32_t slot;
    uint32_t crc;
    vtlk_status_t st;

    if (store == NULL || data == NULL) {
        return VTLK_ERR_NULL_PTR;
    }
    if (len != (size_t)store->block_size) {
        return VTLK_ERR_INVALID_PARAM;
    }

    crc = vtlk_crc32c(data, len);
    meta = vtlk_tier_find_block(store, block_id);

    if (meta != NULL) {
        if (meta->tier == (uint8_t)VTLK_TIER_L1) {
            memcpy(store->l1_mem + ((size_t)meta->slot_index * (size_t)store->block_size),
                   data,
                   len);
            meta->crc32c = crc;
            meta->lru_tick = ++store->lru_clock;
            return VTLK_OK;
        }
        if (meta->tier == (uint8_t)VTLK_TIER_L2) {
            store->l2_slot_used[meta->slot_index] = 0u;
            store->stats.l2_used--;
        } else if (meta->tier == (uint8_t)VTLK_TIER_L3) {
            store->stats.l3_used--;
        }
    } else {
        meta = vtlk_tier_alloc_meta(store);
        if (meta == NULL) {
            return VTLK_ERR_FULL;
        }
    }

    while (store->stats.l1_used >= store->high_watermark) {
        st = vtlk_tier_evict_l1_one(store);
        if (st != VTLK_OK) {
            break;
        }
    }

    st = vtlk_tier_ensure_l1_slot(store, &slot);
    if (st != VTLK_OK) {
        if (!meta->valid) {
            memset(meta, 0, sizeof(*meta));
        }
        return st;
    }

    memcpy(store->l1_mem + ((size_t)slot * (size_t)store->block_size), data, len);
    meta->block_id = block_id;
    meta->tier = (uint8_t)VTLK_TIER_L1;
    meta->valid = 1u;
    meta->slot_index = slot;
    meta->refcount = 0u;
    meta->crc32c = crc;
    meta->lru_tick = ++store->lru_clock;
    store->stats.l1_used++;
    return VTLK_OK;
}

vtlk_status_t vtlk_tier_acquire(vtlk_tier_store_t *store,
                                uint32_t block_id,
                                const void **out_ptr)
{
    vtlk_tier_block_meta_t *meta;
    uint32_t new_slot;
    uint8_t *dst;
    uint32_t actual_crc;
    vtlk_status_t st;

    if (store == NULL || out_ptr == NULL) {
        return VTLK_ERR_NULL_PTR;
    }

    meta = vtlk_tier_find_block(store, block_id);
    if (meta == NULL) {
        return VTLK_ERR_NOT_FOUND;
    }

    if (meta->tier == (uint8_t)VTLK_TIER_L1) {
        meta->refcount++;
        meta->lru_tick = ++store->lru_clock;
        *out_ptr = store->l1_mem + ((size_t)meta->slot_index * (size_t)store->block_size);
        return VTLK_OK;
    }

    st = vtlk_tier_ensure_l1_slot(store, &new_slot);
    if (st != VTLK_OK) {
        return st;
    }
    dst = store->l1_mem + ((size_t)new_slot * (size_t)store->block_size);

    if (meta->tier == (uint8_t)VTLK_TIER_L2) {
        const void *src = store->l2_mem + ((size_t)meta->slot_index * (size_t)store->block_size);
        actual_crc = vtlk_crc32c(src, (size_t)store->block_size);
        if (actual_crc != meta->crc32c) {
            store->l1_slot_used[new_slot] = 0u;
            store->stats.crc_errors++;
            return VTLK_ERR_CRC_MISMATCH;
        }
        memcpy(dst, src, (size_t)store->block_size);
        store->l2_slot_used[meta->slot_index] = 0u;
        store->stats.l2_used--;
    } else if (meta->tier == (uint8_t)VTLK_TIER_L3) {
        st = vtlk_tier_read_l3_block(store, block_id, dst);
        if (st != VTLK_OK) {
            store->l1_slot_used[new_slot] = 0u;
            return st;
        }
        actual_crc = vtlk_crc32c(dst, (size_t)store->block_size);
        if (actual_crc != meta->crc32c) {
            store->l1_slot_used[new_slot] = 0u;
            store->stats.crc_errors++;
            return VTLK_ERR_CRC_MISMATCH;
        }
        store->stats.l3_used--;
    } else {
        store->l1_slot_used[new_slot] = 0u;
        return VTLK_ERR_NOT_FOUND;
    }

    meta->tier = (uint8_t)VTLK_TIER_L1;
    meta->slot_index = new_slot;
    meta->refcount++;
    meta->lru_tick = ++store->lru_clock;
    store->stats.l1_used++;
    store->stats.promotions_to_l1++;
    *out_ptr = dst;
    return VTLK_OK;
}

vtlk_status_t vtlk_tier_release(vtlk_tier_store_t *store,
                                uint32_t block_id)
{
    vtlk_tier_block_meta_t *meta;

    if (store == NULL) {
        return VTLK_ERR_NULL_PTR;
    }
    meta = vtlk_tier_find_block(store, block_id);
    if (meta == NULL) {
        return VTLK_ERR_NOT_FOUND;
    }
    if (meta->refcount == 0u) {
        return VTLK_ERR_INVALID_PARAM;
    }
    meta->refcount--;
    meta->lru_tick = ++store->lru_clock;
    return VTLK_OK;
}

vtlk_status_t vtlk_tier_evict_l1(vtlk_tier_store_t *store,
                                 uint32_t target_evict)
{
    uint32_t i;
    vtlk_status_t st;

    if (store == NULL) {
        return VTLK_ERR_NULL_PTR;
    }
    for (i = 0u; i < target_evict; ++i) {
        st = vtlk_tier_evict_l1_one(store);
        if (st != VTLK_OK) {
            return (i > 0u) ? VTLK_OK : st;
        }
    }
    return VTLK_OK;
}

vtlk_status_t vtlk_tier_stats(const vtlk_tier_store_t *store,
                              vtlk_tier_stats_t *out_stats)
{
    if (store == NULL || out_stats == NULL) {
        return VTLK_ERR_NULL_PTR;
    }
    *out_stats = store->stats;
    return VTLK_OK;
}
