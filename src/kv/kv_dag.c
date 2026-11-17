#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif
#ifndef _DEFAULT_SOURCE
#define _DEFAULT_SOURCE
#endif

#include "vectlink_kv.h"
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
#include <io.h>
#else
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#define VTLK_DAG_INDEX_FILE "vtlk_dag_index.bin"
#define VTLK_DAG_LOCK_FILE  "vtlk_dag.lock"
#define VTLK_DAG_HDR_SIZE   24u
#define VTLK_DAG_ENT_SIZE   (VTLK_KV_DAG_STR_MAX * 4u + 32u)

typedef struct vtlk_dag_file_lock {
#ifdef _WIN32
    HANDLE handle;
#else
    int fd;
#endif
} vtlk_dag_file_lock_t;

static uint64_t vtlk_dag_fnv1a_pair(const char *a, const char *b)
{
    uint64_t h = 14695981039346656037ULL;
    const unsigned char *p;

    if (a != NULL) {
        for (p = (const unsigned char *)a; *p != '\0'; ++p) {
            h ^= (uint64_t)(*p);
            h *= 1099511628211ULL;
        }
    }
    h ^= 0xFFu;
    h *= 1099511628211ULL;
    if (b != NULL) {
        for (p = (const unsigned char *)b; *p != '\0'; ++p) {
            h ^= (uint64_t)(*p);
            h *= 1099511628211ULL;
        }
    }
    return h;
}

static void vtlk_dag_copy_str(char *dst, size_t cap, const char *src)
{
    size_t len;
    if (cap == 0u) {
        return;
    }
    if (src == NULL) {
        dst[0] = '\0';
        return;
    }
    len = strlen(src);
    if (len >= cap) {
        len = cap - 1u;
    }
    memcpy(dst, src, len);
    memset(dst + len, 0, cap - len);
}

static unsigned long vtlk_dag_get_pid(void)
{
#ifdef _WIN32
    return (unsigned long)GetCurrentProcessId();
#else
    return (unsigned long)getpid();
#endif
}

static vtlk_status_t vtlk_dag_mkdir_p(const char *dir)
{
    char buf[VTLK_KV_DAG_PATH_MAX];
    size_t len;
    size_t i;

    if (dir == NULL || dir[0] == '\0') {
        return VTLK_ERR_INVALID_PARAM;
    }
    len = strlen(dir);
    if (len >= sizeof(buf)) {
        return VTLK_ERR_INVALID_PARAM;
    }
    memcpy(buf, dir, len + 1u);

    while (len > 1u && (buf[len - 1u] == '/' || buf[len - 1u] == '\\')) {
        buf[--len] = '\0';
    }

    for (i = 1u; i <= len; ++i) {
        if (buf[i] == '/' || buf[i] == '\\' || buf[i] == '\0') {
            char saved = buf[i];
            if (i == 2u && buf[1] == ':') {
                continue;
            }
            buf[i] = '\0';
#ifdef _WIN32
            if (!CreateDirectoryA(buf, NULL)) {
                DWORD err = GetLastError();
                if (err != ERROR_ALREADY_EXISTS) {
                    return VTLK_ERR_IO;
                }
            }
#else
            if (mkdir(buf, 0700) != 0 && errno != EEXIST) {
                return VTLK_ERR_IO;
            }
#endif
            buf[i] = saved;
        }
    }
    return VTLK_OK;
}

static int vtlk_dag_format_path(char *dst,
                                size_t cap,
                                const char *dir,
                                const char *file)
{
    int n = snprintf(dst, cap, "%s/%s", dir, file);
    return (n >= 0 && (size_t)n < cap) ? 1 : 0;
}

static int vtlk_dag_format_ckpt_path(char *dst,
                                     size_t cap,
                                     const char *dir,
                                     const char *thread_id,
                                     const char *checkpoint_id,
                                     const char *suffix)
{
    uint64_t h = vtlk_dag_fnv1a_pair(thread_id, checkpoint_id);
    int n = snprintf(dst, cap, "%s/ckpt_%016llx.ckpt%s",
                     dir, (unsigned long long)h, suffix ? suffix : "");
    return (n >= 0 && (size_t)n < cap) ? 1 : 0;
}

static vtlk_status_t vtlk_dag_lock_acquire(const char *l3_dir,
                                           int exclusive,
                                           vtlk_dag_file_lock_t *out_lock)
{
    char lock_path[VTLK_KV_DAG_PATH_MAX];
    if (!vtlk_dag_format_path(lock_path, sizeof(lock_path), l3_dir, VTLK_DAG_LOCK_FILE)) {
        return VTLK_ERR_INVALID_PARAM;
    }

#ifdef _WIN32
    {
        HANDLE h;
        OVERLAPPED ov;
        DWORD flags = exclusive ? LOCKFILE_EXCLUSIVE_LOCK : 0u;

        h = CreateFileA(lock_path,
                        GENERIC_READ | GENERIC_WRITE,
                        FILE_SHARE_READ | FILE_SHARE_WRITE,
                        NULL,
                        OPEN_ALWAYS,
                        FILE_ATTRIBUTE_NORMAL,
                        NULL);
        if (h == INVALID_HANDLE_VALUE) {
            return VTLK_ERR_IO;
        }
        memset(&ov, 0, sizeof(ov));
        if (!LockFileEx(h, flags, 0u, 1u, 0u, &ov)) {
            CloseHandle(h);
            return VTLK_ERR_IO;
        }
        out_lock->handle = h;
        return VTLK_OK;
    }
#else
    {
        int fd = open(lock_path, O_CREAT | O_RDWR, 0600);
        if (fd < 0) {
            return VTLK_ERR_IO;
        }
        while (flock(fd, exclusive ? LOCK_EX : LOCK_SH) != 0) {
            if (errno == EINTR) {
                continue;
            }
            close(fd);
            return VTLK_ERR_IO;
        }
        out_lock->fd = fd;
        return VTLK_OK;
    }
#endif
}

static void vtlk_dag_lock_release(vtlk_dag_file_lock_t *lock)
{
#ifdef _WIN32
    if (lock->handle != INVALID_HANDLE_VALUE && lock->handle != NULL) {
        OVERLAPPED ov;
        memset(&ov, 0, sizeof(ov));
        UnlockFileEx(lock->handle, 0u, 1u, 0u, &ov);
        CloseHandle(lock->handle);
        lock->handle = NULL;
    }
#else
    if (lock->fd >= 0) {
        flock(lock->fd, LOCK_UN);
        close(lock->fd);
        lock->fd = -1;
    }
#endif
}

static vtlk_status_t vtlk_dag_atomic_rename(const char *tmp_path,
                                            const char *dst_path)
{
#ifdef _WIN32
    if (!MoveFileExA(tmp_path,
                     dst_path,
                     MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
        return VTLK_ERR_IO;
    }
    return VTLK_OK;
#else
    if (rename(tmp_path, dst_path) != 0) {
        return VTLK_ERR_IO;
    }
    return VTLK_OK;
#endif
}

static vtlk_status_t vtlk_dag_write_file_atomic(const char *tmp_path,
                                                const char *dst_path,
                                                const void *data,
                                                size_t len)
{
    FILE *fp = fopen(tmp_path, "wb");
    if (fp == NULL) {
        return VTLK_ERR_IO;
    }
    if (len > 0u && data != NULL) {
        if (fwrite(data, 1u, len, fp) != len) {
            fclose(fp);
            remove(tmp_path);
            return VTLK_ERR_IO;
        }
    }
    if (fflush(fp) != 0) {
        fclose(fp);
        remove(tmp_path);
        return VTLK_ERR_IO;
    }
#ifdef _WIN32
    (void)_commit(_fileno(fp));
#else
    (void)fsync(fileno(fp));
#endif
    fclose(fp);

    if (vtlk_dag_atomic_rename(tmp_path, dst_path) != VTLK_OK) {
        remove(tmp_path);
        return VTLK_ERR_IO;
    }
    return VTLK_OK;
}

static vtlk_status_t vtlk_dag_ensure_capacity(vtlk_kv_dag_t *dag, uint32_t needed)
{
    uint32_t new_cap;
    vtlk_kv_checkpoint_meta_t *new_arr;

    if (needed <= dag->capacity) {
        return VTLK_OK;
    }
    new_cap = (dag->capacity == 0u) ? 64u : dag->capacity;
    while (new_cap < needed) {
        if (new_cap > 0x3FFFFFFFu) {
            return VTLK_ERR_FULL;
        }
        new_cap *= 2u;
    }
    new_arr = (vtlk_kv_checkpoint_meta_t *)realloc(
        dag->entries, (size_t)new_cap * sizeof(vtlk_kv_checkpoint_meta_t));
    if (new_arr == NULL) {
        return VTLK_ERR_FULL;
    }
    dag->entries = new_arr;
    dag->capacity = new_cap;
    return VTLK_OK;
}

static void vtlk_dag_encode_entry(uint8_t *dst, const vtlk_kv_checkpoint_meta_t *m)
{
    memcpy(dst, m->branch_name, VTLK_KV_DAG_STR_MAX);
    dst += VTLK_KV_DAG_STR_MAX;
    memcpy(dst, m->thread_id, VTLK_KV_DAG_STR_MAX);
    dst += VTLK_KV_DAG_STR_MAX;
    memcpy(dst, m->checkpoint_id, VTLK_KV_DAG_STR_MAX);
    dst += VTLK_KV_DAG_STR_MAX;
    memcpy(dst, m->parent_checkpoint_id, VTLK_KV_DAG_STR_MAX);
    dst += VTLK_KV_DAG_STR_MAX;
    vtlk_write_u64_le(dst, m->snapshot_id);
    dst += 8u;
    vtlk_write_u64_le(dst, m->created_seq);
    dst += 8u;
    vtlk_write_u64_le(dst, m->payload_bytes);
    dst += 8u;
    vtlk_write_u32_le(dst, m->payload_crc32c);
    dst += 4u;
    vtlk_write_u32_le(dst, m->flags);
}

static void vtlk_dag_decode_entry(const uint8_t *src, vtlk_kv_checkpoint_meta_t *m)
{
    memset(m, 0, sizeof(*m));
    memcpy(m->branch_name, src, VTLK_KV_DAG_STR_MAX);
    m->branch_name[VTLK_KV_DAG_STR_MAX - 1u] = '\0';
    src += VTLK_KV_DAG_STR_MAX;

    memcpy(m->thread_id, src, VTLK_KV_DAG_STR_MAX);
    m->thread_id[VTLK_KV_DAG_STR_MAX - 1u] = '\0';
    src += VTLK_KV_DAG_STR_MAX;

    memcpy(m->checkpoint_id, src, VTLK_KV_DAG_STR_MAX);
    m->checkpoint_id[VTLK_KV_DAG_STR_MAX - 1u] = '\0';
    src += VTLK_KV_DAG_STR_MAX;

    memcpy(m->parent_checkpoint_id, src, VTLK_KV_DAG_STR_MAX);
    m->parent_checkpoint_id[VTLK_KV_DAG_STR_MAX - 1u] = '\0';
    src += VTLK_KV_DAG_STR_MAX;

    m->snapshot_id = vtlk_read_u64_le(src);
    src += 8u;
    m->created_seq = vtlk_read_u64_le(src);
    src += 8u;
    m->payload_bytes = vtlk_read_u64_le(src);
    src += 8u;
    m->payload_crc32c = vtlk_read_u32_le(src);
    src += 4u;
    m->flags = vtlk_read_u32_le(src);
}

static vtlk_status_t vtlk_dag_load_index_locked(vtlk_kv_dag_t *dag)
{
    char idx_path[VTLK_KV_DAG_PATH_MAX];
    FILE *fp;
    uint8_t hdr[VTLK_DAG_HDR_SIZE];
    uint32_t magic;
    uint32_t version;
    uint64_t next_seq;
    uint32_t count;
    uint32_t expected_crc;
    uint8_t *body = NULL;
    size_t body_len;
    uint32_t i;
    vtlk_status_t st;

    if (!vtlk_dag_format_path(idx_path, sizeof(idx_path), dag->l3_dir, VTLK_DAG_INDEX_FILE)) {
        return VTLK_ERR_INVALID_PARAM;
    }

    fp = fopen(idx_path, "rb");
    if (fp == NULL) {
        if (errno == ENOENT) {
            dag->count = 0u;
            dag->next_seq = 1u;
            return VTLK_OK;
        }
        return VTLK_ERR_IO;
    }

    if (fread(hdr, 1u, VTLK_DAG_HDR_SIZE, fp) != VTLK_DAG_HDR_SIZE) {
        fclose(fp);
        return VTLK_ERR_IO;
    }

    magic = vtlk_read_u32_le(hdr);
    version = vtlk_read_u32_le(hdr + 4u);
    next_seq = vtlk_read_u64_le(hdr + 8u);
    count = vtlk_read_u32_le(hdr + 16u);
    expected_crc = vtlk_read_u32_le(hdr + 20u);

    if (magic != VTLK_KV_DAG_MAGIC) {
        fclose(fp);
        return VTLK_ERR_INVALID_MAGIC;
    }
    if (version != VTLK_KV_DAG_VERSION) {
        fclose(fp);
        return VTLK_ERR_UNSUPPORTED_VERSION;
    }

    body_len = (size_t)count * VTLK_DAG_ENT_SIZE;
    if (body_len > 0u) {
        body = (uint8_t *)malloc(body_len);
        if (body == NULL) {
            fclose(fp);
            return VTLK_ERR_FULL;
        }
        if (fread(body, 1u, body_len, fp) != body_len) {
            free(body);
            fclose(fp);
            return VTLK_ERR_IO;
        }
        if (vtlk_crc32c(body, body_len) != expected_crc) {
            free(body);
            fclose(fp);
            return VTLK_ERR_CRC_MISMATCH;
        }
    }
    fclose(fp);

    st = vtlk_dag_ensure_capacity(dag, count);
    if (st != VTLK_OK) {
        free(body);
        return st;
    }

    for (i = 0u; i < count; ++i) {
        vtlk_dag_decode_entry(body + (size_t)i * VTLK_DAG_ENT_SIZE, &dag->entries[i]);
    }
    free(body);

    dag->count = count;
    dag->next_seq = (next_seq == 0u) ? 1u : next_seq;
    return VTLK_OK;
}

static vtlk_status_t vtlk_dag_save_index_locked(const vtlk_kv_dag_t *dag)
{
    char idx_path[VTLK_KV_DAG_PATH_MAX];
    char tmp_path[VTLK_KV_DAG_PATH_MAX];
    char tmp_name[128];
    size_t body_len;
    size_t total_len;
    uint8_t *buf;
    uint32_t body_crc = 0u;
    uint32_t i;
    vtlk_status_t st;

    if (!vtlk_dag_format_path(idx_path, sizeof(idx_path), dag->l3_dir, VTLK_DAG_INDEX_FILE)) {
        return VTLK_ERR_INVALID_PARAM;
    }
    snprintf(tmp_name, sizeof(tmp_name), "%s.tmp.%lu",
             VTLK_DAG_INDEX_FILE, vtlk_dag_get_pid());
    if (!vtlk_dag_format_path(tmp_path, sizeof(tmp_path), dag->l3_dir, tmp_name)) {
        return VTLK_ERR_INVALID_PARAM;
    }

    body_len = (size_t)dag->count * VTLK_DAG_ENT_SIZE;
    total_len = VTLK_DAG_HDR_SIZE + body_len;
    buf = (uint8_t *)malloc(total_len);
    if (buf == NULL) {
        return VTLK_ERR_FULL;
    }

    for (i = 0u; i < dag->count; ++i) {
        vtlk_dag_encode_entry(buf + VTLK_DAG_HDR_SIZE + (size_t)i * VTLK_DAG_ENT_SIZE,
                              &dag->entries[i]);
    }
    if (body_len > 0u) {
        body_crc = vtlk_crc32c(buf + VTLK_DAG_HDR_SIZE, body_len);
    }

    vtlk_write_u32_le(buf, VTLK_KV_DAG_MAGIC);
    vtlk_write_u32_le(buf + 4u, VTLK_KV_DAG_VERSION);
    vtlk_write_u64_le(buf + 8u, dag->next_seq);
    vtlk_write_u32_le(buf + 16u, dag->count);
    vtlk_write_u32_le(buf + 20u, body_crc);

    st = vtlk_dag_write_file_atomic(tmp_path, idx_path, buf, total_len);
    free(buf);
    return st;
}

VTLK_API vtlk_status_t vtlk_kv_dag_open(vtlk_kv_dag_t *dag, const char *l3_dir)
{
    vtlk_dag_file_lock_t lock;
    vtlk_status_t st;

    if (dag == NULL || l3_dir == NULL || l3_dir[0] == '\0') {
        return VTLK_ERR_INVALID_PARAM;
    }
    if (strlen(l3_dir) >= VTLK_KV_DAG_PATH_MAX) {
        return VTLK_ERR_INVALID_PARAM;
    }

    memset(dag, 0, sizeof(*dag));
    vtlk_dag_copy_str(dag->l3_dir, sizeof(dag->l3_dir), l3_dir);
    dag->next_seq = 1u;

    st = vtlk_dag_mkdir_p(dag->l3_dir);
    if (st != VTLK_OK) {
        return st;
    }

    st = vtlk_dag_lock_acquire(dag->l3_dir, 0, &lock);
    if (st != VTLK_OK) {
        return st;
    }

    st = vtlk_dag_load_index_locked(dag);
    vtlk_dag_lock_release(&lock);
    if (st != VTLK_OK) {
        free(dag->entries);
        dag->entries = NULL;
        dag->capacity = 0u;
        return st;
    }

    dag->is_open = 1;
    return VTLK_OK;
}

VTLK_API void vtlk_kv_dag_close(vtlk_kv_dag_t *dag)
{
    if (dag == NULL) {
        return;
    }
    free(dag->entries);
    dag->entries = NULL;
    dag->count = 0u;
    dag->capacity = 0u;
    dag->is_open = 0;
}

VTLK_API vtlk_status_t vtlk_kv_dag_put(vtlk_kv_dag_t *dag,
                                       const vtlk_kv_checkpoint_meta_t *meta,
                                       const void *payload,
                                       size_t payload_len)
{
    vtlk_dag_file_lock_t lock;
    vtlk_status_t st;
    vtlk_kv_checkpoint_meta_t entry;
    char ckpt_path[VTLK_KV_DAG_PATH_MAX];
    char tmp_path[VTLK_KV_DAG_PATH_MAX];
    char tmp_suffix[64];
    uint32_t i;
    int found = 0;

    if (dag == NULL || !dag->is_open || meta == NULL) {
        return VTLK_ERR_INVALID_PARAM;
    }
    if (meta->thread_id[0] == '\0' || meta->checkpoint_id[0] == '\0') {
        return VTLK_ERR_INVALID_PARAM;
    }
    if (payload_len > 0u && payload == NULL) {
        return VTLK_ERR_NULL_PTR;
    }

    snprintf(tmp_suffix, sizeof(tmp_suffix), ".tmp.%lu", vtlk_dag_get_pid());
    if (!vtlk_dag_format_ckpt_path(ckpt_path, sizeof(ckpt_path), dag->l3_dir,
                                   meta->thread_id, meta->checkpoint_id, NULL) ||
        !vtlk_dag_format_ckpt_path(tmp_path, sizeof(tmp_path), dag->l3_dir,
                                   meta->thread_id, meta->checkpoint_id, tmp_suffix)) {
        return VTLK_ERR_INVALID_PARAM;
    }

    st = vtlk_dag_lock_acquire(dag->l3_dir, 1, &lock);
    if (st != VTLK_OK) {
        return st;
    }

    st = vtlk_dag_load_index_locked(dag);
    if (st != VTLK_OK) {
        vtlk_dag_lock_release(&lock);
        return st;
    }

    memset(&entry, 0, sizeof(entry));
    vtlk_dag_copy_str(entry.branch_name, sizeof(entry.branch_name), meta->branch_name);
    vtlk_dag_copy_str(entry.thread_id, sizeof(entry.thread_id), meta->thread_id);
    vtlk_dag_copy_str(entry.checkpoint_id, sizeof(entry.checkpoint_id), meta->checkpoint_id);
    vtlk_dag_copy_str(entry.parent_checkpoint_id, sizeof(entry.parent_checkpoint_id),
                      meta->parent_checkpoint_id);
    entry.snapshot_id = meta->snapshot_id;
    entry.created_seq = dag->next_seq++;
    entry.payload_bytes = (uint64_t)payload_len;
    entry.payload_crc32c = (payload_len > 0u) ? vtlk_crc32c(payload, payload_len) : 0u;
    entry.flags = meta->flags;

    st = vtlk_dag_write_file_atomic(tmp_path, ckpt_path, payload, payload_len);
    if (st != VTLK_OK) {
        vtlk_dag_lock_release(&lock);
        return st;
    }

    for (i = 0u; i < dag->count; ++i) {
        if (strcmp(dag->entries[i].thread_id, entry.thread_id) == 0 &&
            strcmp(dag->entries[i].checkpoint_id, entry.checkpoint_id) == 0) {
            dag->entries[i] = entry;
            found = 1;
            break;
        }
    }

    if (!found) {
        st = vtlk_dag_ensure_capacity(dag, dag->count + 1u);
        if (st != VTLK_OK) {
            vtlk_dag_lock_release(&lock);
            return st;
        }
        dag->entries[dag->count++] = entry;
    }

    st = vtlk_dag_save_index_locked(dag);
    vtlk_dag_lock_release(&lock);
    return st;
}

VTLK_API vtlk_status_t vtlk_kv_dag_get(vtlk_kv_dag_t *dag,
                                       const char *thread_id,
                                       const char *checkpoint_id,
                                       vtlk_kv_checkpoint_meta_t *out_meta,
                                       void *out_buf,
                                       size_t buf_cap,
                                       size_t *out_len)
{
    vtlk_dag_file_lock_t lock;
    vtlk_status_t st;
    const vtlk_kv_checkpoint_meta_t *match = NULL;
    char ckpt_path[VTLK_KV_DAG_PATH_MAX];
    FILE *fp;
    size_t need_len;
    uint32_t i;

    if (dag == NULL || !dag->is_open || thread_id == NULL || thread_id[0] == '\0') {
        return VTLK_ERR_INVALID_PARAM;
    }

    st = vtlk_dag_lock_acquire(dag->l3_dir, 0, &lock);
    if (st != VTLK_OK) {
        return st;
    }

    st = vtlk_dag_load_index_locked(dag);
    if (st != VTLK_OK) {
        vtlk_dag_lock_release(&lock);
        return st;
    }

    if (checkpoint_id != NULL && checkpoint_id[0] != '\0') {
        for (i = 0u; i < dag->count; ++i) {
            if (strcmp(dag->entries[i].thread_id, thread_id) == 0 &&
                strcmp(dag->entries[i].checkpoint_id, checkpoint_id) == 0) {
                match = &dag->entries[i];
                break;
            }
        }
    } else {
        for (i = 0u; i < dag->count; ++i) {
            if (strcmp(dag->entries[i].thread_id, thread_id) == 0 &&
                (dag->entries[i].flags & VTLK_KV_DAG_FLAG_WRITES) == 0u) {
                if (match == NULL || dag->entries[i].created_seq > match->created_seq) {
                    match = &dag->entries[i];
                }
            }
        }
    }

    if (match == NULL) {
        vtlk_dag_lock_release(&lock);
        return VTLK_ERR_NOT_FOUND;
    }

    if (out_meta != NULL) {
        *out_meta = *match;
    }
    need_len = (size_t)match->payload_bytes;
    if (out_len != NULL) {
        *out_len = need_len;
    }

    if (need_len > 0u && (out_buf == NULL || buf_cap < need_len)) {
        vtlk_dag_lock_release(&lock);
        return VTLK_ERR_BUFFER_TOO_SMALL;
    }

    if (need_len == 0u) {
        vtlk_dag_lock_release(&lock);
        return VTLK_OK;
    }

    if (!vtlk_dag_format_ckpt_path(ckpt_path, sizeof(ckpt_path), dag->l3_dir,
                                   match->thread_id, match->checkpoint_id, NULL)) {
        vtlk_dag_lock_release(&lock);
        return VTLK_ERR_INVALID_PARAM;
    }

    fp = fopen(ckpt_path, "rb");
    if (fp == NULL) {
        vtlk_dag_lock_release(&lock);
        return VTLK_ERR_IO;
    }
    if (fread(out_buf, 1u, need_len, fp) != need_len) {
        fclose(fp);
        vtlk_dag_lock_release(&lock);
        return VTLK_ERR_IO;
    }
    fclose(fp);

    if (vtlk_crc32c(out_buf, need_len) != match->payload_crc32c) {
        vtlk_dag_lock_release(&lock);
        return VTLK_ERR_CRC_MISMATCH;
    }

    vtlk_dag_lock_release(&lock);
    return VTLK_OK;
}

static int vtlk_dag_cmp_seq_desc(const void *a, const void *b)
{
    const vtlk_kv_checkpoint_meta_t *ma = (const vtlk_kv_checkpoint_meta_t *)a;
    const vtlk_kv_checkpoint_meta_t *mb = (const vtlk_kv_checkpoint_meta_t *)b;
    if (ma->created_seq > mb->created_seq) {
        return -1;
    }
    if (ma->created_seq < mb->created_seq) {
        return 1;
    }
    return 0;
}

VTLK_API vtlk_status_t vtlk_kv_dag_list(vtlk_kv_dag_t *dag,
                                       const char *thread_id,
                                       vtlk_kv_checkpoint_meta_t *out_metas,
                                       uint32_t max_metas,
                                       uint32_t *out_count)
{
    vtlk_dag_file_lock_t lock;
    vtlk_status_t st;
    vtlk_kv_checkpoint_meta_t *tmp = NULL;
    uint32_t match_count = 0u;
    uint32_t i;
    int filter_thread;

    if (dag == NULL || !dag->is_open || out_count == NULL) {
        return VTLK_ERR_INVALID_PARAM;
    }

    filter_thread = (thread_id != NULL && thread_id[0] != '\0');

    st = vtlk_dag_lock_acquire(dag->l3_dir, 0, &lock);
    if (st != VTLK_OK) {
        return st;
    }

    st = vtlk_dag_load_index_locked(dag);
    if (st != VTLK_OK) {
        vtlk_dag_lock_release(&lock);
        return st;
    }

    for (i = 0u; i < dag->count; ++i) {
        if (!filter_thread || strcmp(dag->entries[i].thread_id, thread_id) == 0) {
            match_count++;
        }
    }

    if (out_metas == NULL || max_metas == 0u) {
        *out_count = match_count;
        vtlk_dag_lock_release(&lock);
        return VTLK_OK;
    }

    if (match_count == 0u) {
        *out_count = 0u;
        vtlk_dag_lock_release(&lock);
        return VTLK_OK;
    }

    tmp = (vtlk_kv_checkpoint_meta_t *)malloc(
        (size_t)match_count * sizeof(vtlk_kv_checkpoint_meta_t));
    if (tmp == NULL) {
        vtlk_dag_lock_release(&lock);
        return VTLK_ERR_FULL;
    }

    match_count = 0u;
    for (i = 0u; i < dag->count; ++i) {
        if (!filter_thread || strcmp(dag->entries[i].thread_id, thread_id) == 0) {
            tmp[match_count++] = dag->entries[i];
        }
    }

    qsort(tmp, (size_t)match_count, sizeof(vtlk_kv_checkpoint_meta_t), vtlk_dag_cmp_seq_desc);

    if (match_count > max_metas) {
        match_count = max_metas;
    }
    memcpy(out_metas, tmp, (size_t)match_count * sizeof(vtlk_kv_checkpoint_meta_t));
    *out_count = match_count;

    free(tmp);
    vtlk_dag_lock_release(&lock);
    return VTLK_OK;
}
