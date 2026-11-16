#ifndef VTLK_LLAMA_BRIDGE_COMMON_H
#define VTLK_LLAMA_BRIDGE_COMMON_H

#include "llama.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <sched.h>
#include <time.h>
#endif

#define DEFAULT_SHM_NAME         "/vtlk_llama_bridge"
#define VTLK_FLAG_KV_STATE       (1u << 0)
#define VTLK_LLAMA_KV_META_MAGIC 0x4B564C4Du
#define VTLK_LLAMA_KV_META_SIZE  16u
#define VTLK_GGUF_HASH_PREFIX    65536u

typedef enum vtlk_bridge_mode {
    VTLK_MODE_EMBD = 0,
    VTLK_MODE_KV_STATE = 1
} vtlk_bridge_mode_t;

typedef struct vtlk_llama_kv_meta {
    uint32_t magic;
    uint32_t llama_build;
    int32_t last_token_id;
    uint32_t raw_state_bytes;
} vtlk_llama_kv_meta_t;

static inline uint64_t vtlk_llama_now_ns(void)
{
#ifdef _WIN32
    static LARGE_INTEGER freq = { { 0, 0 } };
    LARGE_INTEGER counter;
    uint64_t q;
    uint64_t f;
    if (freq.QuadPart == 0) {
        QueryPerformanceFrequency(&freq);
    }
    QueryPerformanceCounter(&counter);
    q = (uint64_t)counter.QuadPart;
    f = (uint64_t)freq.QuadPart;
    return (q / f) * 1000000000ULL + ((q % f) * 1000000000ULL) / f;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
#endif
}

static inline void vtlk_llama_cpu_yield(void)
{
#ifdef _WIN32
    SwitchToThread();
#else
    sched_yield();
#endif
}

static inline int vtlk_llama_parse_mode(const char *s, vtlk_bridge_mode_t *out)
{
    if (strcmp(s, "embd") == 0) {
        *out = VTLK_MODE_EMBD;
        return 1;
    }
    if (strcmp(s, "kv-state") == 0) {
        *out = VTLK_MODE_KV_STATE;
        return 1;
    }
    return 0;
}

static inline void vtlk_llama_write_u32_le(uint8_t *dst, uint32_t v)
{
    dst[0] = (uint8_t)(v & 0xFFu);
    dst[1] = (uint8_t)((v >> 8) & 0xFFu);
    dst[2] = (uint8_t)((v >> 16) & 0xFFu);
    dst[3] = (uint8_t)((v >> 24) & 0xFFu);
}

static inline uint32_t vtlk_llama_read_u32_le(const uint8_t *src)
{
    return ((uint32_t)src[0]) |
           ((uint32_t)src[1] << 8) |
           ((uint32_t)src[2] << 16) |
           ((uint32_t)src[3] << 24);
}

static inline void vtlk_llama_kv_meta_encode(const vtlk_llama_kv_meta_t *meta,
                                             uint8_t *dst)
{
    vtlk_llama_write_u32_le(dst + 0u, meta->magic);
    vtlk_llama_write_u32_le(dst + 4u, meta->llama_build);
    vtlk_llama_write_u32_le(dst + 8u, (uint32_t)meta->last_token_id);
    vtlk_llama_write_u32_le(dst + 12u, meta->raw_state_bytes);
}

static inline void vtlk_llama_kv_meta_decode(const uint8_t *src,
                                             vtlk_llama_kv_meta_t *meta)
{
    meta->magic = vtlk_llama_read_u32_le(src + 0u);
    meta->llama_build = vtlk_llama_read_u32_le(src + 4u);
    meta->last_token_id = (int32_t)vtlk_llama_read_u32_le(src + 8u);
    meta->raw_state_bytes = vtlk_llama_read_u32_le(src + 12u);
}

static inline uint64_t vtlk_fnv1a_64_bytes(uint64_t h,
                                           const void *data,
                                           size_t len)
{
    const uint8_t *p = (const uint8_t *)data;
    size_t i;
    for (i = 0u; i < len; ++i) {
        h ^= (uint64_t)p[i];
        h *= 1099511628211ULL;
    }
    return h;
}

static inline uint64_t vtlk_fnv1a_64_u32(uint64_t h, uint32_t v)
{
    uint8_t le[4];
    vtlk_llama_write_u32_le(le, v);
    return vtlk_fnv1a_64_bytes(h, le, sizeof(le));
}

static inline uint64_t vtlk_fnv1a_64_u64(uint64_t h, uint64_t v)
{
    uint8_t le[8];
    le[0] = (uint8_t)(v & 0xFFu);
    le[1] = (uint8_t)((v >> 8) & 0xFFu);
    le[2] = (uint8_t)((v >> 16) & 0xFFu);
    le[3] = (uint8_t)((v >> 24) & 0xFFu);
    le[4] = (uint8_t)((v >> 32) & 0xFFu);
    le[5] = (uint8_t)((v >> 40) & 0xFFu);
    le[6] = (uint8_t)((v >> 48) & 0xFFu);
    le[7] = (uint8_t)((v >> 56) & 0xFFu);
    return vtlk_fnv1a_64_bytes(h, le, sizeof(le));
}

static inline uint32_t vtlk_llama_build_stamp(void)
{
    uint64_t h = 14695981039346656037ULL;
    const char *ver = llama_version();
#ifdef LLAMA_BUILD_NUMBER
    h = vtlk_fnv1a_64_u32(h, (uint32_t)(LLAMA_BUILD_NUMBER));
#endif
    h = vtlk_fnv1a_64_u32(h, (uint32_t)LLAMA_STATE_SEQ_VERSION);
    if (ver != NULL) {
        h = vtlk_fnv1a_64_bytes(h, ver, strlen(ver));
    }
    return (uint32_t)(h ^ (h >> 32));
}

static inline uint64_t vtlk_llama_compute_compat_hash(
    const char *model_path,
    const struct llama_model *model,
    struct llama_context *ctx,
    int is_kv_mode)
{
    uint64_t h = 14695981039346656037ULL;
    char model_desc[256];
    float rope_scale;
    uint32_t rope_scale_bits = 0u;
    FILE *fp;
    uint8_t buf[4096];
    size_t remaining = VTLK_GGUF_HASH_PREFIX;

    if (model_path == NULL || model == NULL) {
        return 0ULL;
    }

    /* 1. Exact GGUF file size + first 64 KB (magic, version, tensor count, metadata) */
    {
#ifdef _WIN32
        struct __stat64 st;
        if (_stat64(model_path, &st) != 0 || st.st_size <= 0) {
            return 0ULL;
        }
        h = vtlk_fnv1a_64_u64(h, (uint64_t)st.st_size);
#else
        struct stat st;
        if (stat(model_path, &st) != 0 || st.st_size <= 0) {
            return 0ULL;
        }
        h = vtlk_fnv1a_64_u64(h, (uint64_t)st.st_size);
#endif
    }

    fp = fopen(model_path, "rb");
    if (fp == NULL) {
        return 0ULL;
    }
    while (remaining > 0u) {
        size_t chunk = (remaining < sizeof(buf)) ? remaining : sizeof(buf);
        size_t got = fread(buf, 1u, chunk, fp);
        if (got == 0u) {
            break;
        }
        h = vtlk_fnv1a_64_bytes(h, buf, got);
        remaining -= got;
    }
    fclose(fp);

    /* 2. Structural model parameters */
    memset(model_desc, 0, sizeof(model_desc));
    llama_model_desc(model, model_desc, sizeof(model_desc));
    h = vtlk_fnv1a_64_bytes(h, model_desc, strlen(model_desc));
    h = vtlk_fnv1a_64_u32(h, (uint32_t)llama_model_n_embd(model));
    h = vtlk_fnv1a_64_u32(h, (uint32_t)llama_model_n_layer(model));
    h = vtlk_fnv1a_64_u32(h, (uint32_t)llama_model_n_head(model));
    h = vtlk_fnv1a_64_u32(h, (uint32_t)llama_model_rope_type(model));

    rope_scale = llama_model_rope_freq_scale_train(model);
    memcpy(&rope_scale_bits, &rope_scale, sizeof(rope_scale_bits));
    h = vtlk_fnv1a_64_u32(h, rope_scale_bits);

    /* 3. Strict KV-Cache runtime environment compatibility */
    if (is_kv_mode && ctx != NULL) {
        h = vtlk_fnv1a_64_u32(h, vtlk_llama_build_stamp());
        h = vtlk_fnv1a_64_u32(h, llama_n_ctx(ctx));
        h = vtlk_fnv1a_64_u64(h, (uint64_t)llama_state_seq_get_size(ctx, 0));
    }

    return h;
}

#endif /* VTLK_LLAMA_BRIDGE_COMMON_H */
