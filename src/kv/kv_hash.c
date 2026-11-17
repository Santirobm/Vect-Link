#include "vectlink_kv.h"

#include <string.h>

static uint64_t vtlk_splitmix64(uint64_t x)
{
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

uint64_t vtlk_kv_hash_tokens(uint64_t prev_hash,
                             const uint32_t *token_ids,
                             size_t token_count)
{
    uint64_t h = prev_hash;
    size_t i;

    if (token_ids == NULL || token_count == 0u) {
        return h;
    }

    for (i = 0u; i < token_count; ++i) {
        uint64_t x = (uint64_t)token_ids[i] + 0x9E3779B97F4A7C15ULL;
        h ^= x + (h << 6) + (h >> 2);
        h = vtlk_splitmix64(h);
    }

    return h;
}

uint64_t vtlk_kv_hash_bytes(const void *data, size_t len)
{
    const uint8_t *p = (const uint8_t *)data;
    uint64_t h = 0x243F6A8885A308D3ULL ^ (uint64_t)len;
    size_t i = 0u;

    if (data == NULL || len == 0u) {
        return 0u;
    }

    while (i + 8u <= len) {
        uint64_t w;
        memcpy(&w, p + i, sizeof(w));
        h ^= vtlk_splitmix64(w + 0x9E3779B97F4A7C15ULL);
        h = (h << 27) | (h >> 37);
        h = h * 0x9E3779B97F4A7C15ULL + 0x517CC1B727220A95ULL;
        i += 8u;
    }

    if (i < len) {
        uint64_t tail = 0u;
        memcpy(&tail, p + i, len - i);
        h ^= vtlk_splitmix64(tail + 0x9E3779B97F4A7C15ULL);
    }

    return vtlk_splitmix64(h ^ (uint64_t)len);
}
