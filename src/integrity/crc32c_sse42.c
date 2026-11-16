#include "integrity/crc32c_internal.h"

#include <nmmintrin.h>
#include <string.h>

uint32_t vtlk_crc32c_sse42_update(uint32_t crc, const void *data, size_t len)
{
    const uint8_t *p = (const uint8_t *)data;
    uint32_t c;

    if (len == 0u || p == NULL) {
        return crc;
    }

    c = ~crc;

    /* 1. Process unaligned leading bytes */
    while (len > 0u && (((uintptr_t)p & 7u) != 0u)) {
        c = _mm_crc32_u8(c, *p);
        ++p;
        --len;
    }

    /* 2. Process 8-byte blocks (or 4-byte blocks on 32-bit x86) */
#if defined(__x86_64__) || defined(_M_X64)
    while (len >= 8u) {
        uint64_t val;
        memcpy(&val, p, sizeof(val));
        c = (uint32_t)_mm_crc32_u64((uint64_t)c, val);
        p += 8;
        len -= 8u;
    }
#else
    while (len >= 4u) {
        uint32_t val;
        memcpy(&val, p, sizeof(val));
        c = _mm_crc32_u32(c, val);
        p += 4;
        len -= 4u;
    }
#endif

    /* 3. Process remaining trailing bytes */
    while (len > 0u) {
        c = _mm_crc32_u8(c, *p);
        ++p;
        --len;
    }

    return ~c;
}

uint32_t vtlk_crc32c_sse42(const void *data, size_t len)
{
    return vtlk_crc32c_sse42_update(0u, data, len);
}
