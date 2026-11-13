#ifndef VECTLINK_INTERNAL_ENDIAN_H
#define VECTLINK_INTERNAL_ENDIAN_H

#include <stdint.h>

static inline void vtlk_write_u16_le(uint8_t *dst, uint16_t val)
{
    dst[0] = (uint8_t)(val & 0xFFu);
    dst[1] = (uint8_t)((val >> 8) & 0xFFu);
}

static inline void vtlk_write_u32_le(uint8_t *dst, uint32_t val)
{
    dst[0] = (uint8_t)(val & 0xFFu);
    dst[1] = (uint8_t)((val >> 8) & 0xFFu);
    dst[2] = (uint8_t)((val >> 16) & 0xFFu);
    dst[3] = (uint8_t)((val >> 24) & 0xFFu);
}

static inline void vtlk_write_u64_le(uint8_t *dst, uint64_t val)
{
    dst[0] = (uint8_t)(val & 0xFFu);
    dst[1] = (uint8_t)((val >> 8) & 0xFFu);
    dst[2] = (uint8_t)((val >> 16) & 0xFFu);
    dst[3] = (uint8_t)((val >> 24) & 0xFFu);
    dst[4] = (uint8_t)((val >> 32) & 0xFFu);
    dst[5] = (uint8_t)((val >> 40) & 0xFFu);
    dst[6] = (uint8_t)((val >> 48) & 0xFFu);
    dst[7] = (uint8_t)((val >> 56) & 0xFFu);
}

static inline uint16_t vtlk_read_u16_le(const uint8_t *src)
{
    return (uint16_t)(((uint16_t)src[0]) |
                      ((uint16_t)src[1] << 8));
}

static inline uint32_t vtlk_read_u32_le(const uint8_t *src)
{
    return ((uint32_t)src[0]) |
           ((uint32_t)src[1] << 8) |
           ((uint32_t)src[2] << 16) |
           ((uint32_t)src[3] << 24);
}

static inline uint64_t vtlk_read_u64_le(const uint8_t *src)
{
    return ((uint64_t)src[0]) |
           ((uint64_t)src[1] << 8) |
           ((uint64_t)src[2] << 16) |
           ((uint64_t)src[3] << 24) |
           ((uint64_t)src[4] << 32) |
           ((uint64_t)src[5] << 40) |
           ((uint64_t)src[6] << 48) |
           ((uint64_t)src[7] << 56);
}

#endif /* VECTLINK_INTERNAL_ENDIAN_H */
