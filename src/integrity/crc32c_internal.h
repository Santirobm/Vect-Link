#ifndef VECTLINK_INTEGRITY_CRC32C_INTERNAL_H
#define VECTLINK_INTEGRITY_CRC32C_INTERNAL_H

#include <stddef.h>
#include <stdint.h>

uint32_t vtlk_crc32c_scalar_update(uint32_t crc, const void *data, size_t len);
uint32_t vtlk_crc32c_scalar(const void *data, size_t len);

uint32_t vtlk_crc32c_sse42_update(uint32_t crc, const void *data, size_t len);
uint32_t vtlk_crc32c_sse42(const void *data, size_t len);

#endif /* VECTLINK_INTEGRITY_CRC32C_INTERNAL_H */
