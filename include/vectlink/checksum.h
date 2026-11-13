#ifndef VECTLINK_CHECKSUM_H
#define VECTLINK_CHECKSUM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Computes the CRC32C (Castagnoli, polynomial 0x82F63B78) checksum
 *        of a contiguous memory buffer.
 *
 * @param data Pointer to the input buffer (may be NULL only if len == 0).
 * @param len  Number of bytes to process.
 * @return 32-bit CRC32C checksum.
 */
uint32_t vtlk_crc32c(const void *data, size_t len);

/**
 * @brief Incrementally updates a running CRC32C (Castagnoli) checksum.
 *
 * Pass crc = 0 for the first chunk, and feed the returned value into
 * subsequent calls.
 *
 * @param crc  Previous CRC32C value (0 for initial call).
 * @param data Pointer to the input buffer (may be NULL only if len == 0).
 * @param len  Number of bytes to process.
 * @return Updated 32-bit CRC32C checksum.
 */
uint32_t vtlk_crc32c_update(uint32_t crc, const void *data, size_t len);

#ifdef __cplusplus
}
#endif

#endif /* VECTLINK_CHECKSUM_H */
