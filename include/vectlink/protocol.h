#ifndef VECTLINK_PROTOCOL_H
#define VECTLINK_PROTOCOL_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define VTLK_HEADER_SIZE      64u
#define VTLK_PROTOCOL_VERSION 0x0001u
#define VTLK_LAYER_FINAL      0xFFFFu

#define VTLK_MAGIC_0 ((uint8_t)'V')
#define VTLK_MAGIC_1 ((uint8_t)'T')
#define VTLK_MAGIC_2 ((uint8_t)'L')
#define VTLK_MAGIC_3 ((uint8_t)'K')

typedef enum vtlk_status {
    VTLK_OK = 0,
    VTLK_ERR_NULL_PTR = -1,
    VTLK_ERR_BUFFER_TOO_SMALL = -2,
    VTLK_ERR_INVALID_MAGIC = -3,
    VTLK_ERR_INVALID_HEADER_SIZE = -4,
    VTLK_ERR_UNSUPPORTED_VERSION = -5,
    VTLK_ERR_CRC_MISMATCH = -6,
    VTLK_ERR_INVALID_PARAM = -7,
    VTLK_ERR_INVALID_ARG = -7,
    VTLK_ERR_FULL = -8,
    VTLK_ERR_EMPTY = -9,
    VTLK_ERR_SHM = -10,
    VTLK_ERR_BUSY = -11
} vtlk_status_t;

typedef enum vtlk_quant_scheme {
    VTLK_QUANT_FP16 = 0,
    VTLK_QUANT_INT4_BLOCK = 1,
    VTLK_QUANT_INT4 = 1,
    VTLK_QUANT_RVQ10 = 2,
    VTLK_QUANT_RAW = 3
} vtlk_quant_scheme_t;

typedef enum vtlk_layout {
    VTLK_LAYOUT_ROW_MAJOR = 0
} vtlk_layout_t;

typedef enum vtlk_codec {
    VTLK_CODEC_NONE = 0
} vtlk_codec_t;

/**
 * @brief Logical representation of the 64-byte Vect-Link packet header.
 *
 * This struct is never cast directly to or from the wire format; all
 * serialization and deserialization uses explicit little-endian offsets.
 */
typedef struct vtlk_header {
    uint8_t magic[4];
    uint16_t header_bytes;
    uint16_t version;
    uint32_t flags;
    uint64_t model_family_hash;
    uint32_t d_model;
    uint32_t n_tokens;
    uint8_t quant_scheme;
    uint8_t layout;
    uint8_t codec;
    uint8_t reserved0;
    uint32_t payload_bytes;
    uint32_t raw_bytes;
    uint64_t sequence;
    uint32_t payload_crc32c;
    uint16_t layer_index;
    uint32_t pos_offset;
    uint16_t reserved1;
    uint32_t header_crc32c;
} vtlk_header_t;

/**
 * @brief Initializes a vtlk_header_t with safe v0.1.0 protocol defaults.
 *
 * Sets magic to "VTLK", header_bytes to 64, version to 0x0001,
 * quant_scheme to VTLK_QUANT_FP16, layout to VTLK_LAYOUT_ROW_MAJOR,
 * codec to VTLK_CODEC_NONE, layer_index to VTLK_LAYER_FINAL, and all
 * remaining fields to 0.
 *
 * @param hdr Pointer to the header struct to initialize (no-op if NULL).
 */
void vtlk_header_init_defaults(vtlk_header_t *hdr);

/**
 * @brief Serializes a vtlk_header_t into a 64-byte little-endian wire buffer.
 *
 * Computes and writes header_crc32c over bytes [0..59] at offset 60.
 *
 * @param hdr     Source header struct (must have d_model > 0 and n_tokens > 0).
 * @param buf     Caller-provided destination buffer.
 * @param buf_len Size of destination buffer in bytes (must be >= 64).
 * @return VTLK_OK on success, or a negative vtlk_status_t error code.
 */
vtlk_status_t vtlk_header_encode(const vtlk_header_t *hdr,
                                 uint8_t *buf,
                                 size_t buf_len);

/**
 * @brief Deserializes and validates a 64-byte little-endian wire header.
 *
 * Validates magic ("VTLK"), header_bytes == 64, header_crc32c over [0..59],
 * protocol version, and basic parameter invariants before populating *hdr.
 *
 * @param buf     Caller-provided source buffer.
 * @param buf_len Size of source buffer in bytes (must be >= 64).
 * @param hdr     Destination header struct (untouched on error).
 * @return VTLK_OK on success, or a negative vtlk_status_t error code.
 */
vtlk_status_t vtlk_header_decode(const uint8_t *buf,
                                 size_t buf_len,
                                 vtlk_header_t *hdr);

#ifdef __cplusplus
}
#endif

#endif /* VECTLINK_PROTOCOL_H */
