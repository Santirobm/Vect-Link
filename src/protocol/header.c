#include "vectlink/protocol.h"
#include "vectlink/checksum.h"
#include "protocol/endian.h"

#include <string.h>

/* Wire format byte offsets */
#define VTLK_OFF_MAGIC             0u
#define VTLK_OFF_HEADER_BYTES      4u
#define VTLK_OFF_VERSION           6u
#define VTLK_OFF_FLAGS             8u
#define VTLK_OFF_MODEL_FAMILY_HASH 12u
#define VTLK_OFF_D_MODEL           20u
#define VTLK_OFF_N_TOKENS          24u
#define VTLK_OFF_QUANT_SCHEME      28u
#define VTLK_OFF_LAYOUT            29u
#define VTLK_OFF_CODEC             30u
#define VTLK_OFF_RESERVED0         31u
#define VTLK_OFF_PAYLOAD_BYTES     32u
#define VTLK_OFF_RAW_BYTES         36u
#define VTLK_OFF_SEQUENCE          40u
#define VTLK_OFF_PAYLOAD_CRC32C    48u
#define VTLK_OFF_LAYER_INDEX       52u
#define VTLK_OFF_POS_OFFSET        54u
#define VTLK_OFF_RESERVED1         58u
#define VTLK_OFF_HEADER_CRC32C     60u

/* Wire format byte offsets (v2, 96 bytes) */
#define VTLK_OFF_V2_PAYLOAD_BYTES  32u
#define VTLK_OFF_V2_RAW_BYTES      40u
#define VTLK_OFF_V2_SEQUENCE       48u
#define VTLK_OFF_V2_PAYLOAD_CRC32C 56u
#define VTLK_OFF_V2_LAYER_INDEX    60u
#define VTLK_OFF_V2_RESERVED1      62u
#define VTLK_OFF_V2_POS_OFFSET     64u
#define VTLK_OFF_V2_RESERVED2      68u
#define VTLK_OFF_V2_RESERVED_EXT   72u
#define VTLK_OFF_V2_HEADER_CRC32C  92u

static int vtlk_validate_common(uint32_t d_model,
                                uint32_t n_tokens,
                                uint8_t quant_scheme,
                                uint8_t layout,
                                uint8_t codec)
{
    if (d_model == 0u || n_tokens == 0u) {
        return 0;
    }
    if (quant_scheme > (uint8_t)VTLK_QUANT_RAW) {
        return 0;
    }
    if (layout != (uint8_t)VTLK_LAYOUT_ROW_MAJOR) {
        return 0;
    }
    if (codec != (uint8_t)VTLK_CODEC_NONE) {
        return 0;
    }
    return 1;
}

static int vtlk_validate_params(const vtlk_header_t *hdr)
{
    if (!vtlk_validate_common(hdr->d_model,
                              hdr->n_tokens,
                              hdr->quant_scheme,
                              hdr->layout,
                              hdr->codec)) {
        return 0;
    }
    if (hdr->reserved0 != 0u || hdr->reserved1 != 0u) {
        return 0;
    }
    return 1;
}

static int vtlk_validate_params_v2(const vtlk_header_v2_t *hdr)
{
    if (!vtlk_validate_common(hdr->d_model,
                              hdr->n_tokens,
                              hdr->quant_scheme,
                              hdr->layout,
                              hdr->codec)) {
        return 0;
    }
    if (hdr->reserved0 != 0u || hdr->reserved1 != 0u || hdr->reserved2 != 0u) {
        return 0;
    }
    return 1;
}

void vtlk_header_init_defaults(vtlk_header_t *hdr)
{
    if (hdr == NULL) {
        return;
    }

    memset(hdr, 0, sizeof(*hdr));
    hdr->magic[0] = VTLK_MAGIC_0;
    hdr->magic[1] = VTLK_MAGIC_1;
    hdr->magic[2] = VTLK_MAGIC_2;
    hdr->magic[3] = VTLK_MAGIC_3;
    hdr->header_bytes = (uint16_t)VTLK_HEADER_SIZE;
    hdr->version = (uint16_t)VTLK_PROTOCOL_VERSION;
    hdr->quant_scheme = (uint8_t)VTLK_QUANT_FP16;
    hdr->layout = (uint8_t)VTLK_LAYOUT_ROW_MAJOR;
    hdr->codec = (uint8_t)VTLK_CODEC_NONE;
    hdr->layer_index = (uint16_t)VTLK_LAYER_FINAL;
}

vtlk_status_t vtlk_header_encode(const vtlk_header_t *hdr,
                                 uint8_t *buf,
                                 size_t buf_len)
{
    uint32_t crc;

    if (hdr == NULL || buf == NULL) {
        return VTLK_ERR_NULL_PTR;
    }
    if (buf_len < VTLK_HEADER_SIZE) {
        return VTLK_ERR_BUFFER_TOO_SMALL;
    }
    if (hdr->version != (uint16_t)VTLK_PROTOCOL_VERSION) {
        return VTLK_ERR_UNSUPPORTED_VERSION;
    }
    if (!vtlk_validate_params(hdr)) {
        return VTLK_ERR_INVALID_PARAM;
    }

    buf[VTLK_OFF_MAGIC + 0u] = VTLK_MAGIC_0;
    buf[VTLK_OFF_MAGIC + 1u] = VTLK_MAGIC_1;
    buf[VTLK_OFF_MAGIC + 2u] = VTLK_MAGIC_2;
    buf[VTLK_OFF_MAGIC + 3u] = VTLK_MAGIC_3;

    vtlk_write_u16_le(buf + VTLK_OFF_HEADER_BYTES, (uint16_t)VTLK_HEADER_SIZE);
    vtlk_write_u16_le(buf + VTLK_OFF_VERSION, hdr->version);
    vtlk_write_u32_le(buf + VTLK_OFF_FLAGS, hdr->flags);
    vtlk_write_u64_le(buf + VTLK_OFF_MODEL_FAMILY_HASH, hdr->model_family_hash);
    vtlk_write_u32_le(buf + VTLK_OFF_D_MODEL, hdr->d_model);
    vtlk_write_u32_le(buf + VTLK_OFF_N_TOKENS, hdr->n_tokens);

    buf[VTLK_OFF_QUANT_SCHEME] = hdr->quant_scheme;
    buf[VTLK_OFF_LAYOUT] = hdr->layout;
    buf[VTLK_OFF_CODEC] = hdr->codec;
    buf[VTLK_OFF_RESERVED0] = 0u;

    vtlk_write_u32_le(buf + VTLK_OFF_PAYLOAD_BYTES, hdr->payload_bytes);
    vtlk_write_u32_le(buf + VTLK_OFF_RAW_BYTES, hdr->raw_bytes);
    vtlk_write_u64_le(buf + VTLK_OFF_SEQUENCE, hdr->sequence);
    vtlk_write_u32_le(buf + VTLK_OFF_PAYLOAD_CRC32C, hdr->payload_crc32c);
    vtlk_write_u16_le(buf + VTLK_OFF_LAYER_INDEX, hdr->layer_index);
    vtlk_write_u32_le(buf + VTLK_OFF_POS_OFFSET, hdr->pos_offset);
    vtlk_write_u16_le(buf + VTLK_OFF_RESERVED1, 0u);

    crc = vtlk_crc32c(buf, VTLK_OFF_HEADER_CRC32C);
    vtlk_write_u32_le(buf + VTLK_OFF_HEADER_CRC32C, crc);

    return VTLK_OK;
}

vtlk_status_t vtlk_header_decode(const uint8_t *buf,
                                 size_t buf_len,
                                 vtlk_header_t *hdr)
{
    vtlk_header_t tmp;
    uint32_t expected_crc;

    if (buf == NULL || hdr == NULL) {
        return VTLK_ERR_NULL_PTR;
    }
    if (buf_len < VTLK_HEADER_SIZE) {
        return VTLK_ERR_BUFFER_TOO_SMALL;
    }

    if (buf[VTLK_OFF_MAGIC + 0u] != VTLK_MAGIC_0 ||
        buf[VTLK_OFF_MAGIC + 1u] != VTLK_MAGIC_1 ||
        buf[VTLK_OFF_MAGIC + 2u] != VTLK_MAGIC_2 ||
        buf[VTLK_OFF_MAGIC + 3u] != VTLK_MAGIC_3) {
        return VTLK_ERR_INVALID_MAGIC;
    }

    tmp.header_bytes = vtlk_read_u16_le(buf + VTLK_OFF_HEADER_BYTES);
    if (tmp.header_bytes != (uint16_t)VTLK_HEADER_SIZE) {
        return VTLK_ERR_INVALID_HEADER_SIZE;
    }

    tmp.header_crc32c = vtlk_read_u32_le(buf + VTLK_OFF_HEADER_CRC32C);
    expected_crc = vtlk_crc32c(buf, VTLK_OFF_HEADER_CRC32C);
    if (tmp.header_crc32c != expected_crc) {
        return VTLK_ERR_CRC_MISMATCH;
    }

    tmp.version = vtlk_read_u16_le(buf + VTLK_OFF_VERSION);
    if (tmp.version != (uint16_t)VTLK_PROTOCOL_VERSION) {
        return VTLK_ERR_UNSUPPORTED_VERSION;
    }

    tmp.magic[0] = buf[VTLK_OFF_MAGIC + 0u];
    tmp.magic[1] = buf[VTLK_OFF_MAGIC + 1u];
    tmp.magic[2] = buf[VTLK_OFF_MAGIC + 2u];
    tmp.magic[3] = buf[VTLK_OFF_MAGIC + 3u];
    tmp.flags = vtlk_read_u32_le(buf + VTLK_OFF_FLAGS);
    tmp.model_family_hash = vtlk_read_u64_le(buf + VTLK_OFF_MODEL_FAMILY_HASH);
    tmp.d_model = vtlk_read_u32_le(buf + VTLK_OFF_D_MODEL);
    tmp.n_tokens = vtlk_read_u32_le(buf + VTLK_OFF_N_TOKENS);
    tmp.quant_scheme = buf[VTLK_OFF_QUANT_SCHEME];
    tmp.layout = buf[VTLK_OFF_LAYOUT];
    tmp.codec = buf[VTLK_OFF_CODEC];
    tmp.reserved0 = buf[VTLK_OFF_RESERVED0];
    tmp.payload_bytes = vtlk_read_u32_le(buf + VTLK_OFF_PAYLOAD_BYTES);
    tmp.raw_bytes = vtlk_read_u32_le(buf + VTLK_OFF_RAW_BYTES);
    tmp.sequence = vtlk_read_u64_le(buf + VTLK_OFF_SEQUENCE);
    tmp.payload_crc32c = vtlk_read_u32_le(buf + VTLK_OFF_PAYLOAD_CRC32C);
    tmp.layer_index = vtlk_read_u16_le(buf + VTLK_OFF_LAYER_INDEX);
    tmp.pos_offset = vtlk_read_u32_le(buf + VTLK_OFF_POS_OFFSET);
    tmp.reserved1 = vtlk_read_u16_le(buf + VTLK_OFF_RESERVED1);

    if (!vtlk_validate_params(&tmp)) {
        return VTLK_ERR_INVALID_PARAM;
    }

    *hdr = tmp;
    return VTLK_OK;
}

void vtlk_header_v2_init_defaults(vtlk_header_v2_t *hdr)
{
    if (hdr == NULL) {
        return;
    }

    memset(hdr, 0, sizeof(*hdr));
    hdr->magic[0] = VTLK_MAGIC_0;
    hdr->magic[1] = VTLK_MAGIC_1;
    hdr->magic[2] = VTLK_MAGIC_2;
    hdr->magic[3] = VTLK_MAGIC_3;
    hdr->header_bytes = (uint16_t)VTLK_HEADER_V2_SIZE;
    hdr->version = (uint16_t)VTLK_PROTOCOL_VERSION_V2;
    hdr->quant_scheme = (uint8_t)VTLK_QUANT_FP16;
    hdr->layout = (uint8_t)VTLK_LAYOUT_ROW_MAJOR;
    hdr->codec = (uint8_t)VTLK_CODEC_NONE;
    hdr->layer_index = (uint16_t)VTLK_LAYER_FINAL;
}

vtlk_status_t vtlk_header_v2_encode(const vtlk_header_v2_t *hdr,
                                    uint8_t *buf,
                                    size_t buf_len)
{
    uint32_t crc;

    if (hdr == NULL || buf == NULL) {
        return VTLK_ERR_NULL_PTR;
    }
    if (buf_len < VTLK_HEADER_V2_SIZE) {
        return VTLK_ERR_BUFFER_TOO_SMALL;
    }
    if (hdr->version != (uint16_t)VTLK_PROTOCOL_VERSION_V2) {
        return VTLK_ERR_UNSUPPORTED_VERSION;
    }
    if (!vtlk_validate_params_v2(hdr)) {
        return VTLK_ERR_INVALID_PARAM;
    }

    buf[VTLK_OFF_MAGIC + 0u] = VTLK_MAGIC_0;
    buf[VTLK_OFF_MAGIC + 1u] = VTLK_MAGIC_1;
    buf[VTLK_OFF_MAGIC + 2u] = VTLK_MAGIC_2;
    buf[VTLK_OFF_MAGIC + 3u] = VTLK_MAGIC_3;

    vtlk_write_u16_le(buf + VTLK_OFF_HEADER_BYTES, (uint16_t)VTLK_HEADER_V2_SIZE);
    vtlk_write_u16_le(buf + VTLK_OFF_VERSION, hdr->version);
    vtlk_write_u32_le(buf + VTLK_OFF_FLAGS, hdr->flags);
    vtlk_write_u64_le(buf + VTLK_OFF_MODEL_FAMILY_HASH, hdr->model_family_hash);
    vtlk_write_u32_le(buf + VTLK_OFF_D_MODEL, hdr->d_model);
    vtlk_write_u32_le(buf + VTLK_OFF_N_TOKENS, hdr->n_tokens);

    buf[VTLK_OFF_QUANT_SCHEME] = hdr->quant_scheme;
    buf[VTLK_OFF_LAYOUT] = hdr->layout;
    buf[VTLK_OFF_CODEC] = hdr->codec;
    buf[VTLK_OFF_RESERVED0] = 0u;

    vtlk_write_u64_le(buf + VTLK_OFF_V2_PAYLOAD_BYTES, hdr->payload_bytes);
    vtlk_write_u64_le(buf + VTLK_OFF_V2_RAW_BYTES, hdr->raw_bytes);
    vtlk_write_u64_le(buf + VTLK_OFF_V2_SEQUENCE, hdr->sequence);
    vtlk_write_u32_le(buf + VTLK_OFF_V2_PAYLOAD_CRC32C, hdr->payload_crc32c);
    vtlk_write_u16_le(buf + VTLK_OFF_V2_LAYER_INDEX, hdr->layer_index);
    vtlk_write_u16_le(buf + VTLK_OFF_V2_RESERVED1, 0u);
    vtlk_write_u32_le(buf + VTLK_OFF_V2_POS_OFFSET, hdr->pos_offset);
    vtlk_write_u32_le(buf + VTLK_OFF_V2_RESERVED2, 0u);
    memset(buf + VTLK_OFF_V2_RESERVED_EXT, 0, sizeof(hdr->reserved_ext));

    crc = vtlk_crc32c(buf, VTLK_OFF_V2_HEADER_CRC32C);
    vtlk_write_u32_le(buf + VTLK_OFF_V2_HEADER_CRC32C, crc);

    return VTLK_OK;
}

vtlk_status_t vtlk_header_v2_decode(const uint8_t *buf,
                                    size_t buf_len,
                                    vtlk_header_v2_t *hdr)
{
    vtlk_header_v2_t tmp;
    uint32_t expected_crc;

    if (buf == NULL || hdr == NULL) {
        return VTLK_ERR_NULL_PTR;
    }
    if (buf_len < VTLK_HEADER_V2_SIZE) {
        return VTLK_ERR_BUFFER_TOO_SMALL;
    }

    if (buf[VTLK_OFF_MAGIC + 0u] != VTLK_MAGIC_0 ||
        buf[VTLK_OFF_MAGIC + 1u] != VTLK_MAGIC_1 ||
        buf[VTLK_OFF_MAGIC + 2u] != VTLK_MAGIC_2 ||
        buf[VTLK_OFF_MAGIC + 3u] != VTLK_MAGIC_3) {
        return VTLK_ERR_INVALID_MAGIC;
    }

    tmp.header_bytes = vtlk_read_u16_le(buf + VTLK_OFF_HEADER_BYTES);
    if (tmp.header_bytes != (uint16_t)VTLK_HEADER_V2_SIZE) {
        return VTLK_ERR_INVALID_HEADER_SIZE;
    }

    tmp.header_crc32c = vtlk_read_u32_le(buf + VTLK_OFF_V2_HEADER_CRC32C);
    expected_crc = vtlk_crc32c(buf, VTLK_OFF_V2_HEADER_CRC32C);
    if (tmp.header_crc32c != expected_crc) {
        return VTLK_ERR_CRC_MISMATCH;
    }

    tmp.version = vtlk_read_u16_le(buf + VTLK_OFF_VERSION);
    if (tmp.version != (uint16_t)VTLK_PROTOCOL_VERSION_V2) {
        return VTLK_ERR_UNSUPPORTED_VERSION;
    }

    tmp.magic[0] = buf[VTLK_OFF_MAGIC + 0u];
    tmp.magic[1] = buf[VTLK_OFF_MAGIC + 1u];
    tmp.magic[2] = buf[VTLK_OFF_MAGIC + 2u];
    tmp.magic[3] = buf[VTLK_OFF_MAGIC + 3u];
    tmp.flags = vtlk_read_u32_le(buf + VTLK_OFF_FLAGS);
    tmp.model_family_hash = vtlk_read_u64_le(buf + VTLK_OFF_MODEL_FAMILY_HASH);
    tmp.d_model = vtlk_read_u32_le(buf + VTLK_OFF_D_MODEL);
    tmp.n_tokens = vtlk_read_u32_le(buf + VTLK_OFF_N_TOKENS);
    tmp.quant_scheme = buf[VTLK_OFF_QUANT_SCHEME];
    tmp.layout = buf[VTLK_OFF_LAYOUT];
    tmp.codec = buf[VTLK_OFF_CODEC];
    tmp.reserved0 = buf[VTLK_OFF_RESERVED0];
    tmp.payload_bytes = vtlk_read_u64_le(buf + VTLK_OFF_V2_PAYLOAD_BYTES);
    tmp.raw_bytes = vtlk_read_u64_le(buf + VTLK_OFF_V2_RAW_BYTES);
    tmp.sequence = vtlk_read_u64_le(buf + VTLK_OFF_V2_SEQUENCE);
    tmp.payload_crc32c = vtlk_read_u32_le(buf + VTLK_OFF_V2_PAYLOAD_CRC32C);
    tmp.layer_index = vtlk_read_u16_le(buf + VTLK_OFF_V2_LAYER_INDEX);
    tmp.reserved1 = vtlk_read_u16_le(buf + VTLK_OFF_V2_RESERVED1);
    tmp.pos_offset = vtlk_read_u32_le(buf + VTLK_OFF_V2_POS_OFFSET);
    tmp.reserved2 = vtlk_read_u32_le(buf + VTLK_OFF_V2_RESERVED2);
    memcpy(tmp.reserved_ext, buf + VTLK_OFF_V2_RESERVED_EXT, sizeof(tmp.reserved_ext));

    if (!vtlk_validate_params_v2(&tmp)) {
        return VTLK_ERR_INVALID_PARAM;
    }

    *hdr = tmp;
    return VTLK_OK;
}
