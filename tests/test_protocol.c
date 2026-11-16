#include "vectlink/protocol.h"
#include "vectlink/checksum.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>

static uint32_t test_read_u32_le(const uint8_t *p)
{
    return ((uint32_t)p[0]) |
           ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) |
           ((uint32_t)p[3] << 24);
}

static void test_crc32c_vectors(void)
{
    static const char rfc_vec[] = "123456789";
    uint8_t zeros[32];
    uint32_t c_empty;
    uint32_t c_rfc;
    uint32_t c_zeros;
    uint32_t c_inc;

    memset(zeros, 0, sizeof(zeros));

    c_empty = vtlk_crc32c(NULL, 0u);
    assert(c_empty == 0x00000000u);

    /* RFC 3720 / iSCSI standard Castagnoli test vector */
    c_rfc = vtlk_crc32c(rfc_vec, 9u);
    assert(c_rfc == 0xE3069283u);

    /* Incremental update must match single-pass */
    c_inc = vtlk_crc32c_update(0u, rfc_vec, 4u);
    c_inc = vtlk_crc32c_update(c_inc, rfc_vec + 4u, 5u);
    assert(c_inc == 0xE3069283u);

    /* 32 zero bytes */
    c_zeros = vtlk_crc32c(zeros, sizeof(zeros));
    assert(c_zeros == 0x8A9136AAu);
}

static void test_roundtrip_and_offsets(void)
{
    vtlk_header_t in;
    vtlk_header_t out;
    uint8_t wire[VTLK_HEADER_SIZE];
    vtlk_status_t st;

    vtlk_header_init_defaults(&in);
    in.flags = 0x11223344u;
    in.model_family_hash = 0x0123456789ABCDEFULL;
    in.d_model = 4096u;
    in.n_tokens = 128u;
    in.quant_scheme = (uint8_t)VTLK_QUANT_INT4_BLOCK;
    in.layout = (uint8_t)VTLK_LAYOUT_ROW_MAJOR;
    in.codec = (uint8_t)VTLK_CODEC_NONE;
    in.payload_bytes = 262144u;
    in.raw_bytes = 1048576u;
    in.sequence = 0xFEDCBA9876543210ULL;
    in.payload_crc32c = 0xCAFEBABEu;
    in.layer_index = 23u;
    in.pos_offset = 512u;

    memset(wire, 0xAA, sizeof(wire));
    st = vtlk_header_encode(&in, wire, sizeof(wire));
    assert(st == VTLK_OK);

    /* Verify exact wire offsets (Little-Endian) */
    /* 0..3: magic "VTLK" */
    assert(wire[0] == 'V');
    assert(wire[1] == 'T');
    assert(wire[2] == 'L');
    assert(wire[3] == 'K');

    /* 4..5: header_bytes == 64 */
    assert(wire[4] == 0x40u && wire[5] == 0x00u);

    /* 6..7: version == 0x0001 */
    assert(wire[6] == 0x01u && wire[7] == 0x00u);

    /* 8..11: flags == 0x11223344 */
    assert(wire[8] == 0x44u && wire[9] == 0x33u &&
           wire[10] == 0x22u && wire[11] == 0x11u);

    /* 12..19: model_family_hash == 0x0123456789ABCDEF */
    assert(wire[12] == 0xEFu && wire[13] == 0xCDu &&
           wire[14] == 0xABu && wire[15] == 0x89u &&
           wire[16] == 0x67u && wire[17] == 0x45u &&
           wire[18] == 0x23u && wire[19] == 0x01u);

    /* 20..23: d_model == 4096 (0x00001000) */
    assert(wire[20] == 0x00u && wire[21] == 0x10u &&
           wire[22] == 0x00u && wire[23] == 0x00u);

    /* 24..27: n_tokens == 128 (0x00000080) */
    assert(wire[24] == 0x80u && wire[25] == 0x00u &&
           wire[26] == 0x00u && wire[27] == 0x00u);

    /* 28..31: quant_scheme, layout, codec, reserved0 */
    assert(wire[28] == 1u);
    assert(wire[29] == 0u);
    assert(wire[30] == 0u);
    assert(wire[31] == 0u);

    /* 32..35: payload_bytes == 262144 (0x00040000) */
    assert(wire[32] == 0x00u && wire[33] == 0x00u &&
           wire[34] == 0x04u && wire[35] == 0x00u);

    /* 36..39: raw_bytes == 1048576 (0x00100000) */
    assert(wire[36] == 0x00u && wire[37] == 0x00u &&
           wire[38] == 0x10u && wire[39] == 0x00u);

    /* 40..47: sequence == 0xFEDCBA9876543210 */
    assert(wire[40] == 0x10u && wire[41] == 0x32u &&
           wire[42] == 0x54u && wire[43] == 0x76u &&
           wire[44] == 0x98u && wire[45] == 0xBAu &&
           wire[46] == 0xDCu && wire[47] == 0xFEu);

    /* 48..51: payload_crc32c == 0xCAFEBABE */
    assert(wire[48] == 0xBEu && wire[49] == 0xBAu &&
           wire[50] == 0xFEu && wire[51] == 0xCAu);

    /* 52..53: layer_index == 23 (0x0017) */
    assert(wire[52] == 0x17u && wire[53] == 0x00u);

    /* 54..57: pos_offset == 512 (0x00000200) */
    assert(wire[54] == 0x00u && wire[55] == 0x02u &&
           wire[56] == 0x00u && wire[57] == 0x00u);

    /* 58..59: reserved1 == 0 */
    assert(wire[58] == 0x00u && wire[59] == 0x00u);

    /* 60..63: header_crc32c == crc32c(wire[0..59]) */
    assert(test_read_u32_le(wire + 60) == vtlk_crc32c(wire, 60u));

    /* Decode and verify field-by-field equality */
    memset(&out, 0x55, sizeof(out));
    st = vtlk_header_decode(wire, sizeof(wire), &out);
    assert(st == VTLK_OK);

    assert(memcmp(out.magic, in.magic, 4u) == 0);
    assert(out.header_bytes == VTLK_HEADER_SIZE);
    assert(out.version == in.version);
    assert(out.flags == in.flags);
    assert(out.model_family_hash == in.model_family_hash);
    assert(out.d_model == in.d_model);
    assert(out.n_tokens == in.n_tokens);
    assert(out.quant_scheme == in.quant_scheme);
    assert(out.layout == in.layout);
    assert(out.codec == in.codec);
    assert(out.reserved0 == 0u);
    assert(out.payload_bytes == in.payload_bytes);
    assert(out.raw_bytes == in.raw_bytes);
    assert(out.sequence == in.sequence);
    assert(out.payload_crc32c == in.payload_crc32c);
    assert(out.layer_index == in.layer_index);
    assert(out.pos_offset == in.pos_offset);
    assert(out.reserved1 == 0u);
    assert(out.header_crc32c == test_read_u32_le(wire + 60));
}

static void test_bit_flip_detection(void)
{
    vtlk_header_t in;
    vtlk_header_t out;
    uint8_t wire[VTLK_HEADER_SIZE];
    uint8_t corrupted[VTLK_HEADER_SIZE];
    size_t byte_idx;
    unsigned int bit_idx;
    vtlk_status_t st;

    vtlk_header_init_defaults(&in);
    in.d_model = 768u;
    in.n_tokens = 16u;
    in.quant_scheme = (uint8_t)VTLK_QUANT_RVQ10;
    in.sequence = 42u;

    st = vtlk_header_encode(&in, wire, sizeof(wire));
    assert(st == VTLK_OK);

    /* Flip every bit in bytes [6..59] and verify header_crc32c catches it */
    for (byte_idx = 6u; byte_idx < 60u; ++byte_idx) {
        for (bit_idx = 0u; bit_idx < 8u; ++bit_idx) {
            memcpy(corrupted, wire, sizeof(wire));
            corrupted[byte_idx] ^= (uint8_t)(1u << bit_idx);

            memset(&out, 0, sizeof(out));
            st = vtlk_header_decode(corrupted, sizeof(corrupted), &out);
            assert(st == VTLK_ERR_CRC_MISMATCH);
        }
    }

    /* Also verify bit flips in header_crc32c itself [60..63] fail CRC */
    for (byte_idx = 60u; byte_idx < VTLK_HEADER_SIZE; ++byte_idx) {
        memcpy(corrupted, wire, sizeof(wire));
        corrupted[byte_idx] ^= 0x01u;
        st = vtlk_header_decode(corrupted, sizeof(corrupted), &out);
        assert(st == VTLK_ERR_CRC_MISMATCH);
    }
}

static void test_rejections_and_edge_cases(void)
{
    vtlk_header_t in;
    vtlk_header_t out;
    uint8_t wire[VTLK_HEADER_SIZE];
    uint8_t bad[VTLK_HEADER_SIZE];
    vtlk_status_t st;

    vtlk_header_init_defaults(&in);
    in.d_model = 256u;
    in.n_tokens = 4u;

    /* NULL pointer checks */
    assert(vtlk_header_encode(NULL, wire, sizeof(wire)) == VTLK_ERR_NULL_PTR);
    assert(vtlk_header_encode(&in, NULL, sizeof(wire)) == VTLK_ERR_NULL_PTR);
    assert(vtlk_header_decode(NULL, sizeof(wire), &out) == VTLK_ERR_NULL_PTR);
    assert(vtlk_header_decode(wire, sizeof(wire), NULL) == VTLK_ERR_NULL_PTR);

    /* Insufficient buffer (< 64 bytes) */
    assert(vtlk_header_encode(&in, wire, 63u) == VTLK_ERR_BUFFER_TOO_SMALL);
    assert(vtlk_header_encode(&in, wire, 0u) == VTLK_ERR_BUFFER_TOO_SMALL);
    assert(vtlk_header_decode(wire, 63u, &out) == VTLK_ERR_BUFFER_TOO_SMALL);
    assert(vtlk_header_decode(wire, 0u, &out) == VTLK_ERR_BUFFER_TOO_SMALL);

    /* Encode parameter validation */
    in.d_model = 0u;
    assert(vtlk_header_encode(&in, wire, sizeof(wire)) == VTLK_ERR_INVALID_PARAM);
    in.d_model = 256u;

    in.n_tokens = 0u;
    assert(vtlk_header_encode(&in, wire, sizeof(wire)) == VTLK_ERR_INVALID_PARAM);
    in.n_tokens = 4u;

    /* Valid encode for decode error tests */
    st = vtlk_header_encode(&in, wire, sizeof(wire));
    assert(st == VTLK_OK);

    /* Invalid magic */
    memcpy(bad, wire, sizeof(wire));
    bad[0] = 'X';
    assert(vtlk_header_decode(bad, sizeof(bad), &out) == VTLK_ERR_INVALID_MAGIC);

    memcpy(bad, wire, sizeof(wire));
    bad[3] = 'Z';
    assert(vtlk_header_decode(bad, sizeof(bad), &out) == VTLK_ERR_INVALID_MAGIC);

    /* Invalid header_bytes != 64 */
    memcpy(bad, wire, sizeof(wire));
    bad[4] = 63u;
    bad[5] = 0u;
    assert(vtlk_header_decode(bad, sizeof(bad), &out) == VTLK_ERR_INVALID_HEADER_SIZE);

    memcpy(bad, wire, sizeof(wire));
    bad[4] = 128u;
    bad[5] = 0u;
    assert(vtlk_header_decode(bad, sizeof(bad), &out) == VTLK_ERR_INVALID_HEADER_SIZE);

    /* Explicit VTLK_QUANT_RAW scheme roundtrip */
    vtlk_header_init_defaults(&in);
    in.d_model = 896u;
    in.n_tokens = 11u;
    in.quant_scheme = (uint8_t)VTLK_QUANT_RAW;
    in.payload_bytes = 135924u;
    in.raw_bytes = 135908u;
    assert(vtlk_header_encode(&in, wire, sizeof(wire)) == VTLK_OK);
    assert(wire[28] == (uint8_t)VTLK_QUANT_RAW);
    assert(vtlk_header_decode(wire, sizeof(wire), &out) == VTLK_OK);
    assert(out.quant_scheme == (uint8_t)VTLK_QUANT_RAW);
}

int main(void)
{
    test_crc32c_vectors();
    test_roundtrip_and_offsets();
    test_bit_flip_detection();
    test_rejections_and_edge_cases();

    printf("All Vect-Link protocol tests passed.\n");
    return 0;
}
