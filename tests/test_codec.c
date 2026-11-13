#ifndef _WIN32
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif
#endif

#include "vectlink/vectlink.h"
#include "codec/int4_internal.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <malloc.h>
#endif

static void *test_alloc_aligned64(size_t bytes)
{
#ifdef _WIN32
    return _aligned_malloc(bytes, 64u);
#else
    void *ptr = NULL;
    if (posix_memalign(&ptr, 64u, bytes) != 0) {
        return NULL;
    }
    return ptr;
#endif
}

static void test_free_aligned64(void *ptr)
{
#ifdef _WIN32
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

static uint32_t test_prng_u32(uint32_t *state)
{
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

/* Box-Muller / Irwin-Hall approximation for Gaussian-like values in [-3.0, +3.0] */
static float test_sample_activation(uint32_t *state)
{
    double sum = 0.0;
    int k;
    for (k = 0; k < 6; ++k) {
        sum += (double)(test_prng_u32(state) & 0xFFFFu) / 65535.0;
    }
    /* sum is in [0, 6], mean 3.0, approx std ~0.707 */
    return (float)(sum - 3.0);
}

static void test_fp16_edge_cases(void)
{
    uint16_t h_pos_zero = vtlk_fp32_to_fp16(0.0f);
    uint16_t h_neg_zero = vtlk_fp32_to_fp16(-0.0f);
    uint16_t h_sub;
    uint16_t h_max;
    uint16_t h_ovf_pos;
    uint16_t h_ovf_neg;
    uint16_t h_nan;
    uint32_t f_bits;
    float f_neg_zero;
    float f_sub;
    float f_nan;
    uint32_t nan_u = 0x7FC01234u;

    assert(h_pos_zero == 0x0000u);
    assert(h_neg_zero == 0x8000u);
    f_neg_zero = vtlk_fp16_to_fp32(h_neg_zero);
    memcpy(&f_bits, &f_neg_zero, sizeof(f_bits));
    assert(f_bits == 0x80000000u);

    /* Subnormal 1e-6f */
    h_sub = vtlk_fp32_to_fp16(1e-6f);
    assert(h_sub != 0u && (h_sub & 0x7C00u) == 0u);
    f_sub = vtlk_fp16_to_fp32(h_sub);
    assert(fabsf(f_sub - 1e-6f) < 1e-7f);

    /* Max finite FP16: 65504.0f -> 0x7BFF */
    h_max = vtlk_fp32_to_fp16(65504.0f);
    assert(h_max == 0x7BFFu);
    assert(vtlk_fp16_to_fp32(h_max) == 65504.0f);

    /* Overflow -> +/- Inf */
    h_ovf_pos = vtlk_fp32_to_fp16(100000.0f);
    h_ovf_neg = vtlk_fp32_to_fp16(-100000.0f);
    assert(h_ovf_pos == 0x7C00u);
    assert(h_ovf_neg == 0xFC00u);
    assert(isinf(vtlk_fp16_to_fp32(h_ovf_pos)));
    assert(isinf(vtlk_fp16_to_fp32(h_ovf_neg)));

    /* NaN preservation */
    memcpy(&f_nan, &nan_u, sizeof(f_nan));
    h_nan = vtlk_fp32_to_fp16(f_nan);
    assert((h_nan & 0x7C00u) == 0x7C00u && (h_nan & 0x03FFu) != 0u);
    assert(isnan(vtlk_fp16_to_fp32(h_nan)));
}

static void test_fp16_tensor_roundtrip(void)
{
    enum { N_TOK = 16, D_MOD = 1024, TOTAL = N_TOK * D_MOD };
    float *src = (float *)malloc((size_t)TOTAL * sizeof(float));
    float *dst = (float *)malloc((size_t)TOTAL * sizeof(float));
    uint8_t *enc = (uint8_t *)malloc((size_t)TOTAL * 2u);
    size_t enc_bytes = 0u;
    size_t written = 0u;
    uint32_t rng = 0x12345678u;
    size_t i;
    double cos_sim;
    double ratio;

    assert(src != NULL && dst != NULL && enc != NULL);

    for (i = 0u; i < (size_t)TOTAL; ++i) {
        src[i] = test_sample_activation(&rng);
    }

    assert(vtlk_codec_encoded_bytes(VTLK_QUANT_FP16, N_TOK, D_MOD, &enc_bytes) == VTLK_OK);
    assert(enc_bytes == (size_t)TOTAL * 2u);

    assert(vtlk_encode_tensor(src, N_TOK, D_MOD, VTLK_QUANT_FP16,
                              enc, enc_bytes, &written) == VTLK_OK);
    assert(written == enc_bytes);

    assert(vtlk_decode_tensor(enc, written, N_TOK, D_MOD, VTLK_QUANT_FP16,
                              dst, (size_t)TOTAL) == VTLK_OK);

    cos_sim = vtlk_cosine_similarity(src, dst, (size_t)TOTAL);
    ratio = (double)((size_t)TOTAL * sizeof(float)) / (double)written;
    assert(cos_sim > 0.99999);
    assert(ratio > 1.999 && ratio < 2.001);
    assert(vtlk_max_abs_error(src, dst, (size_t)TOTAL) < 0.005f);

    free(src);
    free(dst);
    free(enc);
}

static void test_int4_tensor_roundtrip(void)
{
    enum { N_TOK = 16, D_MOD = 1024, TOTAL = N_TOK * D_MOD };
    float *src = (float *)malloc((size_t)TOTAL * sizeof(float));
    float *dst = (float *)malloc((size_t)TOTAL * sizeof(float));
    size_t enc_bytes = 0u;
    size_t written = 0u;
    uint8_t *enc;
    uint32_t rng = 0xDEADBEEFu;
    size_t i;
    double cos_sim;
    double ratio;

    assert(src != NULL && dst != NULL);

    for (i = 0u; i < (size_t)TOTAL; ++i) {
        /* Include all-zero blocks every 512 elements */
        if ((i / 32u) % 16u == 0u) {
            src[i] = 0.0f;
        } else {
            src[i] = test_sample_activation(&rng);
        }
    }

    assert(vtlk_codec_encoded_bytes(VTLK_QUANT_INT4, N_TOK, D_MOD, &enc_bytes) == VTLK_OK);
    assert(enc_bytes == ((size_t)TOTAL / 32u) * 18u);

    enc = (uint8_t *)malloc(enc_bytes);
    assert(enc != NULL);

    assert(vtlk_encode_tensor(src, N_TOK, D_MOD, VTLK_QUANT_INT4,
                              enc, enc_bytes, &written) == VTLK_OK);
    assert(written == enc_bytes);

    assert(vtlk_decode_tensor(enc, written, N_TOK, D_MOD, VTLK_QUANT_INT4,
                              dst, (size_t)TOTAL) == VTLK_OK);

    /* Verify all-zero block reconstructed to exact 0.0f */
    for (i = 0u; i < 32u; ++i) {
        assert(dst[i] == 0.0f);
    }

    cos_sim = vtlk_cosine_similarity(src, dst, (size_t)TOTAL);
    ratio = (double)((size_t)TOTAL * sizeof(float)) / (double)written;
    assert(ratio > 7.10 && ratio < 7.12);
    assert(cos_sim > 0.990);

    free(src);
    free(dst);
    free(enc);
}

static void test_codec_errors(void)
{
    float src[64];
    float dst[64];
    uint8_t enc[128];
    size_t out_bytes = 0u;

    memset(src, 0, sizeof(src));

    /* Not divisible by 32 for INT4 */
    assert(vtlk_codec_encoded_bytes(VTLK_QUANT_INT4, 1u, 33u, &out_bytes) == VTLK_ERR_INVALID_ARG);
    assert(vtlk_encode_tensor(src, 1u, 33u, VTLK_QUANT_INT4, enc, sizeof(enc), &out_bytes) == VTLK_ERR_INVALID_ARG);

    /* Zero dims */
    assert(vtlk_codec_encoded_bytes(VTLK_QUANT_FP16, 0u, 32u, &out_bytes) == VTLK_ERR_INVALID_ARG);
    assert(vtlk_codec_encoded_bytes(VTLK_QUANT_FP16, 1u, 0u, &out_bytes) == VTLK_ERR_INVALID_ARG);

    /* Buffer too small */
    assert(vtlk_encode_tensor(src, 2u, 32u, VTLK_QUANT_FP16, enc, 127u, &out_bytes) == VTLK_ERR_BUFFER_TOO_SMALL);
    assert(vtlk_encode_tensor(src, 2u, 32u, VTLK_QUANT_INT4, enc, 35u, &out_bytes) == VTLK_ERR_BUFFER_TOO_SMALL);
    assert(vtlk_decode_tensor(enc, 35u, 2u, 32u, VTLK_QUANT_INT4, dst, 64u) == VTLK_ERR_BUFFER_TOO_SMALL);
    assert(vtlk_decode_tensor(enc, 36u, 2u, 32u, VTLK_QUANT_INT4, dst, 63u) == VTLK_ERR_BUFFER_TOO_SMALL);
}

static void test_end_to_end_int4_over_spsc(void)
{
    enum { N_TOK = 4, D_MOD = 128, TOTAL = N_TOK * D_MOD };
    float src[TOTAL];
    float dst[TOTAL];
    uint8_t enc[(TOTAL / 32) * 18];
    uint8_t recv_buf[(TOTAL / 32) * 18];
    size_t enc_bytes = 0u;
    uint32_t rng = 0xCAFEBABEu;
    size_t i;

    uint8_t *arena;
    size_t ring_bytes;
    vtlk_ring_t prod;
    vtlk_ring_t cons;
    vtlk_header_t h_in;
    vtlk_header_t h_out;
    double cos_sim;

    for (i = 0u; i < (size_t)TOTAL; ++i) {
        src[i] = test_sample_activation(&rng);
    }

    assert(vtlk_encode_tensor(src, N_TOK, D_MOD, VTLK_QUANT_INT4,
                              enc, sizeof(enc), &enc_bytes) == VTLK_OK);

    ring_bytes = vtlk_ring_required_bytes(4u, (uint32_t)enc_bytes, NULL);
    assert(ring_bytes > 0u);
    arena = (uint8_t *)test_alloc_aligned64(ring_bytes);
    assert(arena != NULL);
    assert(vtlk_ring_init(&prod, arena, ring_bytes, 4u, (uint32_t)enc_bytes) == VTLK_OK);
    assert(vtlk_ring_attach(&cons, arena, ring_bytes) == VTLK_OK);

    vtlk_header_init_defaults(&h_in);
    h_in.n_tokens = N_TOK;
    h_in.d_model = D_MOD;
    h_in.quant_scheme = (uint8_t)VTLK_QUANT_INT4;
    h_in.payload_bytes = (uint32_t)enc_bytes;
    h_in.raw_bytes = (uint32_t)((size_t)TOTAL * sizeof(float));
    h_in.sequence = 1u;

    assert(vtlk_ring_push(&prod, &h_in, enc) == VTLK_OK);
    assert(vtlk_ring_pop(&cons, &h_out, recv_buf, sizeof(recv_buf)) == VTLK_OK);

    assert(h_out.n_tokens == N_TOK);
    assert(h_out.d_model == D_MOD);
    assert(h_out.quant_scheme == (uint8_t)VTLK_QUANT_INT4);
    assert(h_out.payload_bytes == (uint32_t)enc_bytes);

    assert(vtlk_decode_tensor(recv_buf, h_out.payload_bytes,
                              h_out.n_tokens, h_out.d_model,
                              (vtlk_quant_scheme_t)h_out.quant_scheme,
                              dst, TOTAL) == VTLK_OK);

    cos_sim = vtlk_cosine_similarity(src, dst, TOTAL);
    assert(cos_sim > 0.990);
    test_free_aligned64(arena);
}

static void test_int4_boundary_consistency(void)
{
    enum { N_TOK = 2, D_MOD = 64, TOTAL = N_TOK * D_MOD };
    float src[TOTAL];
    float dst[TOTAL];
    uint8_t enc[(TOTAL / 32) * 18 + 8];
    size_t written = 0u;
    size_t i;
    double cos_sim;

    /* Guard bytes after encoded output to verify zero out-of-bounds writes */
    memset(enc, 0xA5, sizeof(enc));

    /* Block 0: extreme boundary values (-8*d, +7*d, small negatives, zeros) */
    for (i = 0u; i < 32u; ++i) {
        if (i == 0u) {
            src[i] = -8.0f;
        } else if (i == 1u) {
            src[i] = 7.0f;
        } else if (i % 4u == 0u) {
            src[i] = 0.0f;
        } else if (i % 4u == 1u) {
            src[i] = -0.05f;
        } else {
            src[i] = (float)((int)(i % 15u) - 7);
        }
    }

    /* Block 1: small negative and positive values */
    for (i = 32u; i < 64u; ++i) {
        src[i] = ((i % 2u) == 0u) ? -0.125f : 0.25f;
    }

    /* Block 2 & 3: mixed scale ramp */
    for (i = 64u; i < (size_t)TOTAL; ++i) {
        src[i] = (float)((int)(i % 16u) - 8) * 0.35f;
    }

    assert(vtlk_encode_tensor(src, N_TOK, D_MOD, VTLK_QUANT_INT4,
                              enc, (TOTAL / 32) * 18u, &written) == VTLK_OK);
    assert(written == (TOTAL / 32) * 18u);

    for (i = written; i < sizeof(enc); ++i) {
        assert(enc[i] == 0xA5u);
    }

    assert(vtlk_decode_tensor(enc, written, N_TOK, D_MOD, VTLK_QUANT_INT4,
                              dst, TOTAL) == VTLK_OK);

    cos_sim = vtlk_cosine_similarity(src, dst, TOTAL);
    assert(cos_sim > 0.995);
}

static void test_int4_scalar_avx2_parity(void)
{
    uint32_t rng = 0x51525354u;
    size_t blk;

    if (!vtlk_int4_has_avx2_runtime()) {
        printf("[INFO] Runtime CPUID+XGETBV: AVX2 unavailable; scalar path active.\n");
        return;
    }

    for (blk = 0u; blk < 1000u; ++blk) {
        float block[32];
        uint8_t out_scalar[18];
        uint8_t out_avx2[18];
        uint8_t out_api[18];
        size_t written = 0u;
        size_t i;

        for (i = 0u; i < 32u; ++i) {
            if (blk == 0u) {
                block[i] = 0.0f;
            } else if (blk == 1u) {
                /* Exact x.5 ties and extremes */
                static const float ties[8] = {
                    -8.0f, -7.5f, -2.5f, -1.5f, 0.5f, 1.5f, 2.5f, 7.0f
                };
                block[i] = ties[i % 8u];
            } else {
                block[i] = test_sample_activation(&rng);
            }
        }

        vtlk_int4_encode_block_scalar(block, out_scalar);
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
        vtlk_int4_encode_block_avx2(block, out_avx2);
#else
        vtlk_int4_encode_block_scalar(block, out_avx2);
#endif
        assert(memcmp(out_scalar, out_avx2, sizeof(out_scalar)) == 0);

        assert(vtlk_encode_tensor(block, 1u, 32u, VTLK_QUANT_INT4,
                                  out_api, sizeof(out_api), &written) == VTLK_OK);
        assert(written == sizeof(out_api));
        assert(memcmp(out_api, out_avx2, sizeof(out_api)) == 0);
    }

    printf("[INFO] AVX2 bit-exact parity verified across 1000 blocks.\n");
}

static void test_int4_nan_inf_underflow_robustness(void)
{
    float block[32];
    float dec[32];
    uint8_t out_scalar[18];
    uint8_t out_avx2[18];
    uint8_t out_api[18];
    size_t i;
    size_t written = 0u;
    uint32_t nan_bits = 0x7FC00001u;
    uint32_t inf_bits = 0x7F800000u;
    uint32_t ninf_bits = 0xFF800000u;
    float f_nan;
    float f_inf;
    float f_ninf;
    uint16_t scale_fp16;

    memcpy(&f_nan, &nan_bits, sizeof(f_nan));
    memcpy(&f_inf, &inf_bits, sizeof(f_inf));
    memcpy(&f_ninf, &ninf_bits, sizeof(f_ninf));

    /* 1. Subnormals & scale underflow to 0 in FP16 (1e-12f, 1e-38f) */
    for (i = 0u; i < 32u; ++i) {
        block[i] = ((i % 2u) == 0u) ? 1e-12f : -1e-38f;
    }
    assert(vtlk_encode_tensor(block, 1u, 32u, VTLK_QUANT_INT4,
                              out_scalar, sizeof(out_scalar), &written) == VTLK_OK);
    assert(written == 18u);
    assert(out_scalar[0] == 0u && out_scalar[1] == 0u);
    for (i = 2u; i < 18u; ++i) {
        assert(out_scalar[i] == 0x88u);
    }

    /* 2. Mixed +Inf, -Inf, NaN with finite values (3.5f, -1.75f) */
    for (i = 0u; i < 32u; ++i) {
        block[i] = 0.0f;
    }
    block[0] = f_inf;
    block[1] = f_ninf;
    block[2] = f_nan;
    block[3] = 3.5f;
    block[4] = -1.75f;

    assert(vtlk_encode_tensor(block, 1u, 32u, VTLK_QUANT_INT4,
                              out_api, sizeof(out_api), &written) == VTLK_OK);
    assert(written == 18u);

    /* Scale in bytes 0..1 must be finite and equal 3.5f / 7.0f = 0.5f */
    scale_fp16 = (uint16_t)((uint16_t)out_api[0] | ((uint16_t)out_api[1] << 8));
    assert(vtlk_fp16_to_fp32(scale_fp16) == 0.5f);

    assert(vtlk_decode_tensor(out_api, written, 1u, 32u, VTLK_QUANT_INT4,
                              dec, 32u) == VTLK_OK);

    /* +Inf, -Inf, and NaN slots must decode deterministically to 0.0f */
    assert(dec[0] == 0.0f);
    assert(dec[1] == 0.0f);
    assert(dec[2] == 0.0f);
    assert(dec[3] == 3.5f);
    assert(isfinite(dec[4]));

    /* Scalar, AVX2, and public API must produce identical 18 bytes */
    vtlk_int4_encode_block_scalar(block, out_scalar);
    assert(memcmp(out_api, out_scalar, sizeof(out_api)) == 0);
    if (vtlk_int4_has_avx2_runtime()) {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
        vtlk_int4_encode_block_avx2(block, out_avx2);
        assert(memcmp(out_scalar, out_avx2, sizeof(out_scalar)) == 0);
#endif
    }
}

int main(void)
{
    test_fp16_edge_cases();
    test_fp16_tensor_roundtrip();
    test_int4_tensor_roundtrip();
    test_codec_errors();
    test_end_to_end_int4_over_spsc();
    test_int4_boundary_consistency();
    test_int4_scalar_avx2_parity();
    test_int4_nan_inf_underflow_robustness();

    printf("All Vect-Link codec tests passed.\n");
    return 0;
}
