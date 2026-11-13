#include "vectlink/codec.h"
#include "protocol/endian.h"

#if defined(__AVX2__)
#include <immintrin.h>
static void vtlk_int4_encode_block_avx2(const float *block_src, uint8_t *block_dst)
{
    __m256 v0 = _mm256_loadu_ps(block_src + 0u);
    __m256 v1 = _mm256_loadu_ps(block_src + 8u);
    __m256 v2 = _mm256_loadu_ps(block_src + 16u);
    __m256 v3 = _mm256_loadu_ps(block_src + 24u);

    __m256 sign_mask = _mm256_set1_ps(-0.0f);
    __m256 m01 = _mm256_max_ps(_mm256_andnot_ps(sign_mask, v0),
                               _mm256_andnot_ps(sign_mask, v1));
    __m256 m23 = _mm256_max_ps(_mm256_andnot_ps(sign_mask, v2),
                               _mm256_andnot_ps(sign_mask, v3));
    __m256 m = _mm256_max_ps(m01, m23);

    __m128 hi = _mm256_extractf128_ps(m, 1);
    __m128 lo = _mm256_castps256_ps128(m);
    __m128 m4 = _mm_max_ps(lo, hi);
    __m128 m2 = _mm_max_ps(m4, _mm_movehl_ps(m4, m4));
    __m128 m1 = _mm_max_ss(m2, _mm_shuffle_ps(m2, m2, 1));
    float max_abs = _mm_cvtss_f32(m1);

    float d = (max_abs > 0.0f) ? (max_abs / 7.0f) : 0.0f;
    uint16_t scale_fp16 = vtlk_fp32_to_fp16(d);
    float d_quant = vtlk_fp16_to_fp32(scale_fp16);
    float inv_d = (d_quant > 0.0f) ? (1.0f / d_quant) : 0.0f;

    __m256 v_inv_d = _mm256_set1_ps(inv_d);
    __m256i v_min = _mm256_set1_epi32(-8);
    __m256i v_max = _mm256_set1_epi32(7);
    __m256i v_off = _mm256_set1_epi32(8);

    __m256i q0 = _mm256_cvtps_epi32(_mm256_round_ps(
        _mm256_mul_ps(v0, v_inv_d), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    __m256i q1 = _mm256_cvtps_epi32(_mm256_round_ps(
        _mm256_mul_ps(v1, v_inv_d), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    __m256i q2 = _mm256_cvtps_epi32(_mm256_round_ps(
        _mm256_mul_ps(v2, v_inv_d), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    __m256i q3 = _mm256_cvtps_epi32(_mm256_round_ps(
        _mm256_mul_ps(v3, v_inv_d), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));

    __m256i p01;
    __m256i p23;
    __m256i bytes32;
    __m256i pairs16;
    __m256i packed;

    q0 = _mm256_add_epi32(_mm256_max_epi32(v_min, _mm256_min_epi32(v_max, q0)), v_off);
    q1 = _mm256_add_epi32(_mm256_max_epi32(v_min, _mm256_min_epi32(v_max, q1)), v_off);
    q2 = _mm256_add_epi32(_mm256_max_epi32(v_min, _mm256_min_epi32(v_max, q2)), v_off);
    q3 = _mm256_add_epi32(_mm256_max_epi32(v_min, _mm256_min_epi32(v_max, q3)), v_off);

    p01 = _mm256_permute4x64_epi64(_mm256_packs_epi32(q0, q1), 0xD8);
    p23 = _mm256_permute4x64_epi64(_mm256_packs_epi32(q2, q3), 0xD8);
    bytes32 = _mm256_permute4x64_epi64(_mm256_packus_epi16(p01, p23), 0xD8);

    pairs16 = _mm256_maddubs_epi16(bytes32, _mm256_set1_epi16(0x1001));
    packed = _mm256_permute4x64_epi64(
        _mm256_packus_epi16(pairs16, _mm256_setzero_si256()), 0xD8);

    vtlk_write_u16_le(block_dst, scale_fp16);
    _mm_storeu_si128((__m128i *)(void *)(block_dst + 2u),
                     _mm256_castsi256_si128(packed));
}
#else
static inline uint8_t vtlk_quantize_nibble_fast(float x, float inv_d)
{
    float scaled = x * inv_d;
    int qi = (int)(scaled + (scaled >= 0.0f ? 0.5f : -0.5f));

    qi = (qi < -8) ? -8 : qi;
    qi = (qi > 7) ? 7 : qi;
    return (uint8_t)(qi + 8);
}

static void vtlk_int4_encode_block_scalar(const float *block_src, uint8_t *block_dst)
{
    float max_abs = 0.0f;
    float d;
    float d_quant;
    float inv_d;
    uint16_t scale_fp16;
    size_t i;
    size_t j;

    for (i = 0u; i < VTLK_INT4_BLOCK_SIZE; ++i) {
        float x = block_src[i];
        float ax = (x < 0.0f) ? -x : x;
        if (ax > max_abs) {
            max_abs = ax;
        }
    }

    d = (max_abs > 0.0f) ? (max_abs / 7.0f) : 0.0f;
    scale_fp16 = vtlk_fp32_to_fp16(d);
    d_quant = vtlk_fp16_to_fp32(scale_fp16);
    inv_d = (d_quant > 0.0f) ? (1.0f / d_quant) : 0.0f;

    vtlk_write_u16_le(block_dst, scale_fp16);

    for (j = 0u; j < (VTLK_INT4_BLOCK_SIZE / 2u); ++j) {
        uint8_t q0 = vtlk_quantize_nibble_fast(block_src[2u * j], inv_d);
        uint8_t q1 = vtlk_quantize_nibble_fast(block_src[2u * j + 1u], inv_d);
        block_dst[2u + j] = (uint8_t)((q0 & 0x0Fu) | ((q1 & 0x0Fu) << 4));
    }
}
#endif

vtlk_status_t vtlk_int4_encode_raw(const float *src,
                                   size_t total_elements,
                                   uint8_t *dst)
{
    size_t num_blocks = total_elements / VTLK_INT4_BLOCK_SIZE;
    size_t b;

    for (b = 0u; b < num_blocks; ++b) {
        const float *block_src = src + b * VTLK_INT4_BLOCK_SIZE;
        uint8_t *block_dst = dst + b * VTLK_INT4_BLOCK_BYTES;
#if defined(__AVX2__)
        vtlk_int4_encode_block_avx2(block_src, block_dst);
#else
        vtlk_int4_encode_block_scalar(block_src, block_dst);
#endif
    }

    return VTLK_OK;
}

vtlk_status_t vtlk_int4_decode_raw(const uint8_t *src,
                                   size_t total_elements,
                                   float *dst)
{
    size_t num_blocks = total_elements / VTLK_INT4_BLOCK_SIZE;
    size_t b;

    for (b = 0u; b < num_blocks; ++b) {
        const uint8_t *block_src = src + b * VTLK_INT4_BLOCK_BYTES;
        float *block_dst = dst + b * VTLK_INT4_BLOCK_SIZE;
        uint16_t scale_fp16 = vtlk_read_u16_le(block_src);
        float d = vtlk_fp16_to_fp32(scale_fp16);
        size_t j;

        for (j = 0u; j < (VTLK_INT4_BLOCK_SIZE / 2u); ++j) {
            uint8_t packed = block_src[2u + j];
            int q0 = (int)(packed & 0x0Fu) - 8;
            int q1 = (int)((packed >> 4) & 0x0Fu) - 8;
            block_dst[2u * j] = (float)q0 * d;
            block_dst[2u * j + 1u] = (float)q1 * d;
        }
    }

    return VTLK_OK;
}
