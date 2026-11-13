#include "vectlink/codec.h"
#include "codec/int4_internal.h"
#include "protocol/endian.h"

#include <float.h>
#include <string.h>

typedef char vtlk_static_assert_f32_size[(sizeof(float) == 4u) ? 1 : -1];
#if FLT_RADIX != 2
#error "IEEE-754 binary floating-point (FLT_RADIX == 2) is required"
#endif

static inline float vtlk_sanitize_finite_f32(float x)
{
    uint32_t u;
    memcpy(&u, &x, sizeof(u));
    if ((u & 0x7F800000u) == 0x7F800000u) {
        return 0.0f;
    }
    return x;
}

#if defined(__AVX2__)
#include <immintrin.h>

static inline __m256 vtlk_sanitize_finite_avx2(__m256 v)
{
    __m256i exp_mask = _mm256_set1_epi32(0x7F800000);
    __m256i is_nonfinite = _mm256_cmpeq_epi32(
        _mm256_and_si256(_mm256_castps_si256(v), exp_mask), exp_mask);
    return _mm256_andnot_ps(_mm256_castsi256_ps(is_nonfinite), v);
}

static void vtlk_int4_encode_block_avx2(const float block_src[32],
                                        uint8_t block_dst[18])
{
    __m256 v0 = vtlk_sanitize_finite_avx2(_mm256_loadu_ps(block_src + 0u));
    __m256 v1 = vtlk_sanitize_finite_avx2(_mm256_loadu_ps(block_src + 8u));
    __m256 v2 = vtlk_sanitize_finite_avx2(_mm256_loadu_ps(block_src + 16u));
    __m256 v3 = vtlk_sanitize_finite_avx2(_mm256_loadu_ps(block_src + 24u));

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

    float d_raw = (max_abs > 0.0f) ? (max_abs / 7.0f) : 0.0f;
    uint16_t scale_fp16 = vtlk_fp32_to_fp16(d_raw);
    float d_eff = vtlk_fp16_to_fp32(scale_fp16);

    if (d_eff <= 0.0f) {
        vtlk_write_u16_le(block_dst, 0u);
        memset(block_dst + 2u, 0x88, VTLK_INT4_BLOCK_SIZE / 2u);
        return;
    }

    {
        float inv_d = 1.0f / d_eff;
        __m256 v_inv_d = _mm256_set1_ps(inv_d);
        __m256 v_fmin = _mm256_set1_ps(-8.0f);
        __m256 v_fmax = _mm256_set1_ps(7.0f);
        __m256i v_off = _mm256_set1_epi32(8);

        __m256 s0 = _mm256_min_ps(v_fmax, _mm256_max_ps(v_fmin, _mm256_mul_ps(v0, v_inv_d)));
        __m256 s1 = _mm256_min_ps(v_fmax, _mm256_max_ps(v_fmin, _mm256_mul_ps(v1, v_inv_d)));
        __m256 s2 = _mm256_min_ps(v_fmax, _mm256_max_ps(v_fmin, _mm256_mul_ps(v2, v_inv_d)));
        __m256 s3 = _mm256_min_ps(v_fmax, _mm256_max_ps(v_fmin, _mm256_mul_ps(v3, v_inv_d)));

        __m256i q0 = _mm256_add_epi32(
            _mm256_cvtps_epi32(_mm256_round_ps(s0, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)),
            v_off);
        __m256i q1 = _mm256_add_epi32(
            _mm256_cvtps_epi32(_mm256_round_ps(s1, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)),
            v_off);
        __m256i q2 = _mm256_add_epi32(
            _mm256_cvtps_epi32(_mm256_round_ps(s2, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)),
            v_off);
        __m256i q3 = _mm256_add_epi32(
            _mm256_cvtps_epi32(_mm256_round_ps(s3, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)),
            v_off);

        __m256i p01 = _mm256_permute4x64_epi64(_mm256_packs_epi32(q0, q1), 0xD8);
        __m256i p23 = _mm256_permute4x64_epi64(_mm256_packs_epi32(q2, q3), 0xD8);
        __m256i bytes32 = _mm256_permute4x64_epi64(_mm256_packus_epi16(p01, p23), 0xD8);
        __m256i pairs16 = _mm256_maddubs_epi16(bytes32, _mm256_set1_epi16(0x1001));
        __m256i packed = _mm256_permute4x64_epi64(
            _mm256_packus_epi16(pairs16, _mm256_setzero_si256()), 0xD8);

        vtlk_write_u16_le(block_dst, scale_fp16);
        _mm_storeu_si128((__m128i *)(void *)(block_dst + 2u),
                         _mm256_castsi256_si128(packed));
    }
}
#endif /* __AVX2__ */

static inline uint8_t vtlk_quantize_nibble_ieee(float x, float inv_d)
{
    const float magic = 12582912.0f; /* 1.5 * 2^23: forces IEEE-754 round-to-nearest-even */
    float scaled;
    float rounded;
    int qi;

    x = vtlk_sanitize_finite_f32(x);
    scaled = x * inv_d;
    if (scaled > 7.0f) {
        scaled = 7.0f;
    }
    if (scaled < -8.0f) {
        scaled = -8.0f;
    }

    rounded = (scaled + magic) - magic;
    qi = (int)rounded;
    return (uint8_t)(qi + 8);
}

void vtlk_int4_encode_block_scalar(const float src[32], uint8_t dst[18])
{
    float max_abs = 0.0f;
    float d_raw;
    float d_eff;
    float inv_d;
    uint16_t scale_fp16;
    size_t i;
    size_t j;

    for (i = 0u; i < VTLK_INT4_BLOCK_SIZE; ++i) {
        float x = vtlk_sanitize_finite_f32(src[i]);
        float ax = (x < 0.0f) ? -x : x;
        if (ax > max_abs) {
            max_abs = ax;
        }
    }

    d_raw = (max_abs > 0.0f) ? (max_abs / 7.0f) : 0.0f;
    scale_fp16 = vtlk_fp32_to_fp16(d_raw);
    d_eff = vtlk_fp16_to_fp32(scale_fp16);

    if (d_eff <= 0.0f) {
        vtlk_write_u16_le(dst, 0u);
        memset(dst + 2u, 0x88, VTLK_INT4_BLOCK_SIZE / 2u);
        return;
    }

    inv_d = 1.0f / d_eff;
    vtlk_write_u16_le(dst, scale_fp16);

    for (j = 0u; j < (VTLK_INT4_BLOCK_SIZE / 2u); ++j) {
        uint8_t q0 = vtlk_quantize_nibble_ieee(src[2u * j], inv_d);
        uint8_t q1 = vtlk_quantize_nibble_ieee(src[2u * j + 1u], inv_d);
        dst[2u + j] = (uint8_t)((q0 & 0x0Fu) | ((q1 & 0x0Fu) << 4));
    }
}

int vtlk_int4_has_avx2_path(void)
{
#if defined(__AVX2__)
    return 1;
#else
    return 0;
#endif
}

void vtlk_int4_encode_block_avx2_dispatch(const float src[32], uint8_t dst[18])
{
#if defined(__AVX2__)
    vtlk_int4_encode_block_avx2(src, dst);
#else
    vtlk_int4_encode_block_scalar(src, dst);
#endif
}

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
