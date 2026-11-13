#include "vectlink/codec.h"
#include "codec/int4_internal.h"
#include "protocol/endian.h"

#include <float.h>
#include <string.h>

typedef char vtlk_static_assert_f32_size[(sizeof(float) == 4u) ? 1 : -1];
#if FLT_RADIX != 2
#error "IEEE-754 binary floating-point (FLT_RADIX == 2) is required"
#endif

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#define VTLK_ARCH_X86 1
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) || defined(__clang__)
#include <cpuid.h>
#endif
#else
#define VTLK_ARCH_X86 0
#endif

#if defined(_MSC_VER)
#include <intrin.h>
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

int vtlk_cpu_supports_avx2(void)
{
#if !VTLK_ARCH_X86 || defined(VTLK_DISABLE_AVX2)
    return 0;
#elif defined(_MSC_VER)
    int info[4];
    unsigned __int64 xcr0;
    int osxsave_avx;

    __cpuidex(info, 0, 0);
    if (info[0] < 7) {
        return 0;
    }

    __cpuidex(info, 1, 0);
    osxsave_avx = (1 << 27) | (1 << 28);
    if ((info[2] & osxsave_avx) != osxsave_avx) {
        return 0;
    }

    xcr0 = _xgetbv(0);
    if ((xcr0 & 0x6u) != 0x6u) {
        return 0;
    }

    __cpuidex(info, 7, 0);
    if ((info[1] & (1 << 5)) == 0) {
        return 0;
    }

    return 1;
#elif defined(__GNUC__) || defined(__clang__)
    unsigned int eax = 0u;
    unsigned int ebx = 0u;
    unsigned int ecx = 0u;
    unsigned int edx = 0u;
    unsigned int xcr0_lo = 0u;
    unsigned int xcr0_hi = 0u;
    unsigned int osxsave_avx = (1u << 27) | (1u << 28);

    if (__get_cpuid_max(0u, NULL) < 7u) {
        return 0;
    }

    if (!__get_cpuid(1u, &eax, &ebx, &ecx, &edx)) {
        return 0;
    }
    if ((ecx & osxsave_avx) != osxsave_avx) {
        return 0;
    }

    __asm__ __volatile__(".byte 0x0f, 0x01, 0xd0"
                         : "=a"(xcr0_lo), "=d"(xcr0_hi)
                         : "c"(0u));
    (void)xcr0_hi;
    if ((xcr0_lo & 0x6u) != 0x6u) {
        return 0;
    }

    if (!__get_cpuid_count(7u, 0u, &eax, &ebx, &ecx, &edx)) {
        return 0;
    }
    if ((ebx & (1u << 5)) == 0u) {
        return 0;
    }

    return 1;
#else
    return 0;
#endif
}

int vtlk_int4_has_avx2_runtime(void)
{
    return vtlk_cpu_supports_avx2();
}

typedef void (*vtlk_int4_block_fn)(const float src[32], uint8_t dst[18]);

/* 0 = unresolved, 1 = scalar, 2 = avx2 */
#if defined(_MSC_VER)
static long g_avx2_state = 0L;
#else
static int g_avx2_state = 0;
#endif

static int vtlk_int4_resolve_state(void)
{
#if defined(__GNUC__) || defined(__clang__)
    int state = __atomic_load_n(&g_avx2_state, __ATOMIC_ACQUIRE);
    if (state == 0) {
        state = vtlk_cpu_supports_avx2() ? 2 : 1;
        __atomic_store_n(&g_avx2_state, state, __ATOMIC_RELEASE);
    }
    return state;
#elif defined(_MSC_VER)
    long state = _InterlockedCompareExchange(&g_avx2_state, 0L, 0L);
    _ReadWriteBarrier();
    if (state == 0L) {
        state = vtlk_cpu_supports_avx2() ? 2L : 1L;
        _ReadWriteBarrier();
        (void)_InterlockedExchange(&g_avx2_state, state);
    }
    return (int)state;
#else
    if (g_avx2_state == 0) {
        g_avx2_state = vtlk_cpu_supports_avx2() ? 2 : 1;
    }
    return g_avx2_state;
#endif
}

vtlk_status_t vtlk_int4_encode_raw(const float *src,
                                   size_t total_elements,
                                   uint8_t *dst)
{
    size_t num_blocks = total_elements / VTLK_INT4_BLOCK_SIZE;
    vtlk_int4_block_fn fn = vtlk_int4_encode_block_scalar;
    size_t b;

#if VTLK_ARCH_X86 && !defined(VTLK_DISABLE_AVX2)
    if (vtlk_int4_resolve_state() == 2) {
        fn = vtlk_int4_encode_block_avx2;
    }
#else
    (void)vtlk_int4_resolve_state();
#endif

    for (b = 0u; b < num_blocks; ++b) {
        fn(src + b * VTLK_INT4_BLOCK_SIZE, dst + b * VTLK_INT4_BLOCK_BYTES);
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
