#include "vectlink/codec.h"
#include "protocol/endian.h"

#include <string.h>

uint16_t vtlk_fp32_to_fp16(float val)
{
    uint32_t u;
    uint16_t sign;
    uint32_t exp_bits;
    uint32_t mant;
    int unbiased;

    memcpy(&u, &val, sizeof(u));
    sign = (uint16_t)((u >> 16) & 0x8000u);
    exp_bits = (u >> 23) & 0xFFu;
    mant = u & 0x007FFFFFu;

    /* Inf or NaN */
    if (exp_bits == 255u) {
        if (mant != 0u) {
            uint16_t nan_payload = (uint16_t)(mant >> 13);
            if (nan_payload == 0u) {
                nan_payload = 1u;
            }
            return (uint16_t)(sign | 0x7E00u | nan_payload);
        }
        return (uint16_t)(sign | 0x7C00u);
    }

    unbiased = (int)exp_bits - 127;

    /* Overflow -> +/- Inf */
    if (unbiased > 15) {
        return (uint16_t)(sign | 0x7C00u);
    }

    /* Normal FP16 */
    if (unbiased >= -14) {
        uint32_t h_exp = (uint32_t)(unbiased + 15);
        uint32_t h_mant = mant >> 13;
        uint32_t rem = mant & 0x1FFFu;

        if (rem > 0x1000u || (rem == 0x1000u && (h_mant & 1u) != 0u)) {
            h_mant++;
            if (h_mant == 0x0400u) {
                h_mant = 0u;
                h_exp++;
                if (h_exp >= 31u) {
                    return (uint16_t)(sign | 0x7C00u);
                }
            }
        }
        return (uint16_t)(sign | (uint16_t)(h_exp << 10) | (uint16_t)h_mant);
    }

    /* Subnormal FP16 or underflow to +/- 0 */
    if (unbiased >= -25 && exp_bits != 0u) {
        uint32_t full_mant = 0x00800000u | mant;
        uint32_t shift = (uint32_t)(-1 - unbiased);
        uint32_t h_mant = full_mant >> shift;
        uint32_t halfway = 1u << (shift - 1u);
        uint32_t rem = full_mant & ((1u << shift) - 1u);

        if (rem > halfway || (rem == halfway && (h_mant & 1u) != 0u)) {
            h_mant++;
        }
        return (uint16_t)(sign | (uint16_t)h_mant);
    }

    return sign;
}

float vtlk_fp16_to_fp32(uint16_t val)
{
    uint32_t sign = ((uint32_t)val & 0x8000u) << 16;
    uint32_t exp_bits = ((uint32_t)val >> 10) & 0x1Fu;
    uint32_t mant = (uint32_t)val & 0x03FFu;
    uint32_t u;
    float res;

    if (exp_bits == 0u) {
        if (mant == 0u) {
            u = sign;
        } else {
            uint32_t e = 127u - 14u;
            uint32_t m = mant;
            while ((m & 0x0400u) == 0u) {
                m <<= 1;
                e--;
            }
            m &= 0x03FFu;
            u = sign | (e << 23) | (m << 13);
        }
    } else if (exp_bits == 31u) {
        if (mant == 0u) {
            u = sign | 0x7F800000u;
        } else {
            u = sign | 0x7FC00000u | (mant << 13);
        }
    } else {
        u = sign | ((exp_bits + (127u - 15u)) << 23) | (mant << 13);
    }

    memcpy(&res, &u, sizeof(res));
    return res;
}

vtlk_status_t vtlk_fp16_encode_raw(const float *src,
                                   size_t total_elements,
                                   uint8_t *dst)
{
    size_t i;
    for (i = 0u; i < total_elements; ++i) {
        vtlk_write_u16_le(dst + i * 2u, vtlk_fp32_to_fp16(src[i]));
    }
    return VTLK_OK;
}

vtlk_status_t vtlk_fp16_decode_raw(const uint8_t *src,
                                   size_t total_elements,
                                   float *dst)
{
    size_t i;
    for (i = 0u; i < total_elements; ++i) {
        dst[i] = vtlk_fp16_to_fp32(vtlk_read_u16_le(src + i * 2u));
    }
    return VTLK_OK;
}
