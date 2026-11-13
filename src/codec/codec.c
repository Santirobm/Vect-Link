#include "vectlink/codec.h"

#include <math.h>

vtlk_status_t vtlk_fp16_encode_raw(const float *src,
                                   size_t total_elements,
                                   uint8_t *dst);
vtlk_status_t vtlk_fp16_decode_raw(const uint8_t *src,
                                   size_t total_elements,
                                   float *dst);
vtlk_status_t vtlk_int4_encode_raw(const float *src,
                                   size_t total_elements,
                                   uint8_t *dst);
vtlk_status_t vtlk_int4_decode_raw(const uint8_t *src,
                                   size_t total_elements,
                                   float *dst);

static vtlk_status_t vtlk_compute_total_elements(uint32_t n_tokens,
                                                 uint32_t d_model,
                                                 size_t *out_total)
{
    if (n_tokens == 0u || d_model == 0u) {
        return VTLK_ERR_INVALID_ARG;
    }
    if ((size_t)n_tokens > SIZE_MAX / (size_t)d_model) {
        return VTLK_ERR_INVALID_ARG;
    }
    *out_total = (size_t)n_tokens * (size_t)d_model;
    return VTLK_OK;
}

vtlk_status_t vtlk_codec_encoded_bytes(vtlk_quant_scheme_t scheme,
                                       uint32_t n_tokens,
                                       uint32_t d_model,
                                       size_t *out_bytes)
{
    size_t total_elements;
    vtlk_status_t st;

    if (out_bytes == NULL) {
        return VTLK_ERR_NULL_PTR;
    }

    st = vtlk_compute_total_elements(n_tokens, d_model, &total_elements);
    if (st != VTLK_OK) {
        return st;
    }

    if (scheme == VTLK_QUANT_FP16) {
        if (total_elements > SIZE_MAX / 2u) {
            return VTLK_ERR_INVALID_ARG;
        }
        *out_bytes = total_elements * 2u;
        return VTLK_OK;
    }

    if (scheme == VTLK_QUANT_INT4_BLOCK) {
        size_t num_blocks;
        if ((total_elements % VTLK_INT4_BLOCK_SIZE) != 0u) {
            return VTLK_ERR_INVALID_ARG;
        }
        num_blocks = total_elements / VTLK_INT4_BLOCK_SIZE;
        if (num_blocks > SIZE_MAX / VTLK_INT4_BLOCK_BYTES) {
            return VTLK_ERR_INVALID_ARG;
        }
        *out_bytes = num_blocks * VTLK_INT4_BLOCK_BYTES;
        return VTLK_OK;
    }

    return VTLK_ERR_INVALID_ARG;
}

vtlk_status_t vtlk_encode_tensor(const float *src,
                                 uint32_t n_tokens,
                                 uint32_t d_model,
                                 vtlk_quant_scheme_t scheme,
                                 uint8_t *dst,
                                 size_t dst_cap,
                                 size_t *out_written)
{
    size_t total_elements;
    size_t required_bytes;
    vtlk_status_t st;

    if (src == NULL || dst == NULL) {
        return VTLK_ERR_NULL_PTR;
    }

    st = vtlk_compute_total_elements(n_tokens, d_model, &total_elements);
    if (st != VTLK_OK) {
        return st;
    }

    st = vtlk_codec_encoded_bytes(scheme, n_tokens, d_model, &required_bytes);
    if (st != VTLK_OK) {
        return st;
    }

    if (dst_cap < required_bytes) {
        return VTLK_ERR_BUFFER_TOO_SMALL;
    }

    if (scheme == VTLK_QUANT_FP16) {
        st = vtlk_fp16_encode_raw(src, total_elements, dst);
    } else if (scheme == VTLK_QUANT_INT4_BLOCK) {
        st = vtlk_int4_encode_raw(src, total_elements, dst);
    } else {
        return VTLK_ERR_INVALID_ARG;
    }

    if (st == VTLK_OK && out_written != NULL) {
        *out_written = required_bytes;
    }
    return st;
}

vtlk_status_t vtlk_decode_tensor(const uint8_t *src,
                                 size_t src_bytes,
                                 uint32_t n_tokens,
                                 uint32_t d_model,
                                 vtlk_quant_scheme_t scheme,
                                 float *dst,
                                 size_t dst_elements)
{
    size_t total_elements;
    size_t required_bytes;
    vtlk_status_t st;

    if (src == NULL || dst == NULL) {
        return VTLK_ERR_NULL_PTR;
    }

    st = vtlk_compute_total_elements(n_tokens, d_model, &total_elements);
    if (st != VTLK_OK) {
        return st;
    }

    st = vtlk_codec_encoded_bytes(scheme, n_tokens, d_model, &required_bytes);
    if (st != VTLK_OK) {
        return st;
    }

    if (src_bytes < required_bytes || dst_elements < total_elements) {
        return VTLK_ERR_BUFFER_TOO_SMALL;
    }

    if (scheme == VTLK_QUANT_FP16) {
        return vtlk_fp16_decode_raw(src, total_elements, dst);
    }
    if (scheme == VTLK_QUANT_INT4_BLOCK) {
        return vtlk_int4_decode_raw(src, total_elements, dst);
    }
    return VTLK_ERR_INVALID_ARG;
}

double vtlk_cosine_similarity(const float *a, const float *b, size_t n)
{
    double dot = 0.0;
    double norm_a = 0.0;
    double norm_b = 0.0;
    size_t i;

    if (a == NULL || b == NULL || n == 0u) {
        return 0.0;
    }

    for (i = 0u; i < n; ++i) {
        double va = (double)a[i];
        double vb = (double)b[i];
        dot += va * vb;
        norm_a += va * va;
        norm_b += vb * vb;
    }

    if (norm_a <= 0.0 || norm_b <= 0.0) {
        return 0.0;
    }

    return dot / (sqrt(norm_a) * sqrt(norm_b));
}

float vtlk_max_abs_error(const float *a, const float *b, size_t n)
{
    float max_err = 0.0f;
    size_t i;

    if (a == NULL || b == NULL || n == 0u) {
        return 0.0f;
    }

    for (i = 0u; i < n; ++i) {
        float err = fabsf(a[i] - b[i]);
        if (err > max_err) {
            max_err = err;
        }
    }

    return max_err;
}
