#ifndef VECTLINK_CODEC_H
#define VECTLINK_CODEC_H

#include "vectlink/protocol.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define VTLK_INT4_BLOCK_SIZE  32u
#define VTLK_INT4_BLOCK_BYTES 18u

/**
 * @brief Portable IEEE 754 binary16 conversion: FP32 -> FP16.
 */
uint16_t vtlk_fp32_to_fp16(float val);

/**
 * @brief Portable IEEE 754 binary16 conversion: FP16 -> FP32.
 */
float vtlk_fp16_to_fp32(uint16_t val);

/**
 * @brief Computes the exact encoded byte size for a tensor under @p scheme.
 *
 * Checks for uint32_t -> size_t multiplication overflow and INT4 block
 * alignment (total_elements % 32 == 0).
 *
 * @param scheme    Quantization scheme (VTLK_QUANT_FP16 or VTLK_QUANT_INT4).
 * @param n_tokens  Number of tokens (> 0).
 * @param d_model   Hidden dimension (> 0).
 * @param out_bytes Output pointer for required byte count.
 * @return VTLK_OK on success, or VTLK_ERR_NULL_PTR / VTLK_ERR_INVALID_ARG.
 */
vtlk_status_t vtlk_codec_encoded_bytes(vtlk_quant_scheme_t scheme,
                                       uint32_t n_tokens,
                                       uint32_t d_model,
                                       size_t *out_bytes);

/**
 * @brief Encodes an FP32 tensor [n_tokens][d_model] into a caller-provided buffer.
 *
 * @param src         Input FP32 elements (n_tokens * d_model elements).
 * @param n_tokens    Number of tokens (> 0).
 * @param d_model     Hidden dimension (> 0).
 * @param scheme      Quantization scheme.
 * @param dst         Caller-provided destination buffer.
 * @param dst_cap     Capacity of @p dst in bytes.
 * @param out_written Optional output for bytes written (may be NULL).
 * @return VTLK_OK on success, or a negative vtlk_status_t error code.
 */
vtlk_status_t vtlk_encode_tensor(const float *src,
                                 uint32_t n_tokens,
                                 uint32_t d_model,
                                 vtlk_quant_scheme_t scheme,
                                 uint8_t *dst,
                                 size_t dst_cap,
                                 size_t *out_written);

/**
 * @brief Decodes a quantized tensor back into an FP32 buffer.
 *
 * @param src          Encoded byte buffer.
 * @param src_bytes    Size of @p src in bytes.
 * @param n_tokens     Number of tokens (> 0).
 * @param d_model      Hidden dimension (> 0).
 * @param scheme       Quantization scheme.
 * @param dst          Caller-provided FP32 output buffer.
 * @param dst_elements Capacity of @p dst in float elements.
 * @return VTLK_OK on success, or a negative vtlk_status_t error code.
 */
vtlk_status_t vtlk_decode_tensor(const uint8_t *src,
                                 size_t src_bytes,
                                 uint32_t n_tokens,
                                 uint32_t d_model,
                                 vtlk_quant_scheme_t scheme,
                                 float *dst,
                                 size_t dst_elements);

/**
 * @brief Computes cosine similarity between two FP32 vectors of length @p n.
 *
 * Returns 0.0 if either vector has zero norm, n == 0, or a pointer is NULL.
 */
double vtlk_cosine_similarity(const float *a, const float *b, size_t n);

/**
 * @brief Computes the maximum absolute error max_i |a[i] - b[i]|.
 *
 * Returns 0.0f if n == 0 or a pointer is NULL.
 */
float vtlk_max_abs_error(const float *a, const float *b, size_t n);

#ifdef __cplusplus
}
#endif

#endif /* VECTLINK_CODEC_H */
