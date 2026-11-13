#ifndef VECTLINK_INTERNAL_INT4_H
#define VECTLINK_INTERNAL_INT4_H

#include <stdint.h>

void vtlk_int4_encode_block_scalar(const float src[32], uint8_t dst[18]);
int vtlk_cpu_supports_avx2(void);
int vtlk_int4_has_avx2_runtime(void);

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
void vtlk_int4_encode_block_avx2(const float src[32], uint8_t dst[18]);
#endif

#endif /* VECTLINK_INTERNAL_INT4_H */
