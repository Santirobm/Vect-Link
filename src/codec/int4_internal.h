#ifndef VECTLINK_INTERNAL_INT4_H
#define VECTLINK_INTERNAL_INT4_H

#include <stdint.h>

void vtlk_int4_encode_block_scalar(const float src[32], uint8_t dst[18]);
int vtlk_int4_has_avx2_path(void);
void vtlk_int4_encode_block_avx2_dispatch(const float src[32], uint8_t dst[18]);

#endif /* VECTLINK_INTERNAL_INT4_H */
