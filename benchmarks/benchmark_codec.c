#ifndef _WIN32
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif
#endif

#include "vectlink/vectlink.h"
#include "bench_timer.h"

#include <stdio.h>
#include <stdlib.h>

#define BENCH_N_TOKENS 32u
#define BENCH_D_MODEL  4096u
#define BENCH_WARMUP   500u
#define BENCH_ITERS    2000u

static uint32_t bench_prng(uint32_t *state)
{
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

static void populate_activations(float *dst, size_t count)
{
    uint32_t rng = 0x9E3779B9u;
    size_t i;

    for (i = 0u; i < count; ++i) {
        double sum = 0.0;
        int k;
        float val;
        for (k = 0; k < 6; ++k) {
            sum += (double)(bench_prng(&rng) & 0xFFFFu) / 65535.0;
        }
        val = (float)(sum - 3.0);
        if ((i % 997u) == 0u) {
            val *= 1.15f;
        }
        dst[i] = val;
    }
}

static void run_codec_bench(const char *name,
                            vtlk_quant_scheme_t scheme,
                            const float *src,
                            float *dst,
                            uint8_t *enc,
                            size_t enc_cap,
                            uint64_t *enc_samples,
                            uint64_t *dec_samples,
                            uint64_t *tot_samples)
{
    size_t total_elems = (size_t)BENCH_N_TOKENS * (size_t)BENCH_D_MODEL;
    size_t raw_bytes = total_elems * sizeof(float);
    size_t written = 0u;
    size_t i;
    vtlk_bench_stats_t s_enc;
    vtlk_bench_stats_t s_dec;
    vtlk_bench_stats_t s_tot;
    double gbps_enc;
    double gbps_dec;
    double cos_sim;
    float max_err;

    for (i = 0u; i < BENCH_WARMUP; ++i) {
        vtlk_encode_tensor(src, BENCH_N_TOKENS, BENCH_D_MODEL, scheme, enc, enc_cap, &written);
        vtlk_decode_tensor(enc, written, BENCH_N_TOKENS, BENCH_D_MODEL, scheme, dst, total_elems);
    }

    for (i = 0u; i < BENCH_ITERS; ++i) {
        uint64_t t0 = vtlk_bench_now_ns();
        vtlk_encode_tensor(src, BENCH_N_TOKENS, BENCH_D_MODEL, scheme, enc, enc_cap, &written);
        uint64_t t1 = vtlk_bench_now_ns();
        vtlk_decode_tensor(enc, written, BENCH_N_TOKENS, BENCH_D_MODEL, scheme, dst, total_elems);
        uint64_t t2 = vtlk_bench_now_ns();

        enc_samples[i] = t1 - t0;
        dec_samples[i] = t2 - t1;
        tot_samples[i] = t2 - t0;
    }

    s_enc = vtlk_bench_compute_stats(enc_samples, BENCH_ITERS);
    s_dec = vtlk_bench_compute_stats(dec_samples, BENCH_ITERS);
    s_tot = vtlk_bench_compute_stats(tot_samples, BENCH_ITERS);

    gbps_enc = (double)raw_bytes / s_enc.mean_ns;
    gbps_dec = (double)raw_bytes / s_dec.mean_ns;
    cos_sim = vtlk_cosine_similarity(src, dst, total_elems);
    max_err = vtlk_max_abs_error(src, dst, total_elems);

    printf("%-10s | %7zu B (%.2fx) | Enc p50: %6.2f us (%.2f GB/s) | "
           "Dec p50: %6.2f us (%.2f GB/s) | Tot p50/p99: %6.2f / %6.2f us | "
           "Cos: %.6f | MaxErr: %.4f\n",
           name,
           written,
           (double)raw_bytes / (double)written,
           (double)s_enc.p50_ns / 1000.0,
           gbps_enc,
           (double)s_dec.p50_ns / 1000.0,
           gbps_dec,
           (double)s_tot.p50_ns / 1000.0,
           (double)s_tot.p99_ns / 1000.0,
           cos_sim,
           max_err);
}

int main(void)
{
    size_t total_elems = (size_t)BENCH_N_TOKENS * (size_t)BENCH_D_MODEL;
    size_t fp16_bytes = 0u;
    size_t int4_bytes = 0u;
    float *src = (float *)malloc(total_elems * sizeof(float));
    float *dst = (float *)malloc(total_elems * sizeof(float));
    uint8_t *enc;
    uint64_t *enc_s = (uint64_t *)malloc(BENCH_ITERS * sizeof(uint64_t));
    uint64_t *dec_s = (uint64_t *)malloc(BENCH_ITERS * sizeof(uint64_t));
    uint64_t *tot_s = (uint64_t *)malloc(BENCH_ITERS * sizeof(uint64_t));

    vtlk_codec_encoded_bytes(VTLK_QUANT_FP16, BENCH_N_TOKENS, BENCH_D_MODEL, &fp16_bytes);
    vtlk_codec_encoded_bytes(VTLK_QUANT_INT4, BENCH_N_TOKENS, BENCH_D_MODEL, &int4_bytes);
    enc = (uint8_t *)malloc(fp16_bytes > int4_bytes ? fp16_bytes : int4_bytes);

    if (src == NULL || dst == NULL || enc == NULL ||
        enc_s == NULL || dec_s == NULL || tot_s == NULL) {
        fprintf(stderr, "benchmark_codec: allocation failed\n");
        free(src);
        free(dst);
        free(enc);
        free(enc_s);
        free(dec_s);
        free(tot_s);
        return 1;
    }

    populate_activations(src, total_elems);

    printf("Vect-Link Codec Benchmark | Tensor: %u x %u (%zu floats, %zu bytes FP32) | "
           "Warmup: %u, Iters: %u\n",
           BENCH_N_TOKENS, BENCH_D_MODEL, total_elems, total_elems * sizeof(float),
           BENCH_WARMUP, BENCH_ITERS);
    printf("------------------------------------------------------------------------"
           "---------------------------------------------------------\n");

    run_codec_bench("FP16", VTLK_QUANT_FP16, src, dst, enc, fp16_bytes, enc_s, dec_s, tot_s);
    run_codec_bench("INT4_BLOCK", VTLK_QUANT_INT4, src, dst, enc, int4_bytes, enc_s, dec_s, tot_s);

    free(src);
    free(dst);
    free(enc);
    free(enc_s);
    free(dec_s);
    free(tot_s);
    return 0;
}
