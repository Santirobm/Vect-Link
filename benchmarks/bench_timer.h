#ifndef VECTLINK_BENCH_TIMER_H
#define VECTLINK_BENCH_TIMER_H

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <time.h>
#endif

typedef struct vtlk_bench_stats {
    uint64_t min_ns;
    uint64_t p50_ns;
    uint64_t p95_ns;
    uint64_t p99_ns;
    double mean_ns;
} vtlk_bench_stats_t;

static inline uint64_t vtlk_bench_now_ns(void)
{
#ifdef _WIN32
    static LARGE_INTEGER freq = { { 0, 0 } };
    LARGE_INTEGER counter;
    uint64_t q;
    uint64_t f;

    if (freq.QuadPart == 0) {
        QueryPerformanceFrequency(&freq);
    }
    QueryPerformanceCounter(&counter);
    q = (uint64_t)counter.QuadPart;
    f = (uint64_t)freq.QuadPart;
    return (q / f) * 1000000000ULL + ((q % f) * 1000000000ULL) / f;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
#endif
}

static int vtlk_bench_cmp_u64(const void *a, const void *b)
{
    uint64_t va = *(const uint64_t *)a;
    uint64_t vb = *(const uint64_t *)b;
    if (va < vb) {
        return -1;
    }
    if (va > vb) {
        return 1;
    }
    return 0;
}

static inline vtlk_bench_stats_t vtlk_bench_compute_stats(uint64_t *samples, size_t n)
{
    vtlk_bench_stats_t st;
    double sum = 0.0;
    size_t i;

    st.min_ns = 0u;
    st.p50_ns = 0u;
    st.p95_ns = 0u;
    st.p99_ns = 0u;
    st.mean_ns = 0.0;

    if (samples == NULL || n == 0u) {
        return st;
    }

    qsort(samples, n, sizeof(uint64_t), vtlk_bench_cmp_u64);

    for (i = 0u; i < n; ++i) {
        sum += (double)samples[i];
    }

    st.min_ns = samples[0];
    st.p50_ns = samples[(size_t)((n - 1u) * 50u / 100u)];
    st.p95_ns = samples[(size_t)((n - 1u) * 95u / 100u)];
    st.p99_ns = samples[(size_t)((n - 1u) * 99u / 100u)];
    st.mean_ns = sum / (double)n;
    return st;
}

#endif /* VECTLINK_BENCH_TIMER_H */
