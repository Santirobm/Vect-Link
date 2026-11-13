#ifndef _WIN32
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif
#elif defined(_MSC_VER) && !defined(_CRT_SECURE_NO_WARNINGS)
#define _CRT_SECURE_NO_WARNINGS
#endif

#include "vectlink/vectlink.h"
#include "transport/atomic_compat.h"
#include "bench_timer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define VTLK_BENCH_CHECK(cond)                                                 \
    do {                                                                       \
        if (!(cond)) {                                                         \
            fprintf(stderr, "Benchmark check failed at %s:%d: %s\n",           \
                    __FILE__, __LINE__, #cond);                                \
            abort();                                                           \
        }                                                                      \
    } while (0)

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
typedef SOCKET vtlk_sock_t;
typedef HANDLE vtlk_thread_t;
#define VTLK_INVALID_SOCK INVALID_SOCKET
#define vtlk_sock_close(s) closesocket(s)
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <pthread.h>
#include <sched.h>
#include <sys/socket.h>
#include <unistd.h>
typedef int vtlk_sock_t;
typedef pthread_t vtlk_thread_t;
#define VTLK_INVALID_SOCK (-1)
#define vtlk_sock_close(s) close(s)
#endif

#define BENCH_N_TOKENS 32u
#define BENCH_D_MODEL  4096u
#define BENCH_ELEMS    ((size_t)BENCH_N_TOKENS * (size_t)BENCH_D_MODEL)
#define BENCH_WARMUP   500u
#define BENCH_N_TCP    2000u
#define BENCH_N_SHM    10000u
#define JSON_BUF_CAP   (3u * 1024u * 1024u)

typedef struct bench_result {
    const char *name;
    size_t wire_bytes;
    double ratio_vs_json;
    vtlk_bench_stats_t stats;
    double throughput_msg_s;
    double cosine_sim;
} bench_result_t;

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

static void bench_cpu_yield(void)
{
#ifdef _WIN32
    SwitchToThread();
#else
    sched_yield();
#endif
}

static void write_u32_le(uint8_t *dst, uint32_t v)
{
    dst[0] = (uint8_t)(v & 0xFFu);
    dst[1] = (uint8_t)((v >> 8) & 0xFFu);
    dst[2] = (uint8_t)((v >> 16) & 0xFFu);
    dst[3] = (uint8_t)((v >> 24) & 0xFFu);
}

static uint32_t read_u32_le(const uint8_t *src)
{
    return ((uint32_t)src[0]) |
           ((uint32_t)src[1] << 8) |
           ((uint32_t)src[2] << 16) |
           ((uint32_t)src[3] << 24);
}

static int tcp_send_all(vtlk_sock_t sock, const void *buf, size_t len)
{
    const uint8_t *p = (const uint8_t *)buf;
    size_t sent = 0u;

    while (sent < len) {
#ifdef _WIN32
        int chunk = (len - sent > (size_t)0x7FFFFFFF) ? 0x7FFFFFFF : (int)(len - sent);
        int n = send(sock, (const char *)(p + sent), chunk, 0);
#else
        ssize_t n = send(sock, p + sent, len - sent, 0);
#endif
        if (n <= 0) {
            return -1;
        }
        sent += (size_t)n;
    }
    return 0;
}

static int tcp_recv_all(vtlk_sock_t sock, void *buf, size_t len)
{
    uint8_t *p = (uint8_t *)buf;
    size_t recvd = 0u;

    while (recvd < len) {
#ifdef _WIN32
        int chunk = (len - recvd > (size_t)0x7FFFFFFF) ? 0x7FFFFFFF : (int)(len - recvd);
        int n = recv(sock, (char *)(p + recvd), chunk, 0);
#else
        ssize_t n = recv(sock, p + recvd, len - recvd, 0);
#endif
        if (n <= 0) {
            return -1;
        }
        recvd += (size_t)n;
    }
    return 0;
}

static void tcp_set_nodelay(vtlk_sock_t sock)
{
    int flag = 1;
#ifdef _WIN32
    setsockopt(sock, IPPROTO_TCP, TCP_NODELAY, (const char *)&flag, (int)sizeof(flag));
#else
    setsockopt(sock, IPPROTO_TCP, TCP_NODELAY, &flag, (socklen_t)sizeof(flag));
#endif
}

static vtlk_sock_t tcp_listen_loopback(uint16_t *out_port)
{
    vtlk_sock_t srv;
    struct sockaddr_in addr;
#ifdef _WIN32
    int alen = (int)sizeof(addr);
#else
    socklen_t alen = (socklen_t)sizeof(addr);
#endif

    srv = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    VTLK_BENCH_CHECK(srv != VTLK_INVALID_SOCK);

    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = 0;

    VTLK_BENCH_CHECK(bind(srv, (struct sockaddr *)&addr, alen) == 0);
    VTLK_BENCH_CHECK(listen(srv, 1) == 0);
    VTLK_BENCH_CHECK(getsockname(srv, (struct sockaddr *)&addr, &alen) == 0);
    *out_port = ntohs(addr.sin_port);
    return srv;
}

static vtlk_sock_t tcp_connect_loopback(uint16_t port)
{
    vtlk_sock_t cli;
    struct sockaddr_in addr;

    cli = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    VTLK_BENCH_CHECK(cli != VTLK_INVALID_SOCK);

    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = htons(port);

    VTLK_BENCH_CHECK(connect(cli, (struct sockaddr *)&addr, (int)sizeof(addr)) == 0);
    tcp_set_nodelay(cli);
    return cli;
}

/* ---------------- Scenario 1: JSON + TCP Loopback ---------------- */

typedef struct json_tcp_ctx {
    vtlk_sock_t srv;
    size_t total_iters;
    float *out_tensor;
} json_tcp_ctx_t;

static size_t encode_json_tensor(const float *src,
                                 uint32_t n_tokens,
                                 uint32_t d_model,
                                 char *dst,
                                 size_t cap)
{
    size_t count = (size_t)n_tokens * (size_t)d_model;
    size_t pos = 0u;
    size_t i;
    int n;

    n = snprintf(dst + pos, cap - pos,
                 "{\"n_tokens\":%u,\"d_model\":%u,\"data\":[", n_tokens, d_model);
    VTLK_BENCH_CHECK(n > 0);
    pos += (size_t)n;

    for (i = 0u; i < count; ++i) {
        if (i + 1u < count) {
            n = snprintf(dst + pos, cap - pos, "%.5f,", (double)src[i]);
        } else {
            n = snprintf(dst + pos, cap - pos, "%.5f]}", (double)src[i]);
        }
        VTLK_BENCH_CHECK(n > 0 && pos + (size_t)n < cap);
        pos += (size_t)n;
    }
    return pos;
}

static void decode_json_tensor(const char *json, float *dst, size_t count)
{
    const char *p = strstr(json, "\"data\":[");
    size_t i;

    VTLK_BENCH_CHECK(p != NULL);
    p += 8;

    for (i = 0u; i < count; ++i) {
        char *endp = NULL;
        dst[i] = strtof(p, &endp);
        VTLK_BENCH_CHECK(endp != p);
        p = (*endp == ',') ? (endp + 1) : endp;
    }
}

#ifdef _WIN32
static DWORD WINAPI json_tcp_consumer(LPVOID arg)
#else
static void *json_tcp_consumer(void *arg)
#endif
{
    json_tcp_ctx_t *ctx = (json_tcp_ctx_t *)arg;
    vtlk_sock_t conn = accept(ctx->srv, NULL, NULL);
    char *rx_buf = (char *)malloc(JSON_BUF_CAP);
    size_t i;
    uint8_t ack = 1u;

    VTLK_BENCH_CHECK(conn != VTLK_INVALID_SOCK);
    VTLK_BENCH_CHECK(rx_buf != NULL);
    tcp_set_nodelay(conn);

    for (i = 0u; i < ctx->total_iters; ++i) {
        uint8_t len_hdr[4];
        uint32_t msg_len;
        VTLK_BENCH_CHECK(tcp_recv_all(conn, len_hdr, 4u) == 0);
        msg_len = read_u32_le(len_hdr);
        VTLK_BENCH_CHECK(msg_len < JSON_BUF_CAP);
        VTLK_BENCH_CHECK(tcp_recv_all(conn, rx_buf, msg_len) == 0);
        rx_buf[msg_len] = '\0';
        decode_json_tensor(rx_buf, ctx->out_tensor, BENCH_ELEMS);
        VTLK_BENCH_CHECK(tcp_send_all(conn, &ack, 1u) == 0);
    }

    free(rx_buf);
    vtlk_sock_close(conn);
#ifdef _WIN32
    return 0;
#else
    return NULL;
#endif
}

static bench_result_t run_json_tcp_bench(const float *src)
{
    uint16_t port = 0u;
    vtlk_sock_t srv = tcp_listen_loopback(&port);
    vtlk_sock_t cli;
    json_tcp_ctx_t ctx;
    vtlk_thread_t th;
    char *tx_buf = (char *)malloc(JSON_BUF_CAP);
    uint64_t *samples = (uint64_t *)malloc(BENCH_N_TCP * sizeof(uint64_t));
    size_t total_iters = BENCH_WARMUP + BENCH_N_TCP;
    size_t last_wire = 0u;
    size_t i;
    bench_result_t res;

    ctx.srv = srv;
    ctx.total_iters = total_iters;
    ctx.out_tensor = (float *)malloc(BENCH_ELEMS * sizeof(float));
    VTLK_BENCH_CHECK(tx_buf != NULL && samples != NULL && ctx.out_tensor != NULL);

#ifdef _WIN32
    th = CreateThread(NULL, 0u, json_tcp_consumer, &ctx, 0u, NULL);
    VTLK_BENCH_CHECK(th != NULL);
#else
    VTLK_BENCH_CHECK(pthread_create(&th, NULL, json_tcp_consumer, &ctx) == 0);
#endif

    cli = tcp_connect_loopback(port);

    for (i = 0u; i < total_iters; ++i) {
        uint64_t t0 = vtlk_bench_now_ns();
        uint8_t len_hdr[4];
        uint8_t ack = 0u;
        size_t jlen = encode_json_tensor(src, BENCH_N_TOKENS, BENCH_D_MODEL,
                                         tx_buf, JSON_BUF_CAP);
        write_u32_le(len_hdr, (uint32_t)jlen);
        VTLK_BENCH_CHECK(tcp_send_all(cli, len_hdr, 4u) == 0);
        VTLK_BENCH_CHECK(tcp_send_all(cli, tx_buf, jlen) == 0);
        VTLK_BENCH_CHECK(tcp_recv_all(cli, &ack, 1u) == 0);
        uint64_t t1 = vtlk_bench_now_ns();

        last_wire = 4u + jlen;
        if (i >= BENCH_WARMUP) {
            samples[i - BENCH_WARMUP] = t1 - t0;
        }
    }

#ifdef _WIN32
    WaitForSingleObject(th, INFINITE);
    CloseHandle(th);
#else
    pthread_join(th, NULL);
#endif

    vtlk_sock_close(cli);
    vtlk_sock_close(srv);

    res.name = "JSON + TCP";
    res.wire_bytes = last_wire;
    res.ratio_vs_json = 1.0;
    res.stats = vtlk_bench_compute_stats(samples, BENCH_N_TCP);
    res.throughput_msg_s = 1e9 / res.stats.mean_ns;
    res.cosine_sim = vtlk_cosine_similarity(src, ctx.out_tensor, BENCH_ELEMS);

    free(tx_buf);
    free(samples);
    free(ctx.out_tensor);
    return res;
}

/* ---------------- Scenario 2: Vect-Link INT4 + TCP Loopback ---------------- */

typedef struct vtlk_tcp_ctx {
    vtlk_sock_t srv;
    size_t total_iters;
    size_t payload_bytes;
    float *out_tensor;
} vtlk_tcp_ctx_t;

#ifdef _WIN32
static DWORD WINAPI vtlk_tcp_consumer(LPVOID arg)
#else
static void *vtlk_tcp_consumer(void *arg)
#endif
{
    vtlk_tcp_ctx_t *ctx = (vtlk_tcp_ctx_t *)arg;
    vtlk_sock_t conn = accept(ctx->srv, NULL, NULL);
    uint8_t *rx_buf = (uint8_t *)malloc(VTLK_HEADER_SIZE + ctx->payload_bytes);
    size_t i;
    uint8_t ack = 1u;

    VTLK_BENCH_CHECK(conn != VTLK_INVALID_SOCK);
    VTLK_BENCH_CHECK(rx_buf != NULL);
    tcp_set_nodelay(conn);

    for (i = 0u; i < ctx->total_iters; ++i) {
        vtlk_header_t hdr;
        uint32_t crc;
        VTLK_BENCH_CHECK(tcp_recv_all(conn, rx_buf, VTLK_HEADER_SIZE) == 0);
        VTLK_BENCH_CHECK(vtlk_header_decode(rx_buf, VTLK_HEADER_SIZE, &hdr) == VTLK_OK);
        VTLK_BENCH_CHECK(hdr.payload_bytes == ctx->payload_bytes);
        VTLK_BENCH_CHECK(tcp_recv_all(conn, rx_buf + VTLK_HEADER_SIZE, hdr.payload_bytes) == 0);
        crc = vtlk_crc32c(rx_buf + VTLK_HEADER_SIZE, hdr.payload_bytes);
        VTLK_BENCH_CHECK(crc == hdr.payload_crc32c);
        VTLK_BENCH_CHECK(vtlk_decode_tensor(rx_buf + VTLK_HEADER_SIZE, hdr.payload_bytes,
                                  hdr.n_tokens, hdr.d_model,
                                  (vtlk_quant_scheme_t)hdr.quant_scheme,
                                  ctx->out_tensor, BENCH_ELEMS) == VTLK_OK);
        VTLK_BENCH_CHECK(tcp_send_all(conn, &ack, 1u) == 0);
    }

    free(rx_buf);
    vtlk_sock_close(conn);
#ifdef _WIN32
    return 0;
#else
    return NULL;
#endif
}

static bench_result_t run_vtlk_tcp_bench(const float *src, size_t json_wire_bytes)
{
    uint16_t port = 0u;
    vtlk_sock_t srv = tcp_listen_loopback(&port);
    vtlk_sock_t cli;
    vtlk_tcp_ctx_t ctx;
    vtlk_thread_t th;
    size_t payload_bytes = 0u;
    uint8_t *tx_buf;
    uint64_t *samples = (uint64_t *)malloc(BENCH_N_TCP * sizeof(uint64_t));
    size_t total_iters = BENCH_WARMUP + BENCH_N_TCP;
    size_t i;
    vtlk_header_t hdr;
    bench_result_t res;

    VTLK_BENCH_CHECK(vtlk_codec_encoded_bytes(VTLK_QUANT_INT4, BENCH_N_TOKENS,
                                    BENCH_D_MODEL, &payload_bytes) == VTLK_OK);

    tx_buf = (uint8_t *)malloc(VTLK_HEADER_SIZE + payload_bytes);
    ctx.srv = srv;
    ctx.total_iters = total_iters;
    ctx.payload_bytes = payload_bytes;
    ctx.out_tensor = (float *)malloc(BENCH_ELEMS * sizeof(float));
    VTLK_BENCH_CHECK(tx_buf != NULL && samples != NULL && ctx.out_tensor != NULL);

#ifdef _WIN32
    th = CreateThread(NULL, 0u, vtlk_tcp_consumer, &ctx, 0u, NULL);
    VTLK_BENCH_CHECK(th != NULL);
#else
    VTLK_BENCH_CHECK(pthread_create(&th, NULL, vtlk_tcp_consumer, &ctx) == 0);
#endif

    cli = tcp_connect_loopback(port);

    vtlk_header_init_defaults(&hdr);
    hdr.n_tokens = BENCH_N_TOKENS;
    hdr.d_model = BENCH_D_MODEL;
    hdr.quant_scheme = (uint8_t)VTLK_QUANT_INT4;
    hdr.payload_bytes = (uint32_t)payload_bytes;
    hdr.raw_bytes = (uint32_t)(BENCH_ELEMS * sizeof(float));

    for (i = 0u; i < total_iters; ++i) {
        uint64_t t0 = vtlk_bench_now_ns();
        size_t written = 0u;
        uint8_t ack = 0u;

        VTLK_BENCH_CHECK(vtlk_encode_tensor(src, BENCH_N_TOKENS, BENCH_D_MODEL, VTLK_QUANT_INT4,
                                  tx_buf + VTLK_HEADER_SIZE, payload_bytes, &written) == VTLK_OK);
        hdr.sequence = (uint64_t)i;
        hdr.payload_crc32c = vtlk_crc32c(tx_buf + VTLK_HEADER_SIZE, written);
        VTLK_BENCH_CHECK(vtlk_header_encode(&hdr, tx_buf, VTLK_HEADER_SIZE) == VTLK_OK);
        VTLK_BENCH_CHECK(tcp_send_all(cli, tx_buf, VTLK_HEADER_SIZE + written) == 0);
        VTLK_BENCH_CHECK(tcp_recv_all(cli, &ack, 1u) == 0);
        uint64_t t1 = vtlk_bench_now_ns();

        if (i >= BENCH_WARMUP) {
            samples[i - BENCH_WARMUP] = t1 - t0;
        }
    }

#ifdef _WIN32
    WaitForSingleObject(th, INFINITE);
    CloseHandle(th);
#else
    pthread_join(th, NULL);
#endif

    vtlk_sock_close(cli);
    vtlk_sock_close(srv);

    res.name = "VTLK INT4 + TCP";
    res.wire_bytes = VTLK_HEADER_SIZE + payload_bytes;
    res.ratio_vs_json = (double)json_wire_bytes / (double)res.wire_bytes;
    res.stats = vtlk_bench_compute_stats(samples, BENCH_N_TCP);
    res.throughput_msg_s = 1e9 / res.stats.mean_ns;
    res.cosine_sim = vtlk_cosine_similarity(src, ctx.out_tensor, BENCH_ELEMS);

    free(tx_buf);
    free(samples);
    free(ctx.out_tensor);
    return res;
}

/* ---------------- Scenario 3: Vect-Link INT4 + SPSC SHM ---------------- */

typedef struct vtlk_shm_ctx {
    const char *shm_name;
    size_t shm_bytes;
    size_t total_iters;
    size_t payload_bytes;
    uint64_t *ack_seq;
    float *out_tensor;
} vtlk_shm_ctx_t;

#ifdef _WIN32
static DWORD WINAPI vtlk_shm_consumer(LPVOID arg)
#else
static void *vtlk_shm_consumer(void *arg)
#endif
{
    vtlk_shm_ctx_t *ctx = (vtlk_shm_ctx_t *)arg;
    vtlk_shm_t shm;
    vtlk_ring_t ring;
    uint8_t *rx_payload = (uint8_t *)malloc(ctx->payload_bytes);
    size_t i;

    VTLK_BENCH_CHECK(rx_payload != NULL);
    VTLK_BENCH_CHECK(vtlk_shm_open(&shm, ctx->shm_name, ctx->shm_bytes) == VTLK_OK);
    VTLK_BENCH_CHECK(vtlk_ring_attach(&ring, shm.addr, shm.size) == VTLK_OK);

    for (i = 0u; i < ctx->total_iters; ++i) {
        vtlk_header_t hdr;
        vtlk_status_t st;
        for (;;) {
            st = vtlk_ring_pop(&ring, &hdr, rx_payload, ctx->payload_bytes);
            if (st == VTLK_OK) {
                break;
            }
            VTLK_BENCH_CHECK(st == VTLK_ERR_EMPTY);
            bench_cpu_yield();
        }
        VTLK_BENCH_CHECK(vtlk_decode_tensor(rx_payload, hdr.payload_bytes,
                                  hdr.n_tokens, hdr.d_model,
                                  (vtlk_quant_scheme_t)hdr.quant_scheme,
                                  ctx->out_tensor, BENCH_ELEMS) == VTLK_OK);
        vtlk_atomic_store_release_u64(ctx->ack_seq, (uint64_t)(i + 1u));
    }

    free(rx_payload);
    vtlk_shm_close(&shm);
#ifdef _WIN32
    return 0;
#else
    return NULL;
#endif
}

static bench_result_t run_vtlk_shm_bench(const float *src, size_t json_wire_bytes)
{
    const char *shm_name = "/vectlink_bench_shm";
    size_t payload_bytes = 0u;
    size_t ring_bytes;
    vtlk_shm_t shm;
    vtlk_ring_t ring;
    vtlk_shm_ctx_t ctx;
    vtlk_thread_t th;
    uint8_t *tx_payload;
    uint64_t *samples = (uint64_t *)malloc(BENCH_N_SHM * sizeof(uint64_t));
    uint64_t ack_seq = 0u;
    size_t total_iters = BENCH_WARMUP + BENCH_N_SHM;
    size_t i;
    vtlk_header_t hdr;
    bench_result_t res;

    VTLK_BENCH_CHECK(vtlk_codec_encoded_bytes(VTLK_QUANT_INT4, BENCH_N_TOKENS,
                                    BENCH_D_MODEL, &payload_bytes) == VTLK_OK);
    ring_bytes = vtlk_ring_required_bytes(8u, (uint32_t)payload_bytes, NULL);
    VTLK_BENCH_CHECK(ring_bytes > 0u);

    VTLK_BENCH_CHECK(vtlk_shm_create(&shm, shm_name, ring_bytes) == VTLK_OK);
    VTLK_BENCH_CHECK(vtlk_ring_init(&ring, shm.addr, shm.size, 8u, (uint32_t)payload_bytes) == VTLK_OK);

    tx_payload = (uint8_t *)malloc(payload_bytes);
    ctx.shm_name = shm_name;
    ctx.shm_bytes = ring_bytes;
    ctx.total_iters = total_iters;
    ctx.payload_bytes = payload_bytes;
    ctx.ack_seq = &ack_seq;
    ctx.out_tensor = (float *)malloc(BENCH_ELEMS * sizeof(float));
    VTLK_BENCH_CHECK(tx_payload != NULL && samples != NULL && ctx.out_tensor != NULL);

#ifdef _WIN32
    th = CreateThread(NULL, 0u, vtlk_shm_consumer, &ctx, 0u, NULL);
    VTLK_BENCH_CHECK(th != NULL);
#else
    VTLK_BENCH_CHECK(pthread_create(&th, NULL, vtlk_shm_consumer, &ctx) == 0);
#endif

    vtlk_header_init_defaults(&hdr);
    hdr.n_tokens = BENCH_N_TOKENS;
    hdr.d_model = BENCH_D_MODEL;
    hdr.quant_scheme = (uint8_t)VTLK_QUANT_INT4;
    hdr.payload_bytes = (uint32_t)payload_bytes;
    hdr.raw_bytes = (uint32_t)(BENCH_ELEMS * sizeof(float));

    for (i = 0u; i < total_iters; ++i) {
        uint64_t t0 = vtlk_bench_now_ns();
        size_t written = 0u;
        vtlk_status_t st;

        VTLK_BENCH_CHECK(vtlk_encode_tensor(src, BENCH_N_TOKENS, BENCH_D_MODEL, VTLK_QUANT_INT4,
                                  tx_payload, payload_bytes, &written) == VTLK_OK);
        hdr.sequence = (uint64_t)i;
        for (;;) {
            st = vtlk_ring_push(&ring, &hdr, tx_payload);
            if (st == VTLK_OK) {
                break;
            }
            VTLK_BENCH_CHECK(st == VTLK_ERR_FULL);
            bench_cpu_yield();
        }
        while (vtlk_atomic_load_acquire_u64(&ack_seq) < (uint64_t)(i + 1u)) {
            bench_cpu_yield();
        }
        uint64_t t1 = vtlk_bench_now_ns();

        if (i >= BENCH_WARMUP) {
            samples[i - BENCH_WARMUP] = t1 - t0;
        }
    }

#ifdef _WIN32
    WaitForSingleObject(th, INFINITE);
    CloseHandle(th);
#else
    pthread_join(th, NULL);
#endif

    vtlk_shm_close(&shm);
    vtlk_shm_unlink(shm_name);

    res.name = "VTLK INT4 + SHM";
    res.wire_bytes = VTLK_HEADER_SIZE + payload_bytes;
    res.ratio_vs_json = (double)json_wire_bytes / (double)res.wire_bytes;
    res.stats = vtlk_bench_compute_stats(samples, BENCH_N_SHM);
    res.throughput_msg_s = 1e9 / res.stats.mean_ns;
    res.cosine_sim = vtlk_cosine_similarity(src, ctx.out_tensor, BENCH_ELEMS);

    free(tx_payload);
    free(samples);
    free(ctx.out_tensor);
    return res;
}

static void print_and_save_results(const bench_result_t *results, size_t count)
{
    FILE *fp = fopen("benchmark_results.csv", "w");
    size_t i;

    printf("\nTensor: %u x %u (%zu floats, %zu bytes FP32) | Warmup: %u | TCP N=%u, SHM N=%u\n",
           BENCH_N_TOKENS, BENCH_D_MODEL, BENCH_ELEMS, BENCH_ELEMS * sizeof(float),
           BENCH_WARMUP, BENCH_N_TCP, BENCH_N_SHM);
    printf("%-16s | %10s | %9s | %9s | %9s | %9s | %12s | %10s\n",
           "Escenario", "Wire (B)", "vs JSON", "p50 (us)", "p95 (us)", "p99 (us)",
           "Msg/s", "Cosine Sim");
    printf("-----------------+------------+-----------+-----------+-----------+-"
           "----------+--------------+-----------\n");

    if (fp != NULL) {
        fprintf(fp, "scenario,wire_bytes,compression_vs_json,p50_us,p95_us,p99_us,throughput_msg_s,cosine_sim\n");
    }

    for (i = 0u; i < count; ++i) {
        double p50 = (double)results[i].stats.p50_ns / 1000.0;
        double p95 = (double)results[i].stats.p95_ns / 1000.0;
        double p99 = (double)results[i].stats.p99_ns / 1000.0;

        printf("%-16s | %10zu | %8.2fx | %9.2f | %9.2f | %9.2f | %12.1f | %10.6f\n",
               results[i].name,
               results[i].wire_bytes,
               results[i].ratio_vs_json,
               p50, p95, p99,
               results[i].throughput_msg_s,
               results[i].cosine_sim);

        if (fp != NULL) {
            fprintf(fp, "%s,%zu,%.4f,%.3f,%.3f,%.3f,%.2f,%.6f\n",
                    results[i].name,
                    results[i].wire_bytes,
                    results[i].ratio_vs_json,
                    p50, p95, p99,
                    results[i].throughput_msg_s,
                    results[i].cosine_sim);
        }
    }

    if (fp != NULL) {
        fclose(fp);
        printf("\nSaved CSV to benchmark_results.csv\n");
    }
}

int main(void)
{
    float *src = (float *)malloc(BENCH_ELEMS * sizeof(float));
    bench_result_t results[3];

#ifdef _WIN32
    WSADATA wsa;
    VTLK_BENCH_CHECK(WSAStartup(MAKEWORD(2, 2), &wsa) == 0);
#endif

    VTLK_BENCH_CHECK(src != NULL);
    populate_activations(src, BENCH_ELEMS);

    results[0] = run_json_tcp_bench(src);
    results[1] = run_vtlk_tcp_bench(src, results[0].wire_bytes);
    results[2] = run_vtlk_shm_bench(src, results[0].wire_bytes);

    print_and_save_results(results, 3u);

    free(src);
#ifdef _WIN32
    WSACleanup();
#endif
    return 0;
}
