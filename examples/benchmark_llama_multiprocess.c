#ifndef _WIN32
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif
#endif

#include "vectlink/vectlink.h"
#include "bench_timer.h"
#include "llama_bridge_common.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

#define MP_DEFAULT_LENGTHS "128,512,1024,2048"
#define MP_DEFAULT_WARMUP  3u
#define MP_DEFAULT_ITERS   15u
#define MP_DEFAULT_THREADS 4
#define MP_DEFAULT_N_BATCH 512u
#define MP_DEFAULT_SHM     "/vtlk_llama_mp_bench"
#define MP_DEFAULT_CSV     "llama_multiprocess_benchmark.csv"
#define MP_MAX_SHM_BYTES   (512u * 1024u * 1024u)
#define MP_CTRL_RING_CAP   4u
#define MP_DATA_RING_CAP   2u
#define MP_CTRL_PAYLOAD_SZ 64u
#define MP_MAX_LENGTHS     16u

#define MP_MSG_READY 1u
#define MP_MSG_DONE  2u

typedef enum mp_role {
    MP_ROLE_RUNNER = 0,
    MP_ROLE_PRODUCER = 1,
    MP_ROLE_CONSUMER = 2
} mp_role_t;

typedef struct mp_ctrl_msg {
    uint32_t kind;
    int32_t top1_token;
    uint64_t receiver_ns;
} mp_ctrl_msg_t;

typedef struct mp_length_summary {
    uint32_t tokens;
    size_t kv_bytes;
    double ref_p50_us;
    double ref_p95_us;
    double ref_p99_us;
    double vtlk_p50_us;
    double vtlk_p95_us;
    double vtlk_p99_us;
    int32_t ref_tok;
    int32_t vtlk_tok;
    int match;
} mp_length_summary_t;

static void print_usage(const char *prog)
{
    fprintf(stderr,
            "Usage: %s --model <path.gguf> [--role runner|producer|consumer] "
            "[--lengths 128,512,1024,2048] [--tokens 512] [--n-ctx 2304] "
            "[--iters 15] [--warmup 3] [--threads 4] "
            "[--shm /vtlk_llama_mp_bench] [--csv llama_multiprocess_benchmark.csv]\n",
            prog);
}

static void sleep_ms_brief(unsigned int ms)
{
#ifdef _WIN32
    Sleep((DWORD)ms);
#else
    struct timespec ts;
    ts.tv_sec = (time_t)(ms / 1000u);
    ts.tv_nsec = (long)((ms % 1000u) * 1000000u);
    nanosleep(&ts, NULL);
#endif
}

static void ctrl_msg_encode(const mp_ctrl_msg_t *m, uint8_t dst[MP_CTRL_PAYLOAD_SZ])
{
    memset(dst, 0, MP_CTRL_PAYLOAD_SZ);
    vtlk_llama_write_u32_le(dst + 0u, m->kind);
    vtlk_llama_write_u32_le(dst + 4u, (uint32_t)m->top1_token);
    vtlk_llama_write_u32_le(dst + 8u, (uint32_t)(m->receiver_ns & 0xFFFFFFFFULL));
    vtlk_llama_write_u32_le(dst + 12u, (uint32_t)(m->receiver_ns >> 32));
}

static void ctrl_msg_decode(const uint8_t src[MP_CTRL_PAYLOAD_SZ], mp_ctrl_msg_t *m)
{
    uint32_t lo;
    uint32_t hi;
    m->kind = vtlk_llama_read_u32_le(src + 0u);
    m->top1_token = (int32_t)vtlk_llama_read_u32_le(src + 4u);
    lo = vtlk_llama_read_u32_le(src + 8u);
    hi = vtlk_llama_read_u32_le(src + 12u);
    m->receiver_ns = ((uint64_t)hi << 32) | (uint64_t)lo;
}

static int ctrl_send(vtlk_ring_t *ring, const mp_ctrl_msg_t *msg, uint64_t seq)
{
    uint8_t buf[MP_CTRL_PAYLOAD_SZ];
    vtlk_header_t hdr;
    vtlk_status_t st;

    ctrl_msg_encode(msg, buf);
    vtlk_header_init_defaults(&hdr);
    hdr.d_model = 1u;
    hdr.n_tokens = 1u;
    hdr.quant_scheme = (uint8_t)VTLK_QUANT_RAW;
    hdr.payload_bytes = MP_CTRL_PAYLOAD_SZ;
    hdr.raw_bytes = MP_CTRL_PAYLOAD_SZ;
    hdr.sequence = seq;

    for (;;) {
        st = vtlk_ring_push(ring, &hdr, buf);
        if (st == VTLK_OK) {
            return 1;
        }
        if (st == VTLK_ERR_FULL) {
            vtlk_llama_cpu_yield();
            continue;
        }
        return 0;
    }
}

static int ctrl_recv(vtlk_ring_t *ring, mp_ctrl_msg_t *msg)
{
    uint8_t buf[MP_CTRL_PAYLOAD_SZ];
    vtlk_header_t hdr;
    vtlk_status_t st;
    uint64_t deadline = vtlk_bench_now_ns() + 60000000000ULL; /* 60s safety timeout */

    for (;;) {
        st = vtlk_ring_pop(ring, &hdr, buf, sizeof(buf));
        if (st == VTLK_OK) {
            ctrl_msg_decode(buf, msg);
            return 1;
        }
        if (st == VTLK_ERR_EMPTY) {
            if (vtlk_bench_now_ns() > deadline) {
                fprintf(stderr, "benchmark_llama_multiprocess: ctrl_recv timed out\n");
                return 0;
            }
            vtlk_llama_cpu_yield();
            continue;
        }
        return 0;
    }
}

static int eval_prompt_chunks(struct llama_context *ctx,
                              const llama_token *tokens,
                              uint32_t n_tokens,
                              uint32_t n_batch)
{
    uint32_t offset = 0u;
    while (offset < n_tokens) {
        uint32_t chunk = n_tokens - offset;
        struct llama_batch batch;
        uint32_t i;
        if (chunk > n_batch) {
            chunk = n_batch;
        }
        batch = llama_batch_init((int32_t)chunk, 0, 1);
        batch.n_tokens = (int32_t)chunk;
        for (i = 0u; i < chunk; ++i) {
            batch.token[i] = tokens[offset + i];
            batch.pos[i] = (llama_pos)(offset + i);
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = ((offset + i) == (n_tokens - 1u)) ? 1 : 0;
        }
        if (llama_decode(ctx, batch) != 0) {
            llama_batch_free(batch);
            return 0;
        }
        llama_batch_free(batch);
        offset += chunk;
    }
    return 1;
}

static int restore_kv_and_decode_last(struct llama_context *ctx,
                                        const uint8_t *raw_state,
                                        size_t state_bytes,
                                        llama_token last_tok,
                                        uint32_t n_tokens)
{
    size_t restored = llama_state_seq_set_data(ctx, raw_state, state_bytes, 0);
    llama_pos last_pos = (llama_pos)(n_tokens - 1u);
    struct llama_batch bridge_batch;

    if (restored == 0u || restored != state_bytes) {
        fprintf(stderr, "benchmark_llama_multiprocess: set_data restored %zu != %zu\n",
                restored, state_bytes);
        return 0;
    }
    if (!llama_memory_seq_rm(llama_get_memory(ctx), 0, last_pos, -1)) {
        fprintf(stderr, "benchmark_llama_multiprocess: llama_memory_seq_rm failed\n");
        return 0;
    }

    bridge_batch = llama_batch_init(1, 0, 1);
    bridge_batch.n_tokens = 1;
    bridge_batch.token[0] = last_tok;
    bridge_batch.pos[0] = last_pos;
    bridge_batch.n_seq_id[0] = 1;
    bridge_batch.seq_id[0][0] = 0;
    bridge_batch.logits[0] = 1;
    if (llama_decode(ctx, bridge_batch) != 0) {
        llama_batch_free(bridge_batch);
        fprintf(stderr, "benchmark_llama_multiprocess: bridge llama_decode failed\n");
        return 0;
    }
    llama_batch_free(bridge_batch);
    return 1;
}

static llama_token sample_greedy_token(struct llama_context *ctx)
{
    struct llama_sampler *smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_token tok;
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
    tok = llama_sampler_sample(smpl, ctx, -1);
    llama_sampler_free(smpl);
    return tok;
}

static char *build_deterministic_prompt(const struct llama_vocab *vocab,
                                        uint32_t target_tokens,
                                        llama_token **out_tokens,
                                        size_t *out_text_bytes)
{
    static const char k_doc_block[] =
        "Vect-Link provides deterministic little-endian framing, Castagnoli CRC32C "
        "verification, and lock-free single-producer single-consumer shared-memory "
        "ring buffers for inter-process tensor and KV-cache handoff without JSON "
        "serialization or kernel socket overhead. ";
    size_t block_len = strlen(k_doc_block);
    size_t cap = block_len * 128u + 1u;
    char *text = (char *)malloc(cap);
    size_t text_len = 0u;
    llama_token *toks = NULL;
    int32_t n_tok = 0;

    if (text == NULL) {
        return NULL;
    }
    text[0] = '\0';

    for (;;) {
        size_t need_bytes = (size_t)target_tokens * 6u;
        while (text_len < need_bytes) {
            if (text_len + block_len + 1u > cap) {
                size_t new_cap = cap * 2u;
                char *grown = (char *)realloc(text, new_cap);
                if (grown == NULL) {
                    free(text);
                    return NULL;
                }
                text = grown;
                cap = new_cap;
            }
            memcpy(text + text_len, k_doc_block, block_len);
            text_len += block_len;
        }
        text[text_len] = '\0';

        n_tok = -llama_tokenize(vocab, text, (int32_t)text_len, NULL, 0, true, false);
        if (n_tok >= (int32_t)target_tokens) {
            break;
        }
        /* Grow once more if dense text yielded fewer tokens */
        if (text_len + block_len * 4u + 1u > cap) {
            size_t new_cap = cap * 2u;
            char *grown = (char *)realloc(text, new_cap);
            if (grown == NULL) {
                free(text);
                return NULL;
            }
            text = grown;
            cap = new_cap;
        }
        memcpy(text + text_len, k_doc_block, block_len);
        text_len += block_len;
    }

    toks = (llama_token *)malloc((size_t)n_tok * sizeof(llama_token));
    if (toks == NULL) {
        free(text);
        return NULL;
    }
    if (llama_tokenize(vocab, text, (int32_t)text_len, toks, n_tok, true, false) < 0) {
        free(toks);
        free(text);
        return NULL;
    }

    *out_tokens = toks;
    *out_text_bytes = text_len;
    return text;
}

static int run_producer(const char *model_path,
                        const char *shm_name,
                        uint32_t n_tokens,
                        uint32_t n_ctx,
                        uint32_t n_batch,
                        int32_t n_threads,
                        uint32_t warmup,
                        uint32_t iters)
{
    char ctrl_name[300];
    struct llama_model_params mparams;
    struct llama_model *model = NULL;
    const struct llama_vocab *vocab = NULL;
    struct llama_context_params cparams;
    struct llama_context *ctx = NULL;
    char *prompt_text = NULL;
    llama_token *tokens = NULL;
    size_t prompt_bytes = 0u;
    uint32_t d_model = 0u;
    uint64_t compat_hash = 0ULL;
    uint32_t local_build = 0u;
    size_t state_size = 0u;
    size_t max_payload_bytes = 0u;
    size_t data_ring_bytes = 0u;
    size_t ctrl_ring_bytes = 0u;
    vtlk_shm_t shm_data;
    vtlk_shm_t shm_ctrl;
    vtlk_ring_t ring_data;
    vtlk_ring_t ring_ctrl;
    int data_open = 0;
    int ctrl_open = 0;
    uint64_t *ref_samples = NULL;
    uint64_t *vtlk_samples = NULL;
    size_t kv_payload_bytes = 0u;
    int32_t ref_tok = 0;
    int32_t vtlk_tok = 0;
    int all_match = 1;
    uint32_t run;
    uint32_t total_runs = warmup + iters;
    int rc = 1;

    snprintf(ctrl_name, sizeof(ctrl_name), "%s_c2p", shm_name);

    llama_backend_init();
    mparams = llama_model_default_params();
    model = llama_model_load_from_file(model_path, mparams);
    if (model == NULL) {
        fprintf(stderr, "producer: failed to load model\n");
        goto cleanup;
    }

    vocab = llama_model_get_vocab(model);
    d_model = (uint32_t)llama_model_n_embd(model);
    prompt_text = build_deterministic_prompt(vocab, n_tokens, &tokens, &prompt_bytes);
    if (prompt_text == NULL) {
        goto cleanup;
    }

    cparams = llama_context_default_params();
    cparams.n_ctx = n_ctx;
    cparams.n_batch = n_batch;
    cparams.n_threads = n_threads;
    cparams.n_threads_batch = n_threads;
    cparams.embeddings = false;

    ctx = llama_init_from_model(model, cparams);
    if (ctx == NULL) {
        goto cleanup;
    }

    compat_hash = vtlk_llama_compute_compat_hash(model_path, model, ctx, 1);
    local_build = vtlk_llama_build_stamp();
    if (compat_hash == 0ULL) {
        goto cleanup;
    }

    if (!eval_prompt_chunks(ctx, tokens, n_tokens, n_batch)) {
        goto cleanup;
    }

    state_size = llama_state_seq_get_size(ctx, 0);
    if (state_size == 0u || state_size > (size_t)(UINT32_MAX - VTLK_LLAMA_KV_META_SIZE - 64u)) {
        goto cleanup;
    }

    max_payload_bytes = (state_size > prompt_bytes) ? state_size : prompt_bytes;
    max_payload_bytes = (max_payload_bytes + (size_t)VTLK_LLAMA_KV_META_SIZE + 63u) & ~(size_t)63u;

    data_ring_bytes = vtlk_ring_required_bytes(MP_DATA_RING_CAP, (uint32_t)max_payload_bytes, NULL);
    ctrl_ring_bytes = vtlk_ring_required_bytes(MP_CTRL_RING_CAP, MP_CTRL_PAYLOAD_SZ, NULL);
    if (data_ring_bytes == 0u || ctrl_ring_bytes == 0u) {
        goto cleanup;
    }

    (void)vtlk_shm_unlink(shm_name);
    (void)vtlk_shm_unlink(ctrl_name);
    if (vtlk_shm_create(&shm_data, shm_name, data_ring_bytes) != VTLK_OK) {
        goto cleanup;
    }
    data_open = 1;
    if (vtlk_ring_init(&ring_data, shm_data.addr, shm_data.size,
                       MP_DATA_RING_CAP, (uint32_t)max_payload_bytes) != VTLK_OK) {
        goto cleanup;
    }
    if (vtlk_shm_create(&shm_ctrl, ctrl_name, ctrl_ring_bytes) != VTLK_OK) {
        goto cleanup;
    }
    ctrl_open = 1;
    if (vtlk_ring_init(&ring_ctrl, shm_ctrl.addr, shm_ctrl.size,
                       MP_CTRL_RING_CAP, MP_CTRL_PAYLOAD_SZ) != VTLK_OK) {
        goto cleanup;
    }

    ref_samples = (uint64_t *)malloc((size_t)iters * sizeof(uint64_t));
    vtlk_samples = (uint64_t *)malloc((size_t)iters * sizeof(uint64_t));
    if (!ref_samples || !vtlk_samples) {
        goto cleanup;
    }

    /* Scenario A: Cross-Process Text Re-Prefill */
    for (run = 0u; run < total_runs; ++run) {
        mp_ctrl_msg_t msg;
        uint64_t t0;
        uint64_t t1;
        vtlk_write_slot_t wslot;
        vtlk_header_t hdr;

        if (!ctrl_recv(&ring_ctrl, &msg) || msg.kind != MP_MSG_READY) {
            goto cleanup;
        }

        t0 = vtlk_bench_now_ns();
        for (;;) {
            int r = vtlk_ring_acquire_write(&ring_data, &wslot);
            if (r == (int)VTLK_OK) {
                break;
            }
            if (r == (int)VTLK_ERR_FULL) {
                vtlk_llama_cpu_yield();
                continue;
            }
            goto cleanup;
        }
        memcpy(wslot.payload, prompt_text, prompt_bytes);
        vtlk_header_init_defaults(&hdr);
        hdr.flags = 0u;
        hdr.model_family_hash = compat_hash;
        hdr.d_model = d_model;
        hdr.n_tokens = n_tokens;
        hdr.quant_scheme = (uint8_t)VTLK_QUANT_RAW;
        hdr.codec = (uint8_t)VTLK_CODEC_NONE;
        hdr.raw_bytes = (uint32_t)prompt_bytes;
        hdr.sequence = (uint64_t)run + 1u;
        if (vtlk_ring_commit_write(&ring_data, &wslot, &hdr, prompt_bytes) != (int)VTLK_OK) {
            (void)vtlk_ring_abort_write(&ring_data, &wslot);
            goto cleanup;
        }

        if (!ctrl_recv(&ring_ctrl, &msg) || msg.kind != MP_MSG_DONE) {
            goto cleanup;
        }
        t1 = vtlk_bench_now_ns();
        ref_tok = msg.top1_token;
        if (run >= warmup) {
            ref_samples[run - warmup] = t1 - t0;
        }
    }

    /* Scenario B: Cross-Process Vect-Link SHM Zero-Copy KV-Cache */
    for (run = 0u; run < total_runs; ++run) {
        mp_ctrl_msg_t msg;
        uint64_t t0;
        uint64_t t1;
        vtlk_write_slot_t wslot;
        uint8_t *wbytes;
        size_t state_cap;
        size_t state_written;
        vtlk_llama_kv_meta_t kv_meta;
        vtlk_header_t hdr;

        if (!ctrl_recv(&ring_ctrl, &msg) || msg.kind != MP_MSG_READY) {
            goto cleanup;
        }

        t0 = vtlk_bench_now_ns();
        for (;;) {
            int r = vtlk_ring_acquire_write(&ring_data, &wslot);
            if (r == (int)VTLK_OK) {
                break;
            }
            if (r == (int)VTLK_ERR_FULL) {
                vtlk_llama_cpu_yield();
                continue;
            }
            goto cleanup;
        }
        wbytes = (uint8_t *)wslot.payload;
        state_cap = wslot.capacity - (size_t)VTLK_LLAMA_KV_META_SIZE;
        state_written = llama_state_seq_get_data(
            ctx, wbytes + VTLK_LLAMA_KV_META_SIZE, state_cap, 0);
        if (state_written == 0u || state_written > state_cap) {
            (void)vtlk_ring_abort_write(&ring_data, &wslot);
            goto cleanup;
        }
        kv_meta.magic = VTLK_LLAMA_KV_META_MAGIC;
        kv_meta.llama_build = local_build;
        kv_meta.last_token_id = (int32_t)tokens[n_tokens - 1u];
        kv_meta.raw_state_bytes = (uint32_t)state_written;
        vtlk_llama_kv_meta_encode(&kv_meta, wbytes);

        kv_payload_bytes = (size_t)VTLK_LLAMA_KV_META_SIZE + state_written;
        vtlk_header_init_defaults(&hdr);
        hdr.flags = VTLK_FLAG_KV_STATE;
        hdr.model_family_hash = compat_hash;
        hdr.d_model = d_model;
        hdr.n_tokens = n_tokens;
        hdr.pos_offset = 0u;
        hdr.quant_scheme = (uint8_t)VTLK_QUANT_RAW;
        hdr.codec = (uint8_t)VTLK_CODEC_NONE;
        hdr.raw_bytes = (uint32_t)state_written;
        hdr.sequence = (uint64_t)(total_runs + run + 1u);

        if (vtlk_ring_commit_write(&ring_data, &wslot, &hdr, kv_payload_bytes) != (int)VTLK_OK) {
            (void)vtlk_ring_abort_write(&ring_data, &wslot);
            goto cleanup;
        }

        if (!ctrl_recv(&ring_ctrl, &msg) || msg.kind != MP_MSG_DONE) {
            goto cleanup;
        }
        t1 = vtlk_bench_now_ns();
        vtlk_tok = msg.top1_token;
        if (vtlk_tok != ref_tok) {
            all_match = 0;
        }
        if (run >= warmup) {
            vtlk_samples[run - warmup] = t1 - t0;
        }
    }

    {
        vtlk_bench_stats_t s_ref = vtlk_bench_compute_stats(ref_samples, iters);
        vtlk_bench_stats_t s_vtlk = vtlk_bench_compute_stats(vtlk_samples, iters);
        printf("RESULT tokens=%u kv_bytes=%zu "
               "ref_p50_us=%.2f ref_p95_us=%.2f ref_p99_us=%.2f "
               "vtlk_p50_us=%.2f vtlk_p95_us=%.2f vtlk_p99_us=%.2f "
               "ref_tok=%d vtlk_tok=%d match=%d\n",
               n_tokens, kv_payload_bytes,
               (double)s_ref.p50_ns / 1000.0,
               (double)s_ref.p95_ns / 1000.0,
               (double)s_ref.p99_ns / 1000.0,
               (double)s_vtlk.p50_ns / 1000.0,
               (double)s_vtlk.p95_ns / 1000.0,
               (double)s_vtlk.p99_ns / 1000.0,
               (int)ref_tok, (int)vtlk_tok, all_match);
        fflush(stdout);
    }

    rc = all_match ? 0 : 1;

cleanup:
    free(vtlk_samples);
    free(ref_samples);
    if (ctrl_open) {
        vtlk_shm_close(&shm_ctrl);
        (void)vtlk_shm_unlink(ctrl_name);
    }
    if (data_open) {
        vtlk_shm_close(&shm_data);
        (void)vtlk_shm_unlink(shm_name);
    }
    free(tokens);
    free(prompt_text);
    if (ctx != NULL) {
        llama_free(ctx);
    }
    if (model != NULL) {
        llama_model_free(model);
    }
    llama_backend_free();
    return rc;
}

static int open_ring_with_retry(vtlk_shm_t *shm,
                                vtlk_ring_t *ring,
                                const char *name,
                                size_t max_bytes)
{
    int retries;
    vtlk_status_t st = VTLK_ERR_SHM;
    const uint8_t *ctrl;
    uint32_t cap;
    uint32_t max_pay;
    size_t full_bytes;

    for (retries = 0; retries < 1500; ++retries) {
        st = vtlk_shm_open(shm, name, VTLK_RING_CTRL_SIZE);
        if (st == VTLK_OK) {
            break;
        }
        sleep_ms_brief(20u);
    }
    if (st != VTLK_OK) {
        return 0;
    }

    ctrl = (const uint8_t *)shm->addr;
    cap = vtlk_llama_read_u32_le(ctrl + 12u);
    max_pay = vtlk_llama_read_u32_le(ctrl + 20u);
    full_bytes = vtlk_ring_required_bytes(cap, max_pay, NULL);
    vtlk_shm_close(shm);

    if (full_bytes == 0u || full_bytes > max_bytes) {
        return 0;
    }
    if (vtlk_shm_open(shm, name, full_bytes) != VTLK_OK) {
        return 0;
    }
    if (vtlk_ring_attach(ring, shm->addr, shm->size) != VTLK_OK) {
        vtlk_shm_close(shm);
        return 0;
    }
    return 1;
}

static int run_consumer(const char *model_path,
                        const char *shm_name,
                        uint32_t n_tokens,
                        uint32_t n_ctx,
                        uint32_t n_batch,
                        int32_t n_threads,
                        uint32_t warmup,
                        uint32_t iters)
{
    char ctrl_name[300];
    struct llama_model_params mparams;
    struct llama_model *model = NULL;
    const struct llama_vocab *vocab = NULL;
    struct llama_context_params cparams;
    struct llama_context *ctx = NULL;
    uint32_t d_model = 0u;
    uint64_t compat_hash = 0ULL;
    uint32_t local_build = 0u;
    vtlk_shm_t shm_data;
    vtlk_shm_t shm_ctrl;
    vtlk_ring_t ring_data;
    vtlk_ring_t ring_ctrl;
    int data_open = 0;
    int ctrl_open = 0;
    llama_token *tok_buf = NULL;
    uint32_t total_runs = warmup + iters;
    uint32_t run;
    uint64_t cseq = 1u;
    int rc = 1;

    snprintf(ctrl_name, sizeof(ctrl_name), "%s_c2p", shm_name);

    llama_backend_init();
    mparams = llama_model_default_params();
    model = llama_model_load_from_file(model_path, mparams);
    if (model == NULL) {
        goto cleanup;
    }

    vocab = llama_model_get_vocab(model);
    d_model = (uint32_t)llama_model_n_embd(model);

    cparams = llama_context_default_params();
    cparams.n_ctx = n_ctx;
    cparams.n_batch = n_batch;
    cparams.n_threads = n_threads;
    cparams.n_threads_batch = n_threads;
    cparams.embeddings = false;

    ctx = llama_init_from_model(model, cparams);
    if (ctx == NULL) {
        goto cleanup;
    }

    compat_hash = vtlk_llama_compute_compat_hash(model_path, model, ctx, 1);
    local_build = vtlk_llama_build_stamp();
    if (compat_hash == 0ULL) {
        goto cleanup;
    }

    if (!open_ring_with_retry(&shm_data, &ring_data, shm_name, MP_MAX_SHM_BYTES)) {
        goto cleanup;
    }
    data_open = 1;
    if (!open_ring_with_retry(&shm_ctrl, &ring_ctrl, ctrl_name, MP_MAX_SHM_BYTES)) {
        goto cleanup;
    }
    ctrl_open = 1;

    int32_t tok_cap = (int32_t)(n_tokens * 2u + 256u);
    tok_buf = (llama_token *)malloc((size_t)tok_cap * sizeof(llama_token));
    if (tok_buf == NULL) {
        goto cleanup;
    }

    /* Scenario A: Cross-Process Text Re-Prefill */
    for (run = 0u; run < total_runs; ++run) {
        mp_ctrl_msg_t ready_msg;
        mp_ctrl_msg_t done_msg;
        vtlk_read_slot_t rslot;
        uint64_t t0;
        uint64_t t1;
        int32_t n_tok_out;
        llama_token top1;

        llama_memory_clear(llama_get_memory(ctx), true);
        ready_msg.kind = MP_MSG_READY;
        ready_msg.top1_token = 0;
        ready_msg.receiver_ns = 0u;
        if (!ctrl_send(&ring_ctrl, &ready_msg, cseq++)) {
            goto cleanup;
        }

        for (;;) {
            int r = vtlk_ring_acquire_read(&ring_data, &rslot);
            if (r == (int)VTLK_OK) {
                break;
            }
            if (r == (int)VTLK_ERR_EMPTY) {
                vtlk_llama_cpu_yield();
                continue;
            }
            goto cleanup;
        }

        t0 = vtlk_bench_now_ns();
        n_tok_out = llama_tokenize(vocab, (const char *)rslot.payload, (int32_t)rslot.bytes,
                                   tok_buf, tok_cap, true, false);
        if (n_tok_out < 0) {
            int32_t needed = -n_tok_out;
            llama_token *grown = (llama_token *)realloc(
                tok_buf, (size_t)needed * sizeof(llama_token));
            if (grown == NULL) {
                (void)vtlk_ring_release_read(&ring_data, &rslot);
                goto cleanup;
            }
            tok_buf = grown;
            tok_cap = needed;
            n_tok_out = llama_tokenize(vocab, (const char *)rslot.payload, (int32_t)rslot.bytes,
                                       tok_buf, tok_cap, true, false);
        }
        (void)vtlk_ring_release_read(&ring_data, &rslot);
        if (n_tok_out < (int32_t)n_tokens) {
            goto cleanup;
        }
        if (!eval_prompt_chunks(ctx, tok_buf, n_tokens, n_batch)) {
            goto cleanup;
        }
        top1 = sample_greedy_token(ctx);
        t1 = vtlk_bench_now_ns();

        done_msg.kind = MP_MSG_DONE;
        done_msg.top1_token = (int32_t)top1;
        done_msg.receiver_ns = t1 - t0;
        if (!ctrl_send(&ring_ctrl, &done_msg, cseq++)) {
            goto cleanup;
        }
    }

    /* Scenario B: Cross-Process Vect-Link SHM Zero-Copy KV-Cache */
    for (run = 0u; run < total_runs; ++run) {
        mp_ctrl_msg_t ready_msg;
        mp_ctrl_msg_t done_msg;
        vtlk_read_slot_t rslot;
        vtlk_llama_kv_meta_t kv_meta;
        uint64_t t0;
        uint64_t t1;
        llama_token top1;

        llama_memory_clear(llama_get_memory(ctx), true);
        ready_msg.kind = MP_MSG_READY;
        ready_msg.top1_token = 0;
        ready_msg.receiver_ns = 0u;
        if (!ctrl_send(&ring_ctrl, &ready_msg, cseq++)) {
            goto cleanup;
        }

        for (;;) {
            int r = vtlk_ring_acquire_read(&ring_data, &rslot);
            if (r == (int)VTLK_OK) {
                break;
            }
            if (r == (int)VTLK_ERR_EMPTY) {
                vtlk_llama_cpu_yield();
                continue;
            }
            goto cleanup;
        }

        t0 = vtlk_bench_now_ns();
        if (rslot.header.model_family_hash != compat_hash ||
            rslot.header.d_model != d_model ||
            (rslot.header.flags & VTLK_FLAG_KV_STATE) == 0u ||
            rslot.header.quant_scheme != (uint8_t)VTLK_QUANT_RAW ||
            rslot.header.pos_offset != 0u ||
            rslot.bytes < VTLK_LLAMA_KV_META_SIZE) {
            (void)vtlk_ring_release_read(&ring_data, &rslot);
            goto cleanup;
        }
        vtlk_llama_kv_meta_decode((const uint8_t *)rslot.payload, &kv_meta);
        if (kv_meta.magic != VTLK_LLAMA_KV_META_MAGIC ||
            kv_meta.llama_build != local_build ||
            rslot.bytes != sizeof(vtlk_llama_kv_meta_t) + (size_t)kv_meta.raw_state_bytes) {
            (void)vtlk_ring_release_read(&ring_data, &rslot);
            goto cleanup;
        }
        if (!restore_kv_and_decode_last(ctx,
                                        (const uint8_t *)rslot.payload + VTLK_LLAMA_KV_META_SIZE,
                                        (size_t)kv_meta.raw_state_bytes,
                                        (llama_token)kv_meta.last_token_id,
                                        n_tokens)) {
            (void)vtlk_ring_release_read(&ring_data, &rslot);
            goto cleanup;
        }
        if (vtlk_ring_release_read(&ring_data, &rslot) != (int)VTLK_OK) {
            goto cleanup;
        }
        top1 = sample_greedy_token(ctx);
        t1 = vtlk_bench_now_ns();

        done_msg.kind = MP_MSG_DONE;
        done_msg.top1_token = (int32_t)top1;
        done_msg.receiver_ns = t1 - t0;
        if (!ctrl_send(&ring_ctrl, &done_msg, cseq++)) {
            goto cleanup;
        }
    }

    rc = 0;

cleanup:
    free(tok_buf);
    if (ctrl_open) {
        vtlk_shm_close(&shm_ctrl);
    }
    if (data_open) {
        vtlk_shm_close(&shm_data);
    }
    if (ctx != NULL) {
        llama_free(ctx);
    }
    if (model != NULL) {
        llama_model_free(model);
    }
    llama_backend_free();
    return rc;
}

static int parse_lengths_csv(const char *csv, uint32_t *out, size_t max_n, size_t *out_n)
{
    const char *p = csv;
    size_t count = 0u;
    while (*p != '\0') {
        char *end = NULL;
        unsigned long val = strtoul(p, &end, 10);
        if (end == p || val < 2ul || count >= max_n) {
            return 0;
        }
        out[count++] = (uint32_t)val;
        if (*end == ',') {
            p = end + 1;
        } else if (*end == '\0') {
            break;
        } else {
            return 0;
        }
    }
    *out_n = count;
    return (count > 0u);
}

static int run_child_pair(const char *exe_path,
                          const char *model_path,
                          const char *shm_name,
                          uint32_t n_tokens,
                          uint32_t n_ctx,
                          uint32_t n_batch,
                          int32_t n_threads,
                          uint32_t warmup,
                          uint32_t iters,
                          mp_length_summary_t *summary)
{
    char prod_err_log[128];
    char cons_err_log[128];
    char out_buf[2048];
    int ok = 0;

    snprintf(prod_err_log, sizeof(prod_err_log), "vtlk_mp_prod_%u.log", n_tokens);
    snprintf(cons_err_log, sizeof(cons_err_log), "vtlk_mp_cons_%u.log", n_tokens);
    memset(out_buf, 0, sizeof(out_buf));

#ifdef _WIN32
    {
        SECURITY_ATTRIBUTES sa;
        HANDLE hReadPipe = NULL;
        HANDLE hWritePipe = NULL;
        HANDLE hProdErr = INVALID_HANDLE_VALUE;
        HANDLE hConsErr = INVALID_HANDLE_VALUE;
        HANDLE hNulOut = INVALID_HANDLE_VALUE;
        PROCESS_INFORMATION piProd;
        PROCESS_INFORMATION piCons;
        STARTUPINFOA siProd;
        STARTUPINFOA siCons;
        char cmdProd[2048];
        char cmdCons[2048];
        DWORD total_read = 0;
        DWORD nread = 0;
        DWORD ecProd = 1;
        DWORD ecCons = 1;

        memset(&piProd, 0, sizeof(piProd));
        memset(&piCons, 0, sizeof(piCons));
        memset(&siProd, 0, sizeof(siProd));
        memset(&siCons, 0, sizeof(siCons));

        sa.nLength = sizeof(sa);
        sa.lpSecurityDescriptor = NULL;
        sa.bInheritHandle = TRUE;

        if (!CreatePipe(&hReadPipe, &hWritePipe, &sa, 0)) {
            return 0;
        }
        SetHandleInformation(hReadPipe, HANDLE_FLAG_INHERIT, 0);

        hProdErr = CreateFileA(prod_err_log, GENERIC_WRITE, FILE_SHARE_READ,
                               &sa, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
        hConsErr = CreateFileA(cons_err_log, GENERIC_WRITE, FILE_SHARE_READ,
                               &sa, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
        hNulOut = CreateFileA("NUL", GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE,
                              &sa, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
        if (hProdErr == INVALID_HANDLE_VALUE || hConsErr == INVALID_HANDLE_VALUE ||
            hNulOut == INVALID_HANDLE_VALUE) {
            goto win_cleanup;
        }

        snprintf(cmdProd, sizeof(cmdProd),
                 "\"%s\" --role producer --model \"%s\" --shm \"%s\" "
                 "--tokens %u --n-ctx %u --n-batch %u --threads %d --warmup %u --iters %u",
                 exe_path, model_path, shm_name, n_tokens, n_ctx, n_batch, n_threads, warmup, iters);
        snprintf(cmdCons, sizeof(cmdCons),
                 "\"%s\" --role consumer --model \"%s\" --shm \"%s\" "
                 "--tokens %u --n-ctx %u --n-batch %u --threads %d --warmup %u --iters %u",
                 exe_path, model_path, shm_name, n_tokens, n_ctx, n_batch, n_threads, warmup, iters);

        siProd.cb = sizeof(siProd);
        siProd.dwFlags = STARTF_USESTDHANDLES;
        siProd.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
        siProd.hStdOutput = hWritePipe;
        siProd.hStdError = hProdErr;

        siCons.cb = sizeof(siCons);
        siCons.dwFlags = STARTF_USESTDHANDLES;
        siCons.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
        siCons.hStdOutput = hNulOut;
        siCons.hStdError = hConsErr;

        if (!CreateProcessA(NULL, cmdProd, NULL, NULL, TRUE, 0, NULL, NULL, &siProd, &piProd)) {
            goto win_cleanup;
        }
        if (!CreateProcessA(NULL, cmdCons, NULL, NULL, TRUE, 0, NULL, NULL, &siCons, &piCons)) {
            TerminateProcess(piProd.hProcess, 1);
            goto win_cleanup;
        }

        CloseHandle(hWritePipe); hWritePipe = NULL;
        CloseHandle(hProdErr); hProdErr = INVALID_HANDLE_VALUE;
        CloseHandle(hConsErr); hConsErr = INVALID_HANDLE_VALUE;
        CloseHandle(hNulOut); hNulOut = INVALID_HANDLE_VALUE;

        while (total_read + 1u < sizeof(out_buf) &&
               ReadFile(hReadPipe, out_buf + total_read,
                        (DWORD)(sizeof(out_buf) - 1u - total_read), &nread, NULL) &&
               nread > 0) {
            total_read += nread;
        }
        out_buf[total_read] = '\0';

        WaitForSingleObject(piProd.hProcess, INFINITE);
        WaitForSingleObject(piCons.hProcess, INFINITE);
        GetExitCodeProcess(piProd.hProcess, &ecProd);
        GetExitCodeProcess(piCons.hProcess, &ecCons);
        ok = (ecProd == 0 && ecCons == 0);

    win_cleanup:
        if (piProd.hThread) CloseHandle(piProd.hThread);
        if (piProd.hProcess) CloseHandle(piProd.hProcess);
        if (piCons.hThread) CloseHandle(piCons.hThread);
        if (piCons.hProcess) CloseHandle(piCons.hProcess);
        if (hReadPipe) CloseHandle(hReadPipe);
        if (hWritePipe) CloseHandle(hWritePipe);
        if (hProdErr != INVALID_HANDLE_VALUE) CloseHandle(hProdErr);
        if (hConsErr != INVALID_HANDLE_VALUE) CloseHandle(hConsErr);
        if (hNulOut != INVALID_HANDLE_VALUE) CloseHandle(hNulOut);
    }
#else
    {
        int pipefd[2];
        pid_t pid_prod = -1;
        pid_t pid_cons = -1;
        char tok_s[32], ctx_s[32], batch_s[32], thr_s[32], warm_s[32], iter_s[32];
        int st_p = 1, st_c = 1;
        size_t total_read = 0u;
        ssize_t nr;

        if (pipe(pipefd) != 0) {
            return 0;
        }
        snprintf(tok_s, sizeof(tok_s), "%u", n_tokens);
        snprintf(ctx_s, sizeof(ctx_s), "%u", n_ctx);
        snprintf(batch_s, sizeof(batch_s), "%u", n_batch);
        snprintf(thr_s, sizeof(thr_s), "%d", n_threads);
        snprintf(warm_s, sizeof(warm_s), "%u", warmup);
        snprintf(iter_s, sizeof(iter_s), "%u", iters);

        pid_prod = fork();
        if (pid_prod == 0) {
            int err_fd = open(prod_err_log, O_WRONLY | O_CREAT | O_TRUNC, 0644);
            const char *args[] = {
                exe_path, "--role", "producer", "--model", model_path, "--shm", shm_name,
                "--tokens", tok_s, "--n-ctx", ctx_s, "--n-batch", batch_s,
                "--threads", thr_s, "--warmup", warm_s, "--iters", iter_s, NULL
            };
            close(pipefd[0]);
            dup2(pipefd[1], STDOUT_FILENO);
            close(pipefd[1]);
            if (err_fd >= 0) {
                dup2(err_fd, STDERR_FILENO);
                close(err_fd);
            }
            execvp(exe_path, (char *const *)args);
            _exit(127);
        }

        pid_cons = fork();
        if (pid_cons == 0) {
            int nul_fd = open("/dev/null", O_WRONLY);
            int err_fd = open(cons_err_log, O_WRONLY | O_CREAT | O_TRUNC, 0644);
            const char *args[] = {
                exe_path, "--role", "consumer", "--model", model_path, "--shm", shm_name,
                "--tokens", tok_s, "--n-ctx", ctx_s, "--n-batch", batch_s,
                "--threads", thr_s, "--warmup", warm_s, "--iters", iter_s, NULL
            };
            close(pipefd[0]);
            close(pipefd[1]);
            if (nul_fd >= 0) {
                dup2(nul_fd, STDOUT_FILENO);
                close(nul_fd);
            }
            if (err_fd >= 0) {
                dup2(err_fd, STDERR_FILENO);
                close(err_fd);
            }
            execvp(exe_path, (char *const *)args);
            _exit(127);
        }

        close(pipefd[1]);
        while (total_read + 1u < sizeof(out_buf) &&
               (nr = read(pipefd[0], out_buf + total_read, sizeof(out_buf) - 1u - total_read)) > 0) {
            total_read += (size_t)nr;
        }
        out_buf[total_read] = '\0';
        close(pipefd[0]);

        if (pid_prod > 0) waitpid(pid_prod, &st_p, 0);
        if (pid_cons > 0) waitpid(pid_cons, &st_c, 0);
        ok = WIFEXITED(st_p) && WEXITSTATUS(st_p) == 0 &&
             WIFEXITED(st_c) && WEXITSTATUS(st_c) == 0;
    }
#endif

    (void)remove(prod_err_log);
    (void)remove(cons_err_log);

    if (!ok) {
        return 0;
    }

    {
        const char *line = strstr(out_buf, "RESULT ");
        unsigned int tok_u = 0u;
        unsigned long long kv_b = 0ULL;
        if (line == NULL) {
            return 0;
        }
        if (sscanf(line,
                   "RESULT tokens=%u kv_bytes=%llu "
                   "ref_p50_us=%lf ref_p95_us=%lf ref_p99_us=%lf "
                   "vtlk_p50_us=%lf vtlk_p95_us=%lf vtlk_p99_us=%lf "
                   "ref_tok=%d vtlk_tok=%d match=%d",
                   &tok_u, &kv_b,
                   &summary->ref_p50_us, &summary->ref_p95_us, &summary->ref_p99_us,
                   &summary->vtlk_p50_us, &summary->vtlk_p95_us, &summary->vtlk_p99_us,
                   &summary->ref_tok, &summary->vtlk_tok, &summary->match) != 11) {
            return 0;
        }
        summary->tokens = (uint32_t)tok_u;
        summary->kv_bytes = (size_t)kv_b;
    }
    return 1;
}

int main(int argc, char **argv)
{
    mp_role_t role = MP_ROLE_RUNNER;
    const char *model_path = NULL;
    const char *lengths_str = MP_DEFAULT_LENGTHS;
    const char *shm_name = MP_DEFAULT_SHM;
    const char *csv_path = MP_DEFAULT_CSV;
    uint32_t single_tokens = 512u;
    uint32_t n_ctx = 0u;
    uint32_t n_batch = MP_DEFAULT_N_BATCH;
    int32_t n_threads = MP_DEFAULT_THREADS;
    uint32_t warmup = MP_DEFAULT_WARMUP;
    uint32_t iters = MP_DEFAULT_ITERS;
    int i;

    for (i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "--role") == 0 && i + 1 < argc) {
            const char *r = argv[++i];
            if (strcmp(r, "runner") == 0) role = MP_ROLE_RUNNER;
            else if (strcmp(r, "producer") == 0) role = MP_ROLE_PRODUCER;
            else if (strcmp(r, "consumer") == 0) role = MP_ROLE_CONSUMER;
            else { print_usage(argv[0]); return 1; }
        } else if (strcmp(argv[i], "--lengths") == 0 && i + 1 < argc) {
            lengths_str = argv[++i];
        } else if (strcmp(argv[i], "--tokens") == 0 && i + 1 < argc) {
            single_tokens = (uint32_t)strtoul(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--n-ctx") == 0 && i + 1 < argc) {
            n_ctx = (uint32_t)strtoul(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--n-batch") == 0 && i + 1 < argc) {
            n_batch = (uint32_t)strtoul(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            n_threads = (int32_t)strtol(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
            warmup = (uint32_t)strtoul(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
            iters = (uint32_t)strtoul(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--shm") == 0 && i + 1 < argc) {
            shm_name = argv[++i];
        } else if (strcmp(argv[i], "--csv") == 0 && i + 1 < argc) {
            csv_path = argv[++i];
        } else {
            print_usage(argv[0]);
            return 1;
        }
    }

    if (model_path == NULL || iters == 0u || n_threads <= 0) {
        print_usage(argv[0]);
        return 1;
    }

    if (role == MP_ROLE_PRODUCER) {
        uint32_t ctx_sz = (n_ctx > 0u) ? n_ctx : (single_tokens + 256u);
        return run_producer(model_path, shm_name, single_tokens, ctx_sz, n_batch,
                            n_threads, warmup, iters);
    }
    if (role == MP_ROLE_CONSUMER) {
        uint32_t ctx_sz = (n_ctx > 0u) ? n_ctx : (single_tokens + 256u);
        return run_consumer(model_path, shm_name, single_tokens, ctx_sz, n_batch,
                            n_threads, warmup, iters);
    }

    /* Runner orchestrator */
    {
        uint32_t lengths[MP_MAX_LENGTHS];
        size_t n_lengths = 0u;
        uint32_t max_len = 0u;
        mp_length_summary_t summaries[MP_MAX_LENGTHS];
        char exe_path[1024];
        size_t li;
        int all_ok = 1;

        if (!parse_lengths_csv(lengths_str, lengths, MP_MAX_LENGTHS, &n_lengths)) {
            fprintf(stderr, "benchmark_llama_multiprocess: invalid --lengths '%s'\n", lengths_str);
            return 1;
        }
        for (li = 0u; li < n_lengths; ++li) {
            if (lengths[li] > max_len) {
                max_len = lengths[li];
            }
        }
        if (n_ctx == 0u) {
            n_ctx = max_len + 256u;
        }

#ifdef _WIN32
        if (GetModuleFileNameA(NULL, exe_path, (DWORD)sizeof(exe_path)) == 0) {
            snprintf(exe_path, sizeof(exe_path), "%s", argv[0]);
        }
#else
        snprintf(exe_path, sizeof(exe_path), "%s", argv[0]);
#endif

        printf("\n=== Vect-Link v0.9.0 Two-Process OS Handoff Benchmark ===\n");
        printf("Model: %s\n", model_path);
        printf("n_ctx: %u | n_batch: %u | threads: %d | warmup: %u | iters: %u\n\n",
               n_ctx, n_batch, n_threads, warmup, iters);
        fflush(stdout);

        for (li = 0u; li < n_lengths; ++li) {
            char child_shm[256];
            snprintf(child_shm, sizeof(child_shm), "%s_%u", shm_name, lengths[li]);
            if (!run_child_pair(exe_path, model_path, child_shm,
                                lengths[li], n_ctx, n_batch, n_threads,
                                warmup, iters, &summaries[li])) {
                fprintf(stderr, "benchmark_llama_multiprocess: child pair failed for %u tokens\n",
                        lengths[li]);
                return 1;
            }
            if (!summaries[li].match) {
                all_ok = 0;
            }
        }

        printf("%-6s | %10s | %25s | %25s | %11s | %11s | %s\n",
               "Tokens", "KV Bytes", "Re-Prefill p50/p95/p99(ms)",
               "Vect-Link p50/p95/p99(ms)", "Speedup p50", "Speedup p99", "Top-1 Match");
        printf("-------|------------|---------------------------|---------------------------|-------------|-------------|------------\n");

        for (li = 0u; li < n_lengths; ++li) {
            const mp_length_summary_t *s = &summaries[li];
            double sp50 = (s->vtlk_p50_us > 0.0) ? (s->ref_p50_us / s->vtlk_p50_us) : 0.0;
            double sp99 = (s->vtlk_p99_us > 0.0) ? (s->ref_p99_us / s->vtlk_p99_us) : 0.0;
            printf("%6u | %10zu | %7.2f / %7.2f / %7.2f | %7.2f / %7.2f / %7.2f | %10.2fx | %10.2fx | %s (%d)\n",
                   s->tokens, s->kv_bytes,
                   s->ref_p50_us / 1000.0, s->ref_p95_us / 1000.0, s->ref_p99_us / 1000.0,
                   s->vtlk_p50_us / 1000.0, s->vtlk_p95_us / 1000.0, s->vtlk_p99_us / 1000.0,
                   sp50, sp99,
                   s->match ? "EXACT" : "MISMATCH", s->vtlk_tok);
        }

        {
            FILE *csv = fopen(csv_path, "w");
            if (csv != NULL) {
                fprintf(csv, "tokens,kv_bytes,ref_p50_ms,ref_p95_ms,ref_p99_ms,vtlk_p50_ms,vtlk_p95_ms,vtlk_p99_ms,speedup_p50,speedup_p99,top1_token,top1_match\n");
                for (li = 0u; li < n_lengths; ++li) {
                    const mp_length_summary_t *s = &summaries[li];
                    double sp50 = (s->vtlk_p50_us > 0.0) ? (s->ref_p50_us / s->vtlk_p50_us) : 0.0;
                    double sp99 = (s->vtlk_p99_us > 0.0) ? (s->ref_p99_us / s->vtlk_p99_us) : 0.0;
                    fprintf(csv, "%u,%zu,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.2f,%.2f,%d,%s\n",
                            s->tokens, s->kv_bytes,
                            s->ref_p50_us / 1000.0, s->ref_p95_us / 1000.0, s->ref_p99_us / 1000.0,
                            s->vtlk_p50_us / 1000.0, s->vtlk_p95_us / 1000.0, s->vtlk_p99_us / 1000.0,
                            sp50, sp99, s->vtlk_tok, s->match ? "EXACT" : "MISMATCH");
                }
                fclose(csv);
                printf("\nExported CSV: %s\n", csv_path);
            }
        }

        return all_ok ? 0 : 1;
    }
}
