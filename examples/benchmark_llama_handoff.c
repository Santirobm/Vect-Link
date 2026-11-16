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

#define DEFAULT_TOKENS    512u
#define DEFAULT_N_CTX     1024u
#define DEFAULT_N_BATCH   512u
#define DEFAULT_THREADS   4
#define DEFAULT_WARMUP    2u
#define DEFAULT_ITERS     10u
#define DEFAULT_BENCH_SHM "/vtlk_llama_handoff_bench"
#define DEFAULT_CSV_PATH  "llama_handoff_benchmark.csv"
#define DEFAULT_DISK_PATH "vtlk_kv_bench_tmp.bin"
#define KV_BENCH_RING_CAP 2u

typedef struct handoff_result {
    const char *name;
    size_t payload_bytes;
    vtlk_bench_stats_t sender;
    vtlk_bench_stats_t receiver;
    vtlk_bench_stats_t total;
    llama_token top1_token;
    int top1_match;
} handoff_result_t;

static void print_usage(const char *prog)
{
    fprintf(stderr,
            "Usage: %s --model <path.gguf> [--tokens 512] [--threads 4] "
            "[--iters 10] [--n-ctx 1024] [--n-batch 512] "
            "[--shm /vtlk_llama_handoff_bench] [--csv llama_handoff_benchmark.csv]\n",
            prog);
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

static int restore_kv_and_decode_last(struct llama_context *ctx_b,
                                        const uint8_t *raw_state,
                                        size_t state_bytes,
                                        llama_token last_tok,
                                        uint32_t n_tokens)
{
    size_t restored = llama_state_seq_set_data(ctx_b, raw_state, state_bytes, 0);
    llama_pos last_pos = (llama_pos)(n_tokens - 1u);
    struct llama_batch bridge_batch;

    if (restored == 0u || restored != state_bytes) {
        fprintf(stderr, "benchmark_llama_handoff: llama_state_seq_set_data restored %zu != %zu\n",
                restored, state_bytes);
        return 0;
    }
    if (!llama_memory_seq_rm(llama_get_memory(ctx_b), 0, last_pos, -1)) {
        fprintf(stderr, "benchmark_llama_handoff: llama_memory_seq_rm failed\n");
        return 0;
    }

    bridge_batch = llama_batch_init(1, 0, 1);
    bridge_batch.n_tokens = 1;
    bridge_batch.token[0] = last_tok;
    bridge_batch.pos[0] = last_pos;
    bridge_batch.n_seq_id[0] = 1;
    bridge_batch.seq_id[0][0] = 0;
    bridge_batch.logits[0] = 1;
    if (llama_decode(ctx_b, bridge_batch) != 0) {
        llama_batch_free(bridge_batch);
        fprintf(stderr, "benchmark_llama_handoff: bridge llama_decode failed\n");
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
                                        uint32_t *out_n_tokens,
                                        size_t *out_text_bytes)
{
    static const char k_doc_block[] =
        "Vect-Link provides deterministic little-endian framing, Castagnoli CRC32C "
        "verification, and lock-free single-producer single-consumer shared-memory "
        "ring buffers for inter-process tensor and KV-cache handoff without JSON "
        "serialization or kernel socket overhead. ";
    size_t block_len = strlen(k_doc_block);
    size_t cap = block_len * 32u + 1u;
    char *text = (char *)malloc(cap);
    size_t text_len = 0u;
    llama_token *toks = NULL;
    int32_t n_tok = 0;

    if (text == NULL) {
        return NULL;
    }
    text[0] = '\0';

    for (;;) {
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
        text[text_len] = '\0';

        n_tok = -llama_tokenize(vocab, text, (int32_t)text_len, NULL, 0, true, false);
        if (n_tok >= (int32_t)target_tokens) {
            break;
        }
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
    *out_n_tokens = target_tokens;
    *out_text_bytes = text_len;
    return text;
}

int main(int argc, char **argv)
{
    const char *model_path = NULL;
    const char *shm_name = DEFAULT_BENCH_SHM;
    const char *csv_path = DEFAULT_CSV_PATH;
    uint32_t target_tokens = DEFAULT_TOKENS;
    uint32_t n_ctx = DEFAULT_N_CTX;
    uint32_t n_batch = DEFAULT_N_BATCH;
    int32_t n_threads = DEFAULT_THREADS;
    uint32_t warmup = DEFAULT_WARMUP;
    uint32_t iters = DEFAULT_ITERS;
    int i;

    struct llama_model_params mparams;
    struct llama_model *model = NULL;
    const struct llama_vocab *vocab = NULL;
    struct llama_context_params cparams;
    struct llama_context *ctx_a = NULL;
    struct llama_context *ctx_b = NULL;

    char *prompt_text = NULL;
    llama_token *tokens = NULL;
    uint32_t n_tokens = 0u;
    size_t prompt_bytes = 0u;
    uint32_t d_model = 0u;
    uint64_t sender_hash = 0ULL;
    uint64_t receiver_hash = 0ULL;
    uint32_t local_build = 0u;

    size_t state_size = 0u;
    size_t max_payload_bytes = 0u;
    size_t ring_bytes = 0u;
    vtlk_shm_t shm_a;
    vtlk_shm_t shm_b;
    vtlk_ring_t ring_a;
    vtlk_ring_t ring_b;
    int shm_a_open = 0;
    int shm_b_open = 0;

    uint8_t *disk_tx_buf = NULL;
    uint8_t *disk_rx_buf = NULL;
    uint8_t *vtlk_tx_buf = NULL;
    uint8_t *vtlk_rx_buf = NULL;

    uint64_t *s_samples = NULL;
    uint64_t *r_samples = NULL;
    uint64_t *t_samples = NULL;
    handoff_result_t results[3];
    llama_token baseline_tok = 0;
    uint32_t run;
    int rc = 1;

    for (i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "--tokens") == 0 && i + 1 < argc) {
            target_tokens = (uint32_t)strtoul(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            n_threads = (int32_t)strtol(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
            iters = (uint32_t)strtoul(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--n-ctx") == 0 && i + 1 < argc) {
            n_ctx = (uint32_t)strtoul(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--n-batch") == 0 && i + 1 < argc) {
            n_batch = (uint32_t)strtoul(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--shm") == 0 && i + 1 < argc) {
            shm_name = argv[++i];
        } else if (strcmp(argv[i], "--csv") == 0 && i + 1 < argc) {
            csv_path = argv[++i];
        } else {
            print_usage(argv[0]);
            return 1;
        }
    }

    if (model_path == NULL || target_tokens < 2u || iters == 0u || n_threads <= 0) {
        print_usage(argv[0]);
        return 1;
    }
    if (n_ctx < target_tokens + 64u) {
        n_ctx = target_tokens + 64u;
    }

    llama_backend_init();
    mparams = llama_model_default_params();
    model = llama_model_load_from_file(model_path, mparams);
    if (model == NULL) {
        fprintf(stderr, "benchmark_llama_handoff: failed to load model '%s'\n", model_path);
        goto cleanup;
    }

    vocab = llama_model_get_vocab(model);
    d_model = (uint32_t)llama_model_n_embd(model);
    prompt_text = build_deterministic_prompt(vocab, target_tokens, &tokens,
                                             &n_tokens, &prompt_bytes);
    if (prompt_text == NULL) {
        fprintf(stderr, "benchmark_llama_handoff: failed to build prompt\n");
        goto cleanup;
    }

    cparams = llama_context_default_params();
    cparams.n_ctx = n_ctx;
    cparams.n_batch = n_batch;
    cparams.n_threads = n_threads;
    cparams.n_threads_batch = n_threads;
    cparams.embeddings = false;

    ctx_a = llama_init_from_model(model, cparams);
    ctx_b = llama_init_from_model(model, cparams);
    if (ctx_a == NULL || ctx_b == NULL) {
        fprintf(stderr, "benchmark_llama_handoff: failed to initialize contexts\n");
        goto cleanup;
    }

    sender_hash = vtlk_llama_compute_compat_hash(model_path, model, ctx_a, 1);
    receiver_hash = vtlk_llama_compute_compat_hash(model_path, model, ctx_b, 1);
    local_build = vtlk_llama_build_stamp();
    if (sender_hash == 0ULL || sender_hash != receiver_hash) {
        fprintf(stderr, "benchmark_llama_handoff: compat hash mismatch between ctx_a and ctx_b\n");
        goto cleanup;
    }

    /* Pre-evaluate prompt on ctx_a (representing Agent A finished prefill) */
    if (!eval_prompt_chunks(ctx_a, tokens, n_tokens, n_batch)) {
        fprintf(stderr, "benchmark_llama_handoff: prefill on ctx_a failed\n");
        goto cleanup;
    }

    state_size = llama_state_seq_get_size(ctx_a, 0);
    if (state_size == 0u || state_size > (size_t)(UINT32_MAX - VTLK_LLAMA_KV_META_SIZE - 64u)) {
        fprintf(stderr, "benchmark_llama_handoff: invalid state_size (%zu)\n", state_size);
        goto cleanup;
    }

    max_payload_bytes = ((size_t)VTLK_LLAMA_KV_META_SIZE + state_size + 63u) & ~(size_t)63u;
    ring_bytes = vtlk_ring_required_bytes(KV_BENCH_RING_CAP, (uint32_t)max_payload_bytes, NULL);
    if (ring_bytes == 0u) {
        fprintf(stderr, "benchmark_llama_handoff: ring_bytes overflow\n");
        goto cleanup;
    }

    (void)vtlk_shm_unlink(shm_name);
    if (vtlk_shm_create(&shm_a, shm_name, ring_bytes) != VTLK_OK) {
        fprintf(stderr, "benchmark_llama_handoff: vtlk_shm_create failed\n");
        goto cleanup;
    }
    shm_a_open = 1;
    if (vtlk_ring_init(&ring_a, shm_a.addr, shm_a.size, KV_BENCH_RING_CAP,
                       (uint32_t)max_payload_bytes) != VTLK_OK) {
        fprintf(stderr, "benchmark_llama_handoff: vtlk_ring_init failed\n");
        goto cleanup;
    }
    if (vtlk_shm_open(&shm_b, shm_name, ring_bytes) != VTLK_OK) {
        fprintf(stderr, "benchmark_llama_handoff: vtlk_shm_open failed\n");
        goto cleanup;
    }
    shm_b_open = 1;
    if (vtlk_ring_attach(&ring_b, shm_b.addr, shm_b.size) != VTLK_OK) {
        fprintf(stderr, "benchmark_llama_handoff: vtlk_ring_attach failed\n");
        goto cleanup;
    }

    disk_tx_buf = (uint8_t *)malloc(state_size);
    disk_rx_buf = (uint8_t *)malloc(state_size);
    vtlk_tx_buf = (uint8_t *)malloc(max_payload_bytes);
    vtlk_rx_buf = (uint8_t *)malloc(max_payload_bytes);
    s_samples = (uint64_t *)malloc((size_t)iters * sizeof(uint64_t));
    r_samples = (uint64_t *)malloc((size_t)iters * sizeof(uint64_t));
    t_samples = (uint64_t *)malloc((size_t)iters * sizeof(uint64_t));
    if (!disk_tx_buf || !disk_rx_buf || !vtlk_tx_buf || !vtlk_rx_buf ||
        !s_samples || !r_samples || !t_samples) {
        fprintf(stderr, "benchmark_llama_handoff: OOM\n");
        goto cleanup;
    }

    /* Scenario 1: Text Re-Prefill */
    for (run = 0u; run < warmup + iters; ++run) {
        uint64_t t0;
        uint64_t t1;
        llama_memory_clear(llama_get_memory(ctx_b), true);
        t0 = vtlk_bench_now_ns();
        if (!eval_prompt_chunks(ctx_b, tokens, n_tokens, n_batch)) {
            goto cleanup;
        }
        t1 = vtlk_bench_now_ns();
        baseline_tok = sample_greedy_token(ctx_b);
        if (run >= warmup) {
            uint32_t idx = run - warmup;
            s_samples[idx] = 0u;
            r_samples[idx] = t1 - t0;
            t_samples[idx] = t1 - t0;
        }
    }
    results[0].name = "Text Re-Prefill (no state)";
    results[0].payload_bytes = prompt_bytes;
    results[0].sender = vtlk_bench_compute_stats(s_samples, iters);
    results[0].receiver = vtlk_bench_compute_stats(r_samples, iters);
    results[0].total = vtlk_bench_compute_stats(t_samples, iters);
    results[0].top1_token = baseline_tok;
    results[0].top1_match = 1;

    /* Scenario 2: KV-Cache via Disk (stdio) */
    {
        size_t disk_bytes = 0u;
        llama_token tok_disk = 0;
        int match_disk = 1;
        for (run = 0u; run < warmup + iters; ++run) {
            uint64_t ts0, ts1, tr0, tr1;
            FILE *fp;
            size_t got;

            llama_memory_clear(llama_get_memory(ctx_b), true);

            ts0 = vtlk_bench_now_ns();
            disk_bytes = llama_state_seq_get_data(ctx_a, disk_tx_buf, state_size, 0);
            if (disk_bytes == 0u || disk_bytes > state_size) {
                goto cleanup;
            }
            fp = fopen(DEFAULT_DISK_PATH, "wb");
            if (fp == NULL) {
                goto cleanup;
            }
            if (fwrite(disk_tx_buf, 1u, disk_bytes, fp) != disk_bytes) {
                fclose(fp);
                goto cleanup;
            }
            fflush(fp);
            fclose(fp);
            ts1 = vtlk_bench_now_ns();

            tr0 = vtlk_bench_now_ns();
            fp = fopen(DEFAULT_DISK_PATH, "rb");
            if (fp == NULL) {
                goto cleanup;
            }
            got = fread(disk_rx_buf, 1u, disk_bytes, fp);
            fclose(fp);
            if (got != disk_bytes ||
                !restore_kv_and_decode_last(ctx_b, disk_rx_buf, disk_bytes,
                                            tokens[n_tokens - 1u], n_tokens)) {
                goto cleanup;
            }
            tr1 = vtlk_bench_now_ns();

            tok_disk = sample_greedy_token(ctx_b);
            if (tok_disk != baseline_tok) {
                match_disk = 0;
            }
            if (run >= warmup) {
                uint32_t idx = run - warmup;
                s_samples[idx] = ts1 - ts0;
                r_samples[idx] = tr1 - tr0;
                t_samples[idx] = (ts1 - ts0) + (tr1 - tr0);
            }
        }
        results[1].name = "KV-Cache Disk (stdio)";
        results[1].payload_bytes = disk_bytes;
        results[1].sender = vtlk_bench_compute_stats(s_samples, iters);
        results[1].receiver = vtlk_bench_compute_stats(r_samples, iters);
        results[1].total = vtlk_bench_compute_stats(t_samples, iters);
        results[1].top1_token = tok_disk;
        results[1].top1_match = match_disk;
    }

    /* Scenario 3: Vect-Link (VTLK_QUANT_RAW + SPSC SHM + CRC32C) */
    {
        size_t vtlk_payload_bytes = 0u;
        llama_token tok_vtlk = 0;
        int match_vtlk = 1;
        for (run = 0u; run < warmup + iters; ++run) {
            uint64_t ts0, ts1, tr0, tr1;
            size_t state_cap = max_payload_bytes - (size_t)VTLK_LLAMA_KV_META_SIZE;
            size_t state_written;
            vtlk_llama_kv_meta_t tx_meta;
            vtlk_llama_kv_meta_t rx_meta;
            vtlk_header_t tx_hdr;
            vtlk_header_t rx_hdr;

            llama_memory_clear(llama_get_memory(ctx_b), true);

            ts0 = vtlk_bench_now_ns();
            state_written = llama_state_seq_get_data(
                ctx_a, vtlk_tx_buf + VTLK_LLAMA_KV_META_SIZE, state_cap, 0);
            if (state_written == 0u || state_written > state_cap) {
                goto cleanup;
            }
            tx_meta.magic = VTLK_LLAMA_KV_META_MAGIC;
            tx_meta.llama_build = local_build;
            tx_meta.last_token_id = (int32_t)tokens[n_tokens - 1u];
            tx_meta.raw_state_bytes = (uint32_t)state_written;
            vtlk_llama_kv_meta_encode(&tx_meta, vtlk_tx_buf);

            vtlk_payload_bytes = (size_t)VTLK_LLAMA_KV_META_SIZE + state_written;
            vtlk_header_init_defaults(&tx_hdr);
            tx_hdr.flags = VTLK_FLAG_KV_STATE;
            tx_hdr.model_family_hash = sender_hash;
            tx_hdr.d_model = d_model;
            tx_hdr.n_tokens = n_tokens;
            tx_hdr.pos_offset = 0u;
            tx_hdr.quant_scheme = (uint8_t)VTLK_QUANT_RAW;
            tx_hdr.codec = (uint8_t)VTLK_CODEC_NONE;
            tx_hdr.payload_bytes = (uint32_t)vtlk_payload_bytes;
            tx_hdr.raw_bytes = (uint32_t)state_written;
            tx_hdr.sequence = (uint64_t)run + 1u;

            if (vtlk_ring_push(&ring_a, &tx_hdr, vtlk_tx_buf) != VTLK_OK) {
                goto cleanup;
            }
            ts1 = vtlk_bench_now_ns();

            tr0 = vtlk_bench_now_ns();
            if (vtlk_ring_pop(&ring_b, &rx_hdr, vtlk_rx_buf, ring_b.max_payload_bytes) != VTLK_OK) {
                goto cleanup;
            }
            if (rx_hdr.model_family_hash != receiver_hash ||
                rx_hdr.d_model != d_model ||
                (rx_hdr.flags & VTLK_FLAG_KV_STATE) == 0u ||
                rx_hdr.quant_scheme != (uint8_t)VTLK_QUANT_RAW ||
                rx_hdr.pos_offset != 0u ||
                rx_hdr.payload_bytes < VTLK_LLAMA_KV_META_SIZE) {
                goto cleanup;
            }
            vtlk_llama_kv_meta_decode(vtlk_rx_buf, &rx_meta);
            if (rx_meta.magic != VTLK_LLAMA_KV_META_MAGIC ||
                rx_meta.llama_build != local_build ||
                (size_t)rx_hdr.payload_bytes != sizeof(vtlk_llama_kv_meta_t) + (size_t)rx_meta.raw_state_bytes) {
                goto cleanup;
            }
            if (!restore_kv_and_decode_last(ctx_b,
                                            vtlk_rx_buf + VTLK_LLAMA_KV_META_SIZE,
                                            (size_t)rx_meta.raw_state_bytes,
                                            (llama_token)rx_meta.last_token_id,
                                            n_tokens)) {
                goto cleanup;
            }
            tr1 = vtlk_bench_now_ns();

            tok_vtlk = sample_greedy_token(ctx_b);
            if (tok_vtlk != baseline_tok) {
                match_vtlk = 0;
            }
            if (run >= warmup) {
                uint32_t idx = run - warmup;
                s_samples[idx] = ts1 - ts0;
                r_samples[idx] = tr1 - tr0;
                t_samples[idx] = (ts1 - ts0) + (tr1 - tr0);
            }
        }
        results[2].name = "Vect-Link SHM (KV_RAW)";
        results[2].payload_bytes = vtlk_payload_bytes;
        results[2].sender = vtlk_bench_compute_stats(s_samples, iters);
        results[2].receiver = vtlk_bench_compute_stats(r_samples, iters);
        results[2].total = vtlk_bench_compute_stats(t_samples, iters);
        results[2].top1_token = tok_vtlk;
        results[2].top1_match = match_vtlk;
    }

    printf("\n=== llama.cpp Multi-Process Handoff Benchmark ===\n");
    printf("Prompt tokens: %u | n_ctx: %u | n_batch: %u | threads: %d | warmup: %u | iters: %u\n\n",
           n_tokens, n_ctx, n_batch, n_threads, warmup, iters);
    printf("%-28s | %12s | %11s | %16s | %16s | %9s | %s\n",
           "Escenario", "Payload (B)", "Sender (us)", "Receiver TTFT(us)",
           "Total p50 (us)", "Speedup", "Top-1 Match");
    printf("-----------------------------|--------------|-------------|------------------|------------------|----------|------------\n");

    for (i = 0; i < 3; ++i) {
        double sender_us = (double)results[i].sender.p50_ns / 1000.0;
        double recv_us = (double)results[i].receiver.p50_ns / 1000.0;
        double tot_us = (double)results[i].total.p50_ns / 1000.0;
        double speedup = (tot_us > 0.0) ? (((double)results[0].total.p50_ns / 1000.0) / tot_us) : 0.0;
        printf("%-28s | %12zu | %11.2f | %16.2f | %16.2f | %7.2fx | %s (%d)\n",
               results[i].name,
               results[i].payload_bytes,
               sender_us,
               recv_us,
               tot_us,
               speedup,
               results[i].top1_match ? "EXACT" : "MISMATCH",
               (int)results[i].top1_token);
    }

    {
        FILE *csv = fopen(csv_path, "w");
        if (csv != NULL) {
            fprintf(csv, "scenario,tokens,threads,payload_bytes,sender_p50_us,receiver_ttft_p50_us,total_p50_us,speedup_vs_reprefill,top1_token,top1_match\n");
            for (i = 0; i < 3; ++i) {
                double sender_us = (double)results[i].sender.p50_ns / 1000.0;
                double recv_us = (double)results[i].receiver.p50_ns / 1000.0;
                double tot_us = (double)results[i].total.p50_ns / 1000.0;
                double speedup = (tot_us > 0.0) ? (((double)results[0].total.p50_ns / 1000.0) / tot_us) : 0.0;
                fprintf(csv, "%s,%u,%d,%zu,%.2f,%.2f,%.2f,%.2f,%d,%s\n",
                        results[i].name, n_tokens, n_threads,
                        results[i].payload_bytes, sender_us, recv_us, tot_us,
                        speedup, (int)results[i].top1_token,
                        results[i].top1_match ? "EXACT" : "MISMATCH");
            }
            fclose(csv);
            printf("\nExported CSV: %s\n", csv_path);
        }
    }

    rc = (results[1].top1_match && results[2].top1_match) ? 0 : 1;

cleanup:
    (void)remove(DEFAULT_DISK_PATH);
    free(t_samples);
    free(r_samples);
    free(s_samples);
    free(vtlk_rx_buf);
    free(vtlk_tx_buf);
    free(disk_rx_buf);
    free(disk_tx_buf);
    if (shm_b_open) {
        vtlk_shm_close(&shm_b);
    }
    if (shm_a_open) {
        vtlk_shm_close(&shm_a);
        (void)vtlk_shm_unlink(shm_name);
    }
    free(tokens);
    free(prompt_text);
    if (ctx_b != NULL) {
        llama_free(ctx_b);
    }
    if (ctx_a != NULL) {
        llama_free(ctx_a);
    }
    if (model != NULL) {
        llama_model_free(model);
    }
    llama_backend_free();
    return rc;
}
