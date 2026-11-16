#ifndef _WIN32
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif
#endif

#include "vectlink/vectlink.h"
#include "llama_bridge_common.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DEFAULT_N_PREDICT  32u
#define MAX_SHM_MAP_BYTES  (512u * 1024u * 1024u)

static void sleep_ms(unsigned int ms)
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

static void verify_against_ref(const char *ref_arg,
                               struct llama_model *model,
                               const struct llama_vocab *vocab,
                               const float *decoded,
                               uint32_t n_tokens,
                               uint32_t d_model)
{
    size_t total_elems = (size_t)n_tokens * (size_t)d_model;
    float *ref = (float *)malloc(total_elems * sizeof(float));
    FILE *fp;

    if (ref == NULL) {
        return;
    }

    fp = fopen(ref_arg, "rb");
    if (fp != NULL) {
        size_t read_n = fread(ref, sizeof(float), total_elems, fp);
        fclose(fp);
        if (read_n == total_elems) {
            printf("[Agent B] Ref Cosine Similarity: %.6f | Max Abs Error: %.5f\n",
                   vtlk_cosine_similarity(ref, decoded, total_elems),
                   vtlk_max_abs_error(ref, decoded, total_elems));
            free(ref);
            return;
        }
    }

    /* Otherwise treat ref_arg as a prompt string and compute reference locally */
    {
        int32_t n_tok = -llama_tokenize(vocab, ref_arg, (int32_t)strlen(ref_arg),
                                        NULL, 0, true, true);
        if (n_tok == (int32_t)n_tokens) {
            llama_token *toks = (llama_token *)malloc((size_t)n_tok * sizeof(llama_token));
            struct llama_context_params rp = llama_context_default_params();
            rp.n_ctx = (n_tokens < 512u) ? 512u : (n_tokens + 64u);
            rp.n_batch = n_tokens;
            rp.embeddings = true;
            rp.pooling_type = LLAMA_POOLING_TYPE_NONE;

            if (toks != NULL &&
                llama_tokenize(vocab, ref_arg, (int32_t)strlen(ref_arg),
                               toks, n_tok, true, true) >= 0) {
                struct llama_context *rctx = llama_init_from_model(model, rp);
                if (rctx != NULL) {
                    struct llama_batch rb = llama_batch_init(n_tok, 0, 1);
                    int32_t k;
                    rb.n_tokens = n_tok;
                    for (k = 0; k < n_tok; ++k) {
                        rb.token[k] = toks[k];
                        rb.pos[k] = (llama_pos)k;
                        rb.n_seq_id[k] = 1;
                        rb.seq_id[k][0] = 0;
                        rb.logits[k] = 1;
                    }
                    if (llama_decode(rctx, rb) == 0) {
                        int ok = 1;
                        for (k = 0; k < n_tok; ++k) {
                            const float *emb = llama_get_embeddings_ith(rctx, k);
                            if (emb == NULL) {
                                ok = 0;
                                break;
                            }
                            memcpy(ref + (size_t)k * (size_t)d_model,
                                   emb, (size_t)d_model * sizeof(float));
                        }
                        if (ok) {
                            printf("[Agent B] Ref Cosine Similarity: %.6f | Max Abs Error: %.5f\n",
                                   vtlk_cosine_similarity(ref, decoded, total_elems),
                                   vtlk_max_abs_error(ref, decoded, total_elems));
                        }
                    }
                    llama_batch_free(rb);
                    llama_free(rctx);
                }
            }
            free(toks);
        } else {
            fprintf(stderr, "[Agent B] --verify-ref token count (%d) != packet n_tokens (%u)\n",
                    n_tok, n_tokens);
        }
    }

    free(ref);
}

static void print_usage(const char *prog)
{
    fprintf(stderr,
            "Usage: %s --model <path.gguf> [--mode embd|kv-state] "
            "[--shm /vtlk_llama_bridge] [--n-predict 32] "
            "[--verify-ref <ref.bin|prompt>]\n",
            prog);
}

int main(int argc, char **argv)
{
    const char *model_path = NULL;
    const char *shm_name = DEFAULT_SHM_NAME;
    const char *verify_ref = NULL;
    vtlk_bridge_mode_t mode = VTLK_MODE_EMBD;
    uint32_t n_predict = DEFAULT_N_PREDICT;
    int i;

    struct llama_model_params mparams;
    struct llama_model *model = NULL;
    const struct llama_vocab *vocab = NULL;
    struct llama_context_params cparams;
    struct llama_context *ctx = NULL;
    struct llama_sampler *smpl = NULL;
    char model_desc[256];
    uint64_t expected_hash;
    int32_t d_model_i32;
    uint32_t expected_d_model;

    vtlk_shm_t shm;
    vtlk_ring_t ring;
    vtlk_header_t hdr;
    uint8_t *rx_payload = NULL;
    float *decoded = NULL;
    size_t rx_cap = MAX_SHM_MAP_BYTES;
    size_t total_elems = 0u;
    vtlk_status_t st;
    int retries;
    uint64_t t0;
    uint64_t t1;
    uint64_t t_pop0;
    uint64_t t_pop1;
    uint64_t t_set0;
    uint64_t t_set1;
    uint32_t step;
    int rc = 1;

    for (i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "--mode") == 0 && i + 1 < argc) {
            if (!vtlk_llama_parse_mode(argv[++i], &mode)) {
                print_usage(argv[0]);
                return 1;
            }
        } else if (strcmp(argv[i], "--shm") == 0 && i + 1 < argc) {
            shm_name = argv[++i];
        } else if (strcmp(argv[i], "--n-predict") == 0 && i + 1 < argc) {
            n_predict = (uint32_t)strtoul(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--verify-ref") == 0 && i + 1 < argc) {
            verify_ref = argv[++i];
        } else {
            print_usage(argv[0]);
            return 1;
        }
    }

    if (model_path == NULL) {
        print_usage(argv[0]);
        return 1;
    }

    llama_backend_init();

    mparams = llama_model_default_params();
    model = llama_model_load_from_file(model_path, mparams);
    if (model == NULL) {
        fprintf(stderr, "llama_agent_b: failed to load model '%s'\n", model_path);
        goto cleanup;
    }

    vocab = llama_model_get_vocab(model);
    d_model_i32 = llama_model_n_embd(model);
    if (d_model_i32 <= 0) {
        fprintf(stderr, "llama_agent_b: invalid model embedding dimension (%d)\n", d_model_i32);
        goto cleanup;
    }
    expected_d_model = (uint32_t)d_model_i32;

    memset(model_desc, 0, sizeof(model_desc));
    llama_model_desc(model, model_desc, sizeof(model_desc));

    printf("[Agent B] Loaded model: %s (d_model=%u, mode=%s)\n",
           model_desc, expected_d_model,
           (mode == VTLK_MODE_KV_STATE) ? "kv-state" : "embd");
    printf("[Agent B] Waiting for SHM '%s'...\n", shm_name);
    fflush(stdout);

    /* Open minimum 192B control block first to inspect ring size, or wait for SHM */
    st = VTLK_ERR_SHM;
    for (retries = 0; retries < 1500; ++retries) {
        st = vtlk_shm_open(&shm, shm_name, VTLK_RING_CTRL_SIZE);
        if (st == VTLK_OK) {
            break;
        }
        sleep_ms(20u);
    }
    if (st != VTLK_OK) {
        fprintf(stderr, "llama_agent_b: timed out opening SHM '%s'\n", shm_name);
        goto cleanup;
    }

    /* Read ring metadata to remap full ring size */
    {
        const uint8_t *ctrl = (const uint8_t *)shm.addr;
        uint32_t cap = ((uint32_t)ctrl[12]) | ((uint32_t)ctrl[13] << 8) |
                       ((uint32_t)ctrl[14] << 16) | ((uint32_t)ctrl[15] << 24);
        uint32_t max_pay = ((uint32_t)ctrl[20]) | ((uint32_t)ctrl[21] << 8) |
                           ((uint32_t)ctrl[22] << 16) | ((uint32_t)ctrl[23] << 24);
        size_t full_bytes = vtlk_ring_required_bytes(cap, max_pay, NULL);
        vtlk_shm_close(&shm);
        if (full_bytes == 0u || full_bytes > rx_cap) {
            fprintf(stderr, "llama_agent_b: invalid SHM ring size (%zu)\n", full_bytes);
            goto cleanup;
        }
        st = vtlk_shm_open(&shm, shm_name, full_bytes);
        if (st != VTLK_OK) {
            fprintf(stderr, "llama_agent_b: vtlk_shm_open full failed (%d)\n", (int)st);
            goto cleanup;
        }
        st = vtlk_ring_attach(&ring, shm.addr, shm.size);
        if (st != VTLK_OK) {
            fprintf(stderr, "llama_agent_b: vtlk_ring_attach failed (%d)\n", (int)st);
            vtlk_shm_close(&shm);
            goto cleanup;
        }
    }

    rx_payload = (uint8_t *)malloc(ring.max_payload_bytes);
    if (rx_payload == NULL) {
        fprintf(stderr, "llama_agent_b: OOM allocating rx_payload\n");
        vtlk_shm_close(&shm);
        goto cleanup;
    }

    for (;;) {
        t_pop0 = vtlk_llama_now_ns();
        st = vtlk_ring_pop(&ring, &hdr, rx_payload, ring.max_payload_bytes);
        if (st == VTLK_OK) {
            t_pop1 = vtlk_llama_now_ns();
            break;
        }
        if (st == VTLK_ERR_EMPTY) {
            vtlk_llama_cpu_yield();
            continue;
        }
        fprintf(stderr, "llama_agent_b: vtlk_ring_pop failed (%d)\n", (int)st);
        vtlk_shm_close(&shm);
        goto cleanup;
    }
    vtlk_shm_close(&shm);

    {
        int pkt_is_kv = ((hdr.flags & VTLK_FLAG_KV_STATE) != 0u);
        int want_kv = (mode == VTLK_MODE_KV_STATE);
        if (pkt_is_kv != want_kv) {
            fprintf(stderr,
                    "[Agent B] REJECTED mode mismatch: packet flags=0x%08x (%s), "
                    "expected --mode %s\n",
                    hdr.flags, pkt_is_kv ? "kv-state" : "embd",
                    want_kv ? "kv-state" : "embd");
            goto cleanup;
        }
    }

    cparams = llama_context_default_params();
    cparams.n_ctx = (hdr.n_tokens + n_predict <= 512u) ? 512u : (hdr.n_tokens + n_predict + 64u);
    cparams.n_batch = (hdr.n_tokens > 0u) ? hdr.n_tokens : 1u;
    cparams.embeddings = false;

    ctx = llama_init_from_model(model, cparams);
    if (ctx == NULL) {
        fprintf(stderr, "llama_agent_b: failed to create llama_context\n");
        goto cleanup;
    }

    /* Validate weight identity + structural params (+ KV env in kv-state mode) */
    expected_hash = vtlk_llama_compute_compat_hash(
        model_path, model, ctx, (mode == VTLK_MODE_KV_STATE) ? 1 : 0);
    if (expected_hash == 0ULL) {
        fprintf(stderr, "llama_agent_b: failed to compute compatibility hash\n");
        goto cleanup;
    }
    if (hdr.model_family_hash != expected_hash || hdr.d_model != expected_d_model) {
        fprintf(stderr,
                "[Agent B] REJECTED incompatible packet: "
                "got hash=0x%016llx d_model=%u, expected hash=0x%016llx d_model=%u\n",
                (unsigned long long)hdr.model_family_hash, hdr.d_model,
                (unsigned long long)expected_hash, expected_d_model);
        goto cleanup;
    }

    if (mode == VTLK_MODE_EMBD) {
        total_elems = (size_t)hdr.n_tokens * (size_t)hdr.d_model;
        decoded = (float *)malloc(total_elems * sizeof(float));
        if (decoded == NULL) {
            fprintf(stderr, "llama_agent_b: OOM allocating decoded tensor\n");
            goto cleanup;
        }

        t0 = vtlk_llama_now_ns();
        st = vtlk_decode_tensor(rx_payload, hdr.payload_bytes,
                                hdr.n_tokens, hdr.d_model,
                                (vtlk_quant_scheme_t)hdr.quant_scheme,
                                decoded, total_elems);
        t1 = vtlk_llama_now_ns();
        if (st != VTLK_OK) {
            fprintf(stderr, "llama_agent_b: vtlk_decode_tensor failed (%d)\n", (int)st);
            goto cleanup;
        }

        printf("[Agent B] Mode: embd (hash=0x%016llx)\n", (unsigned long long)expected_hash);
        printf("[Agent B] Received & decoded %u x %u tensor (%u B INT4) in %.2f us\n",
               hdr.n_tokens, hdr.d_model, hdr.payload_bytes,
               (double)((t_pop1 - t_pop0) + (t1 - t0)) / 1000.0);

        if (verify_ref != NULL) {
            verify_against_ref(verify_ref, model, vocab, decoded, hdr.n_tokens, hdr.d_model);
        }

        /* Inject continuous embeddings directly into layer 0 via batch.embd */
        {
            struct llama_batch embd_batch = llama_batch_init((int32_t)hdr.n_tokens,
                                                             (int32_t)hdr.d_model, 1);
            int32_t k;
            embd_batch.n_tokens = (int32_t)hdr.n_tokens;
            memcpy(embd_batch.embd, decoded, total_elems * sizeof(float));
            for (k = 0; k < (int32_t)hdr.n_tokens; ++k) {
                embd_batch.pos[k] = (llama_pos)k;
                embd_batch.n_seq_id[k] = 1;
                embd_batch.seq_id[k][0] = 0;
                embd_batch.logits[k] = (k == (int32_t)hdr.n_tokens - 1) ? 1 : 0;
            }
            if (llama_decode(ctx, embd_batch) != 0) {
                fprintf(stderr, "llama_agent_b: llama_decode(embd_batch) failed\n");
                llama_batch_free(embd_batch);
                goto cleanup;
            }
            llama_batch_free(embd_batch);
        }
    } else {
        vtlk_llama_kv_meta_t kv_meta;
        uint32_t local_build = vtlk_llama_build_stamp();
        const uint8_t *raw_state_ptr;
        size_t restored;
        llama_token last_tok;
        llama_pos last_pos;
        struct llama_batch bridge_batch;

        if (hdr.quant_scheme != (uint8_t)VTLK_QUANT_RAW) {
            fprintf(stderr,
                    "[Agent B] REJECTED kv-state quant_scheme=%u (expected VTLK_QUANT_RAW=%u)\n",
                    (unsigned)hdr.quant_scheme, (unsigned)VTLK_QUANT_RAW);
            goto cleanup;
        }
        if (hdr.pos_offset != 0u) {
            fprintf(stderr,
                    "[Agent B] REJECTED kv-state nonzero pos_offset=%u\n",
                    hdr.pos_offset);
            goto cleanup;
        }
        if (hdr.payload_bytes < VTLK_LLAMA_KV_META_SIZE) {
            fprintf(stderr,
                    "[Agent B] REJECTED truncated kv-state payload (%u B)\n",
                    hdr.payload_bytes);
            goto cleanup;
        }

        vtlk_llama_kv_meta_decode(rx_payload, &kv_meta);
        if (kv_meta.magic != VTLK_LLAMA_KV_META_MAGIC) {
            fprintf(stderr,
                    "[Agent B] REJECTED invalid kv_meta magic=0x%08x\n",
                    kv_meta.magic);
            goto cleanup;
        }
        if (kv_meta.llama_build != local_build) {
            fprintf(stderr,
                    "[Agent B] REJECTED llama build mismatch: got 0x%08x, expected 0x%08x\n",
                    kv_meta.llama_build, local_build);
            goto cleanup;
        }
        if (kv_meta.raw_state_bytes == 0u ||
            (size_t)hdr.payload_bytes != sizeof(vtlk_llama_kv_meta_t) + (size_t)kv_meta.raw_state_bytes) {
            fprintf(stderr,
                    "[Agent B] REJECTED kv-state size mismatch: payload=%u B, expected %zu B\n",
                    hdr.payload_bytes,
                    sizeof(vtlk_llama_kv_meta_t) + (size_t)kv_meta.raw_state_bytes);
            goto cleanup;
        }

        raw_state_ptr = rx_payload + VTLK_LLAMA_KV_META_SIZE;
        t_set0 = vtlk_llama_now_ns();
        restored = llama_state_seq_set_data(ctx, raw_state_ptr, (size_t)kv_meta.raw_state_bytes, 0);
        t_set1 = vtlk_llama_now_ns();
        if (restored == 0u || restored != (size_t)kv_meta.raw_state_bytes) {
            fprintf(stderr,
                    "llama_agent_b: llama_state_seq_set_data restored %zu B != expected %u B\n",
                    restored, kv_meta.raw_state_bytes);
            goto cleanup;
        }

        printf("[Agent B] Mode: kv-state (hash=0x%016llx, build=0x%08x)\n",
               (unsigned long long)expected_hash, local_build);
        printf("[Agent B] Received & restored KV-Cache (%u B state + %zu B meta, prompt=%u tokens) in %.2f us\n",
               kv_meta.raw_state_bytes, sizeof(vtlk_llama_kv_meta_t), hdr.n_tokens,
               (double)((t_pop1 - t_pop0) + (t_set1 - t_set0)) / 1000.0);

        last_tok = (llama_token)kv_meta.last_token_id;
        last_pos = (llama_pos)(hdr.n_tokens - 1u);

        /* Trim last prompt token position and re-evaluate it to obtain fresh deterministic logits */
        if (!llama_memory_seq_rm(llama_get_memory(ctx), 0, last_pos, -1)) {
            fprintf(stderr, "llama_agent_b: llama_memory_seq_rm failed at pos %d\n",
                    (int)last_pos);
            goto cleanup;
        }

        bridge_batch = llama_batch_init(1, 0, 1);
        bridge_batch.n_tokens = 1;
        bridge_batch.token[0] = last_tok;
        bridge_batch.pos[0] = last_pos;
        bridge_batch.n_seq_id[0] = 1;
        bridge_batch.seq_id[0][0] = 0;
        bridge_batch.logits[0] = 1;
        if (llama_decode(ctx, bridge_batch) != 0) {
            fprintf(stderr, "llama_agent_b: llama_decode(bridge_batch) failed\n");
            llama_batch_free(bridge_batch);
            goto cleanup;
        }
        llama_batch_free(bridge_batch);
    }

    smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    printf("[Agent B] Generated continuation (%u tokens): ", n_predict);
    fflush(stdout);

    {
        struct llama_batch tok_batch = llama_batch_init(1, 0, 1);
        for (step = 0u; step < n_predict; ++step) {
            llama_token next_tok = llama_sampler_sample(smpl, ctx, -1);
            char piece[256];
            int piece_len;

            if (llama_vocab_is_eog(vocab, next_tok)) {
                break;
            }

            piece_len = llama_token_to_piece(vocab, next_tok, piece,
                                             (int32_t)sizeof(piece), 0, true);
            if (piece_len > 0) {
                fwrite(piece, 1u, (size_t)piece_len, stdout);
                fflush(stdout);
            }

            tok_batch.n_tokens = 1;
            tok_batch.token[0] = next_tok;
            tok_batch.pos[0] = (llama_pos)(hdr.n_tokens + step);
            tok_batch.n_seq_id[0] = 1;
            tok_batch.seq_id[0][0] = 0;
            tok_batch.logits[0] = 1;

            if (llama_decode(ctx, tok_batch) != 0) {
                fprintf(stderr, "\nllama_agent_b: decode step %u failed\n", step);
                break;
            }
        }
        llama_batch_free(tok_batch);
    }
    printf("\n");
    rc = 0;

cleanup:
    if (smpl != NULL) {
        llama_sampler_free(smpl);
    }
    free(decoded);
    free(rx_payload);
    if (ctx != NULL) {
        llama_free(ctx);
    }
    if (model != NULL) {
        llama_model_free(model);
    }
    llama_backend_free();
    return rc;
}
