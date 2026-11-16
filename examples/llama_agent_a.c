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

#define DEFAULT_RING_CAP   8u
#define KV_RING_CAP        2u

static void print_usage(const char *prog)
{
    fprintf(stderr,
            "Usage: %s --model <path.gguf> --prompt \"<text>\" "
            "[--mode embd|kv-state] [--shm /vtlk_llama_bridge] "
            "[--dump-ref <file.bin>]\n",
            prog);
}

int main(int argc, char **argv)
{
    const char *model_path = NULL;
    const char *prompt = NULL;
    const char *shm_name = DEFAULT_SHM_NAME;
    const char *dump_ref_path = NULL;
    vtlk_bridge_mode_t mode = VTLK_MODE_EMBD;
    int i;

    struct llama_model_params mparams;
    struct llama_model *model = NULL;
    const struct llama_vocab *vocab = NULL;
    struct llama_context_params cparams;
    struct llama_context *ctx = NULL;
    struct llama_batch batch;
    char model_desc[256];
    uint64_t model_hash;
    int32_t d_model_i32;
    uint32_t d_model;
    int32_t n_tokens_i32;
    uint32_t n_tokens;
    llama_token *tokens = NULL;
    float *latent_tensor = NULL;
    size_t total_elems = 0u;
    size_t max_payload_bytes = 0u;
    size_t written_bytes = 0u;
    size_t ring_bytes;
    uint32_t ring_cap;
    vtlk_shm_t shm;
    vtlk_ring_t ring;
    vtlk_write_slot_t wslot;
    vtlk_header_t hdr;
    vtlk_status_t st;
    int rc_slot;
    uint64_t t0;
    uint64_t t1;
    int rc = 1;

    for (i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "--prompt") == 0 && i + 1 < argc) {
            prompt = argv[++i];
        } else if (strcmp(argv[i], "--mode") == 0 && i + 1 < argc) {
            if (!vtlk_llama_parse_mode(argv[++i], &mode)) {
                print_usage(argv[0]);
                return 1;
            }
        } else if (strcmp(argv[i], "--shm") == 0 && i + 1 < argc) {
            shm_name = argv[++i];
        } else if (strcmp(argv[i], "--dump-ref") == 0 && i + 1 < argc) {
            dump_ref_path = argv[++i];
        } else {
            print_usage(argv[0]);
            return 1;
        }
    }

    if (model_path == NULL || prompt == NULL) {
        print_usage(argv[0]);
        return 1;
    }

    llama_backend_init();

    mparams = llama_model_default_params();
    model = llama_model_load_from_file(model_path, mparams);
    if (model == NULL) {
        fprintf(stderr, "llama_agent_a: failed to load model '%s'\n", model_path);
        goto cleanup;
    }

    vocab = llama_model_get_vocab(model);
    d_model_i32 = llama_model_n_embd(model);
    if (d_model_i32 <= 0) {
        fprintf(stderr, "llama_agent_a: invalid model embedding dimension (%d)\n", d_model_i32);
        goto cleanup;
    }
    d_model = (uint32_t)d_model_i32;

    memset(model_desc, 0, sizeof(model_desc));
    llama_model_desc(model, model_desc, sizeof(model_desc));

    n_tokens_i32 = -llama_tokenize(vocab, prompt, (int32_t)strlen(prompt),
                                   NULL, 0, true, true);
    if (n_tokens_i32 <= 0) {
        fprintf(stderr, "llama_agent_a: tokenization failed\n");
        goto cleanup;
    }
    n_tokens = (uint32_t)n_tokens_i32;

    tokens = (llama_token *)malloc((size_t)n_tokens * sizeof(llama_token));
    if (tokens == NULL) {
        fprintf(stderr, "llama_agent_a: OOM allocating tokens\n");
        goto cleanup;
    }

    if (llama_tokenize(vocab, prompt, (int32_t)strlen(prompt),
                       tokens, n_tokens_i32, true, true) < 0) {
        fprintf(stderr, "llama_agent_a: tokenization pass 2 failed\n");
        goto cleanup;
    }

    cparams = llama_context_default_params();
    cparams.n_ctx = (n_tokens <= 512u) ? 512u : (n_tokens + 64u);
    cparams.n_batch = n_tokens;
    if (mode == VTLK_MODE_EMBD) {
        cparams.embeddings = true;
        cparams.pooling_type = LLAMA_POOLING_TYPE_NONE;
    } else {
        cparams.embeddings = false;
    }

    ctx = llama_init_from_model(model, cparams);
    if (ctx == NULL) {
        fprintf(stderr, "llama_agent_a: failed to create llama_context\n");
        goto cleanup;
    }

    /* Compute compatibility hash before decode so KV initial layout matches Agent B */
    model_hash = vtlk_llama_compute_compat_hash(
        model_path, model, ctx, (mode == VTLK_MODE_KV_STATE) ? 1 : 0);
    if (model_hash == 0ULL) {
        fprintf(stderr, "llama_agent_a: failed to compute compatibility hash\n");
        goto cleanup;
    }

    batch = llama_batch_init(n_tokens_i32, 0, 1);
    batch.n_tokens = n_tokens_i32;
    for (i = 0; i < n_tokens_i32; ++i) {
        batch.token[i] = tokens[i];
        batch.pos[i] = (llama_pos)i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = (mode == VTLK_MODE_EMBD || i == n_tokens_i32 - 1) ? 1 : 0;
    }

    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "llama_agent_a: llama_decode failed\n");
        llama_batch_free(batch);
        goto cleanup;
    }
    llama_batch_free(batch);

    total_elems = (size_t)n_tokens * (size_t)d_model;
    if (mode == VTLK_MODE_EMBD) {
        latent_tensor = (float *)malloc(total_elems * sizeof(float));
        if (latent_tensor == NULL) {
            fprintf(stderr, "llama_agent_a: OOM allocating latent_tensor\n");
            goto cleanup;
        }

        for (i = 0; i < n_tokens_i32; ++i) {
            const float *emb = llama_get_embeddings_ith(ctx, i);
            if (emb == NULL) {
                fprintf(stderr, "llama_agent_a: llama_get_embeddings_ith(%d) returned NULL\n", i);
                goto cleanup;
            }
            memcpy(latent_tensor + (size_t)i * (size_t)d_model,
                   emb,
                   (size_t)d_model * sizeof(float));
        }

        if (dump_ref_path != NULL) {
            FILE *fp = fopen(dump_ref_path, "wb");
            if (fp != NULL) {
                fwrite(latent_tensor, sizeof(float), total_elems, fp);
                fclose(fp);
            }
        }

        st = vtlk_codec_encoded_bytes(VTLK_QUANT_INT4, n_tokens, d_model, &max_payload_bytes);
        if (st != VTLK_OK) {
            fprintf(stderr, "llama_agent_a: vtlk_codec_encoded_bytes failed (%d)\n", (int)st);
            goto cleanup;
        }
        ring_cap = DEFAULT_RING_CAP;
    } else {
        size_t state_size = llama_state_seq_get_size(ctx, 0);
        size_t raw_packet_bytes;
        if (state_size == 0u ||
            state_size > (size_t)(UINT32_MAX - VTLK_LLAMA_KV_META_SIZE - 64u)) {
            fprintf(stderr, "llama_agent_a: invalid KV-cache state size (%zu)\n", state_size);
            goto cleanup;
        }
        /* Align slot max_payload (16B KV meta + state bytes) to 64B cache line */
        raw_packet_bytes = (size_t)VTLK_LLAMA_KV_META_SIZE + state_size;
        max_payload_bytes = (raw_packet_bytes + 63u) & ~(size_t)63u;
        ring_cap = KV_RING_CAP;
    }

    ring_bytes = vtlk_ring_required_bytes(ring_cap, (uint32_t)max_payload_bytes, NULL);
    if (ring_bytes == 0u) {
        fprintf(stderr, "llama_agent_a: vtlk_ring_required_bytes overflow\n");
        goto cleanup;
    }

    (void)vtlk_shm_unlink(shm_name);
    st = vtlk_shm_create(&shm, shm_name, ring_bytes);
    if (st != VTLK_OK) {
        fprintf(stderr, "llama_agent_a: vtlk_shm_create('%s') failed (%d)\n",
                shm_name, (int)st);
        goto cleanup;
    }

    st = vtlk_ring_init(&ring, shm.addr, shm.size, ring_cap, (uint32_t)max_payload_bytes);
    if (st != VTLK_OK) {
        fprintf(stderr, "llama_agent_a: vtlk_ring_init failed (%d)\n", (int)st);
        vtlk_shm_close(&shm);
        vtlk_shm_unlink(shm_name);
        goto cleanup;
    }

    vtlk_header_init_defaults(&hdr);
    hdr.model_family_hash = model_hash;
    hdr.d_model = d_model;
    hdr.n_tokens = n_tokens;
    hdr.sequence = 1u;

    t0 = vtlk_llama_now_ns();
    for (;;) {
        rc_slot = vtlk_ring_acquire_write(&ring, &wslot);
        if (rc_slot == (int)VTLK_OK) {
            break;
        }
        if (rc_slot == (int)VTLK_ERR_FULL) {
            vtlk_llama_cpu_yield();
            continue;
        }
        fprintf(stderr, "llama_agent_a: vtlk_ring_acquire_write failed (%d)\n", rc_slot);
        vtlk_shm_close(&shm);
        vtlk_shm_unlink(shm_name);
        goto cleanup;
    }

    if (mode == VTLK_MODE_EMBD) {
        st = vtlk_encode_tensor(latent_tensor, n_tokens, d_model, VTLK_QUANT_INT4,
                                (uint8_t *)wslot.payload, wslot.capacity, &written_bytes);
        if (st != VTLK_OK) {
            fprintf(stderr, "llama_agent_a: vtlk_encode_tensor failed (%d)\n", (int)st);
            (void)vtlk_ring_abort_write(&ring, &wslot);
            vtlk_shm_close(&shm);
            vtlk_shm_unlink(shm_name);
            goto cleanup;
        }
        hdr.flags = 0u;
        hdr.pos_offset = 0u;
        hdr.quant_scheme = (uint8_t)VTLK_QUANT_INT4;
        hdr.codec = (uint8_t)VTLK_CODEC_NONE;
        hdr.raw_bytes = (uint32_t)(total_elems * sizeof(float));
    } else {
        size_t state_cap = wslot.capacity - (size_t)VTLK_LLAMA_KV_META_SIZE;
        uint8_t *slot_bytes = (uint8_t *)wslot.payload;
        size_t state_written = llama_state_seq_get_data(
            ctx, slot_bytes + VTLK_LLAMA_KV_META_SIZE, state_cap, 0);
        vtlk_llama_kv_meta_t kv_meta;
        if (state_written == 0u || state_written > state_cap) {
            fprintf(stderr, "llama_agent_a: llama_state_seq_get_data failed (%zu)\n",
                    state_written);
            (void)vtlk_ring_abort_write(&ring, &wslot);
            vtlk_shm_close(&shm);
            vtlk_shm_unlink(shm_name);
            goto cleanup;
        }
        kv_meta.magic = VTLK_LLAMA_KV_META_MAGIC;
        kv_meta.llama_build = vtlk_llama_build_stamp();
        kv_meta.last_token_id = (int32_t)tokens[n_tokens - 1u];
        kv_meta.raw_state_bytes = (uint32_t)state_written;
        vtlk_llama_kv_meta_encode(&kv_meta, slot_bytes);

        written_bytes = (size_t)VTLK_LLAMA_KV_META_SIZE + state_written;
        hdr.flags = VTLK_FLAG_KV_STATE;
        hdr.pos_offset = 0u;
        hdr.quant_scheme = (uint8_t)VTLK_QUANT_RAW;
        hdr.codec = (uint8_t)VTLK_CODEC_NONE;
        hdr.raw_bytes = (uint32_t)state_written;
    }

    rc_slot = vtlk_ring_commit_write(&ring, &wslot, &hdr, written_bytes);
    if (rc_slot != (int)VTLK_OK) {
        fprintf(stderr, "llama_agent_a: vtlk_ring_commit_write failed (%d)\n", rc_slot);
        (void)vtlk_ring_abort_write(&ring, &wslot);
        vtlk_shm_close(&shm);
        vtlk_shm_unlink(shm_name);
        goto cleanup;
    }
    t1 = vtlk_llama_now_ns();

    printf("[Agent A] Mode: %s\n", (mode == VTLK_MODE_KV_STATE) ? "kv-state" : "embd");
    printf("[Agent A] Model: %s (hash=0x%016llx, d_model=%u)\n",
           model_desc, (unsigned long long)model_hash, d_model);
    if (mode == VTLK_MODE_EMBD) {
        printf("[Agent A] Tensor: %u tokens x %u dims (%zu B FP32 -> %zu B INT4, %.2fx)\n",
               n_tokens, d_model, total_elems * sizeof(float), written_bytes,
               (double)(total_elems * sizeof(float)) / (double)written_bytes);
        printf("[Agent A] Encode + SHM Push latency: %.2f us\n",
               (double)(t1 - t0) / 1000.0);
    } else {
        printf("[Agent A] KV-Cache state: %zu B (prompt=%u tokens, last_token=%d)\n",
               written_bytes, n_tokens, (int)tokens[n_tokens - 1u]);
        printf("[Agent A] State Extract + SHM Push latency: %.2f us\n",
               (double)(t1 - t0) / 1000.0);
    }
    printf("[Agent A] Waiting for Agent B to attach & consume on '%s'... (press Enter after Agent B finishes)\n",
           shm_name);
    fflush(stdout);
    (void)getchar();

    vtlk_shm_close(&shm);
    vtlk_shm_unlink(shm_name);
    rc = 0;

cleanup:
    free(latent_tensor);
    free(tokens);
    if (ctx != NULL) {
        llama_free(ctx);
    }
    if (model != NULL) {
        llama_model_free(model);
    }
    llama_backend_free();
    return rc;
}
