/**
 * @file llama_kv_paged_bridge.c
 * @brief Paged KV-block bridge between llama.cpp and Vect-Link CoW KVArena.
 *
 * Demonstrates:
 *  1. Partitioning llama_state_seq_get_data() KV-cache into fixed-size paged
 *     blocks stored in vtlk_kv_arena_t (VTLK_KV_DTYPE_RAW_LLAMA).
 *  2. One producer prefilling a system prompt up to pos = N and creating a
 *     vtlk_kv_snapshot_t at pos = N.
 *  3. K child workers performing vtlk_kv_fork() + vtlk_kv_restore() (validating
 *     model_arch_hash and rope_config_hash) to continue decoding from pos = N
 *     with RoPE positions intact.
 *
 * Usage:
 *   llama_kv_paged_bridge --self-test
 *   llama_kv_paged_bridge --model <path.gguf> [--prompt "..."] [--workers 3]
 */

#include "llama.h"
#include "vectlink_kv.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <process.h>
#define VTLK_GETPID() ((int)_getpid())
#else
#include <unistd.h>
#define VTLK_GETPID() ((int)getpid())
#endif

#define BRIDGE_PAGE_BYTES       (64u * 1024u)
#define BRIDGE_DEFAULT_WORKERS  3u
#define BRIDGE_ARENA_BLOCKS     256u

typedef struct bridge_rope_config {
    int32_t rope_type;
    float rope_freq_scale_train;
    int32_t n_ctx_train;
    int32_t n_embd;
    int32_t n_layer;
    int32_t n_head_kv;
} bridge_rope_config_t;

typedef struct bridge_page_header {
    uint32_t total_state_bytes;
    uint32_t page_index;
    uint32_t page_payload_bytes;
    uint32_t pos_n;
} bridge_page_header_t;

static void print_usage(const char *prog)
{
    fprintf(stderr,
            "Usage:\n"
            "  %s --self-test [--workers N]\n"
            "  %s --model <path.gguf> [--prompt <text>] [--workers N]\n",
            prog, prog);
}

static uint64_t compute_synthetic_rope_hash(int32_t rope_type,
                                            float freq_scale,
                                            int32_t n_ctx_train,
                                            int32_t n_embd,
                                            int32_t n_layer,
                                            int32_t n_head_kv)
{
    bridge_rope_config_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.rope_type = rope_type;
    cfg.rope_freq_scale_train = freq_scale;
    cfg.n_ctx_train = n_ctx_train;
    cfg.n_embd = n_embd;
    cfg.n_layer = n_layer;
    cfg.n_head_kv = n_head_kv;
    return vtlk_kv_hash_bytes(&cfg, sizeof(cfg));
}

static int partition_state_into_branch(vtlk_kv_arena_t *arena,
                                       vtlk_kv_branch_t *branch,
                                       const uint8_t *state_buf,
                                       size_t state_bytes,
                                       uint32_t pos_n,
                                       const vtlk_kv_identity_t *base_id)
{
    const size_t max_payload = (size_t)arena->block_size_bytes - sizeof(bridge_page_header_t);
    size_t offset = 0;
    uint32_t page_idx = 0;
    uint64_t rolling_hash = base_id->token_prefix_hash;

    if (max_payload == 0 || state_bytes == 0) {
        return -1;
    }

    while (offset < state_bytes) {
        size_t chunk = state_bytes - offset;
        if (chunk > max_payload) {
            chunk = max_payload;
        }

        uint32_t block_id = VTLK_KV_INVALID_BLOCK_ID;
        vtlk_status_t st = vtlk_kv_block_alloc(arena, &block_id);
        if (st != VTLK_OK) {
            fprintf(stderr, "vtlk_kv_block_alloc failed: %d\n", (int)st);
            return -1;
        }

        uint8_t *dst = (uint8_t *)vtlk_kv_block_data(arena, block_id);
        if (dst == NULL) {
            vtlk_kv_block_release(arena, block_id);
            return -1;
        }

        bridge_page_header_t phdr;
        phdr.total_state_bytes = (uint32_t)state_bytes;
        phdr.page_index = page_idx;
        phdr.page_payload_bytes = (uint32_t)chunk;
        phdr.pos_n = pos_n;

        memcpy(dst, &phdr, sizeof(phdr));
        memcpy(dst + sizeof(phdr), state_buf + offset, chunk);

        /* Chain page hash so every paged block has a distinct content-addressed identity */
        uint32_t page_tag = page_idx + 1u;
        rolling_hash = vtlk_kv_hash_tokens(rolling_hash, &page_tag, 1u);

        vtlk_kv_identity_t blk_id = *base_id;
        blk_id.token_prefix_hash = rolling_hash;
        blk_id.pos_offset = (page_idx == 0) ? 0u : pos_n;
        blk_id.token_count = (page_idx == 0) ? pos_n : 0u;

        st = vtlk_kv_block_seal(arena, block_id, &blk_id);
        if (st != VTLK_OK) {
            fprintf(stderr, "vtlk_kv_block_seal failed: %d\n", (int)st);
            vtlk_kv_block_release(arena, block_id);
            return -1;
        }

        st = vtlk_kv_branch_append_block(arena, branch, block_id);
        vtlk_kv_block_release(arena, block_id);
        if (st != VTLK_OK) {
            fprintf(stderr, "vtlk_kv_branch_append_block failed: %d\n", (int)st);
            return -1;
        }

        offset += chunk;
        page_idx++;
    }

    return 0;
}

static int reassemble_state_from_ptrs(const void *const *ptrs,
                                      uint32_t block_count,
                                      uint32_t prefix_blocks,
                                      uint8_t **out_buf,
                                      size_t *out_bytes,
                                      uint32_t *out_pos_n)
{
    if (prefix_blocks == 0 || prefix_blocks > block_count) {
        return -1;
    }

    const bridge_page_header_t *hdr0 = (const bridge_page_header_t *)ptrs[0];
    size_t total = (size_t)hdr0->total_state_bytes;
    uint8_t *buf = (uint8_t *)malloc(total);
    if (buf == NULL) {
        return -1;
    }

    size_t cursor = 0;
    for (uint32_t i = 0; i < prefix_blocks; ++i) {
        const bridge_page_header_t *ph = (const bridge_page_header_t *)ptrs[i];
        const uint8_t *payload = (const uint8_t *)ptrs[i] + sizeof(bridge_page_header_t);
        if (cursor + ph->page_payload_bytes > total) {
            free(buf);
            return -1;
        }
        memcpy(buf + cursor, payload, ph->page_payload_bytes);
        cursor += ph->page_payload_bytes;
    }

    if (cursor != total) {
        free(buf);
        return -1;
    }

    *out_buf = buf;
    *out_bytes = total;
    *out_pos_n = hdr0->pos_n;
    return 0;
}

static int run_self_test(uint32_t num_workers)
{
    char shm_name[128];
    snprintf(shm_name, sizeof(shm_name), "/vtlk_llama_paged_test_%d", VTLK_GETPID());

    vtlk_kv_arena_t arena;
    vtlk_status_t st = vtlk_kv_arena_create(&arena, shm_name, 64u, 4096u);
    if (st != VTLK_OK) {
        fprintf(stderr, "vtlk_kv_arena_create failed: %d\n", (int)st);
        return 1;
    }

    /* Producer: synthetic prefill of N = 16 system prompt tokens */
    const uint32_t sys_tokens[16] = {
        101, 102, 103, 104, 105, 106, 107, 108,
        109, 110, 111, 112, 113, 114, 115, 116
    };
    const uint32_t pos_n = 16u;

    vtlk_kv_identity_t expected_cfg;
    memset(&expected_cfg, 0, sizeof(expected_cfg));
    expected_cfg.token_prefix_hash = vtlk_kv_hash_tokens(0ULL, sys_tokens, 16u);
    expected_cfg.model_arch_hash = vtlk_kv_hash_bytes("llama-3-8b-test", 15u);
    expected_cfg.rope_config_hash = compute_synthetic_rope_hash(2, 1.0f, 8192, 4096, 32, 8);
    expected_cfg.lora_adapter_hash = 0ULL;
    expected_cfg.kv_dtype = VTLK_KV_DTYPE_RAW_LLAMA;
    expected_cfg.layer_count = 32u;
    expected_cfg.head_dim = 128u;
    expected_cfg.kv_head_count = 8u;

    /* Create a 9000-byte synthetic llama_state_seq_get_data buffer (> 2 pages) */
    const size_t synth_bytes = 9000u;
    uint8_t *synth_state = (uint8_t *)malloc(synth_bytes);
    if (synth_state == NULL) {
        vtlk_kv_arena_close(&arena);
        vtlk_shm_unlink(shm_name);
        return 1;
    }
    for (size_t i = 0; i < synth_bytes; ++i) {
        synth_state[i] = (uint8_t)((i * 37u + 13u) & 0xFFu);
    }

    vtlk_kv_branch_t root_branch;
    memset(&root_branch, 0, sizeof(root_branch));
    root_branch.model_arch_hash = expected_cfg.model_arch_hash;
    root_branch.rope_config_hash = expected_cfg.rope_config_hash;
    root_branch.lora_adapter_hash = expected_cfg.lora_adapter_hash;
    root_branch.kv_dtype = expected_cfg.kv_dtype;
    root_branch.layer_count = expected_cfg.layer_count;
    root_branch.head_dim = expected_cfg.head_dim;
    root_branch.kv_head_count = expected_cfg.kv_head_count;

    if (partition_state_into_branch(&arena, &root_branch, synth_state,
                                    synth_bytes, pos_n, &expected_cfg) != 0) {
        free(synth_state);
        vtlk_kv_branch_release(&arena, &root_branch);
        vtlk_kv_arena_close(&arena);
        vtlk_shm_unlink(shm_name);
        return 1;
    }

    uint64_t snap_id = 0;
    uint32_t root_blocks = root_branch.block_count;
    st = vtlk_kv_snapshot_create(&arena, &root_branch, &snap_id);
    vtlk_kv_branch_release(&arena, &root_branch);
    if (st != VTLK_OK) {
        fprintf(stderr, "vtlk_kv_snapshot_create failed: %d\n", (int)st);
        free(synth_state);
        vtlk_kv_arena_close(&arena);
        vtlk_shm_unlink(shm_name);
        return 1;
    }

    printf("[producer] prefill pos=%u, partitioned %zu B into %u paged blocks, snapshot_id=%llu\n",
           pos_n, synth_bytes, root_blocks, (unsigned long long)snap_id);

    /* N child workers: fork + restore + continue from pos = N with RoPE intact */
    for (uint32_t w = 0; w < num_workers; ++w) {
        vtlk_kv_branch_t forked;
        const void *ptrs[VTLK_KV_MAX_BRANCH_BLOCKS];

        st = vtlk_kv_fork(&arena, snap_id, &forked);
        if (st != VTLK_OK) {
            fprintf(stderr, "worker %u: vtlk_kv_fork failed: %d\n", w, (int)st);
            free(synth_state);
            vtlk_kv_arena_close(&arena);
            vtlk_shm_unlink(shm_name);
            return 1;
        }
        vtlk_kv_branch_release(&arena, &forked);

        st = vtlk_kv_restore(&arena, snap_id, &expected_cfg, &forked,
                             ptrs, VTLK_KV_MAX_BRANCH_BLOCKS);
        if (st != VTLK_OK) {
            fprintf(stderr, "worker %u: vtlk_kv_restore failed: %d\n", w, (int)st);
            free(synth_state);
            vtlk_kv_arena_close(&arena);
            vtlk_shm_unlink(shm_name);
            return 1;
        }

        uint8_t *restored_buf = NULL;
        size_t restored_bytes = 0;
        uint32_t restored_pos = 0;
        if (reassemble_state_from_ptrs(ptrs, forked.block_count, forked.block_count,
                                       &restored_buf, &restored_bytes, &restored_pos) != 0) {
            fprintf(stderr, "worker %u: reassemble failed\n", w);
            vtlk_kv_branch_release(&arena, &forked);
            free(synth_state);
            vtlk_kv_arena_close(&arena);
            vtlk_shm_unlink(shm_name);
            return 1;
        }

        if (restored_pos != pos_n || restored_bytes != synth_bytes ||
            memcmp(restored_buf, synth_state, synth_bytes) != 0) {
            fprintf(stderr, "worker %u: state mismatch!\n", w);
            free(restored_buf);
            vtlk_kv_branch_release(&arena, &forked);
            free(synth_state);
            vtlk_kv_arena_close(&arena);
            vtlk_shm_unlink(shm_name);
            return 1;
        }
        free(restored_buf);

        /* Append worker-specific continuation block at pos = N */
        uint32_t cont_tok = 9000u + w;
        uint32_t cont_bid = VTLK_KV_INVALID_BLOCK_ID;
        if (vtlk_kv_block_alloc(&arena, &cont_bid) != VTLK_OK) {
            vtlk_kv_branch_release(&arena, &forked);
            free(synth_state);
            vtlk_kv_arena_close(&arena);
            vtlk_shm_unlink(shm_name);
            return 1;
        }

        vtlk_kv_identity_t cont_id = expected_cfg;
        cont_id.token_prefix_hash = vtlk_kv_hash_tokens(expected_cfg.token_prefix_hash,
                                                        &cont_tok, 1u);
        cont_id.pos_offset = pos_n;
        cont_id.token_count = 1u;
        vtlk_kv_block_seal(&arena, cont_bid, &cont_id);
        vtlk_kv_branch_append_block(&arena, &forked, cont_bid);
        vtlk_kv_block_release(&arena, cont_bid);

        printf("[worker %u] fork+restore OK (pos=%u, shared_blocks=%u, total_tokens=%u)\n",
               w, restored_pos, forked.block_count - 1u, forked.total_tokens);

        vtlk_kv_branch_release(&arena, &forked);
    }

    free(synth_state);
    vtlk_kv_arena_close(&arena);
    vtlk_shm_unlink(shm_name);
    printf("llama_kv_paged_bridge self-test PASSED (%u workers)\n", num_workers);
    return 0;
}

static llama_token greedy_next_token(struct llama_context *ctx,
                                     int32_t n_vocab,
                                     const float **out_logits)
{
    const float *logits = llama_get_logits_ith(ctx, -1);
    int32_t best_id = 0;
    float best_val = logits[0];
    for (int32_t i = 1; i < n_vocab; ++i) {
        if (logits[i] > best_val) {
            best_val = logits[i];
            best_id = i;
        }
    }
    if (out_logits != NULL) {
        *out_logits = logits;
    }
    return (llama_token)best_id;
}

static int run_llama_model_mode(const char *model_path,
                                const char *prompt,
                                uint32_t num_workers)
{
    static const char *const branch_suffixes[3] = {
        "\n\nRole A (Researcher): Summarize the core invariant concisely:",
        "\n\nRole B (Critic): Identify the main failure mode:",
        "\n\nRole C (Architect): Propose an optimization:"
    };
    const uint32_t gen_len = 16u;

    struct llama_model_params mparams = llama_model_default_params();
    struct llama_model *model = llama_model_load_from_file(model_path, mparams);
    if (model == NULL) {
        fprintf(stderr, "Failed to load GGUF model: %s\n", model_path);
        return 1;
    }

    const struct llama_vocab *vocab = llama_model_get_vocab(model);
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    int32_t n_tok = -llama_tokenize(vocab, prompt, (int32_t)strlen(prompt),
                                    NULL, 0, true, true);
    if (n_tok <= 1) {
        fprintf(stderr, "Tokenization failed or too short\n");
        llama_model_free(model);
        return 1;
    }

    llama_token *tokens = (llama_token *)malloc((size_t)n_tok * sizeof(llama_token));
    if (tokens == NULL ||
        llama_tokenize(vocab, prompt, (int32_t)strlen(prompt),
                       tokens, n_tok, true, true) < 0) {
        free(tokens);
        llama_model_free(model);
        return 1;
    }

    struct llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = (uint32_t)(n_tok + 128);
    cparams.n_batch = (uint32_t)(n_tok + 64);
    cparams.n_threads = 4;
    cparams.n_threads_batch = 4;

    struct llama_context *ctx = llama_init_from_model(model, cparams);
    if (ctx == NULL) {
        free(tokens);
        llama_model_free(model);
        return 1;
    }

    struct llama_batch batch = llama_batch_init(n_tok, 0, 1);
    batch.n_tokens = n_tok;
    for (int32_t i = 0; i < n_tok; ++i) {
        batch.token[i] = tokens[i];
        batch.pos[i] = (llama_pos)i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = (i == n_tok - 1) ? 1 : 0;
    }

    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "Producer llama_decode failed\n");
        llama_batch_free(batch);
        llama_free(ctx);
        free(tokens);
        llama_model_free(model);
        return 1;
    }
    llama_batch_free(batch);

    size_t state_size = llama_state_seq_get_size(ctx, 0);
    uint8_t *state_buf = (uint8_t *)malloc(state_size);
    if (state_buf == NULL || llama_state_seq_get_data(ctx, state_buf, state_size, 0) == 0) {
        fprintf(stderr, "llama_state_seq_get_data failed\n");
        free(state_buf);
        llama_free(ctx);
        free(tokens);
        llama_model_free(model);
        return 1;
    }

    char shm_name[128];
    snprintf(shm_name, sizeof(shm_name), "/vtlk_llama_paged_%d", VTLK_GETPID());

    vtlk_kv_arena_t arena;
    if (vtlk_kv_arena_create(&arena, shm_name, BRIDGE_ARENA_BLOCKS, BRIDGE_PAGE_BYTES) != VTLK_OK) {
        fprintf(stderr, "vtlk_kv_arena_create failed\n");
        free(state_buf);
        llama_free(ctx);
        free(tokens);
        llama_model_free(model);
        return 1;
    }

    vtlk_kv_identity_t expected_cfg;
    memset(&expected_cfg, 0, sizeof(expected_cfg));
    expected_cfg.token_prefix_hash = vtlk_kv_hash_tokens(0ULL, (const uint32_t *)tokens, (size_t)n_tok);
    expected_cfg.model_arch_hash = vtlk_kv_hash_bytes(model_path, strlen(model_path));
    expected_cfg.rope_config_hash = compute_synthetic_rope_hash(
        (int32_t)llama_model_rope_type(model),
        llama_model_rope_freq_scale_train(model),
        llama_model_n_ctx_train(model),
        llama_model_n_embd(model),
        llama_model_n_layer(model),
        llama_model_n_head_kv(model));
    expected_cfg.kv_dtype = VTLK_KV_DTYPE_RAW_LLAMA;
    expected_cfg.layer_count = (uint32_t)llama_model_n_layer(model);
    expected_cfg.kv_head_count = (uint32_t)llama_model_n_head_kv(model);
    expected_cfg.head_dim = (uint32_t)(llama_model_n_embd(model) /
                                       (llama_model_n_head(model) > 0 ? llama_model_n_head(model) : 1));

    vtlk_kv_branch_t root_branch;
    memset(&root_branch, 0, sizeof(root_branch));
    root_branch.model_arch_hash = expected_cfg.model_arch_hash;
    root_branch.rope_config_hash = expected_cfg.rope_config_hash;
    root_branch.kv_dtype = expected_cfg.kv_dtype;
    root_branch.layer_count = expected_cfg.layer_count;
    root_branch.head_dim = expected_cfg.head_dim;
    root_branch.kv_head_count = expected_cfg.kv_head_count;

    uint64_t snap_id = 0;
    if (partition_state_into_branch(&arena, &root_branch, state_buf, state_size,
                                    (uint32_t)n_tok, &expected_cfg) != 0 ||
        vtlk_kv_snapshot_create(&arena, &root_branch, &snap_id) != VTLK_OK) {
        fprintf(stderr, "Failed to snapshot paged KV blocks\n");
        vtlk_kv_branch_release(&arena, &root_branch);
        vtlk_kv_arena_close(&arena);
        vtlk_shm_unlink(shm_name);
        free(state_buf);
        llama_free(ctx);
        free(tokens);
        llama_model_free(model);
        return 1;
    }
    uint32_t paged_blocks = root_branch.block_count;
    vtlk_kv_branch_release(&arena, &root_branch);

    printf("[producer] prefill pos=%d, state=%zu B (%u paged blocks), snapshot_id=%llu\n",
           n_tok, state_size, paged_blocks, (unsigned long long)snap_id);

    int all_exact = 1;
    for (uint32_t w = 0; w < num_workers; ++w) {
        const char *suf = branch_suffixes[w % 3u];
        int32_t n_suf = -llama_tokenize(vocab, suf, (int32_t)strlen(suf),
                                        NULL, 0, false, true);
        llama_token *suf_toks = (llama_token *)malloc((size_t)n_suf * sizeof(llama_token));
        if (suf_toks == NULL ||
            llama_tokenize(vocab, suf, (int32_t)strlen(suf),
                           suf_toks, n_suf, false, true) < 0) {
            free(suf_toks);
            all_exact = 0;
            break;
        }

        /* 1) Baseline from scratch (prefill 0..n_tok-1, then decode suffix n_tok..n_tok+n_suf-1) */
        llama_token base_gen[16];
        float *base_logits = (float *)malloc((size_t)n_vocab * sizeof(float));
        {
            struct llama_context *bctx = llama_init_from_model(model, cparams);
            struct llama_batch bb0 = llama_batch_init(n_tok, 0, 1);
            bb0.n_tokens = n_tok;
            for (int32_t i = 0; i < n_tok; ++i) {
                bb0.token[i] = tokens[i];
                bb0.pos[i] = (llama_pos)i;
                bb0.n_seq_id[i] = 1;
                bb0.seq_id[i][0] = 0;
                bb0.logits[i] = 0;
            }
            llama_decode(bctx, bb0);
            llama_batch_free(bb0);

            struct llama_batch bb = llama_batch_init(n_suf, 0, 1);
            bb.n_tokens = n_suf;
            for (int32_t i = 0; i < n_suf; ++i) {
                bb.token[i] = suf_toks[i];
                bb.pos[i] = (llama_pos)(n_tok + i);
                bb.n_seq_id[i] = 1;
                bb.seq_id[i][0] = 0;
                bb.logits[i] = (i == n_suf - 1) ? 1 : 0;
            }
            llama_decode(bctx, bb);
            llama_batch_free(bb);

            const float *bl0 = NULL;
            base_gen[0] = greedy_next_token(bctx, n_vocab, &bl0);
            memcpy(base_logits, bl0, (size_t)n_vocab * sizeof(float));

            for (uint32_t g = 1; g < gen_len; ++g) {
                struct llama_batch gb = llama_batch_init(1, 0, 1);
                gb.n_tokens = 1;
                gb.token[0] = base_gen[g - 1];
                gb.pos[0] = (llama_pos)(n_tok + n_suf + (int32_t)g - 1);
                gb.n_seq_id[0] = 1;
                gb.seq_id[0][0] = 0;
                gb.logits[0] = 1;
                llama_decode(bctx, gb);
                llama_batch_free(gb);
                base_gen[g] = greedy_next_token(bctx, n_vocab, NULL);
            }
            llama_free(bctx);
        }

        /* 2) Vect-Link CoW Fork + Restore at pos = n_tok */
        vtlk_kv_branch_t forked;
        const void *ptrs[VTLK_KV_MAX_BRANCH_BLOCKS];
        if (vtlk_kv_restore(&arena, snap_id, &expected_cfg, &forked,
                            ptrs, VTLK_KV_MAX_BRANCH_BLOCKS) != VTLK_OK) {
            free(base_logits);
            free(suf_toks);
            all_exact = 0;
            break;
        }

        uint8_t *w_state = NULL;
        size_t w_bytes = 0;
        uint32_t w_pos = 0;
        llama_token cow_gen[16];
        double max_logit_diff = 0.0;
        uint32_t exact_tokens = 0u;

        if (reassemble_state_from_ptrs(ptrs, forked.block_count, forked.block_count,
                                       &w_state, &w_bytes, &w_pos) == 0) {
            struct llama_context *wctx = llama_init_from_model(model, cparams);
            llama_state_seq_set_data(wctx, w_state, w_bytes, 0);

            struct llama_batch wb = llama_batch_init(n_suf, 0, 1);
            wb.n_tokens = n_suf;
            for (int32_t i = 0; i < n_suf; ++i) {
                wb.token[i] = suf_toks[i];
                wb.pos[i] = (llama_pos)(w_pos + (uint32_t)i);
                wb.n_seq_id[i] = 1;
                wb.seq_id[i][0] = 0;
                wb.logits[i] = (i == n_suf - 1) ? 1 : 0;
            }
            llama_decode(wctx, wb);
            llama_batch_free(wb);

            const float *cl0 = NULL;
            cow_gen[0] = greedy_next_token(wctx, n_vocab, &cl0);
            for (int32_t v = 0; v < n_vocab; ++v) {
                double d = (double)cl0[v] - (double)base_logits[v];
                if (d < 0.0) d = -d;
                if (d > max_logit_diff) max_logit_diff = d;
            }

            for (uint32_t g = 1; g < gen_len; ++g) {
                struct llama_batch gb = llama_batch_init(1, 0, 1);
                gb.n_tokens = 1;
                gb.token[0] = cow_gen[g - 1];
                gb.pos[0] = (llama_pos)(n_tok + n_suf + (int32_t)g - 1);
                gb.n_seq_id[0] = 1;
                gb.seq_id[0][0] = 0;
                gb.logits[0] = 1;
                llama_decode(wctx, gb);
                llama_batch_free(gb);
                cow_gen[g] = greedy_next_token(wctx, n_vocab, NULL);
            }

            for (uint32_t g = 0; g < gen_len; ++g) {
                if (cow_gen[g] == base_gen[g]) {
                    exact_tokens++;
                }
            }
            if (exact_tokens != gen_len) {
                all_exact = 0;
            }
            printf("[branch %u] suffix_toks=%d | token_match=%u/%u | max_logit_diff=%.6f\n",
                   w, n_suf, exact_tokens, gen_len, max_logit_diff);

            llama_free(wctx);
            free(w_state);
        }
        vtlk_kv_branch_release(&arena, &forked);
        free(base_logits);
        free(suf_toks);
    }

    vtlk_kv_arena_close(&arena);
    vtlk_shm_unlink(shm_name);
    free(state_buf);
    llama_free(ctx);
    free(tokens);
    llama_model_free(model);
    return all_exact ? 0 : 1;
}

int main(int argc, char **argv)
{
    const char *model_path = NULL;
    const char *prompt = "You are a helpful AI assistant.";
    uint32_t workers = BRIDGE_DEFAULT_WORKERS;
    int self_test = 0;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--self-test") == 0) {
            self_test = 1;
        } else if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "--prompt") == 0 && i + 1 < argc) {
            prompt = argv[++i];
        } else if (strcmp(argv[i], "--workers") == 0 && i + 1 < argc) {
            workers = (uint32_t)atoi(argv[++i]);
            if (workers == 0) {
                workers = 1u;
            }
        } else {
            print_usage(argv[0]);
            return 1;
        }
    }

    if (self_test || model_path == NULL) {
        return run_self_test(workers);
    }

    return run_llama_model_mode(model_path, prompt, workers);
}
