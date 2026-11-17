/**
 * @file bench_llama_cow_apples.c
 * @brief Apples-to-Apples multi-agent KV-cache benchmark against native llama.cpp.
 *
 * Compares N = {1, 2, 4, 8} concurrent agents across 4 architectures:
 *   1. Cold Prefill per agent (N worker processes evaluating ~250-tok prompt)
 *   2. llama.cpp native temp file (llama_state_seq_save_file / load_file)
 *   3. llama.cpp in-process (llama_memory_seq_cp, 1 context, no isolation)
 *   4. Vect-Link CoW (SHM KVArena snapshot + fork + restore, N worker processes)
 *
 * Measures per N:
 *   - Pure fork / load / seq_cp latency (us)
 *   - Total TTFT from start/fork until first generated token (ms)
 *   - Total System RSS summed across all active processes (MiB)
 */

#ifndef _WIN32
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif
#endif

#include "bench_timer.h"
#include "llama.h"
#include "vectlink_kv.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <psapi.h>
#include <process.h>
#define APPLES_GETPID() ((int)_getpid())
#else
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#define APPLES_GETPID() ((int)getpid())
#endif

#define APPLES_MAX_WORKERS     8u
#define APPLES_GEN_TOKENS      16u
#define APPLES_PAGE_BYTES      (64u * 1024u)
#define APPLES_ARENA_BLOCKS    256u
#define APPLES_N_THREADS       2

#ifdef _WIN32
#define APPLES_DEFAULT_MODEL "C:\\Users\\sgi.lm\\Desktop\\models\\qwen2.5-0.5b-instruct-q4_k_m.gguf"
#else
#define APPLES_DEFAULT_MODEL "qwen2.5-0.5b-instruct-q4_k_m.gguf"
#endif

static const char *const APPLES_SYS_PROMPT =
    "You are an expert systems architect specializing in high-performance distributed "
    "inference engines, lock-free shared-memory IPC, copy-on-write KV-cache paging, "
    "and deterministic multi-agent orchestration. When analyzing a system design, you "
    "must evaluate memory footprint, zero-copy invariants, process isolation boundaries, "
    "cache-line contention, NUMA locality, and tail latency (p95/p99) under concurrent "
    "load. Always distinguish between theoretical single-process ceilings and true "
    "multi-process isolation where each agent runs in its own address space. Provide "
    "concise, technically rigorous answers grounded in exact hardware and OS behavior. "
    "Verify that positional encodings (RoPE), attention masks, and sequence identifiers "
    "remain strictly consistent across snapshot creation, branch forking, and state "
    "restoration. Never sacrifice memory safety or deterministic token equivalence for "
    "speculative throughput gains. Keep responses focused, structured, and directly "
    "actionable for systems engineers working on production LLM serving runtimes. "
    "When multiple concurrent agents share a common system prompt prefix, avoid "
    "redundant prefill computation and avoid duplicating immutable KV-cache pages "
    "across process boundaries. Instead, pin sealed KV blocks in shared memory, "
    "validate model architecture and RoPE hashes before zero-copy pointer resolution, "
    "and ensure deterministic greedy decoding matches cold-prefill baselines.";

static const char *const APPLES_SUFFIX_FMT =
    "\n\nAgent %u task: State the primary systems invariant in one short sentence.\nAssistant:";

typedef enum apples_mode {
    MODE_COLD = 0,
    MODE_FILE = 1,
    MODE_SEQ_CP = 2,
    MODE_COW = 3
} apples_mode_t;

typedef struct apples_worker_slot {
    double fork_us;
    double ttft_ms;
    double total_ms;
    uint64_t rss_bytes;
    int32_t first_token;
    int status;
} apples_worker_slot_t;

typedef struct apples_ctrl_shm {
    uint32_t magic;
    uint32_t num_workers;
    uint32_t ready_count;
    uint32_t go_flag;
    uint32_t done_count;
    uint32_t exit_flag;
    uint64_t snap_id;
    uint32_t n_sys_tokens;
    uint32_t reserved;
    vtlk_kv_identity_t expected_cfg;
    char arena_shm_name[128];
    char state_file_path[512];
    apples_worker_slot_t slots[APPLES_MAX_WORKERS];
} apples_ctrl_shm_t;

#define APPLES_CTRL_MAGIC 0x4150504Cu

static uint32_t atomic_load_u32(const volatile uint32_t *p)
{
#ifdef _MSC_VER
    return (uint32_t)_InterlockedCompareExchange((volatile long *)p, 0, 0);
#else
    return __atomic_load_n(p, __ATOMIC_ACQUIRE);
#endif
}

static void atomic_store_u32(volatile uint32_t *p, uint32_t v)
{
#ifdef _MSC_VER
    _InterlockedExchange((volatile long *)p, (long)v);
#else
    __atomic_store_n(p, v, __ATOMIC_RELEASE);
#endif
}

static uint32_t atomic_inc_u32(volatile uint32_t *p)
{
#ifdef _MSC_VER
    return (uint32_t)_InterlockedIncrement((volatile long *)p);
#else
    return __atomic_add_fetch(p, 1u, __ATOMIC_ACQ_REL);
#endif
}

static void apples_sleep_us(uint32_t us)
{
#ifdef _WIN32
    if (us >= 1000u) {
        Sleep(us / 1000u);
    } else {
        Sleep(0);
    }
#else
    usleep(us);
#endif
}

static void quiet_log_cb(enum ggml_log_level level, const char *text, void *user_data)
{
    (void)level;
    (void)text;
    (void)user_data;
}

static uint64_t get_self_rss_bytes(void)
{
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS pmc;
    memset(&pmc, 0, sizeof(pmc));
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        return (uint64_t)pmc.WorkingSetSize;
    }
    return 0u;
#else
    FILE *f = fopen("/proc/self/statm", "r");
    unsigned long size_pages = 0;
    unsigned long rss_pages = 0;
    if (f != NULL) {
        if (fscanf(f, "%lu %lu", &size_pages, &rss_pages) == 2) {
            fclose(f);
            return (uint64_t)rss_pages * (uint64_t)sysconf(_SC_PAGESIZE);
        }
        fclose(f);
    }
    return 0u;
#endif
}

static uint64_t compute_rope_hash(const struct llama_model *model)
{
    struct {
        int32_t rope_type;
        float freq_scale;
        int32_t n_ctx_train;
        int32_t n_embd;
        int32_t n_layer;
        int32_t n_head_kv;
    } cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.rope_type = (int32_t)llama_model_rope_type(model);
    cfg.freq_scale = llama_model_rope_freq_scale_train(model);
    cfg.n_ctx_train = llama_model_n_ctx_train(model);
    cfg.n_embd = llama_model_n_embd(model);
    cfg.n_layer = llama_model_n_layer(model);
    cfg.n_head_kv = llama_model_n_head_kv(model);
    return vtlk_kv_hash_bytes(&cfg, sizeof(cfg));
}

typedef struct page_header {
    uint32_t total_state_bytes;
    uint32_t page_index;
    uint32_t page_payload_bytes;
    uint32_t pos_n;
} page_header_t;

static int partition_state_into_branch(vtlk_kv_arena_t *arena,
                                       vtlk_kv_branch_t *branch,
                                       const uint8_t *state_buf,
                                       size_t state_bytes,
                                       uint32_t pos_n,
                                       const vtlk_kv_identity_t *base_id)
{
    const size_t max_payload = (size_t)arena->block_size_bytes - sizeof(page_header_t);
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
        if (vtlk_kv_block_alloc(arena, &block_id) != VTLK_OK) {
            return -1;
        }

        uint8_t *dst = (uint8_t *)vtlk_kv_block_data(arena, block_id);
        if (dst == NULL) {
            vtlk_kv_block_release(arena, block_id);
            return -1;
        }

        page_header_t phdr;
        phdr.total_state_bytes = (uint32_t)state_bytes;
        phdr.page_index = page_idx;
        phdr.page_payload_bytes = (uint32_t)chunk;
        phdr.pos_n = pos_n;

        memcpy(dst, &phdr, sizeof(phdr));
        memcpy(dst + sizeof(phdr), state_buf + offset, chunk);

        uint32_t page_tag = page_idx + 1u;
        rolling_hash = vtlk_kv_hash_tokens(rolling_hash, &page_tag, 1u);

        vtlk_kv_identity_t blk_id = *base_id;
        blk_id.token_prefix_hash = rolling_hash;
        blk_id.pos_offset = (page_idx == 0) ? 0u : pos_n;
        blk_id.token_count = (page_idx == 0) ? pos_n : 0u;

        if (vtlk_kv_block_seal(arena, block_id, &blk_id) != VTLK_OK ||
            vtlk_kv_branch_append_block(arena, branch, block_id) != VTLK_OK) {
            vtlk_kv_block_release(arena, block_id);
            return -1;
        }
        vtlk_kv_block_release(arena, block_id);

        offset += chunk;
        page_idx++;
    }
    return 0;
}

static int reassemble_state_from_ptrs(const void *const *ptrs,
                                      uint32_t block_count,
                                      uint8_t **out_buf,
                                      size_t *out_bytes,
                                      uint32_t *out_pos_n)
{
    if (block_count == 0) {
        return -1;
    }
    const page_header_t *hdr0 = (const page_header_t *)ptrs[0];
    size_t total = (size_t)hdr0->total_state_bytes;
    uint8_t *buf = (uint8_t *)malloc(total);
    if (buf == NULL) {
        return -1;
    }

    size_t cursor = 0;
    for (uint32_t i = 0; i < block_count; ++i) {
        const page_header_t *ph = (const page_header_t *)ptrs[i];
        const uint8_t *payload = (const uint8_t *)ptrs[i] + sizeof(page_header_t);
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

static llama_token greedy_sample_ith(struct llama_context *ctx, int32_t idx, int32_t n_vocab)
{
    const float *logits = llama_get_logits_ith(ctx, idx);
    int32_t best = 0;
    float best_v = logits[0];
    for (int32_t i = 1; i < n_vocab; ++i) {
        if (logits[i] > best_v) {
            best_v = logits[i];
            best = i;
        }
    }
    return (llama_token)best;
}

static int decode_tokens_seq(struct llama_context *ctx,
                           const llama_token *toks,
                           int32_t n_toks,
                           llama_pos pos_start,
                           llama_seq_id seq_id,
                           int want_last_logits)
{
    struct llama_batch batch = llama_batch_init(n_toks, 0, 1);
    batch.n_tokens = n_toks;
    for (int32_t i = 0; i < n_toks; ++i) {
        batch.token[i] = toks[i];
        batch.pos[i] = pos_start + (llama_pos)i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = seq_id;
        batch.logits[i] = (want_last_logits && i == n_toks - 1) ? 1 : 0;
    }
    int rc = llama_decode(ctx, batch);
    llama_batch_free(batch);
    return rc;
}

static int run_worker_process(const char *model_path,
                              const char *ctrl_shm_name,
                              apples_mode_t mode,
                              uint32_t worker_id)
{
    vtlk_shm_t ctrl_shm;
    if (vtlk_shm_open(&ctrl_shm, ctrl_shm_name, sizeof(apples_ctrl_shm_t)) != VTLK_OK) {
        fprintf(stderr, "[worker %u] vtlk_shm_open failed\n", worker_id);
        return 1;
    }
    apples_ctrl_shm_t *ctrl = (apples_ctrl_shm_t *)ctrl_shm.addr;
    if (ctrl->magic != APPLES_CTRL_MAGIC || worker_id >= ctrl->num_workers) {
        vtlk_shm_close(&ctrl_shm);
        return 1;
    }

    struct llama_model_params mparams = llama_model_default_params();
    struct llama_model *model = llama_model_load_from_file(model_path, mparams);
    if (model == NULL) {
        ctrl->slots[worker_id].status = -1;
        atomic_inc_u32(&ctrl->ready_count);
        atomic_inc_u32(&ctrl->done_count);
        vtlk_shm_close(&ctrl_shm);
        return 1;
    }

    const struct llama_vocab *vocab = llama_model_get_vocab(model);
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);

    int32_t n_sys = -llama_tokenize(vocab, APPLES_SYS_PROMPT, (int32_t)strlen(APPLES_SYS_PROMPT),
                                    NULL, 0, true, true);
    llama_token *sys_toks = (llama_token *)malloc((size_t)n_sys * sizeof(llama_token));
    if (sys_toks == NULL ||
        llama_tokenize(vocab, APPLES_SYS_PROMPT, (int32_t)strlen(APPLES_SYS_PROMPT),
                       sys_toks, n_sys, true, true) < 0) {
        free(sys_toks);
        llama_model_free(model);
        vtlk_shm_close(&ctrl_shm);
        return 1;
    }

    char suf_buf[256];
    snprintf(suf_buf, sizeof(suf_buf), APPLES_SUFFIX_FMT, worker_id);
    int32_t n_suf = -llama_tokenize(vocab, suf_buf, (int32_t)strlen(suf_buf), NULL, 0, false, true);
    llama_token *suf_toks = (llama_token *)malloc((size_t)n_suf * sizeof(llama_token));
    if (suf_toks == NULL ||
        llama_tokenize(vocab, suf_buf, (int32_t)strlen(suf_buf),
                       suf_toks, n_suf, false, true) < 0) {
        free(suf_toks);
        free(sys_toks);
        llama_model_free(model);
        vtlk_shm_close(&ctrl_shm);
        return 1;
    }

    struct llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = (uint32_t)(n_sys + n_suf + (int32_t)APPLES_GEN_TOKENS + 32);
    cparams.n_batch = (uint32_t)(n_sys + n_suf + 16);
    cparams.n_threads = APPLES_N_THREADS;
    cparams.n_threads_batch = APPLES_N_THREADS;

    struct llama_context *ctx = llama_init_from_model(model, cparams);
    if (ctx == NULL) {
        free(suf_toks);
        free(sys_toks);
        llama_model_free(model);
        vtlk_shm_close(&ctrl_shm);
        return 1;
    }

    vtlk_kv_arena_t arena;
    int arena_open = 0;
    if (mode == MODE_COW) {
        if (vtlk_kv_arena_open(&arena, ctrl->arena_shm_name) != VTLK_OK) {
            fprintf(stderr, "[worker %u] vtlk_kv_arena_open failed\n", worker_id);
            llama_free(ctx);
            free(suf_toks);
            free(sys_toks);
            llama_model_free(model);
            vtlk_shm_close(&ctrl_shm);
            return 1;
        }
        arena_open = 1;
    }

    /* Signal ready and wait for synchronized start */
    atomic_inc_u32(&ctrl->ready_count);
    while (atomic_load_u32(&ctrl->go_flag) == 0u) {
        apples_sleep_us(50u);
    }

    uint64_t t0 = vtlk_bench_now_ns();
    uint64_t t_fork_done = t0;
    uint64_t t_first_tok = t0;
    llama_pos cur_pos = (llama_pos)n_sys;
    int ok = 1;
    vtlk_kv_branch_t forked;
    int has_forked = 0;

    if (mode == MODE_COLD) {
        t_fork_done = t0;
        if (decode_tokens_seq(ctx, sys_toks, n_sys, 0, 0, 0) != 0 ||
            decode_tokens_seq(ctx, suf_toks, n_suf, (llama_pos)n_sys, 0, 1) != 0) {
            ok = 0;
        }
        cur_pos = (llama_pos)(n_sys + n_suf);
    } else if (mode == MODE_FILE) {
        llama_token *loaded = (llama_token *)malloc((size_t)(n_sys + 16) * sizeof(llama_token));
        size_t n_loaded = 0;
        if (loaded == NULL ||
            llama_state_seq_load_file(ctx, ctrl->state_file_path, 0,
                                      loaded, (size_t)(n_sys + 16), &n_loaded) == 0) {
            ok = 0;
        }
        t_fork_done = vtlk_bench_now_ns();
        free(loaded);
        if (ok && decode_tokens_seq(ctx, suf_toks, n_suf, (llama_pos)n_sys, 0, 1) != 0) {
            ok = 0;
        }
        cur_pos = (llama_pos)(n_sys + n_suf);
    } else if (mode == MODE_COW) {
        const void *ptrs[VTLK_KV_MAX_BRANCH_BLOCKS];
        if (vtlk_kv_fork(&arena, ctrl->snap_id, &forked) != VTLK_OK) {
            ok = 0;
        } else {
            has_forked = 1;
        }
        t_fork_done = vtlk_bench_now_ns();

        if (ok) {
            if (vtlk_kv_restore(&arena, ctrl->snap_id, &ctrl->expected_cfg,
                                NULL, ptrs, forked.block_count) != VTLK_OK) {
                ok = 0;
            }
        }
        if (ok) {
            uint8_t *w_state = NULL;
            size_t w_bytes = 0;
            uint32_t w_pos = 0;
            if (reassemble_state_from_ptrs(ptrs, forked.block_count,
                                           &w_state, &w_bytes, &w_pos) != 0 ||
                llama_state_seq_set_data(ctx, w_state, w_bytes, 0) == 0) {
                ok = 0;
            }
            free(w_state);
            cur_pos = (llama_pos)w_pos;
        }
        if (ok && decode_tokens_seq(ctx, suf_toks, n_suf, cur_pos, 0, 1) != 0) {
            ok = 0;
        }
        cur_pos += (llama_pos)n_suf;
    }

    llama_token prev_tok = 0;
    if (ok) {
        prev_tok = greedy_sample_ith(ctx, -1, n_vocab);
        t_first_tok = vtlk_bench_now_ns();

        for (uint32_t g = 1u; g < APPLES_GEN_TOKENS; ++g) {
            if (decode_tokens_seq(ctx, &prev_tok, 1, cur_pos, 0, 1) != 0) {
                ok = 0;
                break;
            }
            cur_pos++;
            prev_tok = greedy_sample_ith(ctx, -1, n_vocab);
        }
    }

    uint64_t t_end = vtlk_bench_now_ns();
    uint64_t rss = get_self_rss_bytes();

    ctrl->slots[worker_id].fork_us = (double)(t_fork_done - t0) / 1000.0;
    ctrl->slots[worker_id].ttft_ms = (double)(t_first_tok - t0) / 1000000.0;
    ctrl->slots[worker_id].total_ms = (double)(t_end - t0) / 1000000.0;
    ctrl->slots[worker_id].rss_bytes = rss;
    ctrl->slots[worker_id].first_token = (int32_t)prev_tok;
    ctrl->slots[worker_id].status = ok ? 0 : -1;

    /* Hold process alive until all N workers finish so RSS reflects concurrent residency */
    atomic_inc_u32(&ctrl->done_count);
    while (atomic_load_u32(&ctrl->exit_flag) == 0u) {
        apples_sleep_us(1000u);
    }

    if (has_forked) {
        vtlk_kv_branch_release(&arena, &forked);
    }
    if (arena_open) {
        vtlk_kv_arena_close(&arena);
    }
    llama_free(ctx);
    free(suf_toks);
    free(sys_toks);
    llama_model_free(model);
    vtlk_shm_close(&ctrl_shm);
    return ok ? 0 : 1;
}

static int spawn_and_run_workers(const char *model_path,
                                 apples_ctrl_shm_t *ctrl,
                                 const char *ctrl_shm_name,
                                 apples_mode_t mode,
                                 uint32_t n_workers)
{
#ifdef _WIN32
    char self_exe[MAX_PATH];
    PROCESS_INFORMATION procs[APPLES_MAX_WORKERS];
    if (GetModuleFileNameA(NULL, self_exe, MAX_PATH) == 0) {
        return -1;
    }
    memset(procs, 0, sizeof(procs));

    for (uint32_t w = 0; w < n_workers; ++w) {
        char cmdline[1024];
        STARTUPINFOA si;
        memset(&si, 0, sizeof(si));
        si.cb = sizeof(si);
        snprintf(cmdline, sizeof(cmdline),
                 "\"%s\" --worker %d --id %u --ctrl %s --model \"%s\"",
                 self_exe, (int)mode, w, ctrl_shm_name, model_path);
        if (!CreateProcessA(NULL, cmdline, NULL, NULL, FALSE, 0, NULL, NULL, &si, &procs[w])) {
            fprintf(stderr, "CreateProcessA failed for worker %u (err=%lu)\n",
                    w, (unsigned long)GetLastError());
            atomic_store_u32(&ctrl->go_flag, 1u);
            atomic_store_u32(&ctrl->exit_flag, 1u);
            return -1;
        }
    }

    while (atomic_load_u32(&ctrl->ready_count) < n_workers) {
        apples_sleep_us(500u);
    }
    atomic_store_u32(&ctrl->go_flag, 1u);

    while (atomic_load_u32(&ctrl->done_count) < n_workers) {
        apples_sleep_us(500u);
    }
    atomic_store_u32(&ctrl->exit_flag, 1u);

    for (uint32_t w = 0; w < n_workers; ++w) {
        WaitForSingleObject(procs[w].hProcess, INFINITE);
        CloseHandle(procs[w].hThread);
        CloseHandle(procs[w].hProcess);
    }
#else
    pid_t pids[APPLES_MAX_WORKERS];
    for (uint32_t w = 0; w < n_workers; ++w) {
        pid_t pid = fork();
        if (pid < 0) {
            return -1;
        }
        if (pid == 0) {
            int rc = run_worker_process(model_path, ctrl_shm_name, mode, w);
            _exit(rc);
        }
        pids[w] = pid;
    }

    while (atomic_load_u32(&ctrl->ready_count) < n_workers) {
        apples_sleep_us(500u);
    }
    atomic_store_u32(&ctrl->go_flag, 1u);

    while (atomic_load_u32(&ctrl->done_count) < n_workers) {
        apples_sleep_us(500u);
    }
    atomic_store_u32(&ctrl->exit_flag, 1u);

    for (uint32_t w = 0; w < n_workers; ++w) {
        int st = 0;
        waitpid(pids[w], &st, 0);
    }
#endif
    return 0;
}

typedef struct bench_row {
    const char *method;
    uint32_t n;
    double fork_us;
    double ttft_mean_ms;
    double ttft_max_ms;
    double rss_mib;
} bench_row_t;

static int run_seq_cp_baseline(struct llama_model *model,
                               const llama_token *sys_toks,
                               int32_t n_sys,
                               uint32_t n_workers,
                               bench_row_t *out_row)
{
    const struct llama_vocab *vocab = llama_model_get_vocab(model);
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);

    struct llama_context_params cparams = llama_context_default_params();
    cparams.n_seq_max = n_workers;
    cparams.kv_unified = true;
    cparams.n_ctx = (uint32_t)((int32_t)n_workers * (n_sys + 64));
    cparams.n_batch = (uint32_t)(n_sys + 64);
    cparams.n_threads = APPLES_N_THREADS;
    cparams.n_threads_batch = APPLES_N_THREADS;

    struct llama_context *ctx = llama_init_from_model(model, cparams);
    if (ctx == NULL) {
        return -1;
    }

    if (decode_tokens_seq(ctx, sys_toks, n_sys, 0, 0, 0) != 0) {
        llama_free(ctx);
        return -1;
    }

    llama_memory_t mem = llama_get_memory(ctx);
    uint64_t t0 = vtlk_bench_now_ns();
    for (uint32_t w = 1u; w < n_workers; ++w) {
        llama_memory_seq_cp(mem, 0, (llama_seq_id)w, -1, -1);
    }
    uint64_t t_cp = vtlk_bench_now_ns();

    double ttft_sum = 0.0;
    double ttft_max = 0.0;
    llama_token cur_toks[APPLES_MAX_WORKERS];
    llama_pos cur_pos[APPLES_MAX_WORKERS];

    for (uint32_t w = 0u; w < n_workers; ++w) {
        char suf_buf[256];
        snprintf(suf_buf, sizeof(suf_buf), APPLES_SUFFIX_FMT, w);
        int32_t n_suf = -llama_tokenize(vocab, suf_buf, (int32_t)strlen(suf_buf), NULL, 0, false, true);
        llama_token *suf_toks = (llama_token *)malloc((size_t)n_suf * sizeof(llama_token));
        if (suf_toks == NULL ||
            llama_tokenize(vocab, suf_buf, (int32_t)strlen(suf_buf), suf_toks, n_suf, false, true) < 0 ||
            decode_tokens_seq(ctx, suf_toks, n_suf, (llama_pos)n_sys, (llama_seq_id)w, 1) != 0) {
            free(suf_toks);
            llama_free(ctx);
            return -1;
        }
        free(suf_toks);

        cur_toks[w] = greedy_sample_ith(ctx, -1, n_vocab);
        cur_pos[w] = (llama_pos)(n_sys + n_suf);
        uint64_t t_first = vtlk_bench_now_ns();
        double ttft = (double)(t_first - t0) / 1000000.0;
        ttft_sum += ttft;
        if (ttft > ttft_max) {
            ttft_max = ttft;
        }
    }

    for (uint32_t g = 1u; g < APPLES_GEN_TOKENS; ++g) {
        for (uint32_t w = 0u; w < n_workers; ++w) {
            if (decode_tokens_seq(ctx, &cur_toks[w], 1, cur_pos[w], (llama_seq_id)w, 1) != 0) {
                llama_free(ctx);
                return -1;
            }
            cur_pos[w]++;
            cur_toks[w] = greedy_sample_ith(ctx, -1, n_vocab);
        }
    }

    uint64_t rss = get_self_rss_bytes();
    llama_free(ctx);

    out_row->method = "llama_memory_seq_cp (1 proc, no isol)";
    out_row->n = n_workers;
    out_row->fork_us = (n_workers > 1u) ? ((double)(t_cp - t0) / 1000.0) / (double)(n_workers - 1u) : 0.0;
    out_row->ttft_mean_ms = ttft_sum / (double)n_workers;
    out_row->ttft_max_ms = ttft_max;
    out_row->rss_mib = (double)rss / (1024.0 * 1024.0);
    return 0;
}

static void aggregate_worker_row(const char *label,
                                 const apples_ctrl_shm_t *ctrl,
                                 uint32_t n_workers,
                                 bench_row_t *out_row)
{
    double fork_sum = 0.0;
    double ttft_sum = 0.0;
    double ttft_max = 0.0;
    uint64_t rss_sum = 0u;
    for (uint32_t w = 0; w < n_workers; ++w) {
        fork_sum += ctrl->slots[w].fork_us;
        ttft_sum += ctrl->slots[w].ttft_ms;
        if (ctrl->slots[w].ttft_ms > ttft_max) {
            ttft_max = ctrl->slots[w].ttft_ms;
        }
        rss_sum += ctrl->slots[w].rss_bytes;
    }
    out_row->method = label;
    out_row->n = n_workers;
    out_row->fork_us = fork_sum / (double)n_workers;
    out_row->ttft_mean_ms = ttft_sum / (double)n_workers;
    out_row->ttft_max_ms = ttft_max;
    out_row->rss_mib = (double)rss_sum / (1024.0 * 1024.0);
}

int main(int argc, char **argv)
{
    const char *model_path = APPLES_DEFAULT_MODEL;
    int is_worker = 0;
    apples_mode_t worker_mode = MODE_COLD;
    uint32_t worker_id = 0u;
    const char *ctrl_shm_arg = NULL;
    uint32_t single_n = 0u;
    int verbose = 0;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "--worker") == 0 && i + 1 < argc) {
            is_worker = 1;
            worker_mode = (apples_mode_t)atoi(argv[++i]);
        } else if (strcmp(argv[i], "--id") == 0 && i + 1 < argc) {
            worker_id = (uint32_t)atoi(argv[++i]);
        } else if (strcmp(argv[i], "--ctrl") == 0 && i + 1 < argc) {
            ctrl_shm_arg = argv[++i];
        } else if (strcmp(argv[i], "--n") == 0 && i + 1 < argc) {
            single_n = (uint32_t)atoi(argv[++i]);
        } else if (strcmp(argv[i], "--verbose") == 0) {
            verbose = 1;
        }
    }

    if (!verbose) {
        llama_log_set(quiet_log_cb, NULL);
    }

    if (is_worker) {
        if (ctrl_shm_arg == NULL) {
            return 1;
        }
        return run_worker_process(model_path, ctrl_shm_arg, worker_mode, worker_id);
    }

    /* Parent / Orchestrator */
    struct llama_model_params mparams = llama_model_default_params();
    struct llama_model *model = llama_model_load_from_file(model_path, mparams);
    if (model == NULL) {
        fprintf(stderr, "Failed to load GGUF model: %s\n", model_path);
        return 1;
    }

    const struct llama_vocab *vocab = llama_model_get_vocab(model);
    int32_t n_sys = -llama_tokenize(vocab, APPLES_SYS_PROMPT, (int32_t)strlen(APPLES_SYS_PROMPT),
                                    NULL, 0, true, true);
    llama_token *sys_toks = (llama_token *)malloc((size_t)n_sys * sizeof(llama_token));
    if (sys_toks == NULL ||
        llama_tokenize(vocab, APPLES_SYS_PROMPT, (int32_t)strlen(APPLES_SYS_PROMPT),
                       sys_toks, n_sys, true, true) < 0) {
        free(sys_toks);
        llama_model_free(model);
        return 1;
    }

    /* Producer: prefill system prompt once to obtain state_file and SHM CoW snapshot */
    struct llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = (uint32_t)(n_sys + 64);
    cparams.n_batch = (uint32_t)(n_sys + 16);
    cparams.n_threads = APPLES_N_THREADS;
    cparams.n_threads_batch = APPLES_N_THREADS;

    struct llama_context *pctx = llama_init_from_model(model, cparams);
    if (pctx == NULL || decode_tokens_seq(pctx, sys_toks, n_sys, 0, 0, 0) != 0) {
        fprintf(stderr, "Producer prefill failed\n");
        if (pctx != NULL) llama_free(pctx);
        free(sys_toks);
        llama_model_free(model);
        return 1;
    }

    char state_file[512];
#ifdef _WIN32
    char tmp_dir[MAX_PATH];
    GetTempPathA(MAX_PATH, tmp_dir);
    snprintf(state_file, sizeof(state_file), "%svtlk_apples_seq0_%d.bin", tmp_dir, APPLES_GETPID());
#else
    snprintf(state_file, sizeof(state_file), "/tmp/vtlk_apples_seq0_%d.bin", APPLES_GETPID());
#endif
    size_t saved_bytes = llama_state_seq_save_file(pctx, state_file, 0, sys_toks, (size_t)n_sys);
    if (saved_bytes == 0) {
        fprintf(stderr, "llama_state_seq_save_file failed\n");
        llama_free(pctx);
        free(sys_toks);
        llama_model_free(model);
        return 1;
    }

    size_t state_size = llama_state_seq_get_size(pctx, 0);
    uint8_t *state_buf = (uint8_t *)malloc(state_size);
    if (state_buf == NULL || llama_state_seq_get_data(pctx, state_buf, state_size, 0) == 0) {
        fprintf(stderr, "llama_state_seq_get_data failed\n");
        free(state_buf);
        llama_free(pctx);
        free(sys_toks);
        llama_model_free(model);
        return 1;
    }

    char arena_shm_name[128];
    snprintf(arena_shm_name, sizeof(arena_shm_name), "/vtlk_apples_arena_%d", APPLES_GETPID());
    (void)vtlk_shm_unlink(arena_shm_name);

    vtlk_kv_arena_t arena;
    if (vtlk_kv_arena_create(&arena, arena_shm_name, APPLES_ARENA_BLOCKS, APPLES_PAGE_BYTES) != VTLK_OK) {
        fprintf(stderr, "vtlk_kv_arena_create failed\n");
        free(state_buf);
        llama_free(pctx);
        free(sys_toks);
        llama_model_free(model);
        return 1;
    }

    vtlk_kv_identity_t expected_cfg;
    memset(&expected_cfg, 0, sizeof(expected_cfg));
    expected_cfg.token_prefix_hash = vtlk_kv_hash_tokens(0ULL, (const uint32_t *)sys_toks, (size_t)n_sys);
    expected_cfg.model_arch_hash = vtlk_kv_hash_bytes(model_path, strlen(model_path));
    expected_cfg.rope_config_hash = compute_rope_hash(model);
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
                                    (uint32_t)n_sys, &expected_cfg) != 0 ||
        vtlk_kv_snapshot_create(&arena, &root_branch, &snap_id) != VTLK_OK) {
        fprintf(stderr, "Failed to create CoW snapshot\n");
        vtlk_kv_branch_release(&arena, &root_branch);
        vtlk_kv_arena_close(&arena);
        vtlk_shm_unlink(arena_shm_name);
        free(state_buf);
        llama_free(pctx);
        free(sys_toks);
        llama_model_free(model);
        return 1;
    }
    uint32_t snap_blocks = root_branch.block_count;
    vtlk_kv_branch_release(&arena, &root_branch);
    free(state_buf);
    llama_free(pctx);

    printf("### Vect-Link vs native llama.cpp Apples-to-Apples Benchmark\n");
    printf("- Model: `%s`\n", model_path);
    printf("- Shared System Prompt: `%d` tokens | Suffix: ~15 tokens | Generation: `%u` tokens/agent | Threads/agent: `%d`\n",
           n_sys, APPLES_GEN_TOKENS, APPLES_N_THREADS);
    printf("- KV State Size: `%.2f MiB` (`%zu` B, `%u` CoW pages x `%u` KiB) | File State: `%.2f MiB`\n\n",
           (double)state_size / (1024.0 * 1024.0), state_size,
           snap_blocks, APPLES_PAGE_BYTES / 1024u,
           (double)saved_bytes / (1024.0 * 1024.0));

    printf("| N | Method                                  | Pure Fork/Load (us) | TTFT Mean (ms) | TTFT Max (ms) | Total System RSS (MiB) |\n");
    printf("|---|-----------------------------------------|---------------------|----------------|---------------|------------------------|\n");
    fflush(stdout);

    static const uint32_t n_list[4] = { 1u, 2u, 4u, 8u };
    for (size_t ni = 0; ni < 4u; ++ni) {
        uint32_t n = n_list[ni];
        if (single_n != 0u && n != single_n) {
            continue;
        }

        char ctrl_name[128];
        snprintf(ctrl_name, sizeof(ctrl_name), "/vtlk_apples_ctrl_%d_%u", APPLES_GETPID(), n);
        (void)vtlk_shm_unlink(ctrl_name);

        vtlk_shm_t ctrl_shm;
        if (vtlk_shm_create(&ctrl_shm, ctrl_name, sizeof(apples_ctrl_shm_t)) != VTLK_OK) {
            fprintf(stderr, "vtlk_shm_create failed\n");
            break;
        }
        apples_ctrl_shm_t *ctrl = (apples_ctrl_shm_t *)ctrl_shm.addr;

        static const struct {
            apples_mode_t mode;
            const char *label;
        } mp_modes[3] = {
            { MODE_COLD, "Cold Prefill (N procs, from scratch)" },
            { MODE_FILE, "llama_state_seq_load_file (N procs)" },
            { MODE_COW,  "Vect-Link CoW (N procs, SHM Arena)" }
        };

        for (size_t mi = 0; mi < 3u; ++mi) {
            memset(ctrl, 0, sizeof(*ctrl));
            ctrl->magic = APPLES_CTRL_MAGIC;
            ctrl->num_workers = n;
            ctrl->snap_id = snap_id;
            ctrl->n_sys_tokens = (uint32_t)n_sys;
            ctrl->expected_cfg = expected_cfg;
            snprintf(ctrl->arena_shm_name, sizeof(ctrl->arena_shm_name), "%s", arena_shm_name);
            snprintf(ctrl->state_file_path, sizeof(ctrl->state_file_path), "%s", state_file);

            bench_row_t row;
            if (spawn_and_run_workers(model_path, ctrl, ctrl_name, mp_modes[mi].mode, n) == 0) {
                aggregate_worker_row(mp_modes[mi].label, ctrl, n, &row);
                printf("| %u | %-39s | %19.2f | %14.2f | %13.2f | %22.2f |\n",
                       row.n, row.method, row.fork_us, row.ttft_mean_ms, row.ttft_max_ms, row.rss_mib);
                fflush(stdout);
            }
        }

        bench_row_t seq_row;
        if (run_seq_cp_baseline(model, sys_toks, n_sys, n, &seq_row) == 0) {
            printf("| %u | %-39s | %19.2f | %14.2f | %13.2f | %22.2f |\n",
                   seq_row.n, seq_row.method, seq_row.fork_us, seq_row.ttft_mean_ms, seq_row.ttft_max_ms, seq_row.rss_mib);
            fflush(stdout);
        }

        vtlk_shm_close(&ctrl_shm);
        (void)vtlk_shm_unlink(ctrl_name);
    }

    remove(state_file);
    vtlk_kv_arena_close(&arena);
    (void)vtlk_shm_unlink(arena_shm_name);
    free(sys_toks);
    llama_model_free(model);
    return 0;
}
