# Vect-Link `llama.cpp` Latent Bridge (`llama_latent_bridge`)

Proof-of-concept connecting two independent OS processes running [`llama.cpp`](https://github.com/ggml-org/llama.cpp) over **Vect-Link Shared Memory IPC**, supporting two distinct modes with complementary roles:

- **`--mode kv-state` (Coherent Multi-Process Text Continuation):** Functional demonstration of cross-process inference handoff. Agent A evaluates the prompt (`embeddings = false`), packages the serialized sequence KV-Cache (`llama_state_seq_get_data`) prefixed with a 16-byte little-endian `vtlk_llama_kv_meta_t` header under `VTLK_QUANT_RAW` + `VTLK_FLAG_KV_STATE`, and transfers it via `vtlk_ring_push`. Agent B strictly validates compatibility, verifies `restored == meta.raw_state_bytes` from `llama_state_seq_set_data`, trims the last prompt position (`llama_memory_seq_rm`), re-evaluates `last_token_id` for deterministic logits, and generates semantically coherent continuation tokens without repeating full prefill.
- **`--mode embd` (INT4 Latent Tensor Transport & Reconstruction Benchmark):** Quantized hidden-state transport benchmark. Agent A extracts per-token `llama_get_embeddings_ith` vectors, quantizes `[n_tokens x d_model]` to `VTLK_QUANT_INT4`, and pushes via SHM; Agent B dequantizes back to FP32 and checks cosine similarity (`--verify-ref`). *(See Architecture Note below on why raw layer-N outputs re-injected into layer-0 `batch.embd` collapse semantically during text continuation.)*

Both modes share compatibility/wire helpers in [examples/llama_bridge_common.h](file:///c:/Users/sgi.lm/Desktop/Vect-Link/examples/llama_bridge_common.h), hashing:
1. GGUF exact byte size (`_stat64` / `stat`) + first 64 KB of GGUF headers/metadata,
2. Structural model invariants (`llama_model_desc`, `n_embd`, `n_layer`, `n_head`, `rope_type`, `rope_freq_scale_train`), and
3. In `--mode kv-state`: runtime build stamp (`LLAMA_STATE_SEQ_VERSION` + `llama_version()` / `LLAMA_BUILD_NUMBER`), `llama_n_ctx(ctx)`, and initial `llama_state_seq_get_size(ctx, 0)`.

---

## 1. Build `llama.cpp`

Clone and compile `llama.cpp` with shared libraries enabled so `llama_agent_a` and `llama_agent_b` can link cleanly against `llama` (and `ggml`):

```bash
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
cmake -B build -DBUILD_SHARED_LIBS=ON -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j
```

*(Note the root path to `llama.cpp`, e.g. `C:/src/llama.cpp` on Windows or `/opt/llama.cpp` on Linux — you will pass this as `-DLLAMA_CPP_DIR=<path>`.)*

---

## 2. Download a Compact GGUF Model

Both agents must load the **same** model family and embedding dimension (`d_model`), for example:

- **Qwen2.5 1.5B Instruct (Q4_K_M)** — e.g. `Qwen/Qwen2.5-1.5B-Instruct-GGUF` (`Qwen2.5-1.5B-Instruct-Q4_K_M.gguf`), or
- **Llama 3.2 1B Instruct (Q4_K_M)** — e.g. `bartowski/Llama-3.2-1B-Instruct-GGUF` (`Llama-3.2-1B-Instruct-Q4_K_M.gguf`).

Download with `huggingface-cli` (or browser/curl) into a local folder, e.g. `models/Llama-3.2-1B-Instruct-Q4_K_M.gguf`:

```bash
huggingface-cli download bartowski/Llama-3.2-1B-Instruct-GGUF Llama-3.2-1B-Instruct-Q4_K_M.gguf --local-dir ./models
```

---

## 3. Build Vect-Link with `VTLK_BUILD_LLAMA_BRIDGE=ON`

From the Vect-Link repository root (`VTLK_BUILD_LLAMA_BRIDGE` defaults to `OFF` so standard CI and core builds do not require `llama.cpp`):

### Linux / macOS
```bash
cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DVTLK_BUILD_LLAMA_BRIDGE=ON \
  -DLLAMA_CPP_DIR=/path/to/llama.cpp
cmake --build build -j
```

### Windows (PowerShell + Clang/Ninja or MSVC)
```powershell
cmake -B build `
  -DCMAKE_BUILD_TYPE=Release `
  -DVTLK_BUILD_LLAMA_BRIDGE=ON `
  -DLLAMA_CPP_DIR=C:/path/to/llama.cpp
cmake --build build --config Release -j
```

Make sure the directory containing `llama` / `ggml` shared libraries (`<llama.cpp>/build/bin` or `<llama.cpp>/build/bin/Release`) is in your `PATH` (Windows) or `LD_LIBRARY_PATH` (Linux) when running the executables.

---

## 4. Run Producer and Consumer in Two Terminals

Start **Agent A (Producer)** first and leave it waiting at `(press Enter after Agent B finishes)` while **Agent B (Consumer)** runs in the second terminal. *(On Windows, named shared memory objects are released as soon as the creating handle closes, and Agent B polls for up to 30 seconds on startup.)*

Both processes support `--mode embd` (default: INT4-quantized latent tensor transfer) and `--mode kv-state` (full sequence KV-Cache state transfer via `llama_state_seq_get_data` / `llama_state_seq_set_data` for coherent multi-process continuation without repeating prefill).

### Mode A: `--mode kv-state` (Coherent KV-Cache Transfer, recommended for text continuation)

**Terminal 1 — Producer (`llama_agent_a`):**
```powershell
.\build-llama\llama_agent_a.exe `
  --model C:\Users\sgi.lm\Desktop\models\qwen2.5-0.5b-instruct-q4_k_m.gguf `
  --mode kv-state `
  --prompt "Explain shared-memory zero-copy IPC in one paragraph:" `
  --shm /vtlk_llama_bridge
```

**Terminal 2 — Consumer (`llama_agent_b`):**
```powershell
.\build-llama\llama_agent_b.exe `
  --model C:\Users\sgi.lm\Desktop\models\qwen2.5-0.5b-instruct-q4_k_m.gguf `
  --mode kv-state `
  --shm /vtlk_llama_bridge `
  --n-predict 40
```

### Mode B: `--mode embd` (INT4-Quantized Latent Tensor Transfer)

**Terminal 1 — Producer (`llama_agent_a`):**
```powershell
.\build-llama\llama_agent_a.exe `
  --model C:\Users\sgi.lm\Desktop\models\qwen2.5-0.5b-instruct-q4_k_m.gguf `
  --mode embd `
  --prompt "Explain shared-memory zero-copy IPC in one paragraph:" `
  --shm /vtlk_llama_bridge
```

**Terminal 2 — Consumer (`llama_agent_b`):**
```powershell
.\build-llama\llama_agent_b.exe `
  --model C:\Users\sgi.lm\Desktop\models\qwen2.5-0.5b-instruct-q4_k_m.gguf `
  --mode embd `
  --shm /vtlk_llama_bridge `
  --n-predict 24
```

Once Agent B completes and exits, press **Enter** in Terminal 1 to release the shared-memory handle and exit Agent A cleanly.

---

## 5. Measured End-to-End Results (`qwen2.5-0.5b-instruct-q4_k_m.gguf`)

Measured locally on Windows (`Clang 22.1.8 -O3`, shared `llama.cpp` CPU backend) with prompt `"Explain shared-memory zero-copy IPC in one paragraph:"`:

### `--mode kv-state` (Sequence KV-Cache Transfer)
- **Model metadata validated:** `qwen2 1B Q4_K - Medium` (`model_family_hash=0x35e1f62d1fa40244`, `d_model=896`)
- **Transferred KV-Cache payload (`VTLK_FLAG_KV_STATE`, CRC32C protected):** `135,908 B` (`prompt=11 tokens`, `last_token=25`)
- **Producer (`llama_agent_a`) `llama_state_seq_get_data` + `vtlk_ring_push`:** **`338.20 us`**
- **Consumer (`llama_agent_b`) `vtlk_ring_pop` + `llama_state_seq_set_data`:** **`353.00 us`**
- **Combined KV-Cache IPC latency:** **`691.20 us`** (`~0.69 ms`)
- **Generated continuation (32 tokens):** coherent continuation directly resuming from the restored sequence state (`" Shared-memory zero-copy IPC is a type of inter-process communication (IPC) that allows two processes to communicate without exchanging any data. In this type of IPC,"`).

### `--mode embd` (INT4 Latent Tensor Transfer)
- **Model metadata validated:** `qwen2 1B Q4_K - Medium` (`model_family_hash=0x35e1f62d1fa40244`, `d_model=896`)
- **Tensor shape & compression:** `11 tokens x 896 dims` — `39,424 B FP32 -> 5,544 B INT4` (**`7.11x`** compression)
- **Producer (`llama_agent_a`) Encode + SHM Push:** **`28.80 us`**
- **Consumer (`llama_agent_b`) SHM Pop + INT4 Decode:** **`15.40 us`**
- **Combined IPC + Quantization/Dequantization latency:** **`44.20 us`**
- **Tensor reconstruction fidelity (`--dump-ref` / `--verify-ref`):** Cosine Similarity = **`0.985879`** (`Max Abs Error: 11.92352`)

### Architecture Note: Layer Alignment (`--mode embd` vs `--mode kv-state`)

- In `--mode embd`, `llama_get_embeddings_ith()` reads the **final-layer (Layer N, post-RMSNorm)** output hidden states, whereas `llama_batch.embd` injects vectors into **Layer 0**. Passing Layer N outputs back into Layer 0 collapses autoregressive generation into repetitive tokens (`" is is is..."`).
- `--mode kv-state` resolves this cleanly in `llama.cpp` by moving the accumulated sequence KV-Cache state (`llama_state_seq_get_data` → `VTLK` SHM frame with `VTLK_FLAG_KV_STATE` → `llama_state_seq_set_data`), trimming position `n_tokens - 1` via `llama_memory_seq_rm`, and re-evaluating `last_token` (`header.pos_offset`) to produce deterministic logits without repeating full prefill.

---

## 6. Fidelity Verification (`--dump-ref` / `--verify-ref`, `--mode embd`)

To measure end-to-end cosine similarity between the original FP32 embeddings in `llama_agent_a` and the INT4-reconstructed tensor received by `llama_agent_b`:

1. Run **Agent A** with `--dump-ref ref_embd.bin`:
   ```powershell
   .\build-llama\llama_agent_a.exe `
     --model C:\Users\sgi.lm\Desktop\models\qwen2.5-0.5b-instruct-q4_k_m.gguf `
     --mode embd `
     --prompt "Explain shared-memory zero-copy IPC in one paragraph:" `
     --shm /vtlk_llama_bridge `
     --dump-ref ref_embd.bin
   ```
2. Run **Agent B** with `--verify-ref ref_embd.bin` (or pass a raw prompt string to `--verify-ref` so Agent B computes reference embeddings locally for comparison):
   ```powershell
   .\build-llama\llama_agent_b.exe `
     --model C:\Users\sgi.lm\Desktop\models\qwen2.5-0.5b-instruct-q4_k_m.gguf `
     --mode embd `
     --shm /vtlk_llama_bridge `
     --n-predict 24 `
     --verify-ref ref_embd.bin
   ```

---

## 7. Model & Mode Compatibility Validation (Mismatch Rejection)

`llama_agent_b` validates:
1. `header.model_family_hash == local_family_hash`
2. `header.d_model == local_d_model`
3. `(header.flags & VTLK_FLAG_KV_STATE)` matches its own `--mode` (`embd` vs `kv-state`)

Mismatched models or mismatched `--mode` flags are rejected explicitly before touching `llama_decode` or `llama_state_seq_set_data`.
