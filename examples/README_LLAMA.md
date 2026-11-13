# Vect-Link `llama.cpp` Latent Bridge (`llama_latent_bridge`)

Proof-of-concept connecting two independent OS processes running [`llama.cpp`](https://github.com/ggml-org/llama.cpp) via **Vect-Link Binary INT4 + Shared Memory** instead of text/JSON serialization.

- **Producer (`llama_agent_a`)**: Loads a GGUF model, tokenizes a prompt, executes a forward pass with embeddings enabled (`LLAMA_POOLING_TYPE_NONE`), hashes `llama_model_desc` (64-bit FNV-1a) into `header.model_family_hash` along with `d_model = llama_model_n_embd(model)`, quantizes the `[n_tokens x d_model]` hidden-state / embedding tensor to `VTLK_QUANT_INT4`, and pushes it into the Vect-Link shared-memory ring buffer.
- **Consumer (`llama_agent_b`)**: Loads its GGUF model, attaches to the shared-memory ring buffer, strictly validates `model_family_hash` and `d_model` (rejecting mismatched models), decodes the `VTLK_QUANT_INT4` tensor back to FP32, injects the vectors directly into `llama_batch.embd` (no tokens), runs `llama_decode`, and autoregressively generates `--n-predict` tokens using a greedy sampler.

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

Start **Agent A (Producer)** first and leave it waiting at `(press Enter after Agent B finishes)` while **Agent B (Consumer)** runs in the second terminal. *(On Windows, named shared memory objects are released as soon as the creating handle closes, and Agent B polls for up to 6 seconds on startup.)*

### Terminal 1 — Producer (`llama_agent_a`)
```bash
./build/llama_agent_a \
  --model ./models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  --prompt "Explain shared-memory zero-copy IPC in one paragraph:" \
  --shm /vtlk_llama_bridge
```

### Terminal 2 — Consumer (`llama_agent_b`)
```bash
./build/llama_agent_b \
  --model ./models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  --shm /vtlk_llama_bridge \
  --n-predict 24
```

Once Agent B completes and exits, press **Enter** in Terminal 1 to release the shared-memory handle and exit Agent A cleanly.

---

## 5. Measured End-to-End Results (`qwen2.5-0.5b-instruct-q4_k_m.gguf`)

Measured locally on Windows (`Clang 22.1.8 -O3`, shared `llama.cpp` CPU backend) with prompt `"Explain shared-memory zero-copy IPC in one paragraph:"`:

- **Model metadata validated:** `qwen2 1B Q4_K - Medium` (`model_family_hash=0x35e1f62d1fa40244`, `d_model=896`)
- **Tensor shape & compression:** `11 tokens x 896 dims` — `39,424 B FP32 -> 5,544 B INT4` (**`7.11x`** compression)
- **Producer (`llama_agent_a`) Encode + SHM Push:** **`30.50 us`**
- **Consumer (`llama_agent_b`) SHM Pop + INT4 Decode:** **`22.70 us`**
- **Combined IPC + Quantization/Dequantization latency:** **`53.20 us`**
- **Tensor reconstruction fidelity (`--dump-ref` / `--verify-ref`):** Cosine Similarity = **`0.985879`** (`Max Abs Error: 11.92352`)

### Architecture Note: Layer Alignment & Next Step (KV-Cache Transfer)

This PoC proves live cross-process extraction, `INT4_BLOCK` quantization, shared-memory ring-buffer transport, and strict `model_family_hash` / `d_model` validation across two independent `llama.cpp` OS processes in **~53 us** total.

Notice, however, why raw text continuation from `llama_agent_b` currently collapses into repetitive tokens (e.g. `" is is is..."`):
- `llama_get_embeddings_ith()` reads the **final-layer (Layer N, post-RMSNorm)** output hidden states.
- `llama_batch.embd` in `llama.cpp` injects continuous embeddings into **Layer 0** (the model input space). Re-injecting Layer N representations into Layer 0 passes those vectors through all transformer blocks a second time, producing a layer-misalignment attractor collapse.

**Natural next step:** To achieve 100% semantically coherent multi-process continuation without repeating prefill, either:
1. **Transfer Sequence KV-Cache State:** Extract accumulated `K`/`V` context state with `llama_state_seq_get_size()` / `llama_state_seq_get_data()`, transport it over a `VTLK` frame (`VTLK_QUANT_FP16` or `VTLK_QUANT_INT4`), and restore it in Agent B via `llama_state_seq_set_data()` before calling `llama_decode()` on the final prompt token, or
2. **Layer-to-Layer Adapter:** Train/project a lightweight linear adapter mapping Layer N hidden states back into Layer 0 soft-prompt space.

---

## 6. Fidelity Verification (`--dump-ref` / `--verify-ref`)

To measure end-to-end cosine similarity between the original FP32 embeddings in `llama_agent_a` and the INT4-reconstructed tensor received by `llama_agent_b`:

1. Run **Agent A** with `--dump-ref ref_embd.bin`:
   ```bash
   ./build/llama_agent_a \
     --model ./models/Llama-3.2-1B-Instruct-Q4_K_M.gguf \
     --prompt "Explain shared-memory zero-copy IPC in one paragraph:" \
     --shm /vtlk_llama_bridge \
     --dump-ref ref_embd.bin
   ```
2. Run **Agent B** with `--verify-ref ref_embd.bin` (or pass a raw prompt string to `--verify-ref` so Agent B computes reference embeddings locally for comparison):
   ```bash
   ./build/llama_agent_b \
     --model ./models/Llama-3.2-1B-Instruct-Q4_K_M.gguf \
     --shm /vtlk_llama_bridge \
     --n-predict 32 \
     --verify-ref ref_embd.bin
   ```

---

## 7. Model Compatibility Validation (Mismatch Rejection)

`llama_agent_b` computes the 64-bit FNV-1a hash of its own loaded `llama_model_desc` and checks both:
1. `header.model_family_hash == local_family_hash`
2. `header.d_model == local_d_model`

If you run `llama_agent_a` with e.g. `Qwen2.5-1.5B` and `llama_agent_b` with `Llama-3.2-1B`, `llama_agent_b` rejects the frame with an explicit mismatch diagnostic instead of feeding incompatible latent representations into `llama_decode`.
