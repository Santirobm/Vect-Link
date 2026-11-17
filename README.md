# Vect-Link

**Git for LLM Context — A deterministic, zero-dependency C99 Copy-on-Write (CoW) KV-cache store and zero-copy IPC engine for multi-agent local LLMs.**

[![Language](https://img.shields.io/badge/C-C99-blue.svg)](#building--testing)
[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB.svg)](#python-ffi--langgraph-integration)
[![Platforms](https://img.shields.io/badge/Platforms-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey.svg)](#architecture)
[![Sanitizers](https://img.shields.io/badge/ASan%20%2B%20UBSan-Passing-brightgreen.svg)](#building--testing)

Vect-Link eliminates redundant prefill computation and memory duplication when multiple processes or AI agents share context. Instead of serializing monolithic KV-cache blobs over sockets or disk, Vect-Link manages paged, immutable KV blocks in shared memory with atomic reference counting, instant `Snapshot / Fork / Restore` branching, and transparent L1/L2/L3 tiering.

---

## Why Vect-Link?

When multiple agents share a long system prompt—or when a reasoning pipeline branches into parallel hypotheses (Tree-of-Thought, A/B critics, multi-agent exploration)—traditional inference setups either recompute the prefix per worker or copy tens of megabytes of KV state per branch.

Vect-Link separates the **Data Plane** (paged CoW shared memory arena) from the **Control Plane** (compact descriptor rings and native OS notifications):

* **Zero-Copy Branching (`Fork` in ~4.9–5.9 µs):** Forking a context duplicates only a lightweight table of block IDs (`block_ids[]`) and increments atomic reference counts (`refcount`). Vect-Link never copies tensor data across processes or in intermediate buffers up to the inference API boundary.
* **Bit-Exact RoPE Preservation:** Because all forked branches continue from exact token position $pos = N$, Rotary Position Embeddings (RoPE) remain 100% valid with **zero perplexity or logit degradation** (`max_logit_diff = 0.000000` verified against cold prefill).
* **Strict Block Identity Contract:** Blocks are content-addressed and validated against the full model compatibility tuple—preventing silent corruption across mismatched models, RoPE configurations, quantization formats, or LoRA adapters.
* **Zero External C Dependencies:** Pure C99 core with deterministic memory management, AVX2/SSE4.2 acceleration, and native cross-platform OS primitives (Linux, macOS & Windows).

---

## Architecture

```text
+-----------------------------------------------------------------------+
|                        ECOSYSTEM & BINDINGS                           |
|   llama.cpp Paged Bridge   |   Python FFI (memoryview)   |  LangGraph |
+-----------------------------------------------------------------------+
                                   |
          +------------------------+------------------------+
          v                                                 v
+-----------------------------------+   +-------------------------------+
|          CONTROL PLANE            |   |          DATA PLANE           |
|  - Descriptor Rings (vtlk_desc)   |   |  - Paged SHM Arena (CoW)      |
|  - Protocol v2 (64-bit payloads)  |   |  - 128B Cache-Aligned Headers |
|  - OS Wait/Wake (futex /          |   |  - Content-Addressed HashIdx  |
|    WaitOnAddress)                 |   |  - Lock-Free ABA Free-List    |
+-----------------------------------+   +-------------------------------+
                                                            |
                                        +-------------------+-----------+
                                        |   HIERARCHICAL TIERING (LRU)  |
                                        |  L1: Shared Memory RAM        |
                                        |  L2: Warm mmap / MapViewOfFile|
                                        |  L3: Cold SSD + CRC32C Verify |
                                        +-------------------------------+
```

### Core Subsystems

1. **Paged CoW SHM Arena (`include/vectlink_kv.h`, `src/kv/`):**
   * Fixed-size slab pages managed via a 32-bit ABA-tagged lock-free free-list (`((gen + 1) << 32) | block_id`).
   * **128-byte (2 cache-line) block header (`vtlk_kv_block_header_t`):** Stores `token_prefix_hash`, `model_arch_hash`, `rope_config_hash`, `lora_adapter_hash`, `kv_dtype`, `layer_count`, `head_dim`, `kv_head_count`, `storage_format_version`, CRC32C checksum, and atomic `refcount`.
   * Immutable once sealed: supports 1:N concurrent readers (*Broadcast*) without locks. `lru_stamp` is updated strictly on `1 -> 0` transitions to avoid read-path cache-line bouncing.
2. **Hierarchical L1 / L2 / L3 Storage (`include/vectlink_tier.h`, `src/kv/kv_tier.c`):**
   * **L1 (SHM RAM):** Immediate zero-copy pointer access.
   * **L2 (Warm `mmap`):** Spills inactive (`refcount == 0`) blocks to memory-mapped files when L1 crosses its high watermark.
   * **L3 (Cold SSD):** Persistent disk store with recoverable index (`vtlk_tier_index.bin`) and CRC32C validation upon transparent promotion back to L1 during `vtlk_kv_restore()` / `vtlk_kv_block_retain()`.
3. **OS-Level Security & Synchronization (`include/vectlink_sync.h`, `src/sync/`):**
   * Shared memory segments are created with strict per-user isolation (`0600` permissions + UID check on POSIX; user-SID restricted DACL on Windows).
   * Replaces busy-spinning with `SYS_futex` (Linux) and `WaitOnAddress` (Windows).
   * Preserves full ABI compatibility with v1 SPSC rings (`vtlk_ring_*`) alongside v2 64-bit payload headers (`vtlk_header_v2_t`).

---

## Benchmarks & Verification

### 1. Apples-to-Apples vs. Native `llama.cpp` (N = 1, 2, 4, 8 Concurrent Agents)

Measured via [benchmarks/bench_llama_cow_apples.c](benchmarks/bench_llama_cow_apples.c) and [benchmarks/bench_cow_git.py](benchmarks/bench_cow_git.py) using **`qwen2.5-0.5b-instruct-q4_k_m.gguf`** (`242` shared system prompt tokens, ~15 divergent suffix tokens, `16` generated tokens per agent, `2` threads/agent, `2.84 MiB` KV state = `46` CoW pages × `64 KiB`):

| N | Method | Pure Fork / Load (µs) | TTFT Mean (ms) | TTFT Max (ms) | Total System RSS (MiB) |
|---:|:---|---:|---:|---:|---:|
| **1** | Cold Prefill (`N` procs, from scratch) | 0.00 | 1,425.51 | 1,425.51 | 474.67 |
| **1** | `llama_state_seq_load_file` (`N` procs) | 3,080.10 | 167.70 | 167.70 | 457.49 |
| **1** | **Vect-Link CoW (Direct Zero-Copy)** | **5.80** | **170.12** | **170.12** | **460.48** |
| **1** | Vect-Link CoW (Reassembled Fallback) | 4.50 | 165.51 | 165.51 | 460.46 |
| **1** | `llama_memory_seq_cp` (1 proc, no isolation) | 0.00 | 134.01 | 134.01 | 492.38 |
| **2** | Cold Prefill (`N` procs, from scratch) | 0.00 | 1,453.20 | 1,471.89 | 949.27 |
| **2** | `llama_state_seq_load_file` (`N` procs) | 3,596.10 | 178.59 | 178.66 | 915.07 |
| **2** | **Vect-Link CoW (Direct Zero-Copy)** | **4.90** | **174.67** | **175.80** | **920.83** |
| **2** | Vect-Link CoW (Reassembled Fallback) | 6.30 | 173.76 | 175.26 | 920.97 |
| **2** | `llama_memory_seq_cp` (1 proc, no isolation) | 19.10 | 169.99 | 227.17 | 495.95 |
| **4** | Cold Prefill (`N` procs, from scratch) | 0.00 | 2,817.96 | 3,102.44 | 1,898.62 |
| **4** | `llama_state_seq_load_file` (`N` procs) | 5,077.12 | 319.30 | 388.20 | 1,830.05 |
| **4** | **Vect-Link CoW (Direct Zero-Copy)** | **7.75** | **303.88** | **367.45** | **1,841.71** |
| **4** | Vect-Link CoW (Reassembled Fallback) | 6.12 | 315.60 | 397.39 | 1,841.70 |
| **4** | `llama_memory_seq_cp` (1 proc, no isolation) | 12.13 | 289.70 | 465.60 | 503.17 |
| **8** | Cold Prefill (`N` procs, from scratch) | 0.00 | 3,734.32 | 4,311.97 | 3,797.55 |
| **8** | `llama_state_seq_load_file` (`N` procs) | 8,923.38 | 467.89 | 556.60 | 3,660.35 |
| **8** | **Vect-Link CoW (Direct Zero-Copy)** | **23.85** | **476.97** | **560.24** | **3,683.96** |
| **8** | Vect-Link CoW (Reassembled Fallback) | 12.80 | 462.31 | 524.46 | 3,683.10 |
| **8** | `llama_memory_seq_cp` (1 proc, no isolation) | 11.19 | 530.69 | 940.50 | 520.55 |

* **TTFT vs. Cold Prefill (`7.8×–9.3×`):** Direct Zero-Copy CoW achieves **7.83×–9.27×** lower mean TTFT than Cold Prefill across `N = 1..8` (`170.12 ms` vs `1,425.51 ms` at `N=1`, `303.88 ms` vs `2,817.96 ms` at `N=4`, `476.97 ms` vs `3,734.32 ms` at `N=8`).
* **CoW vs. `llama_state_seq_load_file` (Structural vs. Compute):** In **total TTFT** (which includes evaluating the ~15-token divergent suffix and sampling the first token), Vect-Link CoW is **on par** with `llama_state_seq_load_file` over OS page cache / RAM disk (`~170–477 ms` vs `~168–468 ms`), because once the KV state is restored, wall-clock time is dominated by CPU suffix evaluation. Vect-Link's structural advantages over `load_file` are:
  1. **State Fork & Restore Latency:** **`4.90–23.85 µs`** (zero-copy pointer handoff) vs **`3,080–8,923 µs`** (file deserialization/copy) — **~374×–734× faster** state branching.
  2. **Zero-Duplication Shared Footprint:** `N` workers share the same immutable physical KV pages in SHM (`2.84 MiB` stored once) rather than duplicating state files per worker.
  3. **Multi-Process Tail Latency (`TTFT Max`):** Under contention (`N = 4` and `N = 8`), multi-process CoW outperforms single-process `llama_memory_seq_cp` tail TTFT (`367.45 ms` vs `465.60 ms` at `N=4`; `560.24 ms` vs `940.50 ms` at `N=8`).
* **RSS Note:** In multi-process runs (`N` independent OS processes), each worker process maps the GGUF model weights (~460 MiB/process), while the shared KV prefix (`2.84 MiB` in SHM) is stored once without per-worker duplication.

### 2. Zero-Copy `llama.cpp` GGUF Bit-Exact Validation

Verified with [examples/llama_kv_paged_bridge.c](examples/llama_kv_paged_bridge.c) running **`qwen2.5-0.5b-instruct-q4_k_m.gguf`** (`113`-token shared prompt, `22` contiguous paged blocks = `1,390,508` bytes, `restore_heap_allocs=0`, `copied_bytes=0`):

> **Zero-Copy Scope:** Vect-Link performs zero copies across processes and up to the `llama.cpp` API boundary (`0` heap allocations and `0` intermediate buffers when passing the direct SHM pointer to `llama_state_seq_set_data()`). Internally, `llama.cpp` ingests and copies that restored state into the private KV tensors of each worker's `llama_context`.

| Branch | Divergent Suffix | Token Match vs. Cold Prefill | Max Logit Difference | Mapped Zero-Copy |
| :--- | :---: | :---: | :---: | :---: |
| **Branch 0 (Researcher)** | 17 tokens | **16 / 16 (100%)** | `0.000000` | `1,390,508 B` (`0` heap allocs) |
| **Branch 1 (Critic)** | 13 tokens | **16 / 16 (100%)** | `0.000000` | `1,390,508 B` (`0` heap allocs) |
| **Branch 2 (Architect)** | 12 tokens | **16 / 16 (100%)** | `0.000000` | `1,390,508 B` (`0` heap allocs) |

---

## Quick Start

### C99 CoW Snapshot & Fork API

```c
#include "vectlink_kv.h"

/* 1. Producer seals a shared system prompt into an immutable snapshot */
uint64_t snap_id = 0;
vtlk_kv_snapshot_create(&arena, &base_branch, &snap_id);

/* 2. Spawn 2 divergent agent branches in microseconds (0 tensor bytes copied) */
vtlk_kv_branch_t branch_a, branch_b;
vtlk_kv_fork(&arena, snap_id, &branch_a);
vtlk_kv_fork(&arena, snap_id, &branch_b);

/* 3. Each branch appends its own private KV pages without affecting siblings */
vtlk_kv_branch_append_block(&arena, &branch_a, new_block_a);
vtlk_kv_branch_append_block(&arena, &branch_b, new_block_b);

/* 4. Zero-copy restore validates model/RoPE/LoRA/dtype hashes before access */
const void *ptrs[VTLK_KV_MAX_BRANCH_BLOCKS];
vtlk_kv_restore(&arena, snap_id, &expected_cfg, &branch_a, ptrs, VTLK_KV_MAX_BRANCH_BLOCKS);

/* 5. Atomic refcount release returns unused pages to the free-list / LRU tier */
vtlk_kv_branch_release(&arena, &branch_a);
vtlk_kv_branch_release(&arena, &branch_b);
```

### Python FFI & LangGraph Integration

```python
from vectlink.kv import KVArena, KVIdentity, KVDtype
from vectlink.integrations.langgraph import VectLinkBranchStore

with KVArena.create("/vtlk_llm_ctx", num_blocks=1024, block_size_bytes=65536) as arena:
    store = VectLinkBranchStore(arena)

    # Commit shared base context once
    root = store.commit_root(
        thread_id="tot-session-1",
        checkpoint_id="root",
        block_ids=base_block_ids,
        identity=model_identity,
    )

    # Fork independent reasoning branches (Tree-of-Thought) with zero-copy memoryviews
    with store.fork_branch(root.checkpoint_id) as branch_researcher:
        with arena.block_view(branch_researcher.block_ids[0]) as view:
            prefix_bytes = view.buffer  # Zero-copy memoryview
        # Append divergent blocks...

    with store.fork_branch(root.checkpoint_id) as branch_critic:
        # Explore alternative hypothesis in parallel...
        pass
```

---

## Building & Testing

### C99 Core & Multi-Process Stress Suites

```bash
# Release build (compiles with -std=c99 -Wall -Wextra -Werror -pedantic)
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure

# AddressSanitizer + UndefinedBehaviorSanitizer build
cmake -B build-san -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_C_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer"
cmake --build build-san -j
ctest --test-dir build-san --output-on-failure
```

### Python FFI & Benchmarks

```bash
# Install Python package and run unit tests
pip install ./python
python -m unittest discover -s python/tests -v

# Run C99 and Python CoW benchmarks
./build/bench_cow_git
python benchmarks/bench_cow_git.py
```

---

## Repository Layout

```text
├── include/
│   ├── vectlink/protocol.h        # Wire protocol v1 & v2 (64-bit payload headers)
│   ├── vectlink_kv.h              # CoW Arena, block identity, snapshot/fork/restore API
│   ├── vectlink_sync.h            # OS wait/wake (futex / WaitOnAddress) & descriptor ring
│   └── vectlink_tier.h            # L1 (SHM) -> L2 (mmap) -> L3 (SSD) LRU tiering engine
├── src/
│   ├── kv/                        # kv_arena.c, kv_cow.c, kv_hash.c, kv_tier.c
│   ├── sync/                      # os_notify_posix.c, os_notify_windows.c, desc_ring.c
│   └── transport/                 # SPSC ring, atomic_compat.h, POSIX/Windows SHM ACLs
├── python/vectlink/
│   ├── kv.py                      # Zero-copy Python FFI wrapper (memoryview + BufferError)
│   └── integrations/langgraph.py  # VectLinkBranchStore & VectLinkCheckpointer
├── examples/
│   └── llama_kv_paged_bridge.c    # Paged KV-cache bridge & GGUF bit-exact verifier
├── tests/                         # Unit tests, multi-process CoW stress tests & fixtures
└── benchmarks/                    # bench_cow_git.c & bench_cow_git.py
```

---

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.
