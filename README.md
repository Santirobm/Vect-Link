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

* **Zero-Copy Branching (`Fork` in ~5.9 µs):** Forking a context duplicates only a lightweight table of block IDs (`block_ids[]`) and increments atomic reference counts (`refcount`). Tensor data is never copied.
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

### 1. CoW Branching Latency (9.77 MiB / 10,000-Token Context)

Measured via [benchmarks/bench_cow_git.c](benchmarks/bench_cow_git.c) and [benchmarks/bench_cow_git.py](benchmarks/bench_cow_git.py):

| Operation / Transport | Latency (p50) | Tensor Bytes Copied | Relative Speedup vs. Vect-Link |
| :--- | :---: | :---: | :---: |
| **Vect-Link C99 `restore`** | **0.50 µs** | **0 B** | Baseline (C99) |
| **Vect-Link C99 `fork`** | **5.90 µs** | **0 B** | Baseline (C99) |
| **Vect-Link C99 `snapshot_create`** | **6.30 µs** | **0 B** | Baseline (C99) |
| **Vect-Link Python FFI CoW** | **0.015 ms** | **0 B** | **1.0×** |
| Python `multiprocessing.shared_memory` | 0.517 ms | 9.77 MiB | 34.5× slower |
| TCP Loopback Socket | 4.737 ms | 9.77 MiB | 315.8× slower |
| SSD `mmap` + `os.fsync` | 42.550 ms | 9.77 MiB | 2,836.7× slower |

### 2. End-to-End `llama.cpp` GGUF Validation

Verified with [examples/llama_kv_paged_bridge.c](examples/llama_kv_paged_bridge.c) running **`qwen2.5-0.5b-instruct-q4_k_m.gguf`** with a shared 247-token system prompt (47 paged blocks = 3,038,708 bytes) forked into 3 independent agent branches (`Role A`, `Role B`, `Role C`) and compared against cold baseline runs from scratch:

| Branch | Divergent Suffix | Token Match vs. Cold Prefill | Max Logit Difference |
| :--- | :---: | :---: | :---: |
| **Branch 0 (Researcher)** | 17 tokens | **16 / 16 (100%)** | `0.000000` |
| **Branch 1 (Critic)** | 13 tokens | **16 / 16 (100%)** | `0.000000` |
| **Branch 2 (Architect)** | 12 tokens | **16 / 16 (100%)** | `0.000000` |

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
