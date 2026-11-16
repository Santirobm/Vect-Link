# Vect-Link

Zero-dependency **C99** binary protocol, quantization codecs, and lock-free shared-memory IPC library for low-latency transport of tensors and latent/KV states between processes.

## Highlights

- **64-byte Binary Wire Format:** Little-endian serialized `VTLK` header with Castagnoli **CRC32C** over header `[0..59]` and payload (`VTLK_QUANT_FP16`, `VTLK_QUANT_INT4_BLOCK`, `VTLK_QUANT_RAW`).
- **Lock-Free SPSC Ring Buffer over OS Shared Memory:** Cache-line-isolated 192B control block with acquire/release atomic barriers (`CreateFileMappingA`/`MapViewOfFile` on Windows, `shm_open`/`mmap` on POSIX).
- **Portable FP16 / INT4_BLOCK Codecs with Runtime AVX2 Dispatch:** Branchless `libm`-free C99 scalar fast-path plus translation-unit-isolated AVX2 acceleration (`src/codec/int4_avx2.c`) dispatched dynamically via `CPUID + XGETBV`, deterministic non-finite (`NaN`/`Inf`) sanitization, and bit-exact scalar/AVX2 parity.
- **Real Multi-Process `llama.cpp` IPC Bridge (optional PoC):** Connects two independent `llama.cpp` processes over Vect-Link SHM with strict GGUF weight/metadata & KV-layout compatibility hashing ([examples/README_LLAMA.md](examples/README_LLAMA.md)):
  - **`--mode kv-state`** (coherent continuation via sequence KV-Cache state transfer): `135,908 B` state + `16 B` `vtlk_llama_kv_meta_t`, **`338.20 us`–`343.30 us`** extract+push, coherent continuation without repeating prompt prefill on `qwen2.5-0.5b-instruct-q4_k_m.gguf`.
  - **`--mode embd`** (INT4-quantized tensor transport & reconstruction benchmark): `11 x 896` FP32 -> `5,544 B` INT4 (**`7.11x`** compression), **`~44 us`** combined encode+push+pop+decode (`28.80 us` + `15.40 us`), **`0.985879`** cosine similarity *(demonstrates Layer-N post-RMSNorm vs Layer-0 `batch.embd` alignment requirement)*.

## Baseline Benchmarks (`v0.8.0` / `v0.5.1`)

Measured on **Intel Core i5-13500, 16 GB DDR4-3200, Windows x64, Clang 22.1.8 (`-O3 -march=native`)** transferring a `32 x 4096` hidden-state tensor (`131,072` floats = `524,288` bytes FP32) end-to-end (encode + transport + CRC32C + decode, warmup = 500, TCP `N = 2,000`, SHM `N = 10,000`):

| Scenario | Wire Size (Bytes) | Compression vs JSON | p50 (us) | p95 (us) | p99 (us) | Throughput (msg/s) | Cosine Sim |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **JSON + TCP** | 1,114,216 | 1.00x | 32,845.70 | 33,794.80 | 35,076.20 | 30.6 | 1.000000 |
| **VTLK INT4 + TCP** | 73,792 | 15.10x | 412.80 | 548.90 | 655.00 | 2,299.5 | 0.995645 |
| **VTLK INT4 + SHM** | 73,792 | 15.10x | **376.20** | **395.10** | **484.80** | **2,616.0** | 0.995645 |

### Reproduce the Benchmarks in One Command

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build build && ./build/benchmark_codec && ./build/benchmark_transport
```

*(Outputs `benchmark_results.csv` in the current working directory. Disable native SIMD with `-DVTLK_ENABLE_NATIVE_SIMD=OFF` to benchmark the pure scalar C99 fast-path.)*

## Architecture (`v0.1.0` – `v0.8.5+`)

- **Wire Format (`v0.1.0`+):** Fixed 64-byte `VTLK` header serialized via explicit little-endian offsets (zero struct-packing assumptions, zero dynamic allocations), supporting `VTLK_QUANT_FP16`, `VTLK_QUANT_INT4_BLOCK`, `VTLK_QUANT_RVQ10`, and `VTLK_QUANT_RAW`.
- **Integrity (`v0.2.0`):** Portable Castagnoli **CRC32C** (`0x82F63B78`) over both header bytes `[0..59]` (`header_crc32c`) and payload (`payload_crc32c`).
- **IPC Transport (`v0.3.0`):** Lock-free **SPSC Ring Buffer** over OS shared memory (`CreateFileMappingA`/`MapViewOfFile` on Windows, `shm_open`/`mmap` on POSIX) with 192-byte cache-line-isolated control block and acquire/release atomic barriers.
- **Tensor Codecs (`v0.4.0`–`v0.8.1`):** Portable IEEE 754 **FP16** (`2.0x`) and 32-element block-quantized **INT4_BLOCK** (`7.11x`, 18 bytes/block) with `libm`-free C99 scalar fast-path, `NaN`/`Inf` sanitization, and runtime **`CPUID + XGETBV` AVX2 dispatch** isolated in `src/codec/int4_avx2.c` (`-DVTLK_DISABLE_AVX2=ON` forces scalar fallback cleanly).
- **Verification:** Compiled with `-std=c99 -Wall -Wextra -Werror -pedantic` (GCC/Clang) and `/TC /W4 /WX` (MSVC), tested on Windows + Linux with AddressSanitizer + UndefinedBehaviorSanitizer (`-fsanitize=address,undefined`).

## Repository Layout

```text
include/vectlink/
  vectlink.h      # Umbrella public header
  protocol.h      # VTLK 64-byte header types, status codes, encode/decode API
  checksum.h      # CRC32C Castagnoli API
  codec.h         # FP16 & INT4_BLOCK tensor codecs + cosine similarity metrics
  transport.h     # Lock-free SPSC ring buffer & OS shared memory API
src/
  integrity/      # crc32c.c
  protocol/       # endian.h, header.c
  codec/          # fp16.c, int4.c, int4_avx2.c, codec.c
  transport/      # atomic_compat.h, ring_buffer.c, shm_windows.c, shm_posix.c
tests/
  test_protocol.c
  test_ring_buffer.c
  test_cross_process.c
  test_codec.c
benchmarks/
  bench_timer.h
  benchmark_codec.c
  benchmark_transport.c
examples/
  agent_sender.c
  agent_receiver.c
  llama_bridge_common.h
  llama_agent_a.c
  llama_agent_b.c
  README_LLAMA.md
```

## Build & Run Tests (CMake + Ninja)

```sh
cmake -S . -B build -G Ninja
cmake --build build
ctest --test-dir build --output-on-failure
```

### Strict C99 + Sanitizers (Clang / GCC)

```sh
cmake -S . -B build-san -G Ninja -DCMAKE_C_COMPILER=clang \
  -DCMAKE_C_FLAGS="-O2 -g -fsanitize=address,undefined"
cmake --build build-san
ctest --test-dir build-san --output-on-failure
```

### Two-Process Shared Memory Demo

Terminal 1:
```sh
./build/agent_sender
```

Terminal 2:
```sh
./build/agent_receiver
```

## License

Apache License 2.0 — see [LICENSE](LICENSE).
