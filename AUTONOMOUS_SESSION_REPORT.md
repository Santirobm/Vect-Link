# Autonomous Session Report — Vect-Link CoW «Git for LLM Context» Engine

**Branch:** `feat/cow-kv-git-engine` (base `main` untouched)
**Toolchain:** Windows x64, Clang 22.1.8 (`-std=c99 -Wall -Wextra -Werror -pedantic`), CMake 4.4.3 + Ninja 1.13.2, Python 3.13.5, `numpy 2.3.2`

---

## 1. Arquitectura Implementada y Mapa de Archivos

Vect-Link evolucionó de un transporte IPC SPSC lineal a un motor de almacenamiento y bifurcación de KV-cache con Copy-on-Write (CoW), manteniendo 100% de compatibilidad hacia atrás con la API `vtlk_ring_*` y el formato v1 (`vtlk_header_t`).

### Capa Core C99 (Arena Paginada, Identidad y CoW)
- `include/vectlink_kv.h`: Contrato público del motor CoW:
  - `vtlk_kv_block_header_t` (128 bytes = 2 cache lines de 64B): separa metadatos inmutables (`magic = 0x56544B56`, `storage_format_version = 2`, `block_size_bytes` en `uint64_t`, `token_prefix_hash`, `model_arch_hash`, `rope_config_hash`, `lora_adapter_hash`, `kv_dtype`, `layer_count`, `head_dim`, `kv_head_count`, `payload_crc32c`) de estado atómico mutable (`refcount`, `flags`, `lru_stamp`, `tier_state`, `l2_offset`, `l3_id`).
  - Primitivas Git-for-Context: `vtlk_kv_snapshot_create`, `vtlk_kv_fork`, `vtlk_kv_restore`, `vtlk_kv_branch_append_block`, `vtlk_kv_branch_release`.
- `src/kv/kv_hash.c`: Hash encadenado de 64 bits (`vtlk_kv_hash_tokens`, `vtlk_kv_hash_bytes`) basado en SplitMix64 con propiedad asociativa de prefijos.
- `src/kv/kv_arena.c`: Arena paginada en SHM con free-list lock-free etiquetada (CAS de 64 bits `[gen | block_id]` anti-ABA), tabla hash open-addressing en SHM para deduplicación por identidad completa, sellado inmutable con CRC32C (`vtlk_kv_block_seal`) y revivido seguro de bloques cacheados (`refcount == 0`).
- `src/kv/kv_cow.c`: Registro de snapshots en SHM, `fork` O(N_blocks) copiando solo `block_ids[]` con incremento atómico de `refcount` (0 bytes de tensores copiados) y `restore` con validación estricta de modelo/RoPE/LoRA/dtype.
- `src/transport/atomic_compat.h`: Primitivas atómicas C99 de 32 y 64 bits (`__atomic_*` en Clang/GCC, `_Interlocked*` en MSVC).

### Control Plane, Protocolo v2, Seguridad OS y Persistencia L1/L2/L3
- `include/vectlink/protocol.h`, `src/protocol/header.c`: Preservación de `vtlk_header_t` (v1, 64B) + nuevo `vtlk_header_v2_t` (96B, `VTLK_PROTOCOL_VERSION_V2 = 0x0002`) con `uint64_t payload_bytes` y `uint64_t raw_bytes`, y códigos `VTLK_ERR_TIMEOUT`, `VTLK_ERR_IO`, `VTLK_ERR_NOT_FOUND`.
- `src/transport/shm_windows.c`, `src/transport/shm_posix.c`: Aislamiento estricto por usuario — DACL restringida al SID del usuario actual en Windows (`OpenProcessToken` + `GetTokenInformation(TokenUser)` + `AddAccessAllowedAce`) y `0600` + verificación `st_uid == getuid()` en POSIX.
- `include/vectlink_sync.h`, `src/sync/os_notify_windows.c`, `src/sync/os_notify_posix.c`, `src/sync/desc_ring.c`: Notificaciones nativas del SO (`WaitOnAddress`/`WakeByAddressAll` en Windows, `SYS_futex` en Linux, fallback POSIX) y ring de descriptores livianos `vtlk_desc_ring_t` (`vtlk_kv_descriptor_t`).
- `include/vectlink_tier.h`, `src/kv/kv_tier.c`: Gestor jerárquico `vtlk_tier_store_t` (**L1 SHM RAM** → **L2 Warm `mmap`/`MapViewOfFile`** → **L3 Cold SSD** con índice persistente `vtlk_tier_index.bin` y verificación CRC32C al promover), con evicción LRU estricta solo sobre bloques con `refcount == 0`.

### Ecosistema (`llama.cpp`, Python FFI, LangGraph) y Limpieza
- `tests/fixtures/ref_embd.bin`: Movido desde la raíz del repositorio.
- `examples/llama_kv_paged_bridge.c`: Bridge C11 con `llama.cpp` que particiona el KV-cache por bloques de tokens en `vtlk_kv_arena_t`, permite 1 productor con snapshot en `pos = N` y N workers haciendo `fork` + `restore` manteniendo RoPE intacto.
- `python/vectlink/_ffi.py`, `python/vectlink/kv.py`, `python/vectlink/__init__.py`: Bindings `ctypes` de todas las funciones `vtlk_kv_*` y API Python (`KVArena`, `KVIdentity`, `Snapshot`, `Branch`, `KVBlockView`) con `memoryview` cero-copia y guardas `BufferError` contra vistas colgantes.
- `python/vectlink/integrations/langgraph.py`: `VectLinkBranchStore` y `VectLinkCheckpointer` para bifurcación Tree-of-Thought respaldada por forks CoW (con fallback sin requerir `langgraph` instalado).

### Tests y Benchmarks
- `tests/test_kv_cow.c`: Tests unitarios (hash, arena, dedup, LRU, snapshot/fork/restore, mismatch, desc_ring, L1/L2/L3) + **test de estrés multi-proceso** (1 productor con 10.000 tokens / 625 bloques → snapshot → 3 procesos concurrentes `fork_a`, `fork_b`, `fork_c` verificando bit-exactitud, agregando bloques divergentes y validando `refcount == 0` al finalizar).
- `python/tests/test_kv_cow_and_langgraph.py`: Tests unitarios de la API Python CoW, vistas colgantes (`BufferError`) y `VectLinkBranchStore` / `VectLinkCheckpointer`.
- `benchmarks/bench_cow_git.c`, `benchmarks/bench_cow_git.py`: Suite reproducible de latencia y comparación contra `multiprocessing.shared_memory`, TCP loopback y SSD (`write` + `os.fsync` + `mmap`).

---

## 2. Resultados Reales de Compilación, Sanitizers y Tests

- **Release Build (`cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build build`):**
  Compilación limpia con `-std=c99 -Wall -Wextra -Werror -pedantic`, `0` warnings.
- **CTest Release (`ctest --test-dir build --output-on-failure`):**
  `5/5` suites aprobadas (`test_protocol`, `test_ring_buffer`, `test_cross_process`, `test_codec`, `test_kv_cow`), `100% tests passed`, tiempo total `0.18 s`.
- **Sanitizers (`-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer` en `build-san`):**
  `5/5` suites aprobadas bajo **ASan + UBSan** (`test_kv_cow` multi-proceso incluido), `0` errores detectados, tiempo total `1.15 s`.
- **Python Test Suite (`python -m unittest discover -s python/tests -v`):**
  `15/15` tests aprobados en `0.552 s` (10 tests existentes de ring/IPC + 5 tests nuevos de CoW/dangling-view/LangGraph).
- **`llama.cpp` Paged Bridge (`VTLK_BUILD_LLAMA_BRIDGE=ON`, target `llama_kv_paged_bridge`):**
  Compilación C11 limpia contra `llama.cpp` y ejecución de `--self-test` aprobada (`llama_kv_paged_bridge self-test PASSED (3 workers)`).
- **Validación end-to-end con modelo GGUF real (`qwen2.5-0.5b-instruct-q4_k_m.gguf`, 247 tokens de system prompt → 47 bloques paginados = 3,038,708 B → 3 ramas divergentes):**
  - `[branch 0] suffix_toks=17 | token_match=16/16 | max_logit_diff=0.000000`
  - `[branch 1] suffix_toks=13 | token_match=16/16 | max_logit_diff=0.000000`
  - `[branch 2] suffix_toks=12 | token_match=16/16 | max_logit_diff=0.000000`
- **Acople Arena ↔ Tiering L1/L2/L3 y Hot-Path LRU:**
  - `vtlk_kv_arena_attach_tier()` / `vtlk_kv_arena_demote_lru()` integrados; `vtlk_kv_block_retain()` / `vtlk_kv_restore()` promueven transparentemente bloques desalojados a L2/L3 de vuelta a L1 con verificación CRC32C (`test_arena_tier_transparent_restore` aprobado bajo Release y ASan+UBSan).
  - Eliminado el cache-line bouncing en `vtlk_kv_block_retain()`: `lru_stamp` solo se actualiza al transicionar `refcount` de `1 -> 0` en `vtlk_kv_block_release()`.

*(Nota de plataforma: la ejecución real de tests en esta sesión ocurrió en Windows x64; las rutas POSIX `shm_posix.c`, `os_notify_posix.c` y ramas POSIX de `kv_tier.c` fueron compiladas/revisadas pero no ejecutadas en un kernel Linux en esta sesión.)*

---

## 3. Resultados de Benchmarks (10.000 tokens, 625 bloques × 16 KiB = 9.77 MiB)

### Benchmark C99 (`build/bench_cow_git.exe`, 200 iteraciones)

| Operación | min (us) | p50 (us) | p95 (us) | p99 (us) | mean (us) | Tensor Copiado |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `snapshot_create` | 5.80 | 6.30 | 6.70 | 7.80 | 6.30 | 0 B |
| `fork` | 5.80 | 5.90 | 5.90 | 6.00 | 5.91 | **0 B** |
| `restore` | 0.50 | 0.50 | 0.60 | 0.80 | 0.52 | **0 B** |

### Comparativa Python End-to-End (`benchmarks/bench_cow_git.py`, 30 iteraciones)

| Método | p50 (ms) | p99 (ms) | Throughput Efectivo | Tensor Copiado |
| :--- | ---: | ---: | :--- | :--- |
| **Vect-Link CoW (`Snapshot + Fork + Restore`)** | **0.015** | **0.017** | **67,114.1 ops/s** | **0 B (Zero-Copy)** |
| `multiprocessing.shared_memory` (Full Copy) | 0.517 | 0.915 | 1,934.6 ops/s | 9.77 MiB |
| Local TCP Loopback Socket (`127.0.0.1`) | 4.737 | 5.987 | 211.1 ops/s | 9.77 MiB |
| SSD File (`write` + `os.fsync` + `mmap` read) | 42.550 | 51.685 | 23.5 ops/s | 9.77 MiB |

---

## 4. Decisiones Técnicas y Próximos Pasos Recomendados

- **Decisiones clave:**
  - `vtlk_kv_block_header_t` de 128 bytes (2 cache lines) para aislar metadatos inmutables de `refcount`/`lru_stamp` atómicos y evitar *false sharing* entre lectores concurrentes.
  - Atomics C99 portables vía `atomic_compat.h` (`__atomic_*` / `_Interlocked*`) en lugar de `<stdatomic.h>` para mantener compatibilidad estricta con `-std=c99 -pedantic` en Clang, GCC y MSVC.
  - Desacoplamiento de `vtlk_tier_store_t` del ring/arena para permitir testeo y evicción L1→L2→L3 determinista con CRC32C en promoción.
- **Próximos pasos recomendados:**
  1. Correr la suite completa + TSan/ASan/UBSan en CI Linux para ejercitar en vivo `shm_posix.c` (`0600` + UID) y `os_notify_posix.c` (`SYS_futex`).
  2. Probar `llama_kv_paged_bridge` contra un modelo GGUF real en dos procesos OS separados midiendo tokens/s y paridad Top-1 tras `fork` en `pos = N`.
  3. Añadir cuantización asimétrica específica de KV-cache (estilo KIVI 2-bit/4-bit) y soporte CUDA IPC / `dmabuf` para GPU-direct CoW.


---

## 5. Endurecimiento Pre-Release (`feat/pre-release-hardening`)

### 5.1 Bridge Zero-Copy (`examples/llama_kv_paged_bridge.c`)
- Eliminado el header intercalado por página (`bridge_page_header_t`) y cualquier `malloc` / doble copia en `restore`: la metadata por página reside out-of-band en `vtlk_kv_block_header_t.reserved_imm` y `vtlk_kv_branch_t.total_tokens`.
- `prepare_restore_view()` verifica contigüidad de las páginas mapeadas en SHM (`ptrs[i] == ptrs[i-1] + aligned_blk`) y entrega `(const uint8_t *)ptrs[0]` directo a `llama_state_seq_set_data()` (`mapped_zero_copy_bytes == 1,390,508 B`, `copied_bytes == 0`, `restore_heap_allocs == 0`).
- Validación GGUF real (`qwen2.5-0.5b-instruct-q4_k_m.gguf`, 113 tokens, 22 bloques): `token_match = 16/16 (100%)` y `max_logit_diff = 0.000000` en las 3 ramas divergentes.

### 5.2 Índice Persistente DAG en C/L3 y `VectLinkCheckpointer` Real
- **C99 (`include/vectlink_kv.h`, `src/kv/kv_dag.c`):** `vtlk_kv_dag_t` y `vtlk_kv_checkpoint_meta_t` con `vtlk_kv_dag_open`, `vtlk_kv_dag_close`, `vtlk_kv_dag_put`, `vtlk_kv_dag_get`, `vtlk_kv_dag_list`. Persistencia atómica con lock de archivo OS (`LockFileEx` / `flock`), archivos temporales por PID + rename atómico (`MoveFileExA` / `rename`), e índice `vtlk_dag_index.bin` + `.ckpt` protegidos por CRC32C.
- **Python / LangGraph (`python/vectlink/_ffi.py`, `python/vectlink/integrations/langgraph.py`, `python/pyproject.toml`):** `VectLinkBranchStore` y `VectLinkCheckpointer` reconstruyen todo el DAG y `pending_writes` desde C/L3 sin diccionarios volátiles en memoria.
- **Tests reales sin mocks (`.venv-test`, `langgraph==1.2.22`, `langgraph-checkpoint==4.3.2`, `pytest==9.2.0`):**
  - `pytest python/tests/test_langgraph_real.py -v`: **`3 passed in 7.70s`** (`test_real_langgraph_send_cow_fanout`, `test_process_restart_survival_without_recompute` tras `os._exit(0)`, `test_multiprocess_concurrent_checkpoint_writes` con 4 procesos).
  - `ctest --test-dir build`: **`5/5` passed (`0.40s`)**.
  - `ctest --test-dir build-san` (ASan + UBSan): **`5/5` passed (`1.30s`)**.
  - `python -m unittest discover -s python/tests -v` (entorno base sin `langgraph`): **`16` tests (`15` passed, `1` skipped `test_langgraph_real`)**.

### 5.3 Benchmarks Apples-to-Apples (`benchmarks/bench_llama_cow_apples.c`, `qwen2.5-0.5b-instruct-q4_k_m.gguf`, 242 tokens, KV = 2.84 MiB)

| N | Method | Pure Fork/Load (µs) | TTFT Mean (ms) | TTFT Max (ms) | Total System RSS (MiB) |
|---:|:---|---:|---:|---:|---:|
| 1 | Cold Prefill (N procs) | 0.00 | 1,425.51 | 1,425.51 | 474.67 |
| 1 | `llama_state_seq_load_file` (N procs) | 3,080.10 | 167.70 | 167.70 | 457.49 |
| 1 | **Vect-Link CoW (Direct Zero-Copy)** | **5.80** | **170.12** | **170.12** | **460.48** |
| 1 | Vect-Link CoW (Reassembled Fallback) | 4.50 | 165.51 | 165.51 | 460.46 |
| 1 | `llama_memory_seq_cp` (1 proc, no isol) | 0.00 | 134.01 | 134.01 | 492.38 |
| 2 | Cold Prefill (N procs) | 0.00 | 1,453.20 | 1,471.89 | 949.27 |
| 2 | `llama_state_seq_load_file` (N procs) | 3,596.10 | 178.59 | 178.66 | 915.07 |
| 2 | **Vect-Link CoW (Direct Zero-Copy)** | **4.90** | **174.67** | **175.80** | **920.83** |
| 2 | Vect-Link CoW (Reassembled Fallback) | 6.30 | 173.76 | 175.26 | 920.97 |
| 2 | `llama_memory_seq_cp` (1 proc, no isol) | 19.10 | 169.99 | 227.17 | 495.95 |
| 4 | Cold Prefill (N procs) | 0.00 | 2,817.96 | 3,102.44 | 1,898.62 |
| 4 | `llama_state_seq_load_file` (N procs) | 5,077.12 | 319.30 | 388.20 | 1,830.05 |
| 4 | **Vect-Link CoW (Direct Zero-Copy)** | **7.75** | **303.88** | **367.45** | **1,841.71** |
| 4 | Vect-Link CoW (Reassembled Fallback) | 6.12 | 315.60 | 397.39 | 1,841.70 |
| 4 | `llama_memory_seq_cp` (1 proc, no isol) | 12.13 | 289.70 | 465.60 | 503.17 |
| 8 | Cold Prefill (N procs) | 0.00 | 3,734.32 | 4,311.97 | 3,797.55 |
| 8 | `llama_state_seq_load_file` (N procs) | 8,923.38 | 467.89 | 556.60 | 3,660.35 |
| 8 | **Vect-Link CoW (Direct Zero-Copy)** | **23.85** | **476.97** | **560.24** | **3,683.96** |
| 8 | Vect-Link CoW (Reassembled Fallback) | 12.80 | 462.31 | 524.46 | 3,683.10 |
| 8 | `llama_memory_seq_cp` (1 proc, no isol) | 11.19 | 530.69 | 940.50 | 520.55 |
