# Vect-Link

Zero-dependency **C99** binary protocol and shared-memory IPC library for low-latency transport of tensors and latent states between processes.

## Architecture & Current Status (`v0.3.0`)

- **Wire Format (`v0.1.0`):** Fixed 64-byte `VTLK` header serialized via explicit little-endian offsets (zero struct-packing assumptions, zero dynamic allocations).
- **Integrity (`v0.2.0`):** Portable Castagnoli **CRC32C** (`0x82F63B78`) over both header bytes `[0..59]` (`header_crc32c`) and payload (`payload_crc32c`).
- **IPC Transport (`v0.3.0`):** Lock-free **SPSC Ring Buffer** over OS shared memory (`CreateFileMappingA`/`MapViewOfFile` on Windows, `shm_open`/`mmap` on POSIX) with 192-byte cache-line-isolated control block and acquire/release atomic barriers.
- **Verification:** Compiled with `-std=c99 -Wall -Wextra -Werror -pedantic` (Clang 22.1.8) and `/TC /W4 /WX` (MSVC), and verified with AddressSanitizer + UndefinedBehaviorSanitizer (`-fsanitize=address,undefined`) including 10,000-packet cross-process IPC tests.

## Repository Layout

```text
include/vectlink/
  protocol.h      # VTLK 64-byte header types, status codes, encode/decode API
  checksum.h      # CRC32C Castagnoli API
  transport.h     # Lock-free SPSC ring buffer & OS shared memory API
src/
  integrity/      # crc32c.c
  protocol/       # endian.h, header.c
  transport/      # atomic_compat.h, ring_buffer.c, shm_windows.c, shm_posix.c
tests/
  test_protocol.c
  test_ring_buffer.c
  test_cross_process.c
examples/
  agent_sender.c
  agent_receiver.c
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
