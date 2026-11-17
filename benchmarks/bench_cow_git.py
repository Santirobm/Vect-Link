"""Reproducible Python benchmark comparing Vect-Link CoW Snapshot+Fork+Restore
against multiprocessing.shared_memory (full copy), TCP loopback socket, and
mmap/SSD file serialization (os.fsync + mmap).
"""

from __future__ import annotations

import ctypes
from multiprocessing import shared_memory
import mmap
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
import time
from typing import Callable, List, Tuple


VTLK_KV_MAX_BRANCH_BLOCKS = 1024
VTLK_KV_DTYPE_FP16 = 0

NUM_BLOCKS = 625        # 625 * 16 = 10,000 tokens
TOKENS_PER_BLOCK = 16
BLOCK_SIZE = 16384      # 16 KiB/block => 10,240,000 bytes (~10 MiB)
KV_BYTES = NUM_BLOCKS * BLOCK_SIZE
ITERS = 30


class VtlkShm(ctypes.Structure):
    _fields_ = [
        ("addr", ctypes.c_void_p),
        ("size", ctypes.c_size_t),
        ("name", ctypes.c_char * 256),
        ("is_owner", ctypes.c_int),
        ("handle" if sys.platform == "win32" else "fd",
         ctypes.c_void_p if sys.platform == "win32" else ctypes.c_int),
    ]


class VtlkKvArena(ctypes.Structure):
    _fields_ = [
        ("shm", VtlkShm),
        ("base", ctypes.c_void_p),
        ("total_bytes", ctypes.c_size_t),
        ("num_blocks", ctypes.c_uint32),
        ("block_size_bytes", ctypes.c_uint64),
        ("hash_cap", ctypes.c_uint32),
        ("max_snapshots", ctypes.c_uint32),
        ("owns_shm", ctypes.c_int),
        ("freelist_next_offset", ctypes.c_uint64),
        ("hash_table_offset", ctypes.c_uint64),
        ("snapshots_offset", ctypes.c_uint64),
        ("block_headers_offset", ctypes.c_uint64),
        ("block_data_offset", ctypes.c_uint64),
    ]


class VtlkKvIdentity(ctypes.Structure):
    _fields_ = [
        ("token_prefix_hash", ctypes.c_uint64),
        ("model_arch_hash", ctypes.c_uint64),
        ("rope_config_hash", ctypes.c_uint64),
        ("lora_adapter_hash", ctypes.c_uint64),
        ("pos_offset", ctypes.c_uint32),
        ("token_count", ctypes.c_uint32),
        ("kv_dtype", ctypes.c_uint32),
        ("layer_count", ctypes.c_uint32),
        ("head_dim", ctypes.c_uint32),
        ("kv_head_count", ctypes.c_uint32),
    ]


class VtlkKvBranch(ctypes.Structure):
    _fields_ = [
        ("parent_snapshot_id", ctypes.c_uint64),
        ("block_count", ctypes.c_uint32),
        ("total_tokens", ctypes.c_uint32),
        ("model_arch_hash", ctypes.c_uint64),
        ("rope_config_hash", ctypes.c_uint64),
        ("lora_adapter_hash", ctypes.c_uint64),
        ("kv_dtype", ctypes.c_uint32),
        ("layer_count", ctypes.c_uint32),
        ("head_dim", ctypes.c_uint32),
        ("kv_head_count", ctypes.c_uint32),
        ("block_ids", ctypes.c_uint32 * VTLK_KV_MAX_BRANCH_BLOCKS),
    ]


def load_vectlink_lib() -> ctypes.CDLL:
    env_path = os.environ.get("VECTLINK_LIB_PATH")
    candidates: List[Path] = []
    if env_path:
        candidates.append(Path(env_path))
    repo_root = Path(__file__).resolve().parent.parent
    lib_name = "vectlink.dll" if sys.platform == "win32" else "libvectlink.so"
    candidates.append(repo_root / "build" / lib_name)

    for p in candidates:
        if p.exists():
            lib = ctypes.CDLL(str(p))
            break
    else:
        raise FileNotFoundError(f"Could not find {lib_name}; set VECTLINK_LIB_PATH")

    lib.vtlk_shm_unlink.argtypes = [ctypes.c_char_p]
    lib.vtlk_shm_unlink.restype = ctypes.c_int

    lib.vtlk_kv_hash_bytes.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    lib.vtlk_kv_hash_bytes.restype = ctypes.c_uint64

    lib.vtlk_kv_hash_tokens.argtypes = [ctypes.c_uint64, ctypes.POINTER(ctypes.c_uint32), ctypes.c_size_t]
    lib.vtlk_kv_hash_tokens.restype = ctypes.c_uint64

    lib.vtlk_kv_arena_create.argtypes = [ctypes.POINTER(VtlkKvArena), ctypes.c_char_p, ctypes.c_uint32, ctypes.c_uint64]
    lib.vtlk_kv_arena_create.restype = ctypes.c_int

    lib.vtlk_kv_arena_close.argtypes = [ctypes.POINTER(VtlkKvArena)]
    lib.vtlk_kv_arena_close.restype = None

    lib.vtlk_kv_block_alloc.argtypes = [ctypes.POINTER(VtlkKvArena), ctypes.POINTER(ctypes.c_uint32)]
    lib.vtlk_kv_block_alloc.restype = ctypes.c_int

    lib.vtlk_kv_block_data.argtypes = [ctypes.POINTER(VtlkKvArena), ctypes.c_uint32]
    lib.vtlk_kv_block_data.restype = ctypes.c_void_p

    lib.vtlk_kv_block_seal.argtypes = [ctypes.POINTER(VtlkKvArena), ctypes.c_uint32, ctypes.POINTER(VtlkKvIdentity)]
    lib.vtlk_kv_block_seal.restype = ctypes.c_int

    lib.vtlk_kv_block_release.argtypes = [ctypes.POINTER(VtlkKvArena), ctypes.c_uint32]
    lib.vtlk_kv_block_release.restype = ctypes.c_int

    lib.vtlk_kv_branch_append_block.argtypes = [ctypes.POINTER(VtlkKvArena), ctypes.POINTER(VtlkKvBranch), ctypes.c_uint32]
    lib.vtlk_kv_branch_append_block.restype = ctypes.c_int

    lib.vtlk_kv_branch_release.argtypes = [ctypes.POINTER(VtlkKvArena), ctypes.POINTER(VtlkKvBranch)]
    lib.vtlk_kv_branch_release.restype = ctypes.c_int

    lib.vtlk_kv_snapshot_create.argtypes = [ctypes.POINTER(VtlkKvArena), ctypes.POINTER(VtlkKvBranch), ctypes.POINTER(ctypes.c_uint64)]
    lib.vtlk_kv_snapshot_create.restype = ctypes.c_int

    lib.vtlk_kv_fork.argtypes = [ctypes.POINTER(VtlkKvArena), ctypes.c_uint64, ctypes.POINTER(VtlkKvBranch)]
    lib.vtlk_kv_fork.restype = ctypes.c_int

    lib.vtlk_kv_restore.argtypes = [
        ctypes.POINTER(VtlkKvArena),
        ctypes.c_uint64,
        ctypes.POINTER(VtlkKvIdentity),
        ctypes.POINTER(VtlkKvBranch),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_uint32,
    ]
    lib.vtlk_kv_restore.restype = ctypes.c_int
    return lib


def check(rc: int, op: str) -> None:
    if rc != 0:
        raise RuntimeError(f"{op} failed with status {rc}")


def compute_percentiles(samples_ns: List[int]) -> Tuple[float, float]:
    s = sorted(samples_ns)
    n = len(s)
    p50_ms = s[(n - 1) * 50 // 100] / 1e6
    p99_ms = s[(n - 1) * 99 // 100] / 1e6
    return p50_ms, p99_ms


def bench_vectlink_cow(lib: ctypes.CDLL) -> List[int]:
    shm_name = f"/vtlk_py_bench_{os.getpid()}".encode("utf-8")
    lib.vtlk_shm_unlink(shm_name)

    arena = VtlkKvArena()
    check(lib.vtlk_kv_arena_create(ctypes.byref(arena), shm_name, NUM_BLOCKS + 16, BLOCK_SIZE),
          "vtlk_kv_arena_create")

    arch_h = lib.vtlk_kv_hash_bytes(b"llama-3-8b", 10)
    rope_h = lib.vtlk_kv_hash_bytes(b"rope-theta-500k", 15)
    lora_h = lib.vtlk_kv_hash_bytes(b"base-none", 9)

    base_branch = VtlkKvBranch()
    prefix_h = 0
    tok = ctypes.c_uint32(0)
    bid = ctypes.c_uint32(0)
    payload_chunk = bytes([0xA5]) * BLOCK_SIZE

    for i in range(NUM_BLOCKS):
        tok.value = i + 1
        prefix_h = lib.vtlk_kv_hash_tokens(prefix_h, ctypes.byref(tok), 1)
        ident = VtlkKvIdentity(
            token_prefix_hash=prefix_h,
            model_arch_hash=arch_h,
            rope_config_hash=rope_h,
            lora_adapter_hash=lora_h,
            pos_offset=i * TOKENS_PER_BLOCK,
            token_count=TOKENS_PER_BLOCK,
            kv_dtype=VTLK_KV_DTYPE_FP16,
            layer_count=32,
            head_dim=128,
            kv_head_count=8,
        )
        check(lib.vtlk_kv_block_alloc(ctypes.byref(arena), ctypes.byref(bid)), "alloc")
        ptr = lib.vtlk_kv_block_data(ctypes.byref(arena), bid)
        ctypes.memmove(ptr, payload_chunk, BLOCK_SIZE)
        check(lib.vtlk_kv_block_seal(ctypes.byref(arena), bid, ctypes.byref(ident)), "seal")
        check(lib.vtlk_kv_branch_append_block(ctypes.byref(arena), ctypes.byref(base_branch), bid), "append")
        check(lib.vtlk_kv_block_release(ctypes.byref(arena), bid), "release")

    cfg = VtlkKvIdentity(0, arch_h, rope_h, lora_h, 0, TOKENS_PER_BLOCK, VTLK_KV_DTYPE_FP16, 32, 128, 8)
    snap_id = ctypes.c_uint64(0)
    forked = VtlkKvBranch()
    ptrs = (ctypes.c_void_p * NUM_BLOCKS)()

    samples: List[int] = []
    for _ in range(ITERS):
        t0 = time.perf_counter_ns()
        check(lib.vtlk_kv_snapshot_create(ctypes.byref(arena), ctypes.byref(base_branch), ctypes.byref(snap_id)), "snap")
        check(lib.vtlk_kv_fork(ctypes.byref(arena), snap_id, ctypes.byref(forked)), "fork")
        check(lib.vtlk_kv_restore(ctypes.byref(arena), snap_id, ctypes.byref(cfg), None, ptrs, NUM_BLOCKS), "restore")
        t1 = time.perf_counter_ns()
        check(lib.vtlk_kv_branch_release(ctypes.byref(arena), ctypes.byref(forked)), "branch_release")
        samples.append(t1 - t0)

    check(lib.vtlk_kv_branch_release(ctypes.byref(arena), ctypes.byref(base_branch)), "base_release")
    lib.vtlk_kv_arena_close(ctypes.byref(arena))
    lib.vtlk_shm_unlink(shm_name)
    return samples


def bench_shm_full_copy(src_data: bytes) -> List[int]:
    shm_src = shared_memory.SharedMemory(create=True, size=KV_BYTES)
    shm_dst = shared_memory.SharedMemory(create=True, size=KV_BYTES)
    try:
        shm_src.buf[:KV_BYTES] = src_data
        samples: List[int] = []
        for _ in range(ITERS):
            t0 = time.perf_counter_ns()
            shm_dst.buf[:KV_BYTES] = shm_src.buf[:KV_BYTES]
            t1 = time.perf_counter_ns()
            samples.append(t1 - t0)
        return samples
    finally:
        shm_src.close()
        shm_src.unlink()
        shm_dst.close()
        shm_dst.unlink()


def bench_tcp_loopback(src_data: bytes) -> List[int]:
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]

    recv_buf = bytearray(KV_BYTES)
    done_evt = threading.Event()

    def server_worker() -> None:
        conn, _ = srv.accept()
        with conn:
            mv = memoryview(recv_buf)
            for _ in range(ITERS):
                received = 0
                while received < KV_BYTES:
                    n = conn.recv_into(mv[received:])
                    if n == 0:
                        break
                    received += n
                conn.sendall(b"\x01")
        done_evt.set()

    t = threading.Thread(target=server_worker, daemon=True)
    t.start()

    samples: List[int] = []
    with socket.create_connection(("127.0.0.1", port)) as cli:
        cli.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        for _ in range(ITERS):
            t0 = time.perf_counter_ns()
            cli.sendall(src_data)
            ack = cli.recv(1)
            t1 = time.perf_counter_ns()
            if ack != b"\x01":
                raise RuntimeError("TCP ack mismatch")
            samples.append(t1 - t0)

    done_evt.wait(timeout=5.0)
    srv.close()
    return samples


def bench_mmap_fsync(src_data: bytes) -> List[int]:
    samples: List[int] = []
    with tempfile.TemporaryDirectory() as tmpdir:
        fpath = Path(tmpdir) / "kv_cache_snap.bin"
        dst = bytearray(KV_BYTES)
        for _ in range(ITERS):
            t0 = time.perf_counter_ns()
            with open(fpath, "wb") as f:
                f.write(src_data)
                f.flush()
                os.fsync(f.fileno())
            with open(fpath, "rb") as f:
                with mmap.mmap(f.fileno(), KV_BYTES, access=mmap.ACCESS_READ) as mm:
                    dst[:] = mm[:]
            t1 = time.perf_counter_ns()
            samples.append(t1 - t0)
    return samples


def format_row(name: str, samples: List[int], copied_bytes: int) -> str:
    p50_ms, p99_ms = compute_percentiles(samples)
    ops_per_s = 1000.0 / p50_ms if p50_ms > 0.0 else 0.0
    eff_gb_s = (KV_BYTES / (1024.0 ** 3)) / (p50_ms / 1000.0) if p50_ms > 0.0 else 0.0
    copied_str = "0 B (Zero-Copy)" if copied_bytes == 0 else f"{copied_bytes / (1024.0 ** 2):.2f} MiB"
    return f"| {name:<42} | {p50_ms:8.3f} | {p99_ms:8.3f} | {ops_per_s:9.1f} ops/s ({eff_gb_s:6.2f} GiB/s eff) | {copied_str:<17} |"


def run_apples_to_apples_bench() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    exe_name = "bench_llama_cow_apples.exe" if sys.platform == "win32" else "bench_llama_cow_apples"
    candidates = []
    env_bin = os.environ.get("VECTLINK_APPLES_BENCH")
    if env_bin:
        candidates.append(Path(env_bin))
    candidates.extend([
        repo_root / "build" / exe_name,
        repo_root / exe_name,
    ])

    target = next((p for p in candidates if p.exists()), None)
    if target is None:
        return

    env = os.environ.copy()
    if sys.platform == "win32":
        extra_paths = [
            r"C:\msys64\clang64\bin",
            r"C:\Users\sgi.lm\Desktop\llama.cpp\build\bin",
        ]
        existing = [p for p in extra_paths if os.path.isdir(p)]
        if existing:
            env["PATH"] = os.pathsep.join(existing) + os.pathsep + env.get("PATH", "")

    print()
    proc = subprocess.run(
        [str(target)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.stdout:
        print(proc.stdout.rstrip())
    if proc.returncode != 0 and proc.stderr:
        print(proc.stderr.rstrip(), file=sys.stderr)


def main() -> None:
    lib = load_vectlink_lib()
    src_data = bytes([0x5A]) * KV_BYTES

    s_cow = bench_vectlink_cow(lib)
    s_shm = bench_shm_full_copy(src_data)
    s_tcp = bench_tcp_loopback(src_data)
    s_ssd = bench_mmap_fsync(src_data)

    print(f"### Vect-Link CoW vs Alternatives ({NUM_BLOCKS * TOKENS_PER_BLOCK} tokens, {KV_BYTES / (1024 ** 2):.2f} MiB KV cache, {ITERS} iters)\n")
    print("| Method                                     | p50 (ms) | p99 (ms) | Effective Throughput               | Tensor Copied     |")
    print("|--------------------------------------------|----------|----------|------------------------------------|-------------------|")
    print(format_row("Vect-Link CoW (Snapshot + Fork + Restore)", s_cow, 0))
    print(format_row("multiprocessing.shared_memory (Full Copy)", s_shm, KV_BYTES))
    print(format_row("Local TCP Loopback Socket (127.0.0.1)", s_tcp, KV_BYTES))
    print(format_row("SSD File (write + os.fsync + mmap read)", s_ssd, KV_BYTES))

    run_apples_to_apples_bench()


if __name__ == "__main__":
    main()
