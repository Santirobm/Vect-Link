"""Two-process zero-copy Python example exchanging INT4-quantized tensors via SHM."""

from __future__ import annotations

import array
import math
import multiprocessing
import os
import sys
import time
from pathlib import Path

PYTHON_ROOT = Path(__file__).resolve().parent.parent
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from vectlink import QuantScheme, RingEmptyError, VectLinkRing  # noqa: E402


def consumer_process(shm_name: str, done_q: multiprocessing.Queue) -> None:
    ring = None
    for _ in range(500):
        try:
            ring = VectLinkRing.open(shm_name)
            break
        except Exception:
            time.sleep(0.01)
    if ring is None:
        done_q.put(("error", "Timed out opening SHM"))
        return

    with ring:
        while True:
            try:
                t0 = time.perf_counter_ns()
                with ring.acquire_read() as slot:
                    hdr = slot.header
                    decoded = slot.decode_tensor()
                t1 = time.perf_counter_ns()
                done_q.put(
                    (
                        "ok",
                        hdr.n_tokens,
                        hdr.d_model,
                        hdr.payload_bytes,
                        (t1 - t0) / 1000.0,
                        list(decoded[:4]),
                    )
                )
                return
            except RingEmptyError:
                time.sleep(0.0005)


def main() -> None:
    n_tokens = 32
    d_model = 4096
    total = n_tokens * d_model
    shm_name = f"/vtlk_py_demo_{os.getpid()}"

    tensor = array.array("f", [math.sin(i * 0.017) * 1.8 for i in range(total)])
    slot_cap = (total // 32) * 18 + 64

    ctx = multiprocessing.get_context("spawn")
    done_q: multiprocessing.Queue = ctx.Queue()

    with VectLinkRing.create(shm_name, slot_capacity=slot_cap, slot_count=4) as ring:
        proc = ctx.Process(target=consumer_process, args=(shm_name, done_q))
        proc.start()

        t0 = time.perf_counter_ns()
        with ring.acquire_write() as slot:
            written = slot.encode_tensor(
                tensor,
                n_tokens=n_tokens,
                d_model=d_model,
                quant=QuantScheme.INT4_BLOCK,
                model_hash=0xDEADBEEFCAFEBABE,
            )
        t1 = time.perf_counter_ns()

        status = done_q.get(timeout=10.0)
        proc.join(timeout=10.0)

    if status[0] != "ok":
        raise RuntimeError(status[1])

    _, rx_tok, rx_dim, rx_bytes, rx_us, rx_head = status
    raw_bytes = total * 4
    print(f"[Producer] Tensor: {n_tokens} x {d_model} ({raw_bytes:,} B FP32 -> {written:,} B INT4, {raw_bytes / written:.2f}x)")
    print(f"[Producer] Zero-Copy Encode + Commit: {(t1 - t0) / 1000.0:.2f} us")
    print(f"[Consumer] Zero-Copy Acquire + Decode ({rx_tok} x {rx_dim}, {rx_bytes:,} B): {rx_us:.2f} us")
    print(f"[Consumer] First 4 decoded floats: {[round(x, 4) for x in rx_head]}")


if __name__ == "__main__":
    main()
