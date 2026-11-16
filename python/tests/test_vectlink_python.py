"""Unit and multi-process tests for the Python zero-copy Vect-Link bindings."""

from __future__ import annotations

import array
import math
import multiprocessing
import os
import sys
import time
import unittest
from pathlib import Path

PYTHON_ROOT = Path(__file__).resolve().parent.parent
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from vectlink import (  # noqa: E402
    QuantScheme,
    RingBusyError,
    RingEmptyError,
    RingFullError,
    VectLinkRing,
)
from vectlink._ffi import get_lib  # noqa: E402


def _mp_consumer_worker(shm_name: str, n_tokens: int, d_model: int, q: multiprocessing.Queue) -> None:
    try:
        # Poll briefly until parent creates SHM
        ring = None
        deadline = time.perf_counter() + 5.0
        while time.perf_counter() < deadline:
            try:
                ring = VectLinkRing.open(shm_name)
                break
            except Exception:
                time.sleep(0.01)
        if ring is None:
            q.put(("err", "Failed to open SHM"))
            return

        with ring:
            deadline = time.perf_counter() + 5.0
            while time.perf_counter() < deadline:
                try:
                    with ring.acquire_read() as slot:
                        hdr = slot.header
                        out = slot.decode_tensor()
                        q.put(("ok", hdr.n_tokens, hdr.d_model, hdr.quant_scheme, list(out[:8])))
                        return
                except RingEmptyError:
                    time.sleep(0.001)
            q.put(("err", "Timed out waiting for packet"))
    except Exception as exc:
        q.put(("err", repr(exc)))


class TestVectLinkPython(unittest.TestCase):
    def _shm_name(self, suffix: str) -> str:
        return f"/vtlk_py_test_{os.getpid()}_{suffix}"

    def test_zero_copy_memoryview_and_use_after_release(self) -> None:
        name = self._shm_name("zc")
        with VectLinkRing.create(name, slot_capacity=256, slot_count=4) as prod:
            with VectLinkRing.open(name) as cons:
                # Write directly into SHM via memoryview
                with prod.acquire_write() as wslot:
                    mv = wslot.buffer
                    self.assertEqual(len(mv), 256)
                    mv[0:5] = b"HELLO"
                    wslot.commit(5, n_tokens=1, d_model=1, quant=QuantScheme.RAW, sequence=7)

                    # Use-after-commit must fail on both slot.buffer and the old memoryview
                    with self.assertRaises(ValueError):
                        _ = wslot.buffer
                    with self.assertRaises(ValueError):
                        _ = mv[0]

                with cons.acquire_read() as rslot:
                    rmv = rslot.buffer
                    self.assertEqual(rslot.header.sequence, 7)
                    self.assertEqual(rslot.bytes, 5)
                    self.assertEqual(bytes(rmv), b"HELLO")
                    rslot.release()

                    # Use-after-release must fail
                    with self.assertRaises(ValueError):
                        _ = rslot.buffer
                    with self.assertRaises(ValueError):
                        _ = rmv[0]

    def test_abort_on_exception_and_error_states(self) -> None:
        name = self._shm_name("errors")
        with VectLinkRing.create(name, slot_capacity=128, slot_count=2) as prod:
            with VectLinkRing.open(name) as cons:
                with self.assertRaises(RingEmptyError):
                    cons.acquire_read()

                # Busy on double acquire_write
                w1 = prod.acquire_write()
                try:
                    with self.assertRaises(RingBusyError):
                        prod.acquire_write()
                finally:
                    w1.abort()

                # Exception inside `with acquire_write()` aborts cleanly
                with self.assertRaises(RuntimeError):
                    with prod.acquire_write() as wslot:
                        wslot.write_bytes(b"discarded")
                        raise RuntimeError("boom")
                with self.assertRaises(RingEmptyError):
                    cons.acquire_read()

                # Fill ring to capacity (2) -> 3rd acquire_write raises RingFullError
                with prod.acquire_write() as w:
                    w.write_bytes(b"A")
                with prod.acquire_write() as w:
                    w.write_bytes(b"B")
                with self.assertRaises(RingFullError):
                    prod.acquire_write()

                # Busy on double acquire_read
                r1 = cons.acquire_read()
                try:
                    with self.assertRaises(RingBusyError):
                        cons.acquire_read()
                finally:
                    r1.release()

                with cons.acquire_read() as r2:
                    self.assertEqual(bytes(r2.buffer), b"B")

    def test_encode_decode_int4_and_fp16_roundtrip(self) -> None:
        n_tokens, d_model = 4, 64
        total = n_tokens * d_model
        src = array.array("f", [math.sin(i * 0.13) * 2.5 for i in range(total)])
        lib = get_lib()

        for scheme, min_cos in ((QuantScheme.INT4_BLOCK, 0.99), (QuantScheme.FP16, 0.9999)):
            name = self._shm_name(f"codec_{int(scheme)}")
            with VectLinkRing.create(name, slot_capacity=4096, slot_count=4) as prod:
                with VectLinkRing.open(name) as cons:
                    with prod.acquire_write() as wslot:
                        written = wslot.encode_tensor(
                            src, n_tokens=n_tokens, d_model=d_model, quant=scheme, model_hash=0x1234
                        )
                        self.assertGreater(written, 0)

                    with cons.acquire_read() as rslot:
                        self.assertEqual(rslot.header.model_family_hash, 0x1234)
                        self.assertEqual(rslot.header.quant_scheme, int(scheme))
                        out_arr = array.array("f", [0.0] * total)
                        rslot.decode_tensor(out_arr)

                    # Compute cosine similarity using C library
                    import ctypes

                    a_ptr = (ctypes.c_float * total).from_buffer(src)
                    b_ptr = (ctypes.c_float * total).from_buffer(out_arr)
                    cos = lib.vtlk_cosine_similarity(a_ptr, b_ptr, total)
                    self.assertGreater(cos, min_cos)

    def test_numpy_zero_copy_if_available(self) -> None:
        try:
            import numpy as np
        except ImportError:
            self.skipTest("NumPy not installed")

        n_tokens, d_model = 8, 128
        src = np.linspace(-3.0, 3.0, n_tokens * d_model, dtype=np.float32).reshape(n_tokens, d_model)
        dst = np.empty_like(src)
        name = self._shm_name("numpy")

        with VectLinkRing.create(name, slot_capacity=8192, slot_count=4) as prod:
            with VectLinkRing.open(name) as cons:
                with prod.acquire_write() as wslot:
                    wslot.encode_tensor(src, n_tokens=n_tokens, d_model=d_model, quant=QuantScheme.INT4)
                with cons.acquire_read() as rslot:
                    rslot.decode_tensor(dst)
                cos = float(np.dot(src.ravel(), dst.ravel()) / (np.linalg.norm(src) * np.linalg.norm(dst)))
                self.assertGreater(cos, 0.995)

    def test_multiprocess_ipc(self) -> None:
        name = self._shm_name("mp")
        n_tokens, d_model = 4, 64
        src = array.array("f", [(i % 17 - 8) * 0.25 for i in range(n_tokens * d_model)])
        ctx = multiprocessing.get_context("spawn")
        q: multiprocessing.Queue = ctx.Queue()

        with VectLinkRing.create(name, slot_capacity=2048, slot_count=4) as prod:
            proc = ctx.Process(target=_mp_consumer_worker, args=(name, n_tokens, d_model, q))
            proc.start()
            with prod.acquire_write() as wslot:
                wslot.encode_tensor(src, n_tokens=n_tokens, d_model=d_model, quant=QuantScheme.INT4)
            res = q.get(timeout=5.0)
            proc.join(timeout=5.0)
            self.assertEqual(proc.exitcode, 0)
            self.assertEqual(res[0], "ok")
            self.assertEqual(res[1], n_tokens)
            self.assertEqual(res[2], d_model)
            self.assertEqual(res[3], int(QuantScheme.INT4))
            for a, b in zip(src[:8], res[4]):
                self.assertAlmostEqual(a, b, delta=0.25)


if __name__ == "__main__":
    unittest.main()
