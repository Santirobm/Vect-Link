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
    BufferTooSmallError,
    CrcMismatchError,
    InvalidParamError,
    QuantScheme,
    RingBusyError,
    RingEmptyError,
    RingFullError,
    VectLinkRing,
)
from vectlink._ffi import VTLK_HEADER_SIZE, VTLK_RING_CTRL_SIZE, get_lib  # noqa: E402


def _crash_producer_before_commit(shm_name: str) -> None:
    ring = VectLinkRing.open(shm_name)
    wslot = ring.acquire_write()
    wslot.buffer[0:4] = b"DEAD"
    os._exit(1)


def _crash_consumer_before_release(shm_name: str) -> None:
    ring = VectLinkRing.open(shm_name)
    rslot = ring.acquire_read()
    _ = bytes(rslot.buffer)
    os._exit(1)


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

    def test_dangling_subview_and_numpy_buffer_error(self) -> None:
        name = self._shm_name("dangling")
        with VectLinkRing.create(name, slot_capacity=256, slot_count=4) as prod:
            with VectLinkRing.open(name) as cons:
                # 1. Sub-memoryview slice prevents commit/release until released/deleted
                wslot = prod.acquire_write()
                wslot.write_bytes(b"\x01\x02\x03\x04" * 4)
                sub = wslot.buffer[:16]
                with self.assertRaises(BufferError):
                    wslot.commit()
                sub.release()
                del sub
                wslot.commit()

                rslot = cons.acquire_read()
                rsub = rslot.buffer[:8]
                with self.assertRaises(BufferError):
                    rslot.release()
                rsub.release()
                del rsub
                rslot.release()

                # 2. Dangling np.frombuffer / to_numpy(copy=False) prevents exit/release
                try:
                    import numpy as np
                except ImportError:
                    return

                with prod.acquire_write() as w2:
                    w2.write_bytes(np.arange(16, dtype=np.float32).tobytes())

                r2 = cons.acquire_read()
                arr = r2.to_numpy(dtype=np.float32, copy=False)
                self.assertEqual(float(arr[3]), 3.0)
                with self.assertRaises(BufferError):
                    r2.release()
                del arr
                r2.release()

                # Safe copy=True survives after ReadSlot exits
                with prod.acquire_write() as w3:
                    w3.write_bytes(np.arange(8, dtype=np.float32).tobytes())
                with cons.acquire_read() as r3:
                    safe_arr = r3.to_numpy(dtype=np.float32, copy=True)
                self.assertEqual(float(safe_arr[7]), 7.0)

    def test_close_while_slot_in_flight_raises_busy(self) -> None:
        name = self._shm_name("close_busy")
        prod = VectLinkRing.create(name, slot_capacity=128, slot_count=2)
        cons = VectLinkRing.open(name)
        try:
            wslot = prod.acquire_write()
            with self.assertRaises(RingBusyError):
                prod.close()
            wslot.write_bytes(b"OK")
            wslot.commit()

            rslot = cons.acquire_read()
            with self.assertRaises(RingBusyError):
                cons.close()
            rslot.release()
        finally:
            cons.close()
            prod.close()
            prod.unlink()

    def test_abrupt_process_exit_isolation(self) -> None:
        ctx = multiprocessing.get_context("spawn")

        # 1. Producer crashes before commit -> consumer sees RingEmptyError (no corruption)
        name_p = self._shm_name("crash_prod")
        with VectLinkRing.create(name_p, slot_capacity=128, slot_count=2) as ring:
            p = ctx.Process(target=_crash_producer_before_commit, args=(name_p,))
            p.start()
            p.join(timeout=5.0)
            self.assertNotEqual(p.exitcode, 0)
            with self.assertRaises(RingEmptyError):
                ring.acquire_read()

        # 2. Consumer crashes before release -> producer fills capacity and hits RingFullError
        name_c = self._shm_name("crash_cons")
        with VectLinkRing.create(name_c, slot_capacity=128, slot_count=2) as ring:
            with ring.acquire_write() as w:
                w.write_bytes(b"SLOT0")
            c = ctx.Process(target=_crash_consumer_before_release, args=(name_c,))
            c.start()
            c.join(timeout=5.0)
            self.assertNotEqual(c.exitcode, 0)

            with ring.acquire_write() as w:
                w.write_bytes(b"SLOT1")
            with self.assertRaises(RingFullError):
                ring.acquire_write()

    def test_100_open_close_unlink_cycles(self) -> None:
        name = self._shm_name("cycles100")
        for i in range(100):
            with VectLinkRing.create(name, slot_capacity=64, slot_count=2) as prod:
                with VectLinkRing.open(name) as cons:
                    with prod.acquire_write() as w:
                        w.write_bytes(f"msg-{i}".encode("ascii"))
                    with cons.acquire_read() as r:
                        self.assertEqual(bytes(r.buffer), f"msg-{i}".encode("ascii"))

    def test_hostile_inputs_and_crc_corruption(self) -> None:
        # Invalid SHM names
        for bad_name in ("", "/", "x" * 300):
            with self.assertRaises(InvalidParamError):
                VectLinkRing.create(bad_name, slot_capacity=64, slot_count=2)

        # Invalid capacities
        with self.assertRaises(ValueError):
            VectLinkRing.create(self._shm_name("badcap"), slot_capacity=64, slot_count=3)

        name = self._shm_name("hostile")
        with VectLinkRing.create(name, slot_capacity=128, slot_count=4) as prod:
            with VectLinkRing.open(name) as cons:
                # write_bytes exceeding slot capacity -> BufferTooSmallError
                with prod.acquire_write() as w:
                    with self.assertRaises(BufferTooSmallError):
                        w.write_bytes(b"Z" * 129)

                # encode_tensor with too few floats -> ValueError
                small_floats = array.array("f", [1.0] * 16)
                with prod.acquire_write() as w:
                    with self.assertRaises(ValueError):
                        w.encode_tensor(small_floats, n_tokens=1, d_model=32)

                # decode_tensor into too-small output buffer -> ValueError
                ok_floats = array.array("f", [1.0] * 32)
                with prod.acquire_write() as w:
                    w.encode_tensor(ok_floats, n_tokens=1, d_model=32, quant=QuantScheme.INT4)
                with cons.acquire_read() as r:
                    with self.assertRaises(ValueError):
                        r.decode_tensor(small_floats)

                # Payload corruption in SHM -> CrcMismatchError
                with prod.acquire_write() as w:
                    w.write_bytes(b"INTEGRITY_TEST_PAYLOAD")
                import ctypes

                slot_idx = 1  # second packet landed in slot 1
                slot_off = VTLK_RING_CTRL_SIZE + slot_idx * int(prod._ring.slot_size) + VTLK_HEADER_SIZE
                ptr = ctypes.cast(int(prod._shm.addr) + slot_off, ctypes.POINTER(ctypes.c_uint8))
                ptr[3] ^= 0xFF
                with self.assertRaises(CrcMismatchError):
                    cons.acquire_read()


if __name__ == "__main__":
    unittest.main()
