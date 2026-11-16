# Vect-Link Python Bindings (`vectlink`)

Zero-dependency **zero-copy** Python bindings over `libvectlink` (`v0.9.0`) using `ctypes` + native `memoryview` (and optional NumPy `ndarray` zero-copy interoperability).

## 1. Install the Python Package (`scikit-build-core`)

From the repository root, `pip install ./python` compiles `vectlink_shared` (`vectlink.dll` / `libvectlink.so` / `libvectlink.dylib`) and bundles it directly inside `vectlink/`:

```sh
pip install ./python
# Or with optional NumPy extra:
pip install "./python[numpy]"
```

*(Alternatively, for editable/in-repo use, build `vectlink_shared` manually with `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --target vectlink_shared` or set `VTLK_LIB_PATH=/path/to/vectlink.dll`.)*

## 2. Zero-Copy Slot Loan Usage

```python
import numpy as np
from vectlink import QuantScheme, VectLinkRing

tensor = np.random.randn(32, 4096).astype(np.float32)
out = np.empty_like(tensor)

with VectLinkRing.create("/my_ring", slot_capacity=128 * 1024, slot_count=4) as prod:
    with VectLinkRing.open("/my_ring") as cons:
        # Producer writes INT4 directly into SHM slot (zero intermediate copies)
        with prod.acquire_write() as wslot:
            wslot.encode_tensor(tensor, n_tokens=32, d_model=4096, quant=QuantScheme.INT4)

        # Consumer reads & decodes directly from SHM slot into NumPy buffer
        with cons.acquire_read() as rslot:
            rslot.decode_tensor(out)
            # Or use rslot.to_numpy(dtype=np.float32, copy=True) for a safe decoupled array
```

*(Dangling zero-copy slices or `to_numpy(copy=False)` arrays kept alive past `commit()`/`release()` raise `BufferError`, and closing a ring with active slot loans raises `RingBusyError`.)*

## 3. Run Tests & Two-Process Example

```sh
python -m unittest discover -s python/tests -v
python python/examples/two_agents_zero_copy.py
```
