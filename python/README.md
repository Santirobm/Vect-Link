# Vect-Link Python Bindings (`vectlink`)

Zero-dependency **zero-copy** Python bindings over `libvectlink` (`v0.9.0`) using `ctypes` + native `memoryview` (and optional NumPy `ndarray` zero-copy interoperability).

## 1. Build the Shared C Library (`vectlink_shared`)

From the repository root:

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --target vectlink_shared
```

This produces `build/vectlink.dll` (Windows), `build/libvectlink.so` (Linux), or `build/libvectlink.dylib` (macOS).

## 2. Install the Python Package

```sh
pip install -e ./python
# Or with optional NumPy extra:
pip install -e "./python[numpy]"
```

If your shared library is in a custom directory, set `VTLK_LIB_PATH=/path/to/vectlink.dll` (or `.so`/`.dylib`).

## 3. Zero-Copy Slot Loan Usage

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
```

## 4. Run Tests & Two-Process Example

```sh
python -m unittest discover -s python/tests -v
python python/examples/two_agents_zero_copy.py
```
