# omega-K (WKA) imaging with SAR-DSL

This example expresses the complete wavenumber-domain (omega-K / WKA) SAR
imaging algorithm as a single SAR-DSL kernel and validates it against a
NumPy reference implementation.

| File | Purpose |
|------|---------|
| `wka_dsl.py` | The WKA chain written in the DSL (`build_wka_kernel`) |
| `wka_numpy.py` | NumPy reference implementation (numerical ground truth) |
| `synthetic.py` | Point-target echo simulator for demos and tests |
| `run_synthetic.py` | Focus a synthetic scene, save a PNG |
| `run_alos.py` | Focus the real ALOS-1 San Francisco dataset (16384x16384) |
| `emit_hls.py` | Emit the complete kernel as a Vitis HLS C++ design |
| `data/` | CEOS L1.0 extraction tooling + raw ALOS product |
| `assets/` | Reference imagery |

## Quick start (synthetic)

```bash
# from the repository root, after `make build`
python examples/wka/run_synthetic.py --n 512
```

This simulates three point targets, compiles the WKA kernel with the CPU
backend and writes `wka_synthetic.png`. Each target focuses to roughly one
resolution cell:

![synthetic point targets](assets/wka_synthetic_512.png)

## Real data (ALOS-1)

```bash
cd examples/wka
python data/extract_alos.py     # CEOS L1.0 -> alos_raw_16384x16384.bin (2 GiB)
python run_alos.py              # needs tens of GiB of RAM at full size
```

The full 16384x16384 scene focuses in about 3.6 seconds on a large
multi-core machine (fused element-wise kernels under OpenMP; multithreaded
FFT/Stolt runtime):

![San Francisco Bay](assets/san_francisco_wka.png)

## Processing chain

```
raw --> range FFT --> corner turn --> azimuth FFT --> corner turn
    --> bulk compression (reference phase multiply)
    --> Stolt interpolation (sar.stolt_interp)
    --> range window --> azimuth window
    --> range IFFT --> corner turn --> azimuth IFFT --> corner turn --> |.|
```

Everything runs inside one compiled kernel; the host only precomputes
acquisition metadata (frequency axes via `fftfreq`, Hanning windows) and
passes it as inputs. Scalar radar parameters (`fc`, `Vr`, `R0`, `Kr`, ...)
are baked into the IR at trace time.
