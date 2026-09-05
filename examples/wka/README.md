# omega-K (WKA) imaging with SAR-DSL

The complete wavenumber-domain (omega-K / WKA) SAR imaging algorithm as a single SAR-DSL kernel, validated against a NumPy reference and demonstrated on both synthetic point targets and a real ALOS-1 dataset.

| File | Purpose |
| --- | --- |
| `algorithm.py` | The WKA chain in the DSL (`build_kernel`, `make_inputs`) |
| `reference.py` | NumPy reference implementation (`WKAProcessor`) |
| `run_point_target_cpu.py` | Full cpu-backend flow: simulate, focus, save a PNG |
| `run_point_target_hls.py` | Full hls-backend flow: HLS C++ design + validation package (`hls_project/`) |
| `run_alos_cpu.py` | Focus the real ALOS-1 San Francisco dataset (16384 × 16384) |
| `run_alos_hls.py` | Emit the ALOS-geometry artifact set (see below) |
| `handwritten_hls/` | Standalone FP32 Vitis HLS reference implementation |
| `assets/` | Reference imagery |

## Processing chain

```
raw --> range FFT --> corner turn --> azimuth FFT --> corner turn
    --> bulk compression (reference phase multiply)
    --> Stolt interpolation (sar.stolt_interp)
    --> range window --> azimuth window
    --> range IFFT --> corner turn --> azimuth IFFT --> corner turn --> |.|
```

Everything runs inside one compiled kernel; the host only passes the band-matched Hann tapers as inputs. Acquisition metadata -- scalar radar parameters (`fc`, `Vr`, `R0`, `Kr`, ...) and the frequency axes -- bakes into the IR at trace time.

The optional [`handwritten_hls/`](handwritten_hls/) directory implements the same processing chain directly in Vitis HLS C++. It is kept independent of the compiler and normal example runners, and documents packed AXI, radix-4 row parallelism, fixed stage specialization, fused Stolt processing, and ping-pong corner turns as an optimization reference. Its production top uses the same nine array arguments and eight AXI bundles as generated WKA.

## Running

```bash
PYTHONPATH=python python examples/wka/run_point_target_cpu.py --n 512
PYTHONPATH=python python examples/wka/run_point_target_hls.py --n 256
```

The CPU runner saves a focused image and reports impulse-response metrics. The HLS runner writes `hls_project/wka/`; package contents are listed in the [backend guide](../../docs/backends.md#generated-package). Numerical results are maintained in the [benchmark report](../../benchmarks/README.md).

![synthetic point targets](assets/wka_synthetic_512.png)

## Real data (ALOS-1)

```bash
PYTHONPATH=python python examples/data/extract_alos.py
PYTHONPATH=python python examples/wka/run_alos_cpu.py
PYTHONPATH=python python examples/wka/run_alos_hls.py
```

The runner reports wall time and urban-area contrast (`benchmarks/metrics.py:urban_contrast`, higher is sharper). The effective radar velocity in `ALOS_PARAMS` is autofocus-calibrated by image-contrast maximization over the urban area. The focused image is committed as `assets/san_francisco_wka.png`; download and prepare the matching ALOS-1 granule as described in [the examples guide](../README.md#alos-1-stripmap-data).

`run_alos_hls.py` writes `hls_project/wka_alos/`, specialized to the selected raster and HLS configuration.

Tests (`test/python/test_wka.py`) check numerical equivalence with the reference, point-target focusing, the ALOS parameter set, and HLS C++ emission.
