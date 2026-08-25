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
# from the repository root, after `make build`
python examples/wka/run_point_target_cpu.py --n 512          # focus + PNG
python examples/wka/run_point_target_hls.py --n 256     # design + validation package
```

The CPU runner reports where the brightest target landed and its impulse response; at `--n 512` each target focuses to roughly one resolution cell:

```
peak at (333, 358), expected (333, 358), error (+0, +0)
  range: IRW  2.12 samples, PSLR  -31.7 dB, ISLR  -28.6 dB
azimuth: IRW  2.20 samples, PSLR  -27.1 dB, ISLR  -22.9 dB
```

The HLS runner writes the validation package into `hls_project/wka/`; its contents are listed in [docs/backends.md](../../docs/backends.md#hls-validation-package). C-sim through Vitis HLS is bit-exact against the NumPy reference (max |err| 0.0 over 16384 samples).

![synthetic point targets](assets/wka_synthetic_512.png)

## Real data (ALOS-1)

```bash
# from the repository root; the dataset is shared by all three algorithms
python examples/data/extract_alos.py   # CEOS L1.0 -> alos_raw_...bin (2 GiB)
python examples/wka/run_alos_cpu.py        # tens of GiB of RAM at full size
python examples/wka/run_alos_hls.py   # emit the artifacts at that geometry
```

The runner reports wall time and urban-area contrast (`benchmarks/metrics.py:urban_contrast`, higher is sharper). The effective radar velocity in `ALOS_PARAMS` is autofocus-calibrated by image-contrast maximization over the urban area. The focused image is committed as `assets/san_francisco_wka.png`; the raw product it is made from is not redistributed here.

`run_alos_hls.py` writes one package under `hls_project/wka_alos/`. `wka_alos.h` / `wka_alos.cpp`, its testbench, binary data, manifest, stubs, and C-sim/C-synth/C-RTL scripts plus the portable fallback all describe the same `--n` raster and interface selected by the HLS configuration. The shipped default is AXI; a Python caller can pass different compile options through `emit(...)`.

Tests (`test/python/test_wka.py`) check numerical equivalence with the reference, point-target focusing, the ALOS parameter set, and HLS C++ emission.
