# Chirp Scaling Algorithm (CSA) with SAR-DSL

CSA is _interpolation-free_: range cell migration is equalized by a quadratic phase multiply (chirp scaling) in the range-Doppler domain, so the whole chain is element-wise phase multiplies between FFTs -- it exercises the pure phase-multiply path of the compiler on both backends.

| File | Purpose |
| --- | --- |
| `algorithm.py` | The CSA chain in the DSL (`build_kernel`, `make_inputs`) |
| `reference.py` | NumPy reference implementation (`CSAProcessor`) |
| `assets/` | Reference imagery |
| `run_point_target_cpu.py` | Full cpu-backend flow: simulate, focus, save a PNG |
| `run_point_target_hls.py` | Full hls-backend flow: HLS C++ design + validation package (`hls_project/`) |
| `run_alos_cpu.py` | Focus the real ALOS-1 San Francisco dataset |
| `run_alos_hls.py` | Emit the ALOS-geometry artifact set (see below) |

## Processing chain (zero Doppler centroid)

```
raw --> azimuth FFT
    --> chirp scaling multiply           (equalize migration trajectories)
    --> range FFT
    --> RC + SRC + bulk RCMC multiply    (2-D frequency domain)
    --> range IFFT
    --> azimuth compression multiply
    --> azimuth IFFT --> |.|
```

The Doppler-dependent factors -- migration factor `D(fa)` and modified chirp rate `Km(fa)` -- are computed inside the kernel from the `fa` axis with element-wise ops; the host only provides the `fa`/`fr`/`tau` axes.

The residual phase introduced by the scaling multiply (Cumming & Wong eq. 7.36) is not corrected: the chain outputs a magnitude image, where that term has no effect. Interferometric use would need the extra multiply after azimuth compression.

## Running

```bash
# from the repository root, after `make build`
python examples/csa/run_point_target_cpu.py --n 512          # focus + PNG
python examples/csa/run_point_target_hls.py --n 256     # design + validation package
```

At `--n 512` the brightest target lands on its predicted pixel:

```
peak at (333, 358), expected (333, 358), error (+0, +0)
  range: IRW  2.12 samples, PSLR  -31.7 dB, ISLR  -28.3 dB
azimuth: IRW  2.20 samples, PSLR  -27.2 dB, ISLR  -22.9 dB
```

The HLS runner writes the validation package into `hls_project/csa/`; its contents are listed in [docs/backends.md](../../docs/backends.md#hls-validation-package). C-sim through Vitis HLS matches the NumPy reference to 2.3e-10 over 65536 output samples.

![synthetic point targets](assets/csa_synthetic_512.png)

## Real data (ALOS-1)

```bash
python examples/data/extract_alos.py   # once; shared by all three algorithms
python examples/csa/run_alos_cpu.py
python examples/csa/run_alos_hls.py
```

The runner reports wall time and urban-area contrast. The focused image is committed as `assets/san_francisco_csa.png`; the raw product it is made from is not redistributed here.

`run_alos_hls.py` writes one package under `hls_project/csa_alos/`. `csa_alos.h` / `csa_alos.cpp`, its testbench, binary data, manifest, stubs, and C-sim/C-synth/C-RTL scripts plus the portable fallback all describe the same `--n` raster and interface selected by the HLS configuration.

Tests (`test/python/test_csa.py`) check numerical equivalence with the reference, point-target focusing, cross-algorithm agreement with omega-K, and HLS C++ emission.
