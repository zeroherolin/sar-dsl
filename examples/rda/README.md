# Range-Doppler Algorithm (RDA) with SAR-DSL

The Range-Doppler algorithm as a single SAR-DSL kernel: range compression, then range cell migration correction (RCMC) and azimuth compression in the range-Doppler domain. RCMC is composed from the `sar.interp1d` primitive plus element-wise position arithmetic -- the chain needs no operation the other examples do not already use.

| File | Purpose |
| --- | --- |
| `algorithm.py` | The RDA chain in the DSL (`build_kernel`, `make_inputs`) |
| `reference.py` | NumPy reference implementation (`RDAProcessor`) |
| `assets/` | Reference imagery |
| `run_point_target_cpu.py` | Full cpu-backend flow: simulate, focus, save a PNG |
| `run_point_target_hls.py` | Full hls-backend flow: HLS C++ design + validation package (`hls_project/`) |
| `run_alos_cpu.py` | Focus the real ALOS-1 San Francisco dataset |
| `run_alos_hls.py` | Emit the ALOS-geometry artifact set (see below) |

## Processing chain

```
raw --> range FFT --> matched-filter multiply --> range IFFT
    --> azimuth FFT (range-Doppler domain)
    --> RCMC: positions = column + lambda^2 R fa^2 / (8 Vr^2) * 2 Fs / c
              data = sar.interp1d(data, positions)
    --> azimuth matched filter multiply (Ka(R) per range gate)
    --> azimuth IFFT --> |.|
```

Both the migration correction and the azimuth matched filter are range-dependent (`R = c tau / 2` per gate), which matters across wide swaths: on the 77 km ALOS swath a fixed `Ka(R0)` loses most of the focus away from the reference range.

This is the _basic_ RDA (Cumming & Wong ch. 6): parabolic range-migration model, no secondary range compression. The residual range-azimuth coupling raises the range-axis integrated sidelobes a few dB above omega-K and CSA (see `benchmarks/README.md`); at low squint and moderate bandwidth that is the textbook trade-off of the basic form.

## Running

```bash
# from the repository root, after `make build`
python examples/rda/run_point_target_cpu.py --n 512          # focus + PNG
python examples/rda/run_point_target_hls.py --n 256     # design + validation package
```

At `--n 512` the brightest target lands on its predicted pixel:

```
peak at (333, 358), expected (333, 358), error (+0, +0)
  range: IRW  2.12 samples, PSLR  -31.9 dB, ISLR  -26.1 dB
azimuth: IRW  2.20 samples, PSLR  -27.2 dB, ISLR  -22.9 dB
```

The HLS runner writes the validation package into `hls_project/rda/`; its contents are listed in [docs/backends.md](../../docs/backends.md#hls-validation-package). C-sim through Vitis HLS is bit-exact against the NumPy reference (max |err| 0.0 over 65536 samples).

![synthetic point targets](assets/rda_synthetic_512.png)

## Real data (ALOS-1)

```bash
python examples/data/extract_alos.py   # once; shared by all three algorithms
python examples/rda/run_alos_cpu.py
python examples/rda/run_alos_hls.py
```

The runner reports wall time and urban-area contrast. The focused image is committed as `assets/san_francisco_rda.png`; the raw product it is made from is not redistributed here.

`run_alos_hls.py` writes one package under `hls_project/rda_alos/`. `rda_alos.h` / `rda_alos.cpp`, its testbench, binary data, manifest, stubs, and C-sim/C-synth/C-RTL scripts plus the portable fallback all describe the same `--n` raster and interface selected by the HLS configuration.

Tests (`test/python/test_rda.py`) check numerical equivalence with the reference, point-target focusing, HLS C++ emission, and cross-algorithm agreement: RDA and omega-K must place the same scatterer on the same pixel.
