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
PYTHONPATH=python python examples/rda/run_point_target_cpu.py --n 512
PYTHONPATH=python python examples/rda/run_point_target_hls.py --n 256
```

The CPU runner saves a focused image and reports impulse-response metrics. The HLS runner writes `hls_project/rda/`; package contents are listed in the [backend guide](../../docs/backends.md#generated-package). Numerical results are maintained in the [benchmark report](../../benchmarks/README.md).

![synthetic point targets](assets/rda_synthetic_512.png)

## Real data (ALOS-1)

```bash
PYTHONPATH=python python examples/data/extract_alos.py
PYTHONPATH=python python examples/rda/run_alos_cpu.py
PYTHONPATH=python python examples/rda/run_alos_hls.py
```

The runner reports wall time and urban-area contrast. The focused image is committed as `assets/san_francisco_rda.png`; download and prepare the matching ALOS-1 granule as described in [the examples guide](../README.md#alos-1-stripmap-data).

`run_alos_hls.py` writes `hls_project/rda_alos/`, specialized to the selected raster and HLS configuration.

Tests (`test/python/test_rda.py`) check numerical equivalence with the reference, point-target focusing, HLS C++ emission, and cross-algorithm agreement: RDA and omega-K must place the same scatterer on the same pixel.
