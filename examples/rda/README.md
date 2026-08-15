# Range-Doppler Algorithm (RDA) with SAR-DSL

RDA exists here to prove the dialect generalizes: it introduces **no new
compiler operation** -- range cell migration correction (RCMC) is composed
from the orthogonal `sar.interp1d` primitive plus element-wise position
arithmetic.

| File | Purpose |
|------|---------|
| `algorithm.py` | The RDA chain in the DSL (`build_kernel`, `make_inputs`) |
| `reference.py` | NumPy reference implementation (`RDAProcessor`) |
| `run_point_target_cpu.py` | Full cpu-backend flow: simulate, focus, save a PNG |
| `run_point_target_hls.py` | Full hls-backend flow: HLS C++ design + csim package (`hls_project/`) |
| `run_alos_cpu.py` | Focus the real ALOS-1 San Francisco dataset |

## Processing chain

```
raw --> range FFT --> matched-filter multiply --> range IFFT
    --> azimuth FFT (range-Doppler domain)
    --> RCMC: positions = column + lambda^2 R fa^2 / (8 Vr^2) * 2 Fs / c
              data = sar.interp1d(data, positions)
    --> azimuth matched filter multiply (Ka(R) per range gate)
    --> azimuth IFFT --> |.|
```

Both the migration correction and the azimuth matched filter are
range-dependent (`R = c tau / 2` per gate), which matters across wide
swaths: on the 77 km ALOS swath a fixed `Ka(R0)` loses most of the
focus away from the reference range.

This is the *basic* RDA (Cumming & Wong ch. 6): parabolic
range-migration model, no secondary range compression. The residual
range-azimuth coupling raises the range-axis integrated sidelobes a
few dB above omega-K and CSA (see `benchmarks/README.md`); at low
squint and moderate bandwidth that is the textbook trade-off of the
basic form.

## Running

```bash
# from the repository root, after `make build`
python examples/rda/run_point_target_cpu.py --n 512          # focus + PNG
python examples/rda/run_point_target_hls.py --n 256     # design + csim package
```

![synthetic point targets](assets/rda_synthetic_512.png)

Tests (`test/python/test_rda.py`) check numerical equivalence with the
reference, point-target focusing, and cross-algorithm agreement: RDA and
omega-K must place the same scatterer on the same pixel.
