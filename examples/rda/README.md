# Range-Doppler Algorithm (RDA) with SAR-DSL

The second imaging algorithm in the repository, and deliberately so: RDA
exists to prove the dialect generalizes. It introduces **no new compiler
operation** -- range cell migration correction (RCMC) is composed from the
orthogonal `sar.interp1d` primitive plus element-wise position arithmetic.

| File | Purpose |
|------|---------|
| `rda_dsl.py` | RDA chain in the DSL (`build_rda_kernel`) |
| `rda_numpy.py` | NumPy reference implementation |

Processing chain:

```
raw --> range FFT --> matched-filter multiply --> range IFFT
    --> azimuth FFT (range-Doppler domain)
    --> RCMC: positions = column + (lambda^2 R0 fa^2 / 8 Vr^2)(2 Fs / c)
              data = sar.interp1d(data, positions)
    --> azimuth matched filter multiply --> azimuth IFFT --> |.|
```

Tests (`test/python/test_rda.py`) check numerical equivalence with the
reference, point-target focusing, and cross-algorithm agreement: RDA and
omega-K must place the same scatterer on the same pixel.
