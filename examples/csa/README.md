# Chirp Scaling Algorithm (CSA) with SAR-DSL

CSA is *interpolation-free*: range cell migration is equalized by a
quadratic phase multiply (chirp scaling) in the range-Doppler domain, so
the whole chain is element-wise phase multiplies between FFTs -- it
exercises the pure phase-multiply path of the compiler on both backends.

| File | Purpose |
|------|---------|
| `algorithm.py` | The CSA chain in the DSL (`build_kernel`, `make_inputs`) |
| `reference.py` | NumPy reference implementation (`CSAProcessor`) |
| `run_point_target_cpu.py` | Full cpu-backend flow: simulate, focus, save a PNG |
| `run_point_target_hls.py` | Full hls-backend flow: HLS C++ design + csim package (`hls_project/`) |
| `run_alos_cpu.py` | Focus the real ALOS-1 San Francisco dataset |
| `run_alos_hls.py` | Emit a synthesizable design at the full ALOS raster |

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

The Doppler-dependent factors -- migration factor `D(fa)` and modified
chirp rate `Km(fa)` -- are computed inside the kernel from the `fa` axis
with element-wise ops; the host only provides the `fa`/`fr`/`tau` axes.

The residual phase introduced by the scaling multiply (Cumming & Wong
eq. 7.36) is not corrected: the chain outputs a magnitude image, where
that term has no effect. Interferometric use would need the extra
multiply after azimuth compression.

## Running

```bash
# from the repository root, after `make build`
python examples/csa/run_point_target_cpu.py --n 512          # focus + PNG
python examples/csa/run_point_target_hls.py --n 256     # design + csim package
```

![synthetic point targets](assets/csa_synthetic_512.png)

Tests (`test/python/test_csa.py`) check numerical equivalence with the
reference, point-target focusing, cross-algorithm agreement with omega-K,
and HLS C++ emission.
