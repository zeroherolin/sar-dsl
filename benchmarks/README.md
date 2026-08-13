# Benchmarks

```bash
PYTHONPATH=python python3 benchmarks/bench_wka.py --sizes 1024 4096 --numpy
PYTHONPATH=python python3 benchmarks/point_target_quality.py --n 512
```

## Image quality

Point-target metrics on a 512x512 synthetic scene (Hanning windows, 70%
bandwidth occupancy; measured with 32x upsampled impulse-response cuts;
gated in `test/python/test_quality.py`):

| Algorithm | axis | IRW (samples) | PSLR | ISLR | peak error |
|-----------|------|--------------:|-----:|-----:|:----------:|
| omega-K | range / azimuth | 1.62 / 1.62 | -26.4 / -26.6 dB | -21.7 / -21.9 dB | (0, 0) |
| Range-Doppler | range / azimuth | 1.62 / 1.62 | -25.8 / -26.5 dB | -18.9 / -21.9 dB | (0, 0) |
| Chirp Scaling | range / azimuth | 1.62 / 1.62 | -26.5 / -26.5 dB | -21.7 / -21.9 dB | (0, 0) |

On the real 16384x16384 ALOS-1 scene (urban-area contrast, higher is
sharper; all runs ~3.5 s): omega-K 131.6, Range-Doppler 130.3, Chirp
Scaling 126.1. The autofocus-calibrated effective velocity (Vr = 7072,
see `common/params.py`) was verified to be the contrast optimum of all
three algorithms independently, so the comparison carries no calibration
bias; the residual spread reflects the algorithms' approximation orders
(omega-K is exact for the ideal hyperbolic model, chirp scaling is a
second-order approximation, range-Doppler additionally omits secondary
range compression).

## Reference numbers

Measured on a 240-core x86-64 server (LLVM 21, `-O3 -march=native`,
OpenMP parallelization, linalg elementwise fusion). Full omega-K imaging
chain, synthetic scenes:

| size | compile | run (best) | NumPy reference | speedup |
|------|---------|------------|-----------------|---------|
| 1024x1024 | 0.30 s | 0.21 s | 0.46 s | 2.2x |
| 4096x4096 | 0.29 s | 0.78 s | 6.25 s | 8.0x |
| 16384x16384 (ALOS-1 real data) | 0.3 s | 3.6 s | ~15 min* | ~250x |

\* extrapolated from the scaling of smaller sizes: the reference Stolt
loop dominates and is not practical to run at 16384^2.

History: before elementwise fusion and OpenMP parallelization were added
to the CPU pipeline, the compiled 16384x16384 scene took 55.4 s.
