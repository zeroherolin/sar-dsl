# Benchmarks

| File | Purpose |
|------|---------|
| `algorithms.py` | Registry of the benchmarked imaging chains (setup, run, reference) |
| `metrics.py` | Point-target metrics: IRW / PSLR / ISLR (also used by `test/python/test_quality.py`) |
| `run_performance.py` | Timing table: compiled kernels vs the NumPy references |
| `run_quality.py` | Image-quality table |
| `run_figures.py` | Publication figures into `assets/` |

```bash
PYTHONPATH=python python3 benchmarks/run_performance.py \
    --sizes 1024 4096 --numpy
PYTHONPATH=python python3 benchmarks/run_quality.py --n 512
PYTHONPATH=python python3 benchmarks/run_figures.py
```

## Image quality

![Point-target impulse response](assets/point_target_response.png)

Point-target metrics on a 512x512 synthetic scene (Hann tapers matched
to the occupied 70% bandwidth; measured with 32x upsampled
impulse-response cuts; gated in `test/python/test_quality.py`). IRW
matches the Hann theory line (1.44 bins / 0.70 occupancy = 2.06
samples); PSLR beats the -31.5 dB window bound thanks to the smooth
LFM band edges:

| Algorithm | axis | IRW (samples) | PSLR | ISLR | peak error |
|-----------|------|--------------:|-----:|-----:|:----------:|
| omega-K | range / azimuth | 2.12 / 2.12 | -38.4 / -38.4 dB | -29.0 / -31.8 dB | (0, 0) |
| Range-Doppler | range / azimuth | 2.09 / 2.12 | -38.8 / -38.4 dB | -26.5 / -31.8 dB | (0, 0) |
| Chirp Scaling | range / azimuth | 2.12 / 2.12 | -38.5 / -38.4 dB | -29.0 / -31.8 dB | (0, 0) |

Far-sidelobe floor: at deep display ranges (-60 dB) the images show
faint sidelobe arms at ~-50 dB. This floor is the Fresnel ripple of
the synthetic LFM spectrum -- paired echoes bounded by the scene's
time-bandwidth product (TBP 179 at n=512; the floor drops with TBP,
e.g. below -62 dB at n=4096, and an ideal Hann spectrum measures
-130 dB through the same pipeline, so processing adds nothing).
Range-Doppler sits ~6 dB above the others (~-43 dB, beaded arms):
the residual range-azimuth cross-coupling of the classic
no-secondary-range-compression formulation, invariant under
interpolator taps (2/4/8/16 measured identical) -- an algorithm
property, not a processing artifact.

On the real 16384x16384 ALOS-1 scene (urban contrast: std/mean over
the brightest 512x512 tile of the 4x4-multilooked magnitude image,
`metrics.urban_contrast`, higher is sharper; all runs 3.5-4 s):
omega-K 0.810, Chirp Scaling 0.809, Range-Doppler 0.808. The
autofocus-calibrated effective velocity (Vr = 7072, see
`common/params.py`) was verified to be the contrast optimum of all
three algorithms independently, so the comparison carries no
calibration bias; the ordering follows the algorithms' approximation
orders (omega-K is exact for the ideal hyperbolic model, chirp scaling
is a second-order approximation, range-Doppler additionally omits
secondary range compression), though at this low squint the spread is
small.

## Reference numbers

Measured on a 240-core x86-64 server (LLVM 22, `-O3 -march=native`,
OpenMP parallelization, linalg elementwise fusion). All four compiled
chains vs their NumPy references, synthetic scenes (PFA size is the
polar-grid edge; its image is 2x oversampled):

| Algorithm | size | compile | run (best) | NumPy reference | speedup |
|-----------|------|--------:|-----------:|----------------:|--------:|
| omega-K | 1024 | 0.38 s | 0.22 s | 0.49 s | 2.2x |
| omega-K | 4096 | 0.49 s | 0.80 s | 6.36 s | 8.0x |
| Range-Doppler | 1024 | 0.25 s | 0.32 s | 0.52 s | 1.6x |
| Range-Doppler | 4096 | 0.25 s | 0.71 s | 7.82 s | 11.1x |
| Chirp Scaling | 1024 | 0.29 s | 0.26 s | 0.17 s | 0.7x |
| Chirp Scaling | 4096 | 0.37 s | 0.59 s | 3.36 s | 5.7x |
| PFA | 512 | 0.34 s | 0.13 s | 0.35 s | 2.7x |
| PFA | 1024 | 0.35 s | 0.23 s | 1.18 s | 5.2x |

Small scenes are scheduling-bound: CSA at 1024 is pure FFTs plus fused
element-wise phase multiplies, which numpy also does well, and the
OpenMP dispatch overhead dominates -- the compiled chains pull ahead as
soon as the working set grows.

The omega-K headline on real data:

| size | compile | run (best) | NumPy reference | speedup |
|------|--------:|-----------:|----------------:|--------:|
| 16384x16384 (ALOS-1) | 2.8 s | 3.6 s | ~15 min* | ~250x |

\* extrapolated from the scaling of smaller sizes: the reference Stolt
loop dominates and is not practical to run at 16384^2.

History: before elementwise fusion and OpenMP parallelization were added
to the CPU pipeline, the compiled 16384x16384 scene took 55.4 s.
