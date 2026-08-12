# Benchmarks

```bash
PYTHONPATH=python python3 benchmarks/bench_wka.py --sizes 1024 4096 --numpy
```

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
