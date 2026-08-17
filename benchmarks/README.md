# Benchmarks

| File | Purpose |
|------|---------|
| `algorithms.py` | Registry of the benchmarked imaging chains (setup, run, reference) |
| `metrics.py` | Point-target metrics: IRW / PSLR / ISLR (also used by `test/python/test_quality.py`) |
| `run_performance.py` | Timing and throughput-N curves: compiled kernels vs the NumPy references; reports cold and warm separately |
| `run_accuracy.py` | Cross-backend accuracy: CPU and HLS csim versus the same NumPy reference; `--dtype c64` builds the chains single-precision, completing the backend x precision matrix |
| `run_quality.py` | Image-quality table |
| `run_precision.py` | What single precision costs an image; covers all four chains and both complex widths |
| `run_resources.py` | What each chain costs on an FPGA; `--budget-sweep` for port/memory vs budget |
| `run_figures.py` | Publication figures into `assets/` |

```bash
PYTHONPATH=python python3 benchmarks/run_performance.py \
    --sizes 128 256 512 1024 2048 4096 --numpy
PYTHONPATH=python python3 benchmarks/run_accuracy.py --n 512
PYTHONPATH=python python3 benchmarks/run_quality.py --n 512
PYTHONPATH=python python3 benchmarks/run_precision.py --n 512
PYTHONPATH=python python3 benchmarks/run_resources.py --sizes 512 4096
PYTHONPATH=python python3 benchmarks/run_resources.py \
    --budget-sweep --sweep-size 1024 --sweep-steps 10
PYTHONPATH=python python3 benchmarks/run_figures.py
```

## Precision

Narrowing the data path narrows the kernel *boundary*; inside the body a
phase-multiply that combines a c64 data plane with an f64 geometry plane
widens back to c128 (SAR-DSL promotes on mixed-precision ops, like NumPy).
The "c64/c128 planes" column reports actual complex-plane types in the
emitted IR so the table never claims a narrower path than it runs.

PFA's collection axes bake into the IR as f64 constants (they feed
`interp1d` positions, which the language requires to be f64), so its only
narrowable input is the phase-history plane itself. Geometry stays double
in all chains.

Single precision is a first-class build option: every chain's
`build_kernel` takes `dtype=sar.c64` (default `sar.c128`), which threads
the precision through the annotations, the widening cast and the result
types. The mode label reports what the build narrowed: the stripmap
chains take their window/axis vectors down with the data (`c64+f32`),
PFA narrows the data plane alone (`c64 only`).

Results at N=512, 240-core x86-64:

| Algorithm | precision | IRW | PSLR | ISLR | rel. error | c64/c128 planes |
|-----------|-----------|----:|-----:|-----:|-----------:|----------------:|
| omega-K | f64 | 2.06 | -38.44 dB | -29.01 dB | — | — |
| omega-K | f32 (c64+f32) | 2.06 | -38.44 dB | -29.01 dB | 1.5e-08 | 14 / 40 |
| Range-Doppler | f64 | 2.06 | -38.79 dB | -26.54 dB | — | — |
| Range-Doppler | f32 (c64+f32) | 2.06 | -38.79 dB | -26.54 dB | 1.9e-07 | 28 / 0 |
| Chirp Scaling | f64 | 2.06 | -38.48 dB | -28.98 dB | — | — |
| Chirp Scaling | f32 (c64+f32) | 2.06 | -38.48 dB | -28.98 dB | 4.3e-07 | 6 / 34 |
| PFA | f64 | 1.80 | -13.25 dB | -9.59 dB | — | — |
| PFA | f32 (c64 only) | 1.80 | -13.25 dB | -9.59 dB | 6.3e-08 | 73 / 0 |

IRW and PSLR are unchanged across all four chains. Range-Doppler and
PFA have zero residual c128 planes in the narrowed IR, meaning those
data paths run fully at single precision. omega-K and CSA each carry 40
and 34 c128 planes respectively, because Stolt interpolation and chirp
scaling both multiply the complex data against f64 phase screens computed
from frequency-axis geometry that must stay double.

## FPGA resources

What a design asks of the device, measured from the emitted C++
(`run_resources.py`, 512x512, 4 MiB on-chip budget). Ports are AXI master
pointers, which share bundles by element type:

| Algorithm | ports | bundles | DRAM | on-chip | dataflow regions |
|-----------|------:|--------:|-----:|--------:|-----------------:|
| omega-K | 6 | 2 | 15.0 MiB | 1024 KiB | 12 |
| Range-Doppler | 9 | 2 | 13.0 MiB | 272 KiB | 4 |
| Chirp Scaling | 9 | 2 | 11.0 MiB | 256 KiB | 4 |
| PFA | 5 | 1 | 93.9 MiB | 2480 KiB | 6 |

PFA resamples onto a 2x oversampled polar grid, so its planes are four
times the size of its arguments and more of them have to stream. Its
collection axes bake into the design as constants (they are fixed by the
geometry the kernel was built for), so its ports are the phase-history
planes, the two images and the scratch.

### On-chip budget sweep

![On-chip budget sweep](assets/budget_sweep.png)

`run_resources.py --budget-sweep` walks `on_chip_budget` from 0 to the
chain's full working set, in `--sweep-steps` equal steps, and reports
ports, DRAM traffic and on-chip usage at each point. A budget of 0 means
*unbounded* -- nothing is streamed (see `docs/backends.md`) -- so the
sweep's lower end measures the resident footprint the design would need
to hold everything internally, and that measurement is what bounds the
ladder: past the full working set, raising the budget cannot move
anything.

At N=1024 the AXI master port count is flat across the entire budget
ladder (omega-K: 6, RDA/CSA: 9, PFA: 5). Ports are assigned by element
type, not by how many buffers happen to be resident, so raising or
lowering the budget cannot change them. The on-chip usage is non-monotone:
the buffer-placement heuristic moves whole planes (each 8 MiB at 1024)
in a binary on/off decision, so usage can jump up or down at a budget
boundary. That behavior is a property of the emitter, not a flaw.

## Throughput

![Kernel throughput vs scene size](assets/throughput.png)

Kernel throughput (N² input samples / warm kernel run time, compile time
excluded) against scene size, log-log, one line per chain. Measured as
the minimum of 5 timed runs after 3 warmup passes.

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
`metrics.urban_contrast`, higher is sharper; all runs 2.5-3 s -- warm
times, as observed during the contrast sweep):
omega-K 0.810, Chirp Scaling 0.809, Range-Doppler 0.808. The
autofocus-calibrated effective velocity (Vr = 7072, see
`common/params.py`) was verified to be the contrast optimum of all
three algorithms independently, so the comparison carries no
calibration bias; the ordering follows the algorithms' approximation
orders (omega-K is exact for the ideal hyperbolic model, chirp scaling
is a second-order approximation, range-Doppler additionally omits
secondary range compression), though at this low squint the spread is
small.

## Backend comparison

Scene: synthetic single scatterer, N×N input samples (PFA: polar-grid
edge; image is 2N×2N). Compile time excluded from run times; machine as
in [Reference numbers](#reference-numbers).

### Accuracy (N=512, vs the f64 NumPy reference)

`run_accuracy.py` focuses one scene three ways -- the reference, the CPU
backend, and the emitted design under csim -- at both build precisions
(`--dtype`). The csim leg uses no Vitis: `write_testbench` ships header
stubs that stand in for the Vitis headers, and the package is compiled
with the in-tree clang++ and run as a plain binary.

At f64 the backends sit at double-rounding distance from the reference
(and usually at the identical value, since both compile the same IR; the
CPU runtime's FFT may order its butterflies differently, which is the
Range-Doppler row):

| Algorithm | CPU err | CPU rel | csim err | csim rel | csim |
|-----------|--------:|--------:|---------:|---------:|-----:|
| omega-K | 5.8e-11 | 9.3e-13 | 5.8e-11 | 9.3e-13 | PASS |
| Range-Doppler | 1.2e-10 | 9.7e-14 | 4.7e-10 | 3.9e-13 | PASS |
| Chirp Scaling | 1.5e-11 | 2.3e-13 | 1.5e-11 | 2.3e-13 | PASS |
| PFA | 4.3e-15 | 1.7e-14 | 4.3e-15 | 1.7e-14 | PASS |

With `--dtype c64` the same columns measure what single precision costs
on each backend (relative to peak; the CPU FFT computes f64 butterflies
internally, the HLS FFT computes in the declared width, hence the gap
on the FFT-heavy omega-K). The testbench tolerance scales with the
scene peak here, since f32 rounding is proportional to the signal:

| Algorithm | CPU rel | csim rel | csim |
|-----------|--------:|---------:|-----:|
| omega-K | 1.5e-08 | 6.1e-08 | PASS |
| Range-Doppler | 1.9e-07 | 2.0e-07 | PASS |
| Chirp Scaling | 4.3e-07 | 4.3e-07 | PASS |
| PFA | 6.3e-08 | 1.2e-07 | PASS |

### Performance

CPU wall times (cold and warm, per chain and size) are tabulated once,
in [Reference numbers](#reference-numbers) below.

For HLS there is **no FPGA wall time, and none can be produced here**.
csim is a functional, single-threaded simulation of the emitted C++
running on the host; it validates numerics only. The csim runs above complete
in 0.21–0.74 s at N=512, and those seconds are a property of this host, not of
any device. Device throughput needs synthesis plus either co-simulation or
hardware, and Vitis is not installed on this machine. What each design
asks of a device -- ports, bundles, DRAM, on-chip, dataflow regions --
is derived from the emitted C++ in [FPGA resources](#fpga-resources)
above, bounding what it *asks for* rather than what it costs in
LUT/FF/DSP.

## Reference numbers

Machine: 240-core x86-64, LLVM 22, `-O3 -march=native`, OpenMP
parallelization, linalg elementwise fusion. Every number in this file was
measured on this machine with the plane pool active.

Full size ladder, one fresh process per row. `cold` is that process's first
call, `warm` the minimum of 5 after 3 warmup passes; the gap between them is
the page-fault cost the pool removes, which is why the two are never collapsed
into one figure.

| Algorithm | size | compile | cold | warm | NumPy reference | speedup (warm) |
|-----------|------|--------:|-----:|-----:|----------------:|---------------:|
| omega-K | 128 | 0.69 s | 0.146 s | 0.078 s | 0.02 s | 0.3x |
| omega-K | 256 | 0.67 s | 0.177 s | 0.081 s | 0.05 s | 0.7x |
| omega-K | 512 | 0.86 s | 0.230 s | 0.213 s | 0.20 s | 0.9x |
| omega-K | 1024 | 0.90 s | 0.380 s | 0.245 s | 0.48 s | 1.9x |
| omega-K | 2048 | 0.94 s | 0.627 s | 0.306 s | 1.87 s | 6.1x |
| omega-K | 4096 | 1.21 s | 1.079 s | 0.477 s | 7.57 s | 15.9x |
| Range-Doppler | 128 | 0.70 s | 0.164 s | 0.075 s | 0.03 s | 0.3x |
| Range-Doppler | 256 | 0.66 s | 0.156 s | 0.079 s | 0.05 s | 0.6x |
| Range-Doppler | 512 | 0.85 s | 0.195 s | 0.191 s | 0.18 s | 0.9x |
| Range-Doppler | 1024 | 0.84 s | 0.267 s | 0.235 s | 0.45 s | 1.9x |
| Range-Doppler | 2048 | 0.86 s | 0.487 s | 0.301 s | 1.63 s | 5.4x |
| Range-Doppler | 4096 | 0.85 s | 1.422 s | 0.452 s | 6.60 s | 14.6x |
| Chirp Scaling | 128 | 0.69 s | 0.135 s | 0.068 s | 0.00 s | 0.0x |
| Chirp Scaling | 256 | 0.66 s | 0.119 s | 0.062 s | 0.01 s | 0.1x |
| Chirp Scaling | 512 | 0.95 s | 0.147 s | 0.141 s | 0.05 s | 0.3x |
| Chirp Scaling | 1024 | 0.95 s | 0.201 s | 0.179 s | 0.19 s | 1.1x |
| Chirp Scaling | 2048 | 1.01 s | 0.347 s | 0.225 s | 0.94 s | 4.2x |
| Chirp Scaling | 4096 | 1.17 s | 0.941 s | 0.307 s | 4.35 s | 14.2x |
| PFA | 128 | 0.93 s | 0.227 s | 0.081 s | 0.06 s | 0.7x |
| PFA | 256 | 1.33 s | 0.210 s | 0.095 s | 0.14 s | 1.5x |
| PFA | 512 | 0.91 s | 0.291 s | 0.120 s | 0.48 s | 4.0x |
| PFA | 1024 | 1.40 s | 0.338 s | 0.241 s | 1.50 s | 6.2x |
| PFA | 2048 | 1.79 s | 0.520 s | 0.393 s | 6.17 s | 15.7x |
| PFA | 4096 | 4.66 s | 1.178 s | 0.716 s | 28.02 s | 39.1x |

PFA size is the polar-grid edge; its image is 2x oversampled, so a row at
size N does 4N² of output work.

Small scenes are scheduling-bound: CSA at 128–512 is pure FFTs plus fused
element-wise phase multiplies, which numpy also does well, and the
OpenMP dispatch overhead dominates -- the compiled chains pull ahead as
soon as the working set grows.

The omega-K headline on real data (16384×16384 ALOS-1,
`examples/wka/run_alos_cpu.py`):

| size | compile | cold | NumPy reference | speedup |
|------|--------:|-----:|----------------:|--------:|
| 16384×16384 (ALOS-1) | 2.8 s | 4.6 s | ~15 min* | ~195x |

\* extrapolated from the scaling of smaller sizes: the reference Stolt
loop dominates and is not practical to run at 16384².

The cold 4.6 s is a fresh-process first call: the kernel's planes are
gigabytes each and their pages must be faulted in and zeroed before the
first store. Warm ALOS runs are faster — the pool keeps those pages
mapped — but no warm figure is published here: the repeats were taken on
a heavily contended machine and ranged from 2.6 s to 12 s. Re-run the
example on an idle machine to obtain one.
