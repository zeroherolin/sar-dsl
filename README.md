<div align="center">

# SAR-DSL

**An MLIR-based domain-specific compiler for synthetic aperture radar imaging**

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](CMakeLists.txt)
[![MLIR](https://img.shields.io/badge/built%20on-LLVM%2FMLIR-orange.svg)](https://mlir.llvm.org/)

Write SAR imaging pipelines in Python at the granularity of whole stages —
FFTs, phase multiplies, Stolt interpolation, corner turns — and compile
them to native CPU code or synthesizable FPGA designs.

<img src="examples/wka/assets/san_francisco_wka.png" width="72%"
     alt="San Francisco Bay, ALOS-1 raw echoes focused by the SAR-DSL omega-K kernel"/>

*San Francisco Bay: 16384 x 16384 raw ALOS-1 echoes focused by a single
compiled omega-K kernel in 3.6 seconds.*

</div>

---

## Highlights

- **Two complete imaging algorithms.** omega-K (WKA) and Range-Doppler
  (RDA) chains, each a single compiled kernel, validated numerically
  against NumPy references and demonstrated on real satellite data
  ([examples/wka](examples/wka/), [examples/rda](examples/rda/)).
- **Multi-backend by construction.** A `cpu` backend executes kernels
  natively (linalg fusion, OpenMP, `libsar_runtime` FFT); a `scalehls`
  backend emits Vitis HLS C++ through
  [ScaleHLS-HIDA](https://github.com/UIUC-ChenLab/ScaleHLS-HIDA). Every SAR
  operation has an HLS lowering — FFTs as bit-reversal-free Stockham affine
  loops, interpolation as straight-line masked gathers — so **complete
  imaging chains emit as single FPGA designs**. New targets plug in as
  [backend packages](docs/backends.md) without core changes.
- **A real dialect, not a wrapper.** Domain invariants (power-of-two FFT
  sizes, frequency-axis consistency, precision rules) are enforced by MLIR
  verifiers; bit-exact canonicalization folds; orthogonal primitives
  (`sar.interp1d` underlies both Stolt remapping and RCMC).
- **Tested like a compiler.** FileCheck suites for every pass, per-op
  numerics against NumPy, end-to-end algorithm equivalence, memory-leak
  regression, and a differential fuzzer comparing random DSL programs
  against a NumPy oracle.

## Quick example

```python
import sar

N = 512

@sar.jit
def range_compress(raw: sar.c64[N, N], ref: sar.c64[N, N]) -> sar.c64[N, N]:
    spectrum = sar.fftshift(sar.fft(raw, dim=1), dim=1)
    return sar.ifft(sar.ifftshift(spectrum * ref, dim=1), dim=1)

image = range_compress(raw_np, ref_np)                # JIT on CPU
design = range_compress.compile(backend="scalehls")   # or emit HLS C++
print(design.cpp_path)
```

## How it works

Calling a kernel traces the Python function into a small MLIR dialect of
whole-array SAR operations; the selected backend then runs a pipeline of
compilation *stages* over it (cached on disk, keyed by content):

```mermaid
flowchart TB
    K["@sar.jit kernel<br/><sub>Python tracing, shape/dtype checks</sub>"]
    IR["sar dialect module<br/><sub>textual MLIR, verified by sar-opt</sub>"]
    K --> IR

    IR -- "sar-to-llvm-pipeline" --> CPU1
    IR -- "sar-to-linalg-pipeline<br/><sub>float element-wise</sub>" --> H1
    IR -- "sar-to-affine-pipeline<br/><sub>complex / FFT</sub>" --> H2

    subgraph CPU["cpu backend · native execution"]
        direction TB
        CPU1["linalg fusion · OpenMP parallel loops<br/>FFT and interpolation → libsar_runtime"]
        CPU2["mlir-translate → clang -O3"]
        CPU3["kernel.so<br/><sub>ctypes launcher, memref ABI</sub>"]
        CPU1 --> CPU2 --> CPU3
    end

    subgraph HLS["scalehls backend · FPGA emission"]
        direction TB
        H1["HIDA PyTorch flow<br/><sub>dataflow, tiling</sub>"]
        H2["decomplexify → Stockham FFT affine loops<br/>→ HIDA C++ flow"]
        H3["kernel.hls.cpp<br/><sub>Vitis HLS C++ with directives</sub>"]
        H1 --> H3
        H2 --> H3
    end
```

The design decisions behind this shape — and their trade-offs — are laid
out in [docs/architecture.md](docs/architecture.md). The short version:

| Decision | Why |
|----------|-----|
| Ops on builtin tensors (like TOSA/StableHLO) | no custom type conversions; all upstream MLIR machinery just works |
| Textual IR as the frontend boundary | pure-Python frontend, zero LLVM API coupling; `sar-opt` verifiers remain authoritative |
| Runtime library for CPU FFT/interpolation | the vendor-library pattern (cf. cuFFT): pipeline structure is the compiler's job, leaf transforms are not |
| Split-complex + Stockham FFT for HLS | no bit-reversal means fully affine accesses — synthesizable, and validated against NumPy on the CPU |
| Interpolation as straight-line masked gathers | clamped indices + select-masked taps instead of control flow, keeping HLS loop pipelining applicable |

## Performance

Full omega-K imaging chain, 240-core x86-64 server
([benchmarks/](benchmarks/)):

| Scene | SAR-DSL (cpu) | NumPy reference | Speedup |
|-------|--------------:|----------------:|--------:|
| 1024 x 1024 synthetic | 0.21 s | 0.46 s | 2.2x |
| 4096 x 4096 synthetic | 0.78 s | 6.25 s | 8.0x |
| 16384 x 16384 ALOS-1 | 3.6 s | ~15 min* | ~250x |

<sup>* extrapolated; the reference Stolt loop is impractical at this
size.</sup>

## Getting started

Requirements: cmake >= 3.20, ninja, a C++17 host compiler, Python >= 3.10
with numpy (matplotlib for the examples).

```bash
git clone <repo> && cd sar-dsl
git submodule update --init externals/llvm-project

make llvm        # 1. in-tree LLVM/MLIR/Clang toolchain (one-time, long)
make build       # 2. sar-opt, libsar_runtime, tests
make scalehls    # 3. optional: ScaleHLS-HIDA toolchain for the HLS backend

export PYTHONPATH=$PWD/python:$PYTHONPATH
make test        # lit + pytest, everything should pass
make examples    # focus a 512x512 synthetic scene, writes a PNG
```

Step 3 fetches ScaleHLS' pinned LLVM via the polygeist submodule and
applies a small upstream CMake fix
([scripts/patches/](scripts/patches/scalehls-hida-fixes.patch)).

To reproduce the San Francisco image, place the ALOS-1 CEOS product under
`examples/wka/data/` and run:

```bash
cd examples/wka
python data/extract_alos.py     # CEOS L1.0 -> alos_raw_16384x16384.bin
python run_alos.py
```

## Documentation

| Document | Contents |
|----------|----------|
| [docs/architecture.md](docs/architecture.md) | Layering, design decisions and their rationale |
| [docs/dialect.md](docs/dialect.md) | The `sar` dialect reference: ops, passes, pipelines |
| [docs/backends.md](docs/backends.md) | Backend guide: cpu, scalehls, adding your own |
| [examples/wka/README.md](examples/wka/README.md) | omega-K walkthrough (synthetic + real data) |
| [examples/rda/README.md](examples/rda/README.md) | Range-Doppler walkthrough |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Development setup and conventions |

## Project layout

```
include/sar/, lib/       C++ core: dialect (IR + Transforms), conversions, pipelines
runtime/                 libsar_runtime (FFT, sinc interpolation; C ABI)
tools/sar-opt/           optimizer / pipeline driver
python/sar/              Python package: language, ir, compiler, runtime, backends
third_party/             backend plugins (cpu, scalehls)
examples/                omega-K and Range-Doppler algorithms
benchmarks/              performance suite and reference numbers
test/                    lit suites (MLIR) and pytest suites (Python, e2e, fuzz)
externals/               submodules: llvm-project, ScaleHLS-HIDA
scripts/                 toolchain build scripts and patches
```

## Status and roadmap

SAR-DSL is a research compiler. Current limits, by design or pending work:

- FFT sizes must be powers of two; all shapes are static (one compile per
  geometry).
- HLS designs are emitted and validated numerically (the same IR is
  compiled and checked against NumPy on the CPU), but on-device
  place-and-route/timing closure is left to Vitis and scene sizes are
  bounded by on-chip memory.
- The `cpu` backend targets the *host* machine (JIT); it is tested on
  x86-64 Linux. Other Linux architectures with an LLVM host target should
  work but are unverified; Windows is unsupported.
- Prebuilt wheels are not published; the toolchain builds from source.

Contributions are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT — see [LICENSE](LICENSE). Built on
[LLVM/MLIR](https://mlir.llvm.org/) and
[ScaleHLS-HIDA](https://github.com/UIUC-ChenLab/ScaleHLS-HIDA); sample
imagery derives from JAXA ALOS-1 PALSAR data.
