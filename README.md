<div align="center">

# SAR-DSL — An MLIR compiler for synthetic aperture radar imaging

[![CI](https://github.com/zeroherolin/sar-dsl/actions/workflows/ci.yml/badge.svg)](https://github.com/zeroherolin/sar-dsl/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](CMakeLists.txt)
[![LLVM/MLIR 22](https://img.shields.io/badge/LLVM%2FMLIR-22-orange.svg)](https://mlir.llvm.org/)
[![Backends](https://img.shields.io/badge/backends-cpu%20%7C%20hls-8a2be2.svg)](docs/backends.md)
[![Vitis HLS 2022.2](https://img.shields.io/badge/Vitis%20HLS-2022.2-brightgreen.svg)](docs/backends.md)
[![Platform](https://img.shields.io/badge/platform-Linux%20x86--64-lightgrey.svg)](#getting-started)

Write complete SAR imaging pipelines in Python with whole-array operations —
FFTs, phase operations, interpolation, and corner turns — and compile the same
program to native CPU code or synthesizable Vitis HLS C++.

<img src="examples/wka/assets/san_francisco_wka.png" width="72%"
     alt="San Francisco Bay, ALOS-1 raw echoes focused by the SAR-DSL omega-K kernel"/>

*San Francisco Bay: a 16384 × 16384 ALOS-1 scene focused by the SAR-DSL
omega-K kernel.*

</div>

## Highlights

- **One language, two backends.** The same statically specialized kernel
  compiles to native CPU code or synthesizable Vitis HLS C++; backend
  symmetry is enforced by tests rather than backend-specific DSL operators.
- **Signal-processing structure survives tracing.** FFTs, phase operations,
  interpolation, reductions, and layout changes remain explicit in MLIR long
  enough for backend-wide scheduling and memory decisions.
- **Extensible in Python.** `@sar.op` defines reusable operators by composing
  the same primitives available to kernels, with eager NumPy and compiled
  semantics.
- **Complete imaging chains.** Four examples exercise the language and both
  backends end to end:

| Chain | Imaging mode | Main operations |
|-------|--------------|-----------------|
| [omega-K](examples/wka/) | stripmap | matched filtering, Stolt interpolation |
| [Range-Doppler](examples/rda/) | stripmap | range compression, RCMC |
| [Chirp Scaling](examples/csa/) | stripmap | phase-only range migration correction |
| [Polar Format](examples/pfa/) | spotlight | polar resampling, SVA |

SAR-DSL is a research compiler, not an FPGA deployment stack. This repository
tests native execution, generated C-simulation, and Vitis HLS synthesis.
Vivado implementation, board drivers, and on-device benchmarking are
intentionally outside its scope.

## Quick example

```python
import sar

@sar.func
def range_compress(raw, replica):
    spectrum = sar.fft(raw, axis=1) * replica
    return sar.ifft(spectrum, axis=1)

# Eager specialization and native CPU execution.
image = range_compress(raw_np, replica_np)

# Ahead-of-time specialization and HLS C++ emission.
design = range_compress.specialize(
    sar.c64[512, 512], sar.c64[512, 512]
).compile("hls", options={"interface": "axi"})
print(design.cpp_path)
```

Kernels specialize by static shape and dtype. NumPy arrays may be captured as
constants or passed as arguments. `@sar.op` defines reusable operators with the
same eager NumPy and compiled semantics; see
[Defining operators](docs/defining-ops.md).

## Getting started

The tested host platform is Linux x86-64. Requirements are CMake 3.20+, Ninja,
a C++17 compiler, Python 3.10+, and NumPy. Matplotlib is needed only for
figures; Vitis HLS is optional.

```bash
git clone https://github.com/zeroherolin/sar-dsl.git
cd sar-dsl
git submodule update --init externals/llvm-project
python -m pip install numpy matplotlib pytest

make llvm      # build the pinned LLVM/MLIR toolchain
make build     # build the compiler, runtime, and Python build config
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
make test      # pytest and lit
```

Frontend tracing and diagnostics can be tested without building LLVM:

```bash
PYTHONPATH=python python -m pytest test/python -q
```

Generate the four synthetic example images and benchmark figures with:

```bash
make examples
PYTHONPATH=python:examples python benchmarks/run_figures.py
```

## Compiler architecture

<div align="center">
<img src="docs/assets/how_it_works.png"
     alt="Compilation flow from a Python SAR kernel through the shared SAR
dialect to native CPU code and Vitis HLS C++"/>
</div>

<!-- Diagram source: docs/assets/how_it_works.html. Regenerate the PNG with
     the command in that file's header after changing the architecture. -->

The frontend boundary is textual MLIR, keeping Python independent of LLVM
bindings while leaving verification to `sar-opt`. Both backends consume the
same core operations. Backend symmetry is regression-tested: the DSL does not
expose CPU-only or HLS-only language operators.

The CPU pipeline uses linalg fusion, bufferization, OpenMP, LLVM, and a small
runtime for FFT and interpolation. The HLS pipeline lowers complex values into
real planes, emits affine Stockham FFTs and bounded gathers, plans on-chip and
external buffers, and adds AXI, partition, pipeline, and storage directives.
The compiler can emit a package containing source, testbench, and Vitis Tcl
scripts without requiring Vitis in the user's environment.

Detailed design rationale and IR contracts are in
[Architecture](docs/architecture.md) and [Dialect reference](docs/dialect.md).

## Evaluation

Benchmark runners record raw samples and environment provenance; methodology
and complete reference tables are kept in [benchmarks/](benchmarks/).

### Image formation

The following images are generated by the checked-in CPU examples from the
same 512 × 512 synthetic three-target scene.

<div align="center">
<table>
<tr>
<td align="center" width="50%">
<img src="examples/wka/assets/wka_synthetic_512.png"
     alt="omega-K focusing result"/><br/>
<a href="examples/wka/"><b>omega-K (WKA)</b></a><br/>
<em>exact hyperbolic model, Stolt remapping</em>
</td>
<td align="center" width="50%">
<img src="examples/rda/assets/rda_synthetic_512.png"
     alt="Range-Doppler focusing result"/><br/>
<a href="examples/rda/"><b>Range-Doppler (RDA)</b></a><br/>
<em>range-dependent RCMC + azimuth filter</em>
</td>
</tr>
<tr>
<td align="center" width="50%">
<img src="examples/csa/assets/csa_synthetic_512.png"
     alt="Chirp Scaling focusing result"/><br/>
<a href="examples/csa/"><b>Chirp Scaling (CSA)</b></a><br/>
<em>interpolation-free, phase multiplies only</em>
</td>
<td align="center" width="50%">
<img src="examples/pfa/assets/pfa_synthetic_512.png"
     alt="Polar Format focusing result"/><br/>
<a href="examples/pfa/"><b>Polar Format (PFA) + SVA</b></a><br/>
<em>built from Python-defined operators</em>
</td>
</tr>
</table>
</div>

<div align="center">
<img src="benchmarks/assets/point_target_response.png" width="88%"
     alt="Point-target impulse response of the three algorithms">

*Range and azimuth impulse-response cuts for the three stripmap chains,
measured with 32× upsampling. The dashed −31.5 dB line is the ideal Hann
first-sidelobe reference, not a bound on the complete imaging chain.*
</div>

Peak location, IRW, PSLR, and ISLR thresholds are regression-tested against
the same NumPy references used for cross-backend accuracy checks.

### HLS synthesis

The reference HLS target is `xcvu13p-fhgb2104-2-i`, with a 4 ns clock,
512-bit AXI, and 80% device resource budgets. Vitis HLS 2022.2 is used for
validation but is optional for building and using the compiler.

Complete N=32 designs for all four algorithms pass local Vitis synthesis and
meet the 4 ns target. Their generated packages pass both plain-C++ and Vitis
C-simulation against NumPy golden outputs. Reports and methods are listed in
[benchmarks/](benchmarks/). A full 16384 × 16384 c64 omega-K design also
completes synthesis:

| Estimated clock | Latency at 4 ns target | BRAM18K | URAM | DSP | FF | LUT |
|----------------:|-----------------------:|--------:|-----:|----:|---:|----:|
| 5.698 ns | 47,013,232,685 cycles / 188.053 s | 130 | 552 | 289 | 110,478 | 348,722 |

This large design fits the configured resource budgets but misses the 4 ns
timing target. The result demonstrates synthesizability and exposes the current
research bottleneck; it is not a claim of timing closure or hand-tuned
throughput. The
[machine-readable report summary](benchmarks/results/hls_wka_c64_16384_vitis_2022_2.json)
records its constraints and provenance.

## Documentation

| Document | Contents |
|----------|----------|
| [Python API](docs/python-api.md) | Public language and compiler API |
| [MATLAB guide](docs/matlab-users.md) | Indexing, naming, and convention mapping |
| [Architecture](docs/architecture.md) | Compiler layers and design rationale |
| [Dialect reference](docs/dialect.md) | Operations, invariants, and passes |
| [Defining operators](docs/defining-ops.md) | `@sar.op` and specialization |
| [Backends](docs/backends.md) | CPU, HLS, configuration, and extension API |
| [Benchmarks](benchmarks/README.md) | Methodology and reference measurements |
| [Roadmap](docs/roadmap.md) | Verified scope and remaining work |
| [Contributing](CONTRIBUTING.md) | Build, style, and test conventions |

## Repository layout

```text
include/sar/, lib/   dialects, analyses, conversions, and pipelines
runtime/             CPU runtime with a stable C ABI
tools/               sar-opt, sar-translate, and sar-lsp-server
python/sar/          Python language, compiler driver, and backends
examples/            complete WKA, RDA, CSA, and PFA programs
benchmarks/          accuracy, quality, performance, and HLS report tools
test/                MLIR lit tests and Python tests
externals/           pinned llvm-project submodule
```

## Citation

No archival publication is associated with this repository. GitHub exposes the
software citation in [CITATION.cff](CITATION.cff); include the commit hash used
for an experiment so generated code and measurements remain reproducible.

## License

MIT — see [LICENSE](LICENSE). The compiler builds on
[LLVM/MLIR](https://mlir.llvm.org/). External SAR datasets are not
redistributed by this repository.
