<div align="center">

# SAR-DSL — An MLIR compiler for synthetic aperture radar imaging

[![CI](https://github.com/zeroherolin/sar-dsl/actions/workflows/ci.yml/badge.svg)](https://github.com/zeroherolin/sar-dsl/actions/workflows/ci.yml) [![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE) [![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml) [![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](CMakeLists.txt) [![LLVM/MLIR 22](https://img.shields.io/badge/LLVM%2FMLIR-22-orange.svg)](https://mlir.llvm.org/) [![Backends](https://img.shields.io/badge/backends-cpu%20%7C%20hls-8a2be2.svg)](docs/backends.md) [![Vitis HLS 2022.2](https://img.shields.io/badge/Vitis%20HLS-2022.2-brightgreen.svg)](docs/backends.md) [![Platform](https://img.shields.io/badge/platform-Linux%20x86--64-lightgrey.svg)](#build-and-test)

Express a complete SAR processing chain in Python and specialize the same program for native CPU execution or synthesizable Vitis HLS C++.

<img src="examples/wka/assets/san_francisco_wka.png" width="86%" alt="ALOS-1 San Francisco Bay scene focused by the SAR-DSL omega-K example"/>

_ALOS-1 San Francisco Bay scene focused by the omega-K chain._

</div>

## Abstract

SAR-DSL traces whole-array signal-processing programs into a compact `sar` MLIR dialect. FFTs, interpolation, reductions, phase transforms, layout changes, gathers, and compiled loops remain explicit long enough for fusion, memory planning, and target-specific scheduling. The language is backend-neutral: one statically shaped program feeds both the CPU and HLS paths.

| Layer | Role |
| --- | --- |
| Python DSL | NumPy-style tensors, reusable `@sar.op` compositions, static shape and dtype specialization |
| Shared IR | Domain verification, FFT/interpolation semantics, precision rules, layout and loop contracts |
| CPU backend | Linalg, bufferization, OpenMP, LLVM, and a reusable FFT/interpolation runtime |
| HLS backend | Split-complex affine loops, on-chip/external placement, banking, AXI interfaces, and Vitis HLS source generation |

## Imaging chains

The repository contains four complete chains, each with a SAR-DSL kernel and an independent NumPy reference.

| Algorithm | Collection | Main operations | Example |
| --- | --- | --- | --- |
| Omega-K (WKA) | stripmap | range/azimuth FFTs, bulk compression, Stolt interpolation | [WKA](examples/wka/) |
| Range-Doppler (RDA) | stripmap | range compression, RCMC, azimuth compression | [RDA](examples/rda/) |
| Chirp Scaling (CSA) | stripmap | FFTs and phase-only range migration correction | [CSA](examples/csa/) |
| Polar Format (PFA) | spotlight | polar regridding and spatially variant apodization | [PFA](examples/pfa/) |

<div align="center">
<table width="86%">
<tr>
<td align="center" width="50%"><img src="examples/wka/assets/wka_synthetic_512.png" width="100%" alt="Synthetic point-target image focused with omega-K"/><br/><b>Omega-K</b></td>
<td align="center" width="50%"><img src="examples/rda/assets/rda_synthetic_512.png" width="100%" alt="Synthetic point-target image focused with Range-Doppler"/><br/><b>Range-Doppler</b></td>
</tr>
<tr>
<td align="center" width="50%"><img src="examples/csa/assets/csa_synthetic_512.png" width="100%" alt="Synthetic point-target image focused with Chirp Scaling"/><br/><b>Chirp Scaling</b></td>
<td align="center" width="50%"><img src="examples/pfa/assets/pfa_synthetic_512.png" width="100%" alt="Synthetic point-target image focused with Polar Format and SVA"/><br/><b>Polar Format + SVA</b></td>
</tr>
</table>

_512 × 512 synthetic point-target scenes produced by the checked-in CPU examples._
</div>

The stripmap kernels also process the same 16384 × 16384 ALOS-1 acquisition used by the independent hand-written WKA reference.

<div align="center">
<table width="100%">
<tr>
<td align="center" width="33%"><img src="examples/wka/assets/san_francisco_wka.png" width="100%" alt="ALOS-1 San Francisco Bay scene focused with omega-K"/><br/><b>Omega-K</b></td>
<td align="center" width="33%"><img src="examples/rda/assets/san_francisco_rda.png" width="100%" alt="ALOS-1 San Francisco Bay scene focused with Range-Doppler"/><br/><b>Range-Doppler</b></td>
<td align="center" width="33%"><img src="examples/csa/assets/san_francisco_csa.png" width="100%" alt="ALOS-1 San Francisco Bay scene focused with Chirp Scaling"/><br/><b>Chirp Scaling</b></td>
</tr>
</table>

_ALOS-1 San Francisco Bay focused images from the ASF DAAC granule described below._
</div>

## Compiler architecture

<div align="center">
<img src="docs/assets/how_it_works.png" width="86%" alt="Compilation flow from Python through the shared SAR MLIR dialect to CPU and Vitis HLS backends"/>
</div>

The frontend serializes textual MLIR and does not require compiled MLIR Python bindings. `sar-opt` verifies and transforms the module; `sar-translate` emits target source. The CPU and HLS pipelines share the frontend and domain dialect, then diverge at target-specific lowering, memory planning, and scheduling.

See [Architecture](docs/architecture.md) for design rationale and [Dialect reference](docs/dialect.md) for operation and pass contracts.

## Image quality

The examples are checked against NumPy references and point-target metrics. The stripmap chains reproduce the expected Hann response; PFA additionally demonstrates spatially variant apodization.

<div align="center">
<table width="100%">
<tr>
<td width="58%"><img src="benchmarks/assets/cpu_point_target_response.png" width="100%" alt="Point-target impulse-response cuts for the stripmap chains"/></td>
<td width="42%"><img src="benchmarks/assets/cpu_pfa_sva_response.png" width="100%" alt="Polar Format impulse response before and after spatially variant apodization"/></td>
</tr>
</table>

_Left: range and azimuth impulse-response cuts. Right: PFA range response before and after SVA._
</div>

| Chain | IRW (range / azimuth) | PSLR (range / azimuth) | ISLR (range / azimuth) |
| --- | ---: | ---: | ---: |
| Omega-K | 2.06 / 2.06 | −38.4 / −38.4 dB | −29.0 / −31.8 dB |
| Range-Doppler | 2.06 / 2.07 | −38.8 / −38.4 dB | −26.5 / −31.8 dB |
| Chirp Scaling | 2.06 / 2.07 | −38.5 / −38.4 dB | −29.0 / −31.8 dB |

The complete quality methodology, tolerances, PFA metrics, and precision comparison are maintained in the [benchmark report](benchmarks/README.md).

## CPU performance

The CPU backend emits native code with fused element-wise and layout loops. The following points are warm end-to-end runs at `c128` precision; compile time is excluded.

<div align="center">
<table width="100%">
<tr>
<td width="50%"><img src="benchmarks/assets/cpu_throughput.png" width="100%" alt="CPU throughput versus scene size"/></td>
<td width="50%"><img src="benchmarks/assets/cpu_speedup.png" width="100%" alt="CPU speedup over the NumPy reference versus scene size"/></td>
</tr>
</table>

_Throughput and speedup are redrawn from the versioned CPU measurement record._
</div>

| Chain | Input → output | SAR-DSL CPU | NumPy reference | Speedup |
| --- | ---: | ---: | ---: | ---: |
| Omega-K | 16384² → 16384² | 3.684 s | 186.53 s | 50.6× |
| Range-Doppler | 16384² → 16384² | 3.911 s | 105.00 s | 26.8× |
| Chirp Scaling | 16384² → 16384² | 2.525 s | 73.43 s | 29.1× |
| Polar Format | 8192² → 16384² | 3.761 s | 155.43 s | 41.3× |

The portable FFT/interpolation pool remains capped at 32 workers by default. On the reference dual-socket host, topology-aware placement reaches 31.13 ms for a 2048² RDA run at 120 physical workers, while one-socket SMT oversubscription reaches 172.13 ms. The isolated MKL DFTI leaf is 3.1–4.9× faster than the portable runtime; the detailed comparison is in [`cpu_numa_mkl_2026_08_27.json`](benchmarks/results/cpu_numa_mkl_2026_08_27.json).

## HLS synthesis and resource use

The HLS backend emits a self-contained Vitis HLS package. Complex tensors become split real/imaginary planes; FFTs and gathers become affine loops; full-size intermediates may use compiler-managed external scratch arenas while compact tables and line caches remain on chip.

<div align="center">
<img src="benchmarks/assets/hls_resource_utilization.png" width="86%" alt="HLS production design resource utilization against device budgets"/>

_Synthesized production designs normalized by the configured resource budgets._

<img src="benchmarks/assets/hls_budget_sweep.png" width="86%" alt="HLS resource and interface behavior across on-chip memory budgets"/>

_Compiler-selected strategies across an on-chip budget sweep._
</div>

Production-scale `c64` AXI designs use the shipped `xcvu13p-fhgb2104-2-i`, a 4 ns target clock, 12.5% uncertainty, and 80% resource budgets. The three stripmap chains use ALOS-1 acquisition geometry; PFA uses its spotlight geometry.

| Design | Input → output | Estimated clock | Latency time | BRAM18K | URAM | DSP | FF | LUT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Omega-K, generated | 16384² → 16384² | 3.500 ns | 7.305 s | 1,410 | 960 | 2,668 | 767,232 | 860,085 |
| Omega-K, hand-written | 16384² → 16384² | 3.500 ns | 5.402 s | 672 | 848 | 2,860 | 769,377 | 658,256 |
| Range-Doppler, generated | 16384² → 16384² | 3.500 ns | 5.790 s | 1,736 | 852 | 1,453 | 686,807 | 697,239 |
| Chirp Scaling, generated | 16384² → 16384² | 3.627 ns | 4.467 s | 1,736 | 840 | 2,024 | 593,904 | 638,714 |
| Polar Format, generated | 8192² → 16384² | 3.500 ns | 23.395 s | 65 | 512 | 854 | 292,404 | 420,121 |

All production designs fit the configured resource budgets and the 4 ns board period. CSA is recorded with a 0.127 ns timing shortfall against the 3.5 ns post-uncertainty scheduling goal; timing is reported separately from hard resource-cap violations.

The measured FFT DSE identifies four parallel rows, stage grouping 2, and transfer width 4 as the balanced point for the bounded `c64` probe. The gather DSE shows that a narrow completely banked band can reduce compute II from 4 to 1, but its small `ap_memory` kernels do not model production AXI behavior; production strategy selection remains displacement- and resource-aware.

The full synthesis tables, generated-package contracts, warning counts, DSE records, and provenance are in [benchmarks/](benchmarks/README.md).

## Quick example

```python
import sar

@sar.func
def range_compress(raw, replica):
    spectrum = sar.fft(raw, axis=1) * replica
    return sar.ifft(spectrum, axis=1)

# Call-site arrays specialize and execute the kernel on the CPU.
image = range_compress(raw_np, replica_np)

# The same graph can be specialized and emitted for HLS.
design = range_compress.specialize(
    sar.c64[512, 512], sar.c64[512, 512]
).compile("hls", options={"interface": "axi"})
design.write_synthesis_script("hls_project/range_compress")
```

Kernels specialize by static shape and dtype. NumPy arrays may be runtime arguments or compile-time constants. See the [Python API](docs/python-api.md) and [operator guide](docs/defining-ops.md) for the language surface.

## Build and test

The supported development platform is Linux x86-64. Required tools are CMake 3.20+, Ninja, a C++17 compiler, Python 3.10+, and NumPy. Vitis HLS is optional unless generated hardware packages are being simulated or synthesized.

```bash
git clone https://github.com/zeroherolin/sar-dsl.git
cd sar-dsl
git submodule update --init externals/llvm-project
python -m pip install numpy matplotlib pytest

make llvm
make build
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
make test
```

Frontend-only tests do not require LLVM:

```bash
PYTHONPATH=python python -m pytest test/python -q \
  -m "not requires_cpu and not requires_hls and not requires_vitis"
```

Use `python -m sar doctor` to inspect tool discovery, backend availability, runtime configuration, and the artifact cache.

## ALOS-1 and examples

Each chain has a synthetic point-target example that runs without any external data:

```bash
PYTHONPATH=python python examples/wka/run_point_target_cpu.py --n 512
PYTHONPATH=python python examples/rda/run_point_target_cpu.py --n 512
PYTHONPATH=python python examples/csa/run_point_target_cpu.py --n 512
PYTHONPATH=python python examples/pfa/run_point_target_cpu.py --n 512
```

The stripmap chains additionally include ALOS-1 CPU and HLS runners; PFA is a spotlight example and uses synthetic collection geometry. The real-data path uses ASF DAAC granule [`ALPSRP275140740-L1.0`](https://datapool.asf.alaska.edu/L1.0/A3/ALPSRP275140740-L1.0.zip), an ALOS PALSAR FBS HH CEOS L1.0 product covering San Francisco Bay. Open the [ASF Data Search record](https://search.asf.alaska.edu/#/?search=ALPSRP275140740-L1.0) for the catalog entry; an Earthdata login may be required. Original data is © JAXA/METI; the archive stays outside the repository under the source provider's data terms.

Download and unpack the archive so `examples/data/ALPSRP275140740-L1.0/IMG-HH-ALPSRP275140740-H1.0__A` is available, extract the 16384 × 16384 complex64 raster once, then run a stripmap chain on it (the full-size scene needs tens of GiB of RAM):

```bash
unzip ~/Downloads/ALPSRP275140740-L1.0.zip -d examples/data
PYTHONPATH=python python examples/data/extract_alos.py
PYTHONPATH=python python examples/wka/run_alos_cpu.py
```

See [examples/](examples/) for data preparation, HLS package generation, and algorithm-specific notes. The independent [hand-written WKA implementation](examples/wka/handwritten_hls/) is a comparison design, not a compiler dependency.

## Documentation

| Document | Contents |
| --- | --- |
| [Python API](docs/python-api.md) | Public language and compiler API |
| [Defining operators](docs/defining-ops.md) | `@sar.op`, eager execution, and specialization |
| [MATLAB guide](docs/matlab-users.md) | Indexing and convention mapping |
| [Architecture](docs/architecture.md) | Compiler layers and design rationale |
| [Dialect reference](docs/dialect.md) | Operations, invariants, and pass pipelines |
| [Backends](docs/backends.md) | CPU/HLS use, configuration, artifacts, and extension API |
| [Benchmarks](benchmarks/README.md) | Methodology and complete reference measurements |
| [Project scope](docs/scope.md) | Supported scope and known limits |
| [Contributing](CONTRIBUTING.md) | Development workflow and repository conventions |

## Repository layout

```text
include/sar/, lib/   dialects, analyses, conversions, and pipelines
runtime/             native FFT/interpolation runtime and allocation hooks
tools/               sar-opt, sar-translate, and sar-lsp-server
python/sar/          Python language, compiler driver, runtime, and backends
examples/            complete WKA, RDA, CSA, and PFA programs
benchmarks/          measurement runners, reports, figures, and result data
test/                MLIR lit tests and Python integration tests
docs/                architecture, backend, dialect, and API references
```

## Citation

No archival publication is associated with this repository yet. GitHub exposes the software citation in [CITATION.cff](CITATION.cff); include the commit hash used for an experiment so generated code and measurements remain reproducible.

## License

MIT — see [LICENSE](LICENSE). The compiler builds on [LLVM/MLIR](https://mlir.llvm.org/). External SAR data is obtained separately from its source provider; the ALOS-1 download and preparation path is documented above.
