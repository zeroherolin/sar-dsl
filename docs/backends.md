# Backends

SAR-DSL uses a plugin model: each backend is a small Python
package contributing an ordered set of compilation *stages*.

## Using backends

```python
compiled = kernel.compile(backend="cpu")             # default
design   = kernel.compile(backend="hls",
                          options={"on_chip_budget": 8 << 20})
print(sar.list_backends())                           # discovery
```

## CPU backend (`sar.backends.cpu`)

Compiles kernels for the **host** CPU -- whatever machine the driver runs
on. The lowering is architecture-neutral (LLVM dialect, portable runtime
library, generic memref ABI); clang targets the host triple and native
tuning flags are probed per architecture (`-march=native` /
`-mcpu=native`, with a generic fallback). x86-64 Linux is the tested
platform; other Linux hosts with an LLVM host target (e.g. AArch64) are
expected to work but are not covered by CI. Windows is unsupported.

Stages: `llvm` (sar-opt `--sar-to-llvm-pipeline`: runtime calls, linalg
elementwise fusion, OpenMP parallel loops) -> `ll` (mlir-translate) ->
`shared` (clang `-O3` + native tuning, linked against `libsar_runtime`
and LLVM's `libomp`). The launcher is a `sar.runtime.CompiledKernel`:
numpy arrays are marshalled as strided memref descriptors via ctypes,
results are allocated by the caller (destination-passing style).

Options: `opt_level` (default 3), `native_codegen` (default True).
`OMP_NUM_THREADS` controls loop parallelism at run time.

## HLS backend (`sar.backends.hls`)

Stages: `lower` (`sar-opt --sar-to-affine-pipeline`) -> `hls`
(`sar-opt --hls-pipeline` then `sar-translate --hls-emit-hlscpp`). Returns
an `HLSDesign` handle (`.cpp_path`, `.source()`); it is not executable.

The kernel is decomplexified (complex tensors become re/im float planes),
FFTs become Stockham loop nests and interpolation becomes windowed-sinc
gather loops. The resulting affine IR enters the HLS pipeline, which
builds the dataflow hierarchy, places buffers on or off chip and shapes
the interfaces. Generated top functions take each complex tensor as two
adjacent float arrays (re, im).

Every SAR operation has a lowering, so complete imaging chains (omega-K,
range-Doppler, chirp scaling, polar format) emit as single HLS designs;
the identical IR is validated numerically on the CPU through
`--sar-affine-to-llvm-pipeline`. FFTs of any size >= 2 lower here: powers
of two run Stockham directly, other sizes go through Bluestein's chirp-z
reduction (two padded Stockham transforms with the chirp and kernel
spectrum folded at compile time).

`HLSDesign.write_testbench(inputs, expected, output_dir)` emits a
self-contained C-simulation package: the design, a testbench comparing
every output plane against golden data (from the NumPy reference or the
cpu backend), the data files, a `vitis_hls` csim script (written against
Vitis HLS 2022.2), and Vitis header stand-ins so the package also csims
with any plain C++ compiler
(`c++ -O2 -I stubs <top>.cpp <top>_tb.cpp -o csim -pthread`). The
testbench runs the design on a dedicated 1 GiB stack: in C simulation the
on-chip buffers are stack arrays, which overflow default limits for
larger scenes. The example runners generate one package per algorithm
under `hls_project/<name>/`
(`python examples/wka/run_point_target_hls.py`). All four imaging chains
csim bit-exactly against the NumPy reference.

Options: `top_func`, `axi_interface` (False), `on_chip_budget` (4 MiB),
`axi_bus_bits` (512), `loop_tile_size` (8).

### Where the data lives

Placement is the compiler's decision, not the user's. The backend
measures the resident working set in the lowered kernel -- its arguments
and results plus the full-size intermediates that survive buffer sharing
-- and keeps everything on chip while that fits `on_chip_budget`. Past
the budget the full-size planes are streamed and only the constant
tables (twiddles, interpolation weights) and the one-line transform
scratch stay resident.

The set is measured rather than predicted because it follows the
algorithm, not the signature: the same four-stage chain holds six live
planes under range-Doppler and ten under chirp-scaling.

What the user picks is the interface, and that choice decides how
streamed buffers reach the top function:

| | top signature | use |
|---|---|---|
| `axi_interface=False` | the kernel's own inputs and results | csim packages: the testbench can drive every port |
| `axi_interface=True` | one AXI master port per streamed buffer | designs handed to Vitis |

An imaging chain is a sequence of whole-raster passes, so a plane dies
as soon as the next pass has read it and the ones whose lifetimes do not
overlap share an allocation. That is what keeps the streamed set a
property of the algorithm rather than of the chain's length: adding
passes to a kernel does not add ports. At `16384 x 16384` the omega-K
chain streams eight planes over two AXI bundles, roughly 13 GiB of DRAM,
and holds only the tables and the line scratch on chip.

Testbench generation rejects `axi_interface=True`: the promoted
intermediate ports have no golden data to drive them. Emit the csim
package without it, and compile with it for the design you synthesize.

## Precision contract

Declared dtypes fix the *data-path* precision on every backend: a
`c64` tensor is stored, moved and combined as f32 planes on cpu and
HLS alike. Internal precision of the leaf transforms is a backend
choice:

- interpolation positions/weights are computed in f64 on both
  backends (index arithmetic must not lose fractional bins);
- the cpu FFT runs double-precision butterflies regardless of dtype
  (the runtime library gets f64 for free), while the HLS FFT computes
  in the declared precision (f32 butterflies cost a fraction of the
  DSPs of f64 ones).

Consequently `c128` kernels agree across backends to f64 rounding
(~1e-15, butterfly-ordering differences only) and `c64` kernels to
f32 rounding (~1e-7 relative). Both satisfy the declared precision;
use `c128` where the FFT chain itself must be reproducible bit-for-bit
across backends.

Host data participates in that contract. Promotion follows numpy, so a
float64 array -- numpy's default -- meeting an f32 tensor gives f64, and
every operator and buffer downstream widens with it. The trace reports
this as `sar.PrecisionWarning`, naming the host array rather than the
tensor it met, because the host is the side that can cheaply choose
otherwise:

```python
warnings.simplefilter("error", sar.PrecisionWarning)  # treat as a bug
```

A mixed-precision chain also costs interfaces: buffers of different
element types cannot share an allocation, so the streamed set is the sum
of the two rather than the larger of them.

## Adding a backend

1. Create a backend package -- `python/sar/backends/<name>/compiler.py`
   for a built-in one, any directory for an out-of-tree one:

```python
from sar.backends.base import BaseBackend, KernelMetadata

class Backend(BaseBackend):
    name = "mytarget"

    @classmethod
    def is_available(cls):
        ...  # probe for tools/devices

    def add_stages(self, stages, metadata: KernelMetadata):
        stages["lowered"] = self._stage_lower   # artifact -> artifact
        stages["binary"] = self._stage_codegen

    def make_launcher(self, artifact, metadata):
        ...  # callable for execution targets, handle for emission targets
```

2. Add `__init__.py` re-exporting `Backend`.

That's all: subpackages of `sar.backends` are discovered automatically,
and `SAR_DSL_BACKEND_PATH` (os.pathsep-separated directories) picks up
out-of-tree ones.

Useful building blocks: `sar.compiler.toolchain.find_tool` / `run_tool`
(tool discovery + subprocess execution with good errors), the per-kernel
artifact cache handed to every stage, and `sar.runtime` for memref
marshalling if the target executes on the host.

## Toolchain discovery

Tools are located in this order:

1. `SAR_DSL_TOOL_<NAME>` environment variables (e.g. `SAR_DSL_TOOL_SAR_OPT`);
2. paths from the CMake-generated `sar/_build_config.py` (build tree);
3. `SAR_DSL_TOOL_PATH` directories, then `PATH`.

## Environment variables

| Variable | Effect |
|----------|--------|
| `SAR_DSL_TOOL_<NAME>` | Absolute path to one tool, overriding discovery |
| `SAR_DSL_TOOL_DIR` | Build-tree `bin` directory (written by CMake into `sar/_build_config.py`; set it to run against another build) |
| `SAR_DSL_TOOL_PATH` | Extra directories to search for tools |
| `SAR_DSL_RUNTIME_LIB` | Path to `libsar_runtime.so` |
| `SAR_DSL_BACKEND_PATH` | Extra directories to search for backend packages |
| `SAR_DSL_CACHE_DIR` | Artifact cache root (default `~/.cache/sar-dsl`) |
| `SAR_DSL_DISABLE_CACHE` | `1` skips cache lookups (artifacts are still written) |
| `SAR_DSL_CACHE_MAX_SIZE` | Cache eviction threshold in bytes (default 2 GiB) |
| `SAR_DSL_OMP_LIB` | Path to `libomp.so` for the cpu backend |
| `SAR_RT_NUM_THREADS` | Runtime worker threads; falls back to `OMP_NUM_THREADS`, then the hardware concurrency |
