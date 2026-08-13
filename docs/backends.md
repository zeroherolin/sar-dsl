# Backends

SAR-DSL uses a FlagTree-style plugin model: each backend is a small Python
package contributing an ordered set of compilation *stages*.

## Using backends

```python
compiled = kernel.compile(backend="cpu")             # default
design   = kernel.compile(backend="scalehls",
                          options={"loop_tile_size": 16})
print(sar.list_backends())                           # discovery
```

## CPU backend (`third_party/cpu`)

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

## ScaleHLS backend (`third_party/scalehls`)

Stages: `check` (subset validation + flow selection) -> `lower` -> `hls`
(scalehls-opt HIDA + scalehls-translate). Returns an `HLSDesign` handle
(`.cpp_path`, `.source()`, `.flow`); it is not executable.

Two lowering flows, selected automatically:

- **linalg** (float-only element-wise kernels): `--sar-to-linalg-pipeline`
  into `-hida-pytorch-pipeline` (dataflow decomposition, tiling).
- **affine** (complex arithmetic, FFTs, interpolation):
  `--sar-to-affine-pipeline` (decomplexify + Stockham FFT loop nests +
  windowed-sinc gathers) into `-hida-cpp-pipeline`. Generated top
  functions take each complex tensor as two adjacent float arrays
  (re, im).

Every SAR operation has a lowering in the affine flow, so complete
imaging chains (omega-K, range-Doppler, chirp scaling) emit as single
HLS designs; the identical IR is validated numerically on the CPU
through `--sar-affine-to-llvm-pipeline`.

Options: `flow` ("auto"), `top_func`, `loop_tile_size` (8),
`loop_unroll_factor` (4).

## Adding a backend

1. Create `third_party/<name>/backend/compiler.py`:

```python
from sar.backends.base import BaseBackend, KernelMetadata

class Backend(BaseBackend):
    name = "mytarget"

    @classmethod
    def is_available(cls):
        ...  # probe for tools/devices

    def add_stages(self, stages, metadata: KernelMetadata):
        stages["lowered"] = self._stage_lower     # (artifact, metadata, cache) -> artifact
        stages["binary"] = self._stage_codegen

    def make_launcher(self, artifact, metadata):
        ...  # callable for execution targets, handle for emission targets
```

2. Add `__init__.py` re-exporting `Backend`.

That's all: source-tree checkouts discover `third_party/*/backend`
automatically; `setup.py` copies the package into `sar/backends/<name>` for
installation; `SAR_DSL_BACKEND_PATH` supports out-of-tree development.

Useful building blocks: `sar.compiler.toolchain.find_tool` / `run_tool`
(tool discovery + subprocess execution with good errors), the per-kernel
artifact cache handed to every stage, and `sar.runtime` for memref
marshalling if the target executes on the host.

## Toolchain discovery

Tools are located in this order:

1. `SAR_DSL_TOOL_<NAME>` environment variables (e.g. `SAR_DSL_TOOL_SAR_OPT`);
2. paths from the CMake-generated `sar/_build_config.py` (build tree);
3. `SAR_DSL_TOOL_PATH` directories, then `PATH`.

`SAR_DSL_RUNTIME_LIB` overrides the runtime library location;
`SAR_DSL_CACHE_DIR` / `SAR_DSL_DISABLE_CACHE` control the artifact cache.
