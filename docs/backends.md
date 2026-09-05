# Backends

SAR-DSL provides a native CPU backend and a Vitis HLS source-emission backend. Both consume the same traced `sar` module and support the complete public language.

## Common interface

```python
compiled = kernel.compile("cpu")
design = kernel.compile("hls", options={"interface": "axi"})

print(sar.list_backends())
print(sar.get_backend("cpu"))
```

Compilation is cached by kernel IR, backend, resolved options, Python driver content, toolchain binaries, runtime library, and host code-generation identity. Repeated compilation with the same inputs reuses the cached artifacts.

Use `Kernel.dump_pipeline(directory, backend=..., options=...)` to export intermediate IR, kernel facts, resolved options, and a stage manifest. `python -m sar doctor` reports tool discovery, backend availability, runtime configuration, and cache location.

## CPU backend

The CPU backend lowers a kernel to LLVM IR, links it with `libsar_runtime`, and returns a callable `sar.runtime.CompiledKernel`. Input arrays must match the specialized shape and dtype. Non-contiguous inputs are copied to contiguous storage before execution; results are returned as new NumPy arrays.

```python
compiled = kernel.compile("cpu", options={
    "opt_level": 3,
    "native_codegen": True,
})
result = compiled(input_array)
```

| Option | Default | Meaning |
| --- | --- | --- |
| `opt_level` | `3` | Clang optimization level, from 0 to 3 |
| `native_codegen` | `true` | Use the supported native CPU tuning flag when available |

Generated array loops use OpenMP. FFT and interpolation calls use a separate reusable runtime pool. The mechanisms execute successive stages rather than nested teams.

- `OMP_NUM_THREADS` controls generated OpenMP loops.
- `SAR_RT_NUM_THREADS` controls FFT/interpolation runtime participants. When unset, it falls back to `OMP_NUM_THREADS`, then to `min(load-time process affinity, 32)`.
- `SAR_RT_FFT_PLAN_CACHE_MAX` bounds cached FFT lengths; `0` disables plan caching. `sar.runtime.clear_fft_plan_cache()` and `fft_plan_cache_size()` manage the in-process plan cache.
- `SAR_RT_POOL_MAX_BYTES` bounds retained intermediate planes; `0` disables plane pooling.

On NUMA hosts, bind memory consistently with the selected CPUs. Measurements on the reference dual-socket system are summarized in the [benchmark report](../benchmarks/README.md); portable defaults remain conservative and topology-independent.

## HLS backend

The HLS backend returns an `HLSDesign` handle. It does not require Vitis to emit source.

```python
design = kernel.compile("hls", options={"interface": "axi"})

print(design.source())
print(design.header_source())
print(design.interface_schema())

design.write_testbench(inputs, expected, "hls_project/kernel")
design.write_synthesis_script("hls_project/kernel")
```

### Interfaces

| Interface | Use |
| --- | --- |
| `ap_memory` | Plain array ports; the complete working set must fit the configured on-chip memory caps |
| `axi` | Addressed AXI4 memory-mapped ports with compiler-managed scratch arenas for external intermediates |
| `stream` | AXI4-Stream ports; valid only when every public port is consumed or produced by one complete monotonic row-major sweep |

Transforms, transposes, and data-dependent gathers require addressed storage and are rejected early for `interface="stream"`.

### Generated package

`write_testbench()` creates:

- `<top>.h`, `<top>.cpp`, and optional `<top>_tables.h`;
- `<top>_tb.cpp` and native-endian binary input/golden planes;
- Vitis C-simulation, synthesis, and RTL co-simulation Tcl scripts;
- a portable C++ functional-simulation script and Vitis header stubs;
- `design_manifest.json`, containing kernel types, physical interfaces, resolved configuration, implementation decisions, and source hashes.

Floating outputs are compared with a double-precision oracle using `rtol` and `atol`. Integer outputs are compared exactly. `SAR_DSL_HLS_TESTBENCH_MAX_BYTES` bounds the static arrays generated for simulation; pass `max_bytes=0` only when a production-scale host simulation is intentional.

Run a generated package with:

```bash
cd hls_project/kernel
vitis_hls -f kernel_hls_csim.tcl
vitis_hls -f kernel_csynth.tcl
```

The synthesis script fails when a configured resource cap is exceeded. A timing miss is reported but does not discard the report because HLS timing is a pre-route estimate.

## HLS resource contract

The default target is `xcvu13p-fhgb2104-2-i`, with 80% resource budgets, a 4 ns clock, and 12.5% clock uncertainty.

For a device listed in `sar.backends.hls.devices.DEVICES`, name the part and optionally a utilization percentage:

```python
design = kernel.compile("hls", options={
    "part": "xczu9eg-ffvb1156-2-e",
    "utilization": 70,
})
```

The compiler derives all six resource budgets and the memory primitive geometry from the device table.

For an unlisted part, provide the complete contract: target, all six budgets, and BRAM/URAM primitive sizes.

```python
design = kernel.compile("hls", options={
    "part": "xcvu5p-flva2104-1-e",
    "bram_bytes": 4 << 20,
    "uram_bytes": 8 << 20,
    "lutram_bytes": 1 << 16,
    "dsp": 512,
    "ff": 200_000,
    "lut": 100_000,
    "bram_block_bytes": 4608,
    "uram_block_bytes": 36864,
})
```

Use `uram_bytes=0, uram_block_bytes=0` for a target without UltraRAM. Use a nonzero primitive size with `uram_bytes=0` when the device supports UltraRAM but the design is not allowed to consume it.

Naming a part with only some explicit budgets is rejected. When `part` is omitted, individual budget overrides are interpreted as restrictions on the default target; this mode is useful for resource sweeps.

A project-wide configuration file (`options={"config": path}` or `$SAR_DSL_HLS_CONFIG`) overrides the keys it names; a key restated at its shipped default is a no-op, so a copy of `hls_config.yaml` behaves exactly like the defaults, and editing its `utilization` or `part` re-derives the budgets that combination implies.

`design.config.resource_contract()` and `design_manifest.json` report the target, budget mode, byte budgets, primitive sizes, and equivalent memory primitive counts.

### Constraint options

| Option | Default | Meaning |
| --- | ---: | --- |
| `part` | `xcvu13p-fhgb2104-2-i` | Vitis target part |
| `utilization` | `80` | Percentage used when deriving budgets from a listed part |
| `bram_bytes` | `9907200` | Block RAM cap |
| `uram_bytes` | `37748736` | UltraRAM cap |
| `lutram_bytes` | `2883584` | Distributed RAM cap |
| `dsp` | `9830` | DSP synthesis-report cap |
| `ff` | `2764800` | Flip-flop synthesis-report cap |
| `lut` | `1382400` | LUT synthesis-report cap |
| `bram_block_bytes` | `4608` | Block RAM primitive size used for placement and banking costs |
| `uram_block_bytes` | `36864` | UltraRAM primitive size; zero when unavailable |
| `interface` | `axi` | Top-level protocol: `ap_memory`, `axi`, or `stream` |
| `axi_bus_bits` | `512` | AXI data width |
| `axi_max_burst_length` | `64` | Maximum beats per burst, also bounded by the AXI 4 KiB rule |
| `axi_max_outstanding` | `16` | Maximum bursts in flight per direction |
| `precision` | `native` | Required signal precision: `native`, `f32`, or `f64` |
| `clock_ns` | `4.0` | Target clock period |
| `clock_uncertainty_percent` | `12.5` | Scheduling margin as a percentage of `clock_ns` |
| `top_func` | `null` | Generated top-function name; null uses the kernel name |

Strategy options such as FFT lane count, stage grouping, banking, tiling, gather caching, and storage thresholds are derived from kernel facts and resource constraints. They may be pinned through compile options for controlled synthesis experiments; `design.config.provenance` distinguishes user values from compiler decisions.

## Backend discovery

Built-in backend packages are discovered under `sar.backends`. `SAR_DSL_BACKEND_PATH` adds out-of-tree backend package directories. A backend implements `BaseBackend`:

```python
class Backend(BaseBackend):
    name = "mytarget"

    @classmethod
    def is_available(cls): ...

    def add_stages(self, stages, metadata): ...

    def make_launcher(self, artifact, metadata): ...
```

Stages receive the previous artifact, `KernelMetadata`, and a content-addressed `KernelCache`. Execution backends return a callable; source-emission backends return an artifact handle.

## Tool discovery

Compiler tools are resolved in this order:

1. `SAR_DSL_TOOL_<NAME>` for an individual executable;
2. the CMake-generated build configuration;
3. directories in `SAR_DSL_TOOL_PATH`;
4. `PATH`.

The runtime library follows `SAR_DSL_RUNTIME_LIB`, the active build configuration, then the prefix containing `sar-opt`. `SAR_DSL_OMP_LIB` overrides the OpenMP runtime used by the CPU backend.

## Environment variables

| Variable | Effect |
| --- | --- |
| `SAR_DSL_TOOL_<NAME>` | Override one tool path |
| `SAR_DSL_TOOL_PATH` | Additional tool search directories |
| `SAR_DSL_TOOL_TIMEOUT_SECONDS` | Per-stage timeout; non-positive disables it |
| `SAR_DSL_BUILD_CONFIG` | Explicit generated `_build_config.py` |
| `SAR_DSL_BUILD_DIR` | Build directory containing `python/sar/_build_config.py` |
| `SAR_DSL_RUNTIME_LIB` | Runtime shared library |
| `SAR_DSL_OMP_LIB` | OpenMP shared library |
| `SAR_DSL_BACKEND_PATH` | Out-of-tree backend packages |
| `SAR_DSL_CACHE_DIR` | Artifact cache root |
| `SAR_DSL_DISABLE_CACHE` | Disable cache reads while still writing artifacts |
| `SAR_DSL_CACHE_MAX_SIZE` | Artifact-cache LRU limit; default 2 GiB |
| `SAR_DSL_SPECIALIZATIONS_MAX` | In-process shape/dtype variants per generic kernel; default 64 |
| `SAR_DSL_OP_VARIANTS_MAX` | Eager constant variants per `@sar.op`; default 64 |
| `SAR_DSL_COMPILED_MAX` | Compiled launchers per specialized kernel; default 16 |
| `SAR_DSL_HLS_CONFIG` | Project HLS configuration file |
| `SAR_DSL_HLS_TESTBENCH_MAX_BYTES` | Generated HLS testbench memory limit |
| `OMP_NUM_THREADS` | OpenMP loop team and runtime-pool fallback |
| `SAR_RT_NUM_THREADS` | FFT/interpolation runtime participants |
| `SAR_RT_FFT_PLAN_CACHE_MAX` | Cached CPU FFT lengths; default 16, `0` disables |
| `SAR_RT_POOL_MAX_BYTES` | Retained CPU intermediate-plane bytes |

`python -m sar cache` reports cache contents. `python -m sar cache --clear` removes entries not locked by active compilers.
