# Backends

SAR-DSL uses a plugin model: each backend is a small Python
package contributing an ordered set of compilation *stages*.

## Using backends

```python
compiled = kernel.compile(backend="cpu")             # default
design   = kernel.compile(backend="hls",
                          options={"bram_bytes": 4 << 20})
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
NumPy arrays are marshalled as strided memref descriptors via ctypes,
results are allocated by the caller (destination-passing style).

Options: `opt_level` (default 3), `native_codegen` (default True).
`OMP_NUM_THREADS` controls generated OpenMP loops. FFT and interpolation
use a process-wide reusable worker pool controlled by
`SAR_RT_NUM_THREADS`, then `OMP_NUM_THREADS`; requests are capped by the
process affinity and the default is at most 32 workers.

### Buffer pool

A kernel's intermediate planes are gigabytes each. Left to libc, every
call unmaps them on return and the next call faults and zeroes every
page before the first useful store — on a 16384² kernel that first
touch costs more than the imaging itself (~1.5 s per 4 GiB plane vs.
~0.03 s to rewrite).

The runtime pools freed blocks rather than returning them to libc. A
request that matches a pooled size gets back that block with its pages
already mapped, so the fault cost is paid once per process rather than
once per call. The pool never holds more than the kernel's peak live
footprint, so a process that could run the kernel can hold the cache.

Set `SAR_RT_POOL_MAX_BYTES=0` to disable pooling (e.g. for memory
profiling). The first call in a process always pays the fault cost
regardless; measure warm performance for a fair comparison.

## HLS backend (`sar.backends.hls`)

Stages: `lower` (`sar-opt --sar-to-affine-pipeline`) -> `hls`
(`sar-opt --hls-pipeline` then `sar-translate --hls-emit-hlscpp`). Returns
an `HLSDesign` handle (`.cpp_path`, `.source()`); it is not executable.

The kernel is decomplexified (complex tensors become re/im float planes),
FFTs become Stockham loop nests, interpolation and the 2-D gather become
clamped straight-line loops, and compiled loops (`sar.iterate`) lose
their carries to side effects (`sar-demote-loop-carries`) since a
dataflow task may not yield values. The resulting affine IR enters the
HLS pipeline, which builds the dataflow hierarchy, places buffers on or
off chip and shapes the interfaces. Generated top functions take each
complex tensor as two adjacent float arrays (re, im), and the ports are
named after the
kernel's Python parameters (`raw` becomes `raw_re`, `raw_im`); result
planes are `out0`, `out1`, ... in declaration order. A parameter name
the emitter cannot use verbatim -- a C++ keyword, or the shape of a name
it generates itself -- falls back to the generated scheme.

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
cpu backend), the data files, `vitis_hls` csim and csynth scripts
(written against Vitis HLS 2022.2), and Vitis header stand-ins so the
package also csims with any plain C++ compiler
(`c++ -O2 -I stubs <top>.cpp <top>_tb.cpp -o csim -pthread`). The data
files in `<top>_tb_data/` are named by position (`in0_re.dat`,
`in0_im.dat`, ..., `out0_re.dat`, ...), while the design's ports keep
the kernel's parameter names. The
testbench runs the design on a dedicated 1 GiB stack: in C simulation the
on-chip buffers are stack arrays, which overflow default limits for
larger scenes. The example runners generate one package per algorithm
under `hls_project/<name>/`
(`python examples/wka/run_point_target_hls.py`). All four imaging chains
pass their testbenches, matching the NumPy reference to double-rounding
distance (`benchmarks/README.md` tabulates the errors).

Options: see "HLS configuration" below for the full schema, the shipped
defaults and how to override them.

### Where the data lives

Placement is the compiler's decision, not the user's. The backend
measures the resident working set in the lowered kernel -- its arguments
and results plus the full-size intermediates that survive buffer sharing
-- and keeps everything on chip while that fits the tier caps
(`bram_bytes` + `uram_bytes` + `lutram_bytes`). Past them the full-size
planes are streamed and only the constant tables (twiddles,
interpolation weights) and line-sized transform scratch stay
resident.

The set is measured rather than predicted because it follows the
algorithm, not the signature: the same four-stage chain holds six live
planes under range-Doppler and ten under chirp-scaling.

What the user picks is the interface, and that choice decides how
streamed buffers reach the top function:

| | top signature | use |
|---|---|---|
| `interface='ap_memory'` | the kernel's own inputs and results | csim packages: the testbench can drive every port |
| `interface='axi'` | one AXI master per I/O plane, plus compiler-managed typed scratch arenas | memory-mapped designs handed to Vitis |
| `interface='stream'` | AXI4-Stream ports for pure inputs and outputs; scratch arenas remain AXI masters | streaming radar front ends |

Internal buffers that spill to DRAM never surface as ports of their own:
they are distributed between two fixed arenas for their element type and
carved at aligned offsets. Each arena's array bound is the storage size the
host must bind. One-element placeholders preserve both masters when nothing
spills, so placement decisions do not change the interface.

An imaging chain is a sequence of whole-raster passes, so a plane dies
as soon as the next pass has read it and the ones whose lifetimes do not
overlap share an allocation. That is what keeps the streamed set a
property of the algorithm rather than of the chain's length: adding
passes to a kernel does not add ports. At `16384 x 16384` the omega-K
chain streams its full-size working planes through typed arenas and holds
only tables and line scratch on chip. Every port drives its own AXI master
bundle: ports sharing a bundle serialize their bus requests, starving
loops that read two planes concurrently.

Testbench generation rejects `interface='axi'` -- the compiler-managed
scratch allocation has no golden input data to drive it -- and
`interface='stream'`, whose `hls::stream<>` ports need a FIFO-feeding
harness the array testbench is not. Emit the csim package with
`interface='ap_memory'` (the default; the numerics are identical), and
compile with `'axi'` or `'stream'` for the design you synthesize.

### Validating a design in Vitis HLS

C simulation proves the numerics; synthesis proves the design. The two
run from generated scripts, and neither needs hand-written Tcl:

```python
design = kernel.compile(backend="hls", options={"interface": "axi"})
design.write_synthesis_script("hls_project/mykernel")
# then, on a machine with Vitis HLS 2022.2:
#   cd hls_project/mykernel && vitis_hls -f mykernel_csynth.tcl
```

`write_synthesis_script` works for every interface -- synthesis needs no
golden data, so the AXI and stream designs that cannot be csim'd are
exactly the ones it exists for. The csim package carries the same script
for its own raster, and the ALOS example runners emit one next to each
`_axi.cpp` design. The script names the `part` and `clock_ns` from the
configuration, so the budgets the compiler placed against and the device
the tool builds for stay the same device.
Each package includes `design_manifest.json` with the resolved configuration,
its provenance, and the generated source SHA-256.

Reports land in `<top>_csynth_proj/sol1/syn/report/`; the checklist,
in the order failures usually appear:

1. **Synthesis completes.** The emitter's output is affine loops, static
   arrays and straight-line arithmetic, all synthesizable by
   construction; a failure here is a compiler bug and worth a report.
2. **Timing.** The estimated clock in `<top>_csynth.rpt` must meet
   `clock_ns`. The shipped target is 4 ns; a miss is reported as a failed
   constraint rather than being hidden by the emitter.
3. **Initiation intervals.** Pipelined loops should report the II the
   pragma asked for (`II=1` throughout the element-wise and FFT nests;
   the interpolation gather may settle at II=2 on a dual-port band).
   A large II names the loop that needs attention.
4. **Memory utilization.** BRAM/URAM usage must sit within the
   `bram_bytes`/`uram_bytes` caps. The caps are hard: the compiler
   charges dataflow buffers at primitive granularity, rechecks the final
   banked layout, and reduces automatic partition factors before spilling
   more planes. When the working set
   cannot fit, the backend first retries with every full-size plane
   streamed, then fails compilation rather than emit a design that
   cannot fit the device (the fix is raising the caps to a larger
   part, or shrinking the kernel's resident tables).
   The FFT twiddle tables are file-scope `const` arrays; with the stage
   butterflies unrolled they constant-fold into the datapath and cost
   no memory primitives at all. Only dynamically-indexed tables (the
   axis ROMs) should appear as memories, at one or two BRAM primitives
   per reading process -- dozens means the tool replicated ROMs that
   ought to be shared, which is worth a report.
   Exactly linear axes use a compact `constexpr` generator; non-exact ramps
   remain explicit arrays so source compaction cannot change a floating-point
   value.
5. **Interfaces.** One `m_axi` port per I/O plane plus the typed scratch
   arenas, on the bundles the design declared; burst length and outstanding
   depths as configured (see the header comment of the emitted C++).

`benchmarks/run_resources.py` estimates memory from the emitted C++
without Vitis; the synthesis report is the measured truth, and a large
gap between the two is itself a finding worth recording.

Two things csim cannot decide and only this flow can: whether the tool
shares the twiddle ROMs (step 4), and whether the pass-pipeline option
`balance-dataflow=false` -- which saves copy nodes and on-chip bytes --
is safe under hardware dataflow concurrency, which needs co-simulation
(`cosim_design`) rather than the sequential csim.

### HLS memory access patterns

An access is **cross-row** when the innermost loop's induction variable
does not drive the fastest-varying index of a multi-dimensional buffer.
Each innermost iteration then moves by at least a whole row: on an FPGA
that is a memory transaction per element instead of a burst, and it
forces the buffer to stay resident rather than stream.

A corner turn has to stride *something* -- that is what a corner turn
is. What can be chosen is what it strides, and the answer that scales is
a small on-chip block: `sar-stage-transposes` fills the block along the
source rows and drains it along the destination rows, so both full-size
planes are swept contiguously and the block is the only thing left
striding. The invariant is therefore not zero cross-row accesses, but
that every one of them lands on a staged block and never on a
full-scene plane; `test/python/test_hls_access_patterns.py` gates it on
the omega-K chain.

Two design choices uphold the invariant:

- `sar.fftshift` lowers to a `linalg.generic` whose *input indexing map*
  carries the rotation (`(d0, d1) -> (d0, (d1 + n/2) mod n)`) rather
  than a gather in the body, so the read stays an affine function of
  the loop indices and fuses into neighbouring maps instead of hiding
  behind an `arith.remui`.
- `sar-stage-transposes` accepts any index expression that reads exactly
  one induction variable: it re-expresses the access map rather than
  interpreting it, so a corner turn with a fused shift stages exactly
  like a bare one.

What remains strided in the omega-K chain at `512 x 512`:

| Access | Count | Why it is inherent |
|---|---|---|
| Block fills of the staged corner turns | 10 | a transpose must stride one side; the block bounds the stride by the staging budget rather than the scene |
| Stolt band gather reads | 4 | not cross-row (the innermost loop drives the fastest axis) but non-affine: the address is clamped against the raster edge, and the position it clamps is computed from the data |

### FFT line blocks, lanes and stage grouping

A power-of-two Stockham transform uses radix-4 stages plus one radix-2 stage
when the exponent is odd, for `ceil(log2(N)/2)` butterfly passes. Lines are
processed in blocks of `fft_parallel_rows` lanes: a prefetch sweep copies the
block from the source planes into on-chip line buffers with unit-stride
external accesses (`fft_io_unroll` elements per access, so an AXI master can
burst), the butterfly stages run entirely on chip with the lane loop
innermost -- one twiddle fetch per butterfly is shared by every lane -- and a
mirrored sweep writes the block back. Between prefetch and write-back, with
`fft_stage_group=0`, each intermediate pass writes its own scratch block.

`fft_stage_group` trades that overlap for area. With `k > 0` the stages
are packed into groups that share scratch, cutting the live buffers to
roughly `ceil(log2(N)/2)/k`. Reusing a line puts a write-after-read edge between
the stages that share it, so the backend can no longer run them
concurrently -- the cost is pipeline depth, not arithmetic, and the
result is bit-identical either way.

Two scratch lines are the floor whenever more than one stage writes
scratch: a Stockham butterfly reads `X[q + sp]`, `X[q + s(p + m)]` and
writes `Y[q + 2sp]`, `Y[q + s(2p+1)]`, so a stage whose source and
destination were the same line would overwrite values a later iteration
still has to read. A grouping at or above the mixed-radix stage count
saturates at that floor rather than collapsing to one buffer.

`fft_parallel_rows` is bounded by the DSP budget (about six slices per
butterfly lane and stage, charged across every transform site) and by what
the banked line blocks cost the block-RAM tiers. Each lane owns its banks --
the line buffers are hinted complete over the lane dimension, which no local
access analysis could recover from the compact lane loop -- and the generated
C++ keeps one lane loop with an HLS unroll directive instead of cloning every
stage in source. `fft_io_unroll` is half a bus beat per plane, since the real
and imaginary sweeps run in one loop. Values may be pinned for synthesis
experiments, but they are compiler strategy rather than required user
constraints.

Each stage keeps its own affine loop nest whatever the grouping: stage
t+1 reads elements that many different iterations of stage t produce, so
folding several stages into one iteration space would violate the
dependence. Grouping changes which buffer a stage writes, not how the
stages are shaped.

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
(~1e-15, butterfly-ordering differences only) and `c64` kernels within
the f32 error envelope (up to ~1e-6 relative on the measured chains).
Use `c128` where the FFT chain requires double-precision cross-backend
agreement. The example chains expose the choice as
`build_kernel(..., dtype=sar.c64)`. A WKA f32 C-simulation smoke test is
gated in `test_hls_backend.py`; `benchmarks/run_accuracy.py --dtype c64`
produces the complete four-chain matrix.

Host data participates in that contract. Promotion follows NumPy, so a
float64 array -- NumPy's default -- meeting an f32 tensor gives f64, and
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
2. directories from the CMake-generated `sar/_build_config.py` (`SAR_DSL_TOOL_DIR`
   and `LLVM_TOOL_DIR`), then directories listed in `SAR_DSL_TOOL_PATH`;
3. `PATH`.

## Environment variables

| Variable | Effect |
|----------|--------|
| `SAR_DSL_TOOL_<NAME>` | Absolute path to one tool, overriding discovery |
| `SAR_DSL_TOOL_PATH` | Extra directories to search for tools (before `PATH`, after the build tree) |
| `SAR_DSL_TOOL_TIMEOUT_SECONDS` | Per-stage subprocess timeout (default 1800; non-positive disables it) |
| `SAR_DSL_BUILD_CONFIG` | Explicit CMake-generated `_build_config.py` path |
| `SAR_DSL_BUILD_DIR` | Non-default build tree containing `python/sar/_build_config.py` |
| `SAR_DSL_RUNTIME_LIB` | Path to `libsar_runtime.so` |
| `SAR_DSL_BACKEND_PATH` | Extra directories to search for backend packages |
| `SAR_DSL_CACHE_DIR` | Artifact cache root (default `~/.cache/sar-dsl`) |
| `SAR_DSL_DISABLE_CACHE` | `1` skips cache lookups (artifacts are still written) |
| `SAR_DSL_CACHE_MAX_SIZE` | Cache eviction threshold in bytes (default 2 GiB) |
| `SAR_DSL_OMP_LIB` | Path to `libomp.so` for the cpu backend |
| `SAR_DSL_HLS_CONFIG` | HLS configuration file overriding the shipped defaults |
| `SAR_RT_NUM_THREADS` | Reusable runtime workers; falls back to `OMP_NUM_THREADS`, then min(process affinity, 32) |
| `SAR_RT_POOL_MAX_BYTES` | Plane-pool retention bound in bytes; `0` disables pooling (see "Buffer pool") |

## HLS configuration

Every knob the HLS backend hands to the passes has a default in
`python/sar/backends/hls/hls_config.yaml`, which ships with the package.
Configuration files are flat YAML mappings of option names to scalar values.
Resolution order, weakest first:

```
hls_config.yaml  <  user config file  <  compile options
```

The user file comes from `options={"config": "device.yaml"}` or, failing
that, `$SAR_DSL_HLS_CONFIG`; it overrides the keys it names and leaves
the rest at the shipped default. The resolved set is what the design
reports:

```python
design = kernel.compile(backend="hls",
                        options={"interface": "axi", "axi_bus_bits": 256})
print(design.config.axi_bus_bits)     # 256
print(dict(design.config))            # every key, as compiled
print(design.config.provenance)       # and who decided each one
```

Every value is validated. An unknown key, a value of the wrong type and a
value outside the range the passes accept all raise `HLSConfigError` (a
`sar.SARError` and a `ValueError`), naming the file or the option that
carried it -- an option the user believed in and the backend ignored is
the failure this schema exists to prevent.

### Constraints

These are facts about the device and the deliverable. The compiler
cannot discover them, so they are the schema the user actually sets.
Defaults target `xcvu13p-fhgb2104-2-i` and budget 80% of each resource,
leaving 20% for control, interconnect and placement. Memory placement
charges whole primitives; DSP is checked from the synthesis report because
operator binding happens inside Vitis.

| Key | Default | Controls |
|---|---|---|
| `bram_bytes` | `9907200` | 2150 of 2688 BRAM36 primitives; hard cap |
| `uram_bytes` | `37748736` | 1024 of 1280 UltraRAM primitives; hard cap |
| `lutram_bytes` | `2883584` | 80% of the distributed-RAM capacity; hard cap |
| `dsp` | `9830` | 80% DSP synthesis-report budget |
| `ff` | `2764800` | 80% flip-flop synthesis-report budget |
| `lut` | `1382400` | 80% lookup-table synthesis-report budget |
| `interface` | `ap_memory` | Protocol the top-function ports speak: `ap_memory` (plain arrays), `axi` (AXI4 memory-mapped masters), or `stream` (AXI4-Stream) |
| `axi_bus_bits` | `512` | Data width of the AXI masters, in bits |
| `axi_max_burst_length` | `256` | Beats per burst at full bus width (the AXI4 maximum) |
| `axi_max_outstanding` | `16` | Full-length bursts in flight per direction (Vitis caps it at 32) |
| `precision` | `native` | Data path the kernel must carry: `native`, `f32` or `f64` |
| `part` | `xcvu13p-fhgb2104-2-i` | Device part the generated Vitis scripts name |
| `clock_ns` | `4.0` | Target clock period of the generated scripts |
| `top_func` | `null` | Name of the emitted top function; `null` takes the kernel's name |

`axi_bus_bits`, `axi_max_burst_length` and `axi_max_outstanding` are a
buffering budget rather than a per-port setting: the emitter shapes each
port from the buffer it serves -- a beat cannot widen past the contiguous
run, a burst cannot outrun the row -- and a port whose bursts come out
shorter than `axi_max_burst_length` gets proportionally more of them in
flight, up to the Vitis cap.

The caps are the whole constraint surface: placement charges each tier
in whole primitives (a 36 Kb block holding one kilobyte is spent),
overflows one block tier into the other before spilling anything, and
fails the design when a buffer that cannot stream fits no tier. Vitis binds
operators and control after emission, so DSP, FF, and LUT budgets are
validated from `*_csynth.xml`; `precision` is the pre-synthesis lever when
datapath pressure matters.

`precision` is a gate, not a conversion: the declared dtypes *are* the
data path (see "Precision contract"), so `precision="f32"` rejects a
kernel carrying f64 planes rather than narrowing it -- an f64 butterfly
costs several times the DSPs and block RAM of an f32 one, and nothing
downstream reports it.

### What the compiler decides for itself

Optimization strategy is not configured. FFT grouping and row parallelism,
loop tiling, banking, banded gathers, and buffer thresholds
are derived from the constraints above and from what the compiler measures
in the kernel: plane sizes, transform lengths, element widths, buffer
lifetimes. They are absent from `hls_config.yaml` because no value written
there would be a better guess than the computed one, and because knowing
what `fft_stage_group=2` trades away is a compiler-team question, not a
user's.

`design.config` reports what was chosen and `design.config.provenance`
marks those keys `derived`. The policy lives in
`python/sar/backends/hls/autotune.py`, one function per parameter:

| Derived | From |
|---------|------|
| `fft_stage_group` | the least grouping whose Stockham scratch fits the working share of the on-chip budget; full unroll where it fits |
| `fft_parallel_rows` | lanes per prefetched line block, bounded by the DSP budget across all transform sites and by the banked line buffers' block-RAM cost |
| `fft_io_unroll` | elements per external access in the FFT transfer sweeps: half a bus beat per plane |
| `loop_tile_size` | the element count in one bus beat, bounded by the pass limits |
| `lutram_max_bytes` | one bus beat: a bank no larger than one transfer does not earn a block RAM primitive |
| `array_partition_max_factor` | the largest power-of-two bank count no wider than one AXI beat; reduced automatically if final primitive accounting exceeds a cap |
| `interp_banded_gather` | on: the pass proves a bounded displacement per operation and falls back on its own when it cannot |
| `reuse_buffer_min_elements` | a full-scene plane against what the budget can afford to keep private |
| `recompute_min_elements` | the same measure -- storage traded for arithmetic instead of for sharing |
| `external_buffer_threshold` | the measured working set against the budget, which is what lets scene size outgrow the device |

Each is still accepted as a compile option, which is how an ablation
pins one (`options={"fft_stage_group": 2}`) and how an expert who has
synthesised a design overrides a pre-synthesis estimate. That is a
diagnostic act: a pinned value reaches the passes unchanged, and
`provenance` records that the user, not the compiler, chose it.
