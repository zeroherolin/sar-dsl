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
| `interface='ap_memory'` (or the deprecated `'bram'`) | the kernel's own inputs and results | csim packages: the testbench can drive every port |
| `interface='axi'` | one AXI master port per I/O plane, plus one trailing scratch port | designs handed to Vitis |
| `interface='stream'` | AXI4-Stream ports for the pure inputs and outputs; the scratch stays an AXI master, since a stream is unidirectional and consumed once | streaming radar front ends |

Internal buffers that spill to DRAM never surface as ports of their own:
they are carved into the one trailing scratch allocation, whose extent is
the port's own array bound -- that is the size the host must bind. The
port exists even when nothing spilled (at a one-element placeholder), so
the signature does not depend on what the optimizer decided this round.

An imaging chain is a sequence of whole-raster passes, so a plane dies
as soon as the next pass has read it and the ones whose lifetimes do not
overlap share an allocation. That is what keeps the streamed set a
property of the algorithm rather than of the chain's length: adding
passes to a kernel does not add ports. At `16384 x 16384` the omega-K
chain streams eight planes over two AXI bundles, roughly 13 GiB of DRAM,
and holds only the tables and the line scratch on chip.

Testbench generation rejects `interface='axi'` and `interface='stream'`:
the promoted intermediate ports have no golden data to drive them. Emit
the csim package with `interface='ap_memory'` (the default), and compile
with `'axi'` or `'stream'` for the design you synthesize.

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

Reports land in `<top>_csynth_proj/sol1/syn/report/`; the checklist,
in the order failures usually appear:

1. **Synthesis completes.** The emitter's output is affine loops, static
   arrays and straight-line arithmetic, all synthesizable by
   construction; a failure here is a compiler bug and worth a report.
2. **Timing.** The estimated clock in `<top>_csynth.rpt` must meet
   `clock_ns`. The shipped 10 ns is conservative; tighten it once the
   design closes, and treat a miss at 10 ns as a bug.
3. **Initiation intervals.** Pipelined loops should report the II the
   pragma asked for (`II=1` throughout the element-wise and FFT nests;
   the interpolation gather may settle at II=2 on a dual-port band).
   A large II names the loop that needs attention.
4. **Memory utilization.** BRAM/URAM usage must sit within the
   `bram_bytes`/`uram_bytes` budgets -- placement charged whole
   primitives, so the report should come in at or under the charge.
   The FFT twiddle tables are file-scope `const` arrays and should map
   to ROM: a size-N transform costs a handful of BRAM primitives for
   its cos/sin tables, not dozens (dozens means the tool replicated
   ROMs that ought to be shared, which is worth a report).
5. **Interfaces.** One `m_axi` port per I/O plane plus the scratch, on
   the bundles the design declared; burst length and outstanding
   depths as configured (see the header comment of the emitted C++).

`benchmarks/run_resources.py` estimates memory from the emitted C++
without Vitis; the synthesis report is the measured truth, and a large
gap between the two is itself a finding worth recording.

Two things csim cannot decide and only this flow can: whether the tool
shares the twiddle ROMs (step 4), and whether `balance_dataflow=False`
-- which saves copy nodes and on-chip bytes -- is safe under hardware
dataflow concurrency, which needs co-simulation (`cosim_design`) rather
than the sequential csim.

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

### FFT stage grouping

A Stockham transform of length N is log2(N) butterfly passes. By default
each pass writes its own scratch line, which makes the passes a chain: a
dataflow backend can overlap one line's late stages with the next line's
early ones. It also means log2(N)-1 live scratch buffers.

`fft_stage_group` trades that overlap for area. With `k > 0` the stages
are packed into groups that share scratch, cutting the live buffers to
roughly log2(N)/k. Reusing a line puts a write-after-read edge between
the stages that share it, so the backend can no longer run them
concurrently -- the cost is pipeline depth, not arithmetic, and the
result is bit-identical either way.

Two scratch lines are the floor whenever more than one stage writes
scratch: a Stockham butterfly reads `X[q + sp]`, `X[q + s(p + m)]` and
writes `Y[q + 2sp]`, `Y[q + s(2p+1)]`, so a stage whose source and
destination were the same line would overwrite values a later iteration
still has to read. `k` at or above log2(N) saturates at that floor
rather than collapsing to one buffer.

Omega-K at `512 x 512`, measured from the emitted C++ by
`benchmarks/run_resources.py`:

| `fft_stage_group` | on-chip | vs. default | dataflow regions |
|---|---|---|---|
| 0 (default) | 1024.5 KiB | -- | 12 |
| 1 | 1024.5 KiB | 0.0% | 12 |
| 2 | 896.5 KiB | -12.5% | 8 |
| 4 | 832.5 KiB | -18.7% | 8 |
| 8 | 832.5 KiB | -18.7% | 8 |

`k=1` gives every stage its own slot, so it is the full unroll's buffer
count under another name. The saving arrives at `k=2` and flattens once
the pool reaches its floor. The dataflow-region count falling from 12 to
8 is the throughput being spent.

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
(~1e-15, butterfly-ordering differences only) and `c64` kernels to
f32 rounding (~1e-7 relative). Both satisfy the declared precision;
use `c128` where the FFT chain itself must be reproducible bit-for-bit
across backends. The example chains expose the choice as
`build_kernel(..., dtype=sar.c64)`, and the cross-backend agreement at
f32 is gated by csim in `test_hls_backend.py`.

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
2. directories from the CMake-generated `sar/_build_config.py` (`SAR_DSL_TOOL_DIR`
   and `LLVM_TOOL_DIR`), then directories listed in `SAR_DSL_TOOL_PATH`;
3. `PATH`.

## Environment variables

| Variable | Effect |
|----------|--------|
| `SAR_DSL_TOOL_<NAME>` | Absolute path to one tool, overriding discovery |
| `SAR_DSL_TOOL_PATH` | Extra directories to search for tools (before `PATH`, after the build tree) |
| `SAR_DSL_RUNTIME_LIB` | Path to `libsar_runtime.so` |
| `SAR_DSL_BACKEND_PATH` | Extra directories to search for backend packages |
| `SAR_DSL_CACHE_DIR` | Artifact cache root (default `~/.cache/sar-dsl`) |
| `SAR_DSL_DISABLE_CACHE` | `1` skips cache lookups (artifacts are still written) |
| `SAR_DSL_CACHE_MAX_SIZE` | Cache eviction threshold in bytes (default 2 GiB) |
| `SAR_DSL_OMP_LIB` | Path to `libomp.so` for the cpu backend |
| `SAR_DSL_HLS_CONFIG` | HLS configuration file overriding the shipped defaults |
| `SAR_RT_NUM_THREADS` | Runtime worker threads; falls back to `OMP_NUM_THREADS`, then the hardware concurrency |
| `SAR_RT_POOL_MAX_BYTES` | Plane-pool retention bound in bytes; `0` disables pooling (see "Buffer pool") |

## HLS configuration

Every knob the HLS backend hands to the passes has a default in
`python/sar/backends/hls/hls_config.yaml`, which ships with the package.
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
Defaults describe half a Virtex UltraScale+ VU13P: the device carries
94.5 Mb of block RAM, 360 Mb of UltraRAM and 12288 DSP slices, and
placement runs before synthesis, charging buffers in whole primitives.
Half leaves room for that estimate to be wrong.

| Key | Default | Controls |
|---|---|---|
| `bram_bytes` | `6193152` | Block RAM the design may occupy (0 = unbounded) |
| `uram_bytes` | `23592960` | UltraRAM the design may occupy (0 = unbounded) |
| `lutram_bytes` | `901120` | Distributed RAM the design may occupy; a quarter of the device rather than half, since this tier is carved out of the SLICEM LUTs the datapath is built from |
| `on_chip_budget` | `null` (29.3 MiB) | Total on-chip bytes before full-scene planes are streamed; `null` sums the three tiers, `0` keeps everything resident |
| `uram_min_bytes` | `36864` | Buffer size at or above which UltraRAM is used; a device fact (one 288 Kb URAM block), so retargeting states a different value |
| `interface` | `ap_memory` | Protocol the top-function ports speak: `ap_memory` (plain arrays, csim-able; `bram` is a deprecated alias), `axi` (AXI4 memory-mapped masters), or `stream` (AXI4-Stream) |
| `axi_bus_bits` | `512` | Data width of the AXI masters, in bits |
| `axi_max_burst_length` | `256` | Beats per burst at full bus width (the AXI4 maximum) |
| `axi_max_outstanding` | `16` | Full-length bursts in flight per direction (Vitis caps it at 32) |
| `precision` | `native` | Data path the kernel must carry: `native`, `f32` or `f64` |
| `part` | `xcvu13p-fhgb2104-2-e` | Device part the generated Vitis scripts name; must be the device the budgets describe |
| `clock_ns` | `10.0` | Target clock period of the generated Vitis scripts |
| `top_func` | `null` | Name of the emitted top function; `null` takes the kernel's name |

`axi_interface=True/False` predates `interface` and still selects `axi` /
`ap_memory`; giving both a conflicting value is an error.

`axi_bus_bits`, `axi_max_burst_length` and `axi_max_outstanding` are a
buffering budget rather than a per-port setting: the emitter shapes each
port from the buffer it serves -- a beat cannot widen past the contiguous
run, a burst cannot outrun the row -- and a port whose bursts come out
shorter than `axi_max_burst_length` gets proportionally more of them in
flight, up to the Vitis cap.

The tier budgets sum into `on_chip_budget` when no total is given; the
split between them is a placement decision `-hls-pipeline` does not
currently take an option for, so a per-tier ceiling is not yet enforced
on its own. There is no DSP budget: Vitis binds the float operators of
an imaging chain itself, so the emitter has no binding decision to
ration.

`precision` is a gate, not a conversion: the declared dtypes *are* the
data path (see "Precision contract"), so `precision="f32"` rejects a
kernel carrying f64 planes rather than narrowing it -- an f64 butterfly
costs several times the DSPs and block RAM of an f32 one, and nothing
downstream reports it.

### What the compiler decides for itself

Optimization strategy is not configured. FFT stage grouping, loop tiling,
banded gathers, and the buffer-sharing, recompute and off-chip thresholds
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
| `loop_tile_size` | bus width over element width, so a tile is a whole number of beats |
| `lutram_max_bytes` | one bus beat: a bank that cannot fill a single transfer does not earn a block RAM primitive, so below a beat it lands in distributed RAM |
| `interp_banded_gather` | on: the pass proves a bounded displacement per operation and falls back on its own when it cannot |
| `reuse_buffer_min_elements` | a full-scene plane against what the budget can afford to keep private |
| `recompute_min_elements` | the same measure -- storage traded for arithmetic instead of for sharing |
| `external_buffer_threshold` | the measured working set against the budget, which is what lets scene size outgrow the device |

Each is still accepted as a compile option, which is how an ablation
pins one (`options={"fft_stage_group": 2}`) and how an expert who has
synthesised a design overrides a pre-synthesis estimate. That is a
diagnostic act: a pinned value reaches the passes unchanged, and
`provenance` records that the user, not the compiler, chose it.
