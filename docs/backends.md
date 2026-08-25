# Backends

SAR-DSL uses a plugin model: each backend is a small Python package contributing an ordered set of compilation _stages_.

## Using backends

```python
compiled = kernel.compile(backend="cpu")             # default
design   = kernel.compile(backend="hls",
                          options={"bram_bytes": 4 << 20})
print(sar.list_backends())                           # discovery
```

## CPU backend (`sar.backends.cpu`)

Compiles kernels for the **host** CPU -- whatever machine the driver runs on. The lowering is architecture-neutral (LLVM dialect, portable runtime library, generic memref ABI); clang targets the host triple and native tuning flags are probed per architecture (`-march=native` / `-mcpu=native`, with a generic fallback). x86-64 Linux is the tested platform; other Linux hosts with an LLVM host target (e.g. AArch64) are expected to work but are not covered by CI. Windows is unsupported.

Stages: `llvm` (sar-opt `--sar-to-llvm-pipeline`: runtime calls, linalg elementwise fusion, OpenMP parallel loops) -> `ll` (mlir-translate) -> `shared` (clang `-O3` + native tuning, linked against `libsar_runtime` and LLVM's `libomp`). The launcher is a `sar.runtime.CompiledKernel`: NumPy arrays are marshalled as strided memref descriptors via ctypes, results are allocated by the caller (destination-passing style).

Options: `opt_level` (default 3), `native_codegen` (default True).

The CPU backend has two successive parallel execution mechanisms. Generated element-wise, reduction, and layout loops use an OpenMP team controlled by `OMP_NUM_THREADS`; when it is unset, the OpenMP runtime chooses its default from the CPUs visible to the process. FFT and interpolation lower to calls into `libsar_runtime` and use a separate process-wide reusable pool. Its participant count is the first positive value from `SAR_RT_NUM_THREADS` and `OMP_NUM_THREADS`, or `min(process affinity, 32)` when neither is set. The count includes the calling thread, so a value of 32 creates 31 background threads and uses up to 32 software threads total; a call with fewer independent rows or transform lines activates fewer participants. The pool is not pinned, and “worker” means a software thread schedulable on one affinity-visible logical CPU, not a guaranteed physical core.

OpenMP loops and FFT/interpolation calls normally run as successive stages, not as nested teams, so their limits are not additive. `SAR_RT_NUM_THREADS=32` alone does not make the whole CPU backend a 32-thread run because OpenMP remains independently configured. Set both `OMP_NUM_THREADS=N` and `SAR_RT_NUM_THREADS=N` for an explicit common limit; setting only `OMP_NUM_THREADS=N` also reaches the runtime pool through its fallback unless `SAR_RT_NUM_THREADS` is set. The runtime pool clamps an explicit request to the process affinity. OpenMP is configured by its own runtime and can oversubscribe if `OMP_NUM_THREADS` is set above the available logical CPUs, so use a value no greater than the process affinity when comparing runs.

### Buffer pool

A kernel's intermediate planes are gigabytes each. Left to libc, every call unmaps them on return and the next call faults and zeroes every page before the first useful store — on a 16384 × 16384 kernel that first touch costs more than the imaging itself (~1.5 s per 4 GiB plane vs. ~0.03 s to rewrite).

The runtime pools freed blocks rather than returning them to libc. A request that matches a pooled size gets back that block with its pages already mapped, so the fault cost is paid once per process rather than once per call. The pool never holds more than the kernel's peak live footprint, so a process that could run the kernel can hold the cache.

Set `SAR_RT_POOL_MAX_BYTES=0` to disable pooling (e.g. for memory profiling). The first call in a process always pays the fault cost regardless; measure warm performance for a fair comparison.

## HLS backend (`sar.backends.hls`)

Stages: `lower` (`sar-opt --sar-to-affine-pipeline`) -> `hls` (`sar-opt --hls-pipeline` then `sar-translate --hls-emit-hlscpp`). Returns an `HLSDesign` handle (`.cpp_path`, `.source()`, `.header_source()`); it is not executable.

The kernel is decomplexified (complex tensors become re/im float planes), FFTs become Stockham loop nests, interpolation and the 2-D gather become clamped straight-line loops, and compiled loops (`sar.iterate`) lose their carries to side effects (`sar-demote-loop-carries`) since a dataflow task may not yield values. The resulting affine IR enters the HLS pipeline, which builds the dataflow hierarchy, places buffers on or off chip and shapes the interfaces. Generated top functions take each complex tensor as two adjacent float arrays (re, im), and the ports are named after the kernel's Python parameters (`raw` becomes `raw_re`, `raw_im`); result planes are `out0`, `out1`, ... in declaration order. A parameter name the emitter cannot use verbatim -- a C++ keyword, or the shape of a name it generates itself -- falls back to the generated scheme.

Every SAR operation has a lowering, so complete imaging chains (omega-K, range-Doppler, chirp scaling, polar format) emit as single HLS designs; the identical IR is validated numerically on the CPU through `--sar-affine-to-llvm-pipeline`. FFTs of any size >= 2 lower here: powers of two run Stockham directly, other sizes go through Bluestein's chirp-z reduction (two padded Stockham transforms with the chirp and kernel spectrum folded at compile time).

### HLS validation package

`HLSDesign.write_testbench(inputs, expected, output_dir)` emits a self-contained validation package: a `<top>.h` declaration file, a `<top>_tables.h` holding the design's constant (ROM) tables when it has any, a top-first `<top>.cpp` implementation, a testbench comparing every output plane against golden data (from the NumPy reference or the cpu backend), the data files, `<top>_hls_csim.tcl`, the C-synth/C-RTL scripts (written against Vitis HLS 2022.2), and `<top>_portable_cpp_sim.sh`. Tests and benchmarks run C-sim through Vitis HLS whenever `vitis_hls` is installed. The portable C++ script, together with its Vitis header stand-ins, is the explicit fallback for environments without Vitis HLS. The data files in `<top>_tb_data/` are native-endian binary planes named by position (`in0_re.bin`, `in0_im.bin`, ..., `out0_re.bin`, ...), while the design's ports keep the kernel's parameter names. The testbench runs the design on a dedicated 1 GiB stack: in C simulation the on-chip buffers are stack arrays, which overflow default limits for larger scenes. Floating planes are compared against a double-precision oracle within `rtol`/`atol`; integer planes are compared exactly, since a 64-bit value neither survives a round trip through double nor has a meaningful tolerance. The example runners generate one package per algorithm under `hls_project/<name>/` (`python examples/wka/run_point_target_hls.py`). All four imaging chains pass their testbenches, matching the NumPy reference to double-rounding distance (`benchmarks/README.md` tabulates the errors).

The package also carries `<top>_cosim.tcl`, which runs the generated RTL against the same testbench and golden data at that package's unchanged raster. Thus an ALOS package emitted at 16384 × 16384 has a cosim script at the same dimensions; the repository generates it but does not execute that production run. The two levels cost very different amounts, so automated validation is split as follows:

|  | needs | covers |
| --- | --- | --- |
| C-sim | Vitis HLS | every port protocol, at every raster the suite builds |
| Portable C++ fallback | any C++ compiler | the same numerical oracle when Vitis HLS is unavailable |
| RTL co-simulation | Vitis HLS | generated at the package raster; one separate 256 × 256 transform is executed opt-in via `SAR_DSL_TEST_VITIS=1` |

Co-simulation time grows with the raster, so a production grid is not run by the test suite. The script still ships at the full selected raster: a user who wants to execute it runs `vitis_hls -f <top>_cosim.tcl` on it. Neither building the compiler nor running hosted CI requires Vitis; those environments select the portable fallback explicitly through tool discovery.

Options: see "HLS configuration" below for the full schema, the shipped defaults and how to override them.

### Where the data lives

Placement is the compiler's decision, not the user's. The backend measures the resident working set in the lowered kernel -- its arguments and results plus the full-size intermediates that survive buffer sharing -- and keeps everything on chip while that fits the tier caps (`bram_bytes` + `uram_bytes` + `lutram_bytes`). Past them the full-size planes are streamed and only the constant tables (twiddles, interpolation weights) and line-sized transform scratch stay resident.

The set is measured rather than predicted because it follows the algorithm, not the signature: the same four-stage chain holds six live planes under range-Doppler and ten under chirp-scaling.

What the user picks is the interface, and that choice decides how streamed buffers reach the top function:

|  | top signature | use |
| --- | --- | --- |
| `interface='ap_memory'` | the kernel's own inputs and results as block-memory arrays | directly connected on-chip-memory designs |
| `interface='axi'` | one AXI master per I/O plane, plus the scratch arenas the spilled buffers need | memory-mapped designs handed to Vitis |
| `interface='stream'` | AXI4-Stream ports for pure inputs and outputs; scratch arenas remain AXI masters | streaming radar front ends |

A stream has no addresses, so `interface='stream'` requires every public port to be swept exactly once in row-major order. That holds for an element-wise front end and not for a transform: an FFT reads a plane along one axis and a corner turn along the other, so a chain containing either is rejected with `cannot use AXI4-Stream without one complete monotonic row-major access`. Those chains take `axi`, where the same planes are addressed rather than streamed.

A port is a platform resource: it is what a board integrator wires to a physical memory channel, and a channel spent on a compiler convenience is one the algorithm cannot use. So the port list is the algorithm's own I/O and nothing more. Internal buffers that spill to DRAM never surface as ports of their own: they are carved at aligned offsets out of scratch arenas, and an element type that never spills contributes no port. Each arena's array bound is the storage size the host must bind.

Two things set the arena count, both about aliasing rather than convenience. A typed C++ signature needs one pointer per element type, since an arena carries one scalar type and one pointer cannot be both. And a buffer written by a dataflow node cannot share its pointer with another buffer active in that node: HLS must otherwise assume each load or write may alias a store still in flight, and the requests serialize behind their own responses -- an initiation interval of tens of cycles where the arithmetic wanted one. Pure read fan-in may share an arena to preserve prefetch locality. Splitting on the remaining conflicts is the ping-pong a hand-written design allocates by hand, and it is what the compiler allocates too.

The conflict graph is colored exactly while it is small. Its physical master count is then bounded by the derived `max_scratch_arenas` strategy so an artificially deep graph cannot grow the platform interface; if compaction has to merge a conflict, the implementation summary records that fact. Compiler-owned arenas have no public scalar layout, so their aligned movers use packed vector words. Unusual gathers and scatters retain scalar semantics through word extract or read-modify-write adapters. Dynamic clamped tap windows are recognized as a bounded two-word family: all taps select from one of two adjacent packed words, even when unrelated planes' loads are interleaved. A source-size guard leaves pathological graphs scalar.

An imaging chain is a sequence of whole-raster passes, so a plane dies as soon as the next pass has read it and the ones whose lifetimes do not overlap share an allocation. That is what keeps the streamed set a property of the algorithm rather than of the chain's length: adding passes to a kernel does not add ports. At 16384 × 16384 the omega-K chain streams its full-size working planes through four f32 arenas -- the depth of its read/write ping-pong and the accesses one node makes at once, not of its pass list -- and holds only tables and line scratch on chip. Masters are bounded by the ports rather than equal to them. A full-size plane drives its own: it is swept for a whole pass, and sharing would serialize two such sweeps, starving loops that read two planes concurrently. The small read-only tables a chain carries -- an axis, a window, a reference chirp -- share one instead, the way a hand-written design does: they are read a row at a time and their bursts interleave.

Testbench generation supports scalar-public `interface='axi'` designs even when compiler-owned scratch is packed: scratch arrays are allocated by the harness, and the package includes both C-simulation and Verilog RTL co-simulation scripts. Packed public AXI words are packed and unpacked by the generated harness too. `interface='stream'` has the same numerical oracle when each port is proven to make one complete monotonic row-major sweep; addressed, repeated or random accesses are rejected instead of being mislabeled as a stream.

### Validating a design in Vitis HLS

C simulation proves the numerics; synthesis proves the design. The two run from generated scripts, and neither needs hand-written Tcl:

```python
design = kernel.compile(backend="hls", options={"interface": "axi"})
design.write_synthesis_script("hls_project/mykernel")
# then, on a machine with Vitis HLS 2022.2:
#   cd hls_project/mykernel && vitis_hls -f mykernel_csynth.tcl
```

`write_synthesis_script` works for every interface and needs no golden data. It writes `<top>.h`, `<top>.cpp`, the manifest, and the synthesis Tcl; function declarations live in the header while the top definition precedes helper definitions in the implementation. A design with constant tables also gets `<top>_tables.h`: twiddle, axis and window ROMs routinely outweigh the logic that reads them, so they are kept out of the implementation file. The HLS validation package carries the same script for its own raster, and the ALOS example runners emit one next to each `_axi.cpp` design. The script names the `part`, `clock_ns`, and explicit clock uncertainty from the configuration, so the budgets the compiler placed against and the device the tool builds for stay the same device. The script parses the resulting XML and exits nonzero when the report is incomplete or a configured resource budget is violated. A timing miss is reported with its shortfall while the result remains available for inspection, because the estimate is a pre-route optimization goal rather than proof that a device cannot hold the design. Each package includes `design_manifest.json` with logical and physical port shapes, the resolved configuration and provenance, and SHA-256 hashes for the generated source and header.

Reports land in `<top>_csynth_proj/sol1/syn/report/`; the checklist, in the order failures usually appear:

1. **Synthesis completes.** The emitter's output is affine loops, static arrays and straight-line arithmetic, so it is synthesizable by construction; report a failure here as a compiler bug.
2. **Timing.** Compare the estimated clock in `<top>_csynth.rpt` with the effective scheduling budget: `clock_ns` minus its uncertainty margin. The shipped 4 ns target reserves 12.5%, so Vitis schedules against 3.5 ns. A miss is reported explicitly with its shortfall rather than being hidden by the emitter.
3. **Initiation intervals.** Pipelined loops should report the II the pragma asked for (`II=1` throughout scalar element-wise nests). Packed interpolation loops are interpreted per logical sample: production caches settle at II=8 for an eight-sample word and II=4 for a four-sample word, or about one sample per cycle. A larger normalized II names the loop that needs attention.

   The shared-line-buffer Stockham microarchitecture has a proven minimum II of 2, which is attached to the loop rather than requesting an unattainable II=1. A radix-4 stage reads four taps a stride apart, and that stride is a power of the radix at every stage but the last -- so a cyclic banking that separates one stage's taps maps another stage's onto a single bank, and a block banking separates only the first. No static partitioning of one line buffer serves every stage, which is why the butterfly scratch is hinted one bank per lane and takes the II. The lane parallelism absorbs it: `fft_parallel_rows` lines are in flight per iteration, so the II costs throughput only against a design that banked every stage at once, which would need one buffer and banking per stage -- a different point on the resource/latency curve, not a free one.

4. **Memory utilization.** BRAM/URAM usage must sit within the `bram_bytes`/`uram_bytes` caps. The caps are hard: the compiler charges dataflow buffers at primitive granularity, rechecks the final banked layout, and reduces automatic partition factors before spilling more planes. When the working set cannot fit, the backend first retries with every full-size plane streamed, then fails compilation rather than emit a design that cannot fit the device (the fix is raising the caps to a larger part, or shrinking the kernel's resident tables). The FFT twiddles are file-scope `const` arrays split by Stockham stage. Constant 1/0 entries fold into the datapath; dynamically indexed stages map only their own slice to ROM, rather than making every concurrent reader replicate the full transform table. Axis ROMs and stage tables should consume memory in proportion to the entries each process reads -- dozens of copies of a full table indicate unintended replication. Exactly linear axes use a compact `constexpr` generator; non-exact ramps remain explicit arrays so source compaction cannot change a floating-point value.
5. **Interfaces.** One `m_axi` port per I/O plane plus one per scratch arena, on the bundles the design declared; burst length and outstanding depths as configured (see the header comment of the emitted C++). Every addressed master has one AXI-Lite base-address register, and those registers, scalar arguments and `return` share the single `ctrl` bundle; they never create a second control interface or another memory master. The arena count follows how the kernel's planes alias, never how many passes it has; report the latter as a compiler bug.

`benchmarks/run_hls_resources.py` estimates memory from the emitted C++ without Vitis; the synthesis report is the measured figure.

Step 4 is the one question C simulation cannot answer: whether the tool shares the twiddle ROMs across the stages that read them, which decides whether the block-RAM figure matches the compiler's estimate.

### HLS memory access patterns

An access is **cross-row** when the innermost loop's induction variable does not drive the fastest-varying index of a multi-dimensional buffer. Each innermost iteration then moves by at least a whole row: on an FPGA that is a memory transaction per element instead of a burst, and it forces the buffer to stay resident rather than stream.

A corner turn has to stride _something_ -- that is what a corner turn is. What can be chosen is what it strides, and the answer that scales is a small on-chip block: `sar-stage-transposes` fills the block along the source rows and drains it along the destination rows, so both full-size planes are swept contiguously and the block is the only thing left striding. The invariant is therefore not zero cross-row accesses, but that every one of them lands on a staged block and never on a full-scene plane; `test/python/test_hls_access_patterns.py` gates it on the omega-K chain.

Two design choices uphold the invariant:

- `sar.fftshift` lowers to a `linalg.generic` whose _input indexing map_ carries the rotation (`(d0, d1) -> (d0, (d1 + n/2) mod n)`) rather than a gather in the body, so the read stays an affine function of the loop indices and fuses into neighbouring maps instead of hiding behind an `arith.remui`.
- `sar-stage-transposes` accepts any index expression that reads exactly one induction variable: it re-expresses the access map rather than interpreting it, so a corner turn with a fused shift stages exactly like a bare one.

What remains strided in the omega-K chain, counted by the 128 × 128 audit in `test/python/test_hls_access_patterns.py`:

| Access | Count | Why it is inherent |
| --- | --- | --- |
| Block fills of the staged corner turns | 10 | a transpose must stride one side; the block bounds the stride by the staging budget rather than the scene |
| Stolt band gather reads | 4 | not cross-row (the innermost loop drives the fastest axis) but non-affine: the address is clamped against the raster edge, and the position it clamps is computed from the data |

### FFT line blocks, lanes and stage grouping

A power-of-two Stockham transform uses radix-4 stages plus one radix-2 stage when the exponent is odd, for `ceil(log2(N)/2)` butterfly passes. Lines are processed in staged blocks. On the fastest-varying axis the block has `fft_parallel_rows` lines and `fft_io_unroll` adjacent elements move per access. On a slow axis, adjacent lines are the contiguous direction: the transfer block therefore covers at least one packed word, while the `fft_parallel_rows` compute lanes visit it in sub-blocks. The butterfly stages stay entirely on chip, one twiddle fetch per butterfly is shared by every lane, and a mirrored sweep writes the block back. Between prefetch and write-back, with `fft_stage_group=0`, each intermediate pass writes its own scratch block.

`fft_stage_group` trades that overlap for area. With `k > 0` the stages are packed into groups that share scratch, cutting the live buffers to roughly `ceil(log2(N)/2)/k`. Reusing a line puts a write-after-read edge between the stages that share it, so the backend can no longer run them concurrently -- the cost is pipeline depth, not arithmetic, and the result is bit-identical either way.

How much scratch stage unrolling may hold is half the block-RAM tiers. Full stage unroll consumes materially more memory without removing a full-plane transfer, so the grouping decision keeps the half-tier ceiling. Lane-parallel line buffers may use five eighths: unlike another stage slot, another lane removes complete row iterations, and the final banking pass rebalances BRAM and URAM before enforcing both hard caps. Grouping decides how many intermediate lines a transform holds _on chip_; it does not change how often a plane crosses the external bus, and at production rasters that traffic is what sets the latency. Scratch past a shallow stage chain therefore buys pipeline depth nothing waits on, while lanes are the knob that can convert spare storage into latency.

Two scratch lines are the floor whenever more than one stage writes scratch: a Stockham butterfly reads `X[q + sp]`, `X[q + s(p + m)]` and writes `Y[q + 2sp]`, `Y[q + s(2p+1)]`, so a stage whose source and destination were the same line would overwrite values a later iteration still has to read. A grouping at or above the mixed-radix stage count saturates at that floor rather than collapsing to one buffer.

`fft_parallel_rows` is bounded by the DSP budget (about six slices per butterfly lane and stage, charged across every transform site) and by what the banked line blocks cost the block-RAM tiers. Each lane owns its banks -- the line buffers are hinted complete over the lane dimension, which no local access analysis could recover from the compact lane loop -- and the generated C++ keeps one lane loop with an HLS unroll directive instead of cloning every stage in source. `fft_io_unroll` is bounded by a full beat on each independent real/imaginary master and by the synthesis-validated lane cap. Values may be pinned for synthesis experiments, but they are compiler strategy rather than required user constraints.

Those two bounds are the transform's own; they say nothing about what the rest of the chain is holding. The staged blocks a wide transfer fills are resident alongside every other live plane, and the tier caps apply to the total. So the transfer width is also the last thing the placement retry spends: if a design still overruns after automatic banking has been reduced and every full-size plane streamed, `fft_io_unroll` halves and the kernel is lowered again, one step at a time, because a narrower transfer costs bandwidth on every line. The width the design was finally built with is what its manifest reports.

Each stage keeps its own affine loop nest whatever the grouping: stage t+1 reads elements that many different iterations of stage t produce, so folding several stages into one iteration space would violate the dependence. Grouping changes which buffer a stage writes, not how the stages are shaped.

## Precision contract

Declared dtypes fix the _data-path_ precision on every backend: a `c64` tensor is stored, moved and combined as f32 planes on cpu and HLS alike. Internal precision of the leaf transforms is a backend choice:

- interpolation positions/weights are computed in f64 on both backends (index arithmetic must not lose fractional bins);
- the cpu FFT runs double-precision butterflies regardless of dtype (the runtime library gets f64 for free), while the HLS FFT computes in the declared precision (f32 butterflies cost a fraction of the DSPs of f64 ones).

Consequently `c128` kernels agree across backends to f64 rounding (~1e-15, butterfly-ordering differences only) and `c64` kernels within the f32 error envelope (up to ~1e-6 relative on the measured chains). Use `c128` where the FFT chain requires double-precision cross-backend agreement. The example chains expose the choice as `build_kernel(..., dtype=sar.c64)`. A WKA f32 C-simulation regression test is gated in `test_hls_backend.py`; `benchmarks/run_cpu_hls_accuracy.py --dtype c64` produces the complete four-chain matrix.

Host data participates in that contract. Promotion follows NumPy, so a float64 array -- NumPy's default -- meeting an f32 tensor gives f64, and every operator and buffer downstream widens with it. The trace reports this as `sar.PrecisionWarning`, naming the host array rather than the tensor it met, because the host is the side that can cheaply choose otherwise:

```python
warnings.simplefilter("error", sar.PrecisionWarning)  # treat as a bug
```

A mixed-precision chain also costs interfaces: buffers of different element types cannot share an allocation, so the streamed set is the sum of the two rather than the larger of them.

## Adding a backend

1. Create a backend package -- `python/sar/backends/<name>/compiler.py` for a built-in one, any directory for an out-of-tree one:

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

That's all: subpackages of `sar.backends` are discovered automatically, and `SAR_DSL_BACKEND_PATH` (os.pathsep-separated directories) picks up out-of-tree ones.

Useful building blocks: `sar.compiler.toolchain.find_tool` / `run_tool` (tool discovery + subprocess execution with good errors), the per-kernel artifact cache handed to every stage, and `sar.runtime` for memref marshalling if the target executes on the host.

## Toolchain discovery

Tools are located in this order:

1. `SAR_DSL_TOOL_<NAME>` environment variables (e.g. `SAR_DSL_TOOL_SAR_OPT`);
2. directories from the CMake-generated `sar/_build_config.py` (`SAR_DSL_TOOL_DIR` and `LLVM_TOOL_DIR`), then directories listed in `SAR_DSL_TOOL_PATH`;
3. `PATH`.

## Environment variables

| Variable | Effect |
| --- | --- |
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
| `OMP_NUM_THREADS` | OpenMP team limit for generated CPU loops; also the FFT/interpolation pool fallback when `SAR_RT_NUM_THREADS` is unset |
| `SAR_DSL_HLS_CONFIG` | HLS configuration file overriding the shipped defaults |
| `SAR_DSL_HLS_TESTBENCH_MAX_BYTES` | Ceiling on the static arrays a generated testbench declares (default 1 GiB); `write_testbench(..., max_bytes=0)` explicitly disables the guard for a production-scale simulation |
| `SAR_RT_NUM_THREADS` | FFT/interpolation runtime participants, including the calling thread; falls back to `OMP_NUM_THREADS`, then `min(process affinity, 32)`, and does not itself limit generated OpenMP loops |
| `SAR_RT_POOL_MAX_BYTES` | Plane-pool retention bound in bytes; `0` disables pooling (see "Buffer pool") |

## HLS configuration

Every knob the HLS backend hands to the passes has a default in `python/sar/backends/hls/hls_config.yaml`, which ships with the package. Configuration files are flat YAML mappings of option names to scalar values. Resolution order, weakest first:

```
hls_config.yaml  <  user config file  <  compile options
```

The user file comes from `options={"config": "device.yaml"}` or, failing that, `$SAR_DSL_HLS_CONFIG`; it overrides the keys it names and leaves the rest at the shipped default. The resolved set is what the design reports:

```python
design = kernel.compile(backend="hls",
                        options={"interface": "axi", "axi_bus_bits": 256})
print(design.config.axi_bus_bits)     # 256
print(dict(design.config))            # every key, as compiled
print(design.config.provenance)       # and who decided each one
```

Every value is validated. An unknown key, a value of the wrong type and a value outside the range the passes accept all raise `HLSConfigError` (a `sar.SARError` and a `ValueError`), naming the file or the option that carried it -- an option the user believed in and the backend ignored is the failure this schema exists to prevent.

### Constraints

These are facts about the device and the deliverable. The compiler cannot discover them, so they are the schema the user actually sets. Defaults target `xcvu13p-fhgb2104-2-i` and budget 80% of each resource, leaving 20% for control, interconnect and placement. Memory placement charges whole primitives; DSP is checked from the synthesis report because operator binding happens inside Vitis.

#### Stating the resource budgets

The six budgets and `part` describe one device, so compile options state one or the other, never both.

**Name the part**, and the device table sizes every budget. `utilization` is how much of the device the design may claim; it defaults to the shipped 80%, leaving the rest for control, interconnect and placement:

```python
kernel.compile("hls", options={"part": "xczu9eg-ffvb1156-2-e"})
kernel.compile("hls", options={"part": "xcu280-fsvh2892-2L-e",
                               "utilization": 60})
```

```python
design.config.bram_bytes                  # 3359232 -- 80% of the Zynq
design.config.uram_bytes                  # 0 -- the family has no UltraRAM
design.config.provenance["bram_bytes"]    # 'derived'
```

`sar.backends.hls.devices.DEVICES` is that table. It also supplies the primitive geometry the storage estimates round to, so a family without UltraRAM charges banks in block RAM rather than assuming the reference part's 288 Kb blocks. A part the table does not carry is an error naming the alternative below, not a silent fallback to another device's sizing.

**Or state the budgets outright**, and they are used as written — for a device the table does not carry, or to plan against something other than a whole part:

```python
kernel.compile("hls", options={"bram_bytes": 4 << 20, "uram_bytes": 0,
                               "lutram_bytes": 1 << 16, "dsp": 512,
                               "ff": 200000, "lut": 100000})
```

Naming a part _and_ a budget is refused rather than ranked: a part says the budgets are the device's, a budget says they are the caller's, and keeping one of the two would plan against a device that is partly each.

With neither, `hls_config.yaml` applies as it stands — its six values are the reference VU13P at 80%, and a project can replace the file wholesale through `options={"config": path}` or `$SAR_DSL_HLS_CONFIG`.

| Key | Default | Controls |
| --- | --- | --- |
| `bram_bytes` | `9907200` | 2150 of 2688 BRAM36 primitives; hard cap |
| `uram_bytes` | `37748736` | 1024 of 1280 UltraRAM primitives; hard cap |
| `lutram_bytes` | `2883584` | 80% of the distributed-RAM capacity; hard cap |
| `dsp` | `9830` | 80% DSP synthesis-report budget |
| `ff` | `2764800` | 80% flip-flop synthesis-report budget |
| `lut` | `1382400` | 80% lookup-table synthesis-report budget |
| `interface` | `axi` | Protocol the top-function ports speak: `ap_memory` (plain arrays), `axi` (AXI4 memory-mapped masters), or `stream` (AXI4-Stream, only for kernels that sweep every port once in row-major order) |
| `axi_bus_bits` | `512` | Data width of the AXI masters, in bits |
| `axi_max_burst_length` | `64` | Maximum beats per burst, also capped at the AXI 4 KiB boundary |
| `axi_max_outstanding` | `16` | Maximum bursts in flight per direction (Vitis caps it at 32) |
| `precision` | `native` | Data path the kernel must carry: `native`, `f32` or `f64` |
| `part` | `xcvu13p-fhgb2104-2-i` | Device part the generated Vitis scripts name; naming one in compile options derives the six budgets from it |
| `utilization` | `80` | Percent of a named part the compiler may plan against; applies only when compile options name a `part` |
| `clock_ns` | `4.0` | Target clock period of the generated scripts |
| `clock_uncertainty_percent` | `12.5` | Portion of `clock_ns` reserved as the Vitis scheduling/place-and-route margin; the effective default budget is 3.5 ns |
| `top_func` | `null` | Name of the emitted top function; `null` takes the kernel's name. C++ identifiers that are HDL keywords receive a `sar_` prefix so the generated `set_top` remains legal |

`axi_bus_bits`, `axi_max_burst_length` and `axi_max_outstanding` are a buffering budget rather than a per-port setting: the emitter shapes each port from the buffer it serves -- a beat cannot widen past the contiguous run, a burst cannot outrun the row -- and a port whose bursts come out shorter than `axi_max_burst_length` gets proportionally more of them in flight, up to the Vitis cap.

The caps are the whole constraint surface: placement charges each tier in whole primitives (a 36 Kb block holding one kilobyte is spent), overflows one block tier into the other before spilling anything, and fails the design when a buffer that cannot stream fits no tier. Vitis binds operators and control after emission, so DSP, FF, and LUT budgets are validated from `*_csynth.xml`; `precision` is the pre-synthesis lever when datapath pressure matters.

`clock_ns` is the one constraint that is not a cap. The compiler and generated scripts optimize against `clock_ns * (1 - clock_uncertainty_percent / 100)`; the full period remains the board clock used to convert cycles to seconds. Resources decide whether a design can be placed on the device at all, so overrunning one aborts the flow; the period is what the compiler optimizes toward, and the synthesis figure is a pre-route estimate. A design that misses it is therefore reported -- by the generated script, by `run_hls_resources.py --reports`, and in `benchmarks/`'s recorded results -- and left for the project to weigh against its own margin. Only the resource budgets bound what the tuner may spend.

`precision` is a gate, not a conversion: the declared dtypes _are_ the data path (see "Precision contract"), so `precision="f32"` rejects a kernel carrying f64 planes rather than narrowing it -- an f64 butterfly costs several times the DSPs and block RAM of an f32 one, and nothing downstream reports it.

### What the compiler decides for itself

Optimization strategy is not configured. FFT grouping and row parallelism, loop tiling, banking, banded gathers, and buffer thresholds are derived from the constraints above and from what the compiler measures in the kernel: plane sizes, transform lengths, element widths, buffer lifetimes. They are absent from `hls_config.yaml` because no value written there would be a better guess than the computed one.

`design.config` reports what was chosen and `design.config.provenance` marks those keys `derived`. The policy lives in `python/sar/backends/hls/autotune.py`.

The values split by how the compiler arrives at them, and a generated manifest reports them under those two headings (`optimization_plan.tuned` and `.fixed`).

**Tuned** — modelled per kernel against the resource budgets. This is the search space where the latency/area trade-off is actually made: each of these can move when the kernel or the budgets do.

| Tuned | From |
| --- | --- |
| `fft_stage_group` | the least grouping whose Stockham scratch fits the working share of the on-chip budget; full unroll where it fits |
| `fft_parallel_rows` | lanes per prefetched line block, bounded by DSP, by the transform storage ceiling, and by the routing cap implied by `clock_ns` |
| `fft_io_unroll` | elements per FFT transfer, bounded by one full per-plane beat and the validated vector-lane cap; halved further if the placed design overruns the on-chip caps |
| `fuse_sibling_sweeps` | fuse identical affine scans before allocation reuse, except in multi-regrid graphs where the larger live ranges lose in synthesis |
| `max_unrolled_ops` / `max_unroll_factor` | source-size guard for pipelined inner loops; gather-heavy graphs use a tighter derived budget to reduce Vitis scheduling complexity |
| `external_vector_max_lanes` | common packed width supported by every scalar plane type; multi-gather graphs use a bounded four-lane scratch representation |
| `external_vector_pack_outputs` | keep public output arrays scalar in multi-gather graphs while scratch and read-only inputs use bounded packed words |
| `external_vector_compute_lanes` | unroll only a resource-bounded subset of lanes inside a packed transfer word for gather-heavy loops |
| `external_vector_min_elements` | minimum array size that amortizes changing the physical AXI ABI |
| `loop_tile_size` | the element count in one bus beat, bounded by the pass limits |
| `lutram_max_bytes` | one bus beat: a bank no larger than one transfer does not earn a block RAM primitive |
| `array_partition_max_factor` | the largest power-of-two bank count no wider than one AXI beat; reduced automatically if final primitive accounting exceeds a cap |
| `interp_full_row_max_bytes` | one working-buffer share of BRAM+URAM divided across interpolation sites; a gather uses an immutable complete-row cache when its replicas fit and a narrow fully banked band is unavailable |
| `interp_cache_copies` | up to four band/row-cache replicas, following packed compute lanes; each lane reads its own copy so arbitrary indices cannot create conservative cross-lane bank conflicts |
| `interp_complete_bank_max_elements` | largest power-of-two band whose dynamic-read mux fits one quarter of LUT/FF budgets and the clock-dependent routing cap (16 elements at 4 ns); wider narrow bands use cyclic `taps × compute lanes` banking |
| `reuse_buffer_min_elements` | initially disables sharing, then moves to the full-plane streaming threshold only when the measured unshared lowering spills planes to DRAM; on-chip dataflow channels remain private |
| `recompute_min_elements` | a full-scene plane against what the budget can afford: storage traded for arithmetic |
| `external_buffer_threshold` | the measured working set against the budget, which is what lets scene size outgrow the device |

**Fixed** — the same for every kernel. Each encodes a property of the lowering or of the synthesis tool rather than a per-kernel choice, so it carries a rationale instead of a cost model, and changes only if that property does.

| Fixed | Value | Why it does not vary |
| --- | --- | --- |
| `interp_banded_gather` | on | Range proof, complete-row staging and the direct fallback are all semantics-preserving; enabling the ladder is a lowering capability rather than a per-kernel trade-off. |
| `max_scratch_arenas` | 4 | Read/write ping-pong for split scalar planes needs four masters; more would grow the platform interface with graph depth, which is what the arenas exist to prevent. |

Each is also accepted as a compile option. A pinned value reaches the passes unchanged, and `provenance` records that it was supplied by the user.
