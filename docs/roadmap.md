# Roadmap

The goal is a compiler that can express and compile most SAR / signal
processing algorithms. Coverage comes from a small set of orthogonal
primitives -- not from per-algorithm ops -- and every construct in the
language compiles to every backend.

## Done

- **Primitive completeness.** Complex access (`conj`/`real`/`imag`/
  `complex`), transcendentals (`exp`/`log`/`atan2`), reductions
  (`sum`/`max`/`min`/`argmax`), selection (`where` + comparisons, the
  numpy masking idiom), layout (strided `slice`/`concat`/`pad`),
  axis-parametric interpolation with selectable kernels (`nearest`/
  `linear`/`cubic`/windowed `sinc` with `rect`/`hann`/`hamming`/`kaiser`
  tapers, on both cpu and HLS paths), FFTs of any size >= 2 on both
  backends (radix-2 where the size allows, Bluestein's chirp-z reduction
  otherwise).
- **Numeric interop.** `cast` bridges int and float tensors, so
  `argmax`/`argmin` indices participate in kernel arithmetic;
  `sign`/`floor`/`ceil`/`round` and numpy `norm=` conventions on
  `fft`/`ifft` are trace-time compositions.
- **Minimal IR, rich surface.** Everything expressible as a composition
  traces to primitives in Python instead of owning an IR op: `expj`,
  `angle`, negation, `maximum`/`minimum`/`clip`, `multilook`, and the
  omega-K Stolt remapping itself (position computation + `interp1d` +
  phase ramps).
- **Operator definition in Python.** `@sar.op` declares operators as
  compositions of primitives: they inline into kernels at trace time,
  fuse with their surroundings and compile to every backend
  ([defining-ops.md](defining-ops.md)). No per-backend native code;
  what composition cannot express is a primitive gap, tracked below.
- **Matlab/scipy-familiar vocabulary.** `fft2`/`ifft2`, `flip`,
  `circshift`, `mean`/`std`/`var`, dB conversions (both directions),
  `sinc`, `hypot`, `dechirp`,
  `matched_filter` and window constants (incl. Taylor), all decomposing
  into IR primitives at trace time.
- **Domain diagnostics.** Trace-time per-axis time/frequency tracking
  warns on double FFTs, inverse transforms of centered spectra and
  mixed-domain arithmetic (`sar.DomainWarning`).
- **HLS designs csim bit-exact.** All four imaging chains pass their
  generated C-simulation testbenches. Two bugs in the derived HLS passes were
  root-caused and fixed in our fork: the HLS C++ emitter printed
  floating-point constants with 6 decimals (quantizing FFT twiddles
  and flushing small phase coefficients to zero), and multi-consumer
  forking redirected reads across buffer redefinitions to stale data.
- **Scene size bounded by DRAM, not by the device.** The full
  16384 x 16384 ALOS raster emits as a single design
  (`examples/wka/run_alos_hls.py`). The backend budgets on-chip
  memory itself: buffers stay resident while the working set fits
  `on_chip_budget`, and past it the full-size planes -- including the
  FFT scratch -- move behind AXI masters, leaving only the constant
  tables on chip. Getting there took four fixes in the fork: subview
  types now follow their source's memory space through buffer
  placement, the unfinished DRAM depth/tap path gave way to the general
  buffer chain, loop tiling was re-enabled in the C++ pipeline, and AXI
  ports of one element type now share a bundle instead of each taking
  its own. The Stockham lowering also lost its copy-in and copy-out
  passes, reading the input and writing the result directly.
- **The HLS dialect moved in-tree.** What the pipeline actually uses --
  the dialect, 32 passes and the C++ emitter -- now lives under
  `lib/Dialect/HLS` and `lib/Target/HLS`, built by the same CMake as the
  rest. The TOSA and PyTorch frontends, design-space exploration and QoR
  estimation went with the submodule; `sar-opt` and `sar-translate`
  replaced the two vendor tools.

## Next

- Scan and order-statistic primitives, the known expressiveness holes:
  no `cumsum` (azimuth integration in motion compensation) and no
  `sort`/`median` (speckle median filtering, CFAR detection) can be
  built from the current primitives. Each needs a real IR op (a scan
  and a windowed rank filter) with lowerings on both backends, keeping
  every DSL construct backend-symmetric.
- Compiled counted loops with tensor carries (sub-aperture and block
  processing). Requires an HLS lowering story for loop nests with
  tensor carries so the construct stays backend-symmetric.
- Generalized gather beyond per-line interpolation, towards time-domain
  backprojection.
- Interpolation boundary policies beyond zero (edge replication for
  image-domain resampling); global interpolators (splines) are not
  local gathers and would be a separate primitive.
- Bit-exact spectral rewrites in canonicalization (shift calculus
  extensions); reassociating rewrites (phase-multiply merging) behind an
  explicit accuracy budget.

## Non-goals (for now)

- Data-dependent control flow inside kernels (autofocus-style iteration
  stays host-orchestrated; the Vr calibration workflow is the pattern).
- Dynamic shapes: one compile per geometry is the model.
- Fixed-point HLS types.
- ML-flavored ops: every op must have signal-processing semantics and
  naming; upstream lowering convenience alone does not justify one.
