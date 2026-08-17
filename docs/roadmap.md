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
  tapers) and boundary policies (`zero`/`edge`/`reflect`), on both cpu
  and HLS paths, FFTs of any size >= 2 on both backends (radix-2 where
  the size allows, Bluestein's chirp-z reduction otherwise).
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
  mixed-domain arithmetic (`sar.DomainWarning`). Element-wise operands
  that agree on domain but disagree on centering are diagnosed too: the
  phase reference then differs by a half-band rotation, which corrupts
  the product silently. Warnings are attributed to the nearest frame
  outside the package, so the report lands on the kernel line whatever
  path reached the op (`_warn_user`).
- **Banded interpolation gathers.** Displacement-range analysis
  (`lib/Analysis/DisplacementRange.cpp`) bounds `|positions - j|` at compile
  time -- symbolically for affine position fields, and by folding the plane
  when it is constant, which is what lets it see through the `sqrt` in a
  Stolt remapping. A bounded field gathers through a narrow on-chip band
  rather than materializing the whole source plane; unbounded fields keep
  the full-plane path.
- **HLS designs csim-validated.** All four imaging chains pass their
  generated C-simulation testbenches against the NumPy reference, at
  both build precisions.
- **Scene size bounded by DRAM, not by the device.** The full
  16384 x 16384 ALOS raster emits as a single design
  (`examples/wka/run_alos_hls.py`). The backend budgets on-chip
  memory itself: buffers stay resident while the working set fits
  `on_chip_budget`, and past it the full-size planes -- including the
  FFT scratch -- move behind AXI masters, leaving only the constant
  tables on chip.
- **Scan and order-statistic primitives.** `cumsum` (azimuth integration in
  motion compensation) and `rank_filter`/`median_filter` (speckle median
  filtering, CFAR detection) are IR ops with lowerings on both backends.
  The scan is sequential -- a loop nest carrying the running sum through
  the result buffer. The rank filter sorts its window with a
  compare-exchange network unrolled over the static window size, so the
  body stays straight-line.
- **The HLS dialect is in-tree.** The dialect, its transform passes and
  the C++ emitter live under `lib/Dialect/HLS` and `lib/Target/HLS`,
  built by the same CMake as the rest and driven by `sar-opt` and
  `sar-translate`; no vendor tools or second toolchain are involved.
- **HLS config system with auto-derived strategy.** `hls_config.yaml`
  ships device-specific on-chip budgets (BRAM / URAM / LUTRAM tiers for
  Virtex UltraScale+ VU13P); `autotune.py` derives the optimization
  strategy (stage grouping, tiling, banded gather, on-chip placement) from
  the kernel itself so the user needs to supply only constraints, not
  strategy. Both are tested in `test/python/test_hls_config.py` and
  `test/python/test_hls_autotune.py`.
- **Compiled counted loops with tensor carries.** `sar.iterate` applies a
  body `trips` times feeding each iteration's results to the next, and
  stays a single loop in the design instead of unrolling at trace time.
  On the CPU path it lowers through `scf.for` tensor carries; on the HLS
  path `sar-demote-loop-carries` turns the carry into side effects (the
  body iterates in the init buffer, a per-iteration copy replaces the
  yield), since a dataflow task may not yield values. The body cannot
  yet observe the iteration index, so per-iteration addressing
  (sub-aperture slicing) remains host-orchestrated.
- **2-D gather (`sar.gather2d`).** Both coordinates data-dependent --
  the access pattern of time-domain backprojection -- with nearest and
  bilinear kernels and zero/edge boundaries. Lowers through one
  linalg.generic on both backends; the gather loads are clamped and
  select-masked like the interpolation's, so the body stays
  straight-line. The full source plane stays resident (or streams);
  a banded variant needs the displacement analysis generalized to 2-D.

## Next

- Iteration-index access inside `sar.iterate` (per-iteration slicing for
  sub-aperture and block processing), which needs dynamic offsets in the
  slice lowering.
- Scratch carving through loop carries: a carried buffer that spills to
  DRAM currently keeps its own AXI port (with a warning) instead of
  joining the scratch allocation, because the carving cannot yet redirect
  accesses through an `scf.for` region.
- A banded variant of `sar.gather2d`, generalizing the 1-D
  displacement-range analysis to both coordinates.
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
