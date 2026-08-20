# Hand-written Vitis HLS omega-K reference

This directory contains a self-contained FP32 omega-K implementation written
directly for Vitis HLS 2022.2. It follows the same WKA semantics and ALOS
parameters as the parent SAR-DSL example, but it is not compiler output and is
not a dependency of SAR-DSL. It serves as an independent baseline: a
hand-tuned design at the same geometry, constraints and device that the
generated one can be measured against.

The production contract is fixed at:

| Constraint | Value |
|------------|-------|
| Raster | 16384 × 16384 |
| Samples | complex FP32 |
| Device | `xcvu13p-fhgb2104-2-i` |
| Clock | 4 ns |
| External interface | 512-bit AXI |

## Design

The top function reuses one row-transform engine and one corner-turn engine
across an eight-pass schedule:

```text
range FFT → transpose → azimuth FFT → transpose
→ bulk compression + Stolt + range window + range IFFT
→ transpose → azimuth window + azimuth IFFT → magnitude transpose
```

The design rests on:

- 512-bit packed complex I/O with contiguous bursts and explicit outstanding
  transaction limits;
- three full-frame DDR buffers with a fixed ping-pong lifetime schedule;
- one radix-4 FFT engine reused across all transforms, processing 16 rows in
  parallel at production size;
- banked URAM line buffers and shared ROM twiddle/window tables;
- a fused bulk-compression, Stolt-interpolation, window, and inverse-transform
  row pass;
- a 256 × 256 packed corner turn with BRAM ping-pong tiles and load/store
  `DATAFLOW`;
- reduced C-simulation and RTL co-simulation against an independent NumPy
  reference, separate from production synthesis.

## Source layout

| Path | Purpose |
|------|---------|
| `wka_top.cpp` | Eight-pass schedule and top-level AXI interfaces |
| `fft_core.cpp` | Reusable multi-row radix-4 FFT/IFFT engine |
| `stolt_interpolation.cpp` | Bulk phase, Stolt interpolation, and windows |
| `corner_turn.cpp` | Packed, banked, ping-pong transpose |
| `config.h` | Size profiles, radar constants, and hardware constraints |
| `generated/wka_luts.h` | ROMs created deterministically by `make luts` |
| `reference/wka_reference.py` | Independent NumPy functional reference |
| `hls/run_hls.tcl` | C-sim, synthesis, and RTL co-simulation flow |

## Run

Vitis HLS is optional for the rest of SAR-DSL but required for this directory.
Run the targets from this directory:

```bash
make csim-smoke       # N=64 end-to-end C-simulation
make cosim-smoke      # N=64 Verilog C/RTL co-simulation
make csynth-corner    # production corner-turn module
make csynth-row       # production row engine
make csynth-top       # complete N=16384 design
make reports          # collect XML summaries
```

`make luts` regenerates `generated/wka_luts.h`. Build products go under
`work/`; generated test vectors and report collections go under `reports/`.

The production testbench can read
`data/alos_raw_16384x16384.bin`. The dataset is intentionally not included;
use the [shared extractor](../../data/extract_alos.py) to create it from the
ALOS CEOS product.

```bash
mkdir -p data
python ../../data/extract_alos.py --input <IMG-file> \
  --output data/alos_raw_16384x16384.bin
```

## Measured production synthesis

The checked configuration completes Vitis HLS 2022.2 synthesis:

| Metric | Result |
|--------|-------:|
| Estimated clock | 3.500 ns |
| Best / worst latency | 940,925,067 / 2,033,373,247 cycles |
| BRAM18K / URAM | 384 / 296 |
| DSP / FF / LUT | 1,135 / 210,325 / 373,962 |
| `csynth_design` elapsed time | 201.24 s |

These are HLS estimates, not post-route or board measurements. The
machine-readable constraints and top-level report values are in
`reports/production_csynth.json`.
