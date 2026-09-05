# Hand-written Vitis HLS omega-K reference

This directory contains a self-contained FP32 omega-K implementation written directly for Vitis HLS 2022.2. It follows the same WKA semantics and ALOS parameters as the parent SAR-DSL example, but it is not compiler output and is not a dependency of SAR-DSL. It serves as an independent baseline: a hand-tuned design at the same geometry, constraints and device that the generated one can be measured against.

The production contract is fixed at:

| Constraint         | Value                       |
| ------------------ | --------------------------- |
| Raster             | 16384 × 16384               |
| Samples            | complex FP32                |
| Device             | `xcvu13p-fhgb2104-2-i`      |
| Clock              | 4 ns                        |
| External interface | 9 arrays over 8 AXI masters |

## Design

The top function executes a statically selected eight-pass schedule:

```text
range FFT → transpose → azimuth FFT → transpose
→ bulk compression + Stolt + range window + range IFFT
→ transpose → azimuth window + azimuth IFFT → magnitude transpose
```

The design rests on:

- the same split-complex ABI as generated WKA: `raw_re`, `raw_im`, `win_r`, `win_a`, `out0`, and four scratch planes on bundles `axi_0` through `axi_7`;
- paired 256-bit real/imaginary plane transfers, equivalent to 512 bits of complex data per cycle, with contiguous bursts and explicit outstanding transaction limits;
- four full-frame scratch planes forming two complex ping-pong frames;
- fixed forward and inverse radix-4 engines, processing 16 rows in parallel at production size, with no runtime transform mode;
- banked URAM line buffers, shared ROM twiddles, and top-level window inputs cached on chip;
- a fused bulk-compression, Stolt-interpolation, window, and inverse-transform row pass; Stolt coordinates, Hann-sinc weights, and accumulation use f64 like generated WKA, while stored complex planes and FFT butterflies use f32;
- a 256 × 256 packed corner turn with BRAM ping-pong tiles and load/store `DATAFLOW`;
- full-size C-simulation and synthesis at 16384 × 16384, plus 256 × 256 RTL co-simulation against an independent NumPy reference.

At 256 × 256, the C++ result also agrees with the main example's `WKAProcessor` to `2.93e-7` NRMSE (correlation `0.99999999999996`). This cross-check prevents the standalone reference and the main project reference from drifting together unnoticed.

The comparison fixes external and numerical contracts rather than internal parallelism. Generated WKA uses eight FFT row lanes; the hand-written baseline uses sixteen. The resource table includes this microarchitecture difference.

## Source layout

| Path | Purpose |
| --- | --- |
| `wka_top.cpp` | Eight-pass schedule and top-level AXI interfaces |
| `fft_core.cpp` | Reusable multi-row radix-4 FFT/IFFT engine |
| `stolt_interpolation.cpp` | Bulk phase, Stolt interpolation, and windows |
| `corner_turn.cpp` | Packed, banked, ping-pong transpose |
| `config.h` | Size profiles, radar constants, and hardware constraints |
| `generated/wka_luts.h` | ROMs created deterministically by `make luts` |
| `reference/wka_reference.py` | Independent NumPy functional reference |
| `hls/run_hls.tcl` | C-sim, synthesis, and RTL co-simulation flow |

## Run

Vitis HLS is optional for the rest of SAR-DSL but required for this directory. Run the targets from this directory:

```bash
make hls_csim         # N=16384 C-sim through Vitis HLS
make cosim            # N=256 Verilog C/RTL co-simulation
make csynth           # complete N=16384 design
make csynth_corner    # production corner-turn module
make csynth_row       # production row engine
make reports          # collect XML summaries
```

`make luts` regenerates `generated/wka_luts.h`. Build products go under `work/`; generated test vectors and report collections go under `reports/`.

The production C-simulation reads the shared `examples/data/alos_raw_16384x16384.bin`. Download and unpack the matching [ALPSRP275140740-L1.0 ALOS-1 product from ASF DAAC](https://datapool.asf.alaska.edu/L1.0/A3/ALPSRP275140740-L1.0.zip), then use the [shared extractor](../../data/extract_alos.py) to create the raster from its CEOS image file. The expected layout and catalog link are in the [examples guide](../../README.md#alos-1-stripmap-data).

```bash
python ../../data/extract_alos.py --input <IMG-file> \
  --output ../../data/alos_raw_16384x16384.bin
```

## Measured production synthesis

The checked configuration completes Vitis HLS 2022.2 synthesis:

| Metric                       |                                 Result |
| ---------------------------- | -------------------------------------: |
| Estimated clock              |                               3.500 ns |
| Fixed latency                | 1,350,448,211 cycles (5.402 s at 4 ns) |
| Initiation interval          |                   1,350,448,212 cycles |
| BRAM18K / URAM               |                              672 / 848 |
| DSP / FF / LUT               |              2,860 / 769,377 / 658,256 |
| Array ports / AXI masters    |                                  9 / 8 |
| `csynth_design` elapsed time |                               419.16 s |

These are HLS estimates, not post-route or board measurements. The machine-readable constraints and top-level report values are in `reports/production_csynth.json`.
