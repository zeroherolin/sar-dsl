# Imaging examples

Four complete SAR imaging algorithms, each following the same layout so
they can be compared side by side (`pfa` doubles as the showcase
for Python-defined operators):

```
examples/
├── common/            shared infrastructure
│   ├── params.py      RadarParams, ALOS_PARAMS, alos_params, synthetic_params
│   ├── simulate.py    point-target echo simulator
│   ├── quality.py     focus-quality / scene-contrast reporting
│   ├── hls_artifacts.py  artifact set of the ALOS-scale HLS runners
│   ├── plot.py        dB-scale image saving
│   └── alos.py        ALOS-1 raw loading and scene post-processing
├── data/              shared dataset: CEOS extraction tooling + ALOS product
├── wka/               omega-K (wavenumber domain)
├── rda/               Range-Doppler
├── csa/               Chirp Scaling (interpolation-free)
└── pfa/               Polar Format built from Python-defined operators
```

Every algorithm directory contains:

| File | Purpose |
|------|---------|
| `README.md` | the algorithm's own notes and result images |
| `algorithm.py` | the imaging chain in the DSL: `build_kernel(n, params, name=..., dtype=...)`, `make_inputs(n, params)`; `dtype=sar.c64` builds the single-precision variant |
| `reference.py` | NumPy reference implementation (numerical ground truth) |
| `run_point_target_cpu.py` | **full flow on the cpu backend**: simulate, compile, focus, measure, save a PNG |
| `run_point_target_hls.py` | **full flow on the hls backend**: emit a Vitis HLS C++ design plus its C-simulation package |

`pfa` additionally carries `geometry.py` (polar-grid setup).

Runner scripts are named `run_<scene>_<backend>.py`: the scene is either
`point_target` (a synthetic scene generated on the spot) or `alos` (the
real dataset), and the backend is `cpu` or `hls`. Every runner takes
`--output`; every runner whose raster is a free choice takes `--n`, and
the HLS runners take `--no-testbench`:

```bash
# from the repository root, after `make build`; PYTHONPATH must include python/
PYTHONPATH=python python3 examples/wka/run_point_target_cpu.py --n 512
PYTHONPATH=python python3 examples/wka/run_point_target_hls.py --n 256
PYTHONPATH=python python3 examples/rda/run_point_target_cpu.py
PYTHONPATH=python python3 examples/csa/run_point_target_hls.py
```

The CPU runners print where the brightest target landed and its impulse
response; the HLS runners write a package that C-simulates against
golden data from the NumPy reference.

## Point-target results

`--n 512` synthetic geometry for the stripmap chains, `--n 512`
spotlight collection for PFA. Peak position error is `(0, 0)` for all
four:

| | range IRW | range PSLR | azimuth IRW | azimuth PSLR | csim max abs err (n=256) |
|---|---|---|---|---|---|
| omega-K | 2.12 | -31.7 dB | 2.20 | -27.1 dB | 2.3e-10 |
| Range-Doppler | 2.12 | -31.9 dB | 2.20 | -27.2 dB | 0.0 (bit-exact) |
| Chirp Scaling | 2.12 | -31.7 dB | 2.20 | -27.2 dB | 2.3e-10 |
| PFA (uniform) | 1.80 | -13.3 dB | 1.80 | -13.3 dB | 1.9e-15 / 3.8e-15 |

The PSLRs here are measured on the runners' multi-target scene, where
neighbouring targets' sidelobes overlap; on the single-target benchmark
scene the same stripmap chains measure ~-38.4 dB
([benchmarks/README.md](../benchmarks/README.md#image-quality)).

Sidelobe control differs by design: the three stripmap chains apply
the textbook band-matched Hann taper, while PFA demonstrates 2-D SVA --
per-pixel data-dependent weighting that nulls the sinc sidelobes of its
uniformly weighted, 2x-oversampled image (PSLR -13.3 dB -> -23.3 dB with
no mainlobe broadening).

Each `assets/<algo>_synthetic_512.png` regenerates with its chain's CPU
runner, e.g. `python examples/wka/run_point_target_cpu.py --n 512
--output examples/wka/assets/wka_synthetic_512.png`.

## Real data (ALOS-1)

The three stripmap algorithms also process the real ALOS-1 San Francisco
dataset (16384 x 16384 raw echoes, extracted once via
`data/extract_alos.py`; original data © JAXA/METI, obtained from JAXA --
the product itself is not part of this repository). Measured on a
240-core machine, urban-area image contrast from
`benchmarks/metrics.py:urban_contrast` (higher is sharper):

| | focus time | urban contrast |
|---|---|---|
| omega-K | 4.6 s | 0.810 |
| Chirp Scaling | 3.0 s | 0.809 |
| Range-Doppler | 3.5 s | 0.808 |

Each also has a `run_alos_hls.py`, which emits **two** designs into
`hls_project/<algo>_alos/`, because no single design can be both
synthesizable at scene size and simulatable:

- `<algo>_alos_axi.cpp` -- the 16384 x 16384 design, compiled with
  `axi_interface=True`. Every full-size buffer, including the FFT
  scratch planes, sits behind an AXI master backed by DRAM and only the
  constant tables stay on chip; see
  [docs/backends.md](../docs/backends.md) for how that placement is
  decided. It is emitted for synthesis and not simulated: the promoted
  ports are kernel scratch, and golden data exists only for the kernel's
  own inputs and results.
- `<algo>_alos.cpp` with `<algo>_alos_tb.cpp`, the csim/csynth scripts,
  `<algo>_alos_tb_data/` and `stubs/` -- a C-simulation package at
  `--csim-n` (default 1024), small enough to keep every plane on chip so
  the top function is the kernel's own signature.

Both designs come with a `_csynth.tcl` synthesis script; the validation
checklist for reading the reports is in
[docs/backends.md](../docs/backends.md#validating-a-design-in-vitis-hls).

The reduced package is the same radar: `common.params.alos_params`
changes only `t_shift`, where R0 sits in the sampled window, because a
window shorter than 4800 samples would not contain the echo at all.
`fc`, `Fs`, PRF, `Vr`, `R0`, `Kr` and `Tp` are the acquisition's at
every size, and the frequency axes span +/- Fs/2 and +/- PRF/2
regardless of raster, so the Stolt map and the migration corrections are
evaluated with the real constants. What shrinks is the sampled window
and the dwell, so the target focuses at coarser azimuth resolution than
the full aperture gives. At 1024 x 1024 all three csim-match the NumPy
reference:

| | csim max abs err (1024 x 1024) |
|---|---|
| omega-K | 4.7e-10 |
| Range-Doppler | 1.9e-09 |
| Chirp Scaling | 3.7e-09 |

## Why PFA has no ALOS runner

PFA is a spotlight algorithm: it assumes the antenna stares at one
fixed scene center, and the ALOS-1 product here is a stripmap
collection with no common scene center to reformat around. The full
argument -- and why a sub-aperture decomposition would be a different
algorithm -- is in [pfa/README.md](pfa/README.md); the directory
synthesizes its spotlight phase history in `geometry.py` instead.

All four algorithms are validated in `test/python/`: DSL vs reference
equivalence, point-target focusing quality, cross-algorithm agreement on
target positions, and the artifact set each HLS runner emits.
