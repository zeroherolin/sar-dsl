# Imaging examples

Four complete SAR imaging algorithms, each following the same layout so they can be compared side by side (`pfa` doubles as the showcase for Python-defined operators):

```
examples/
├── common/            shared infrastructure
│   ├── params.py      RadarParams, ALOS_PARAMS, alos_params, synthetic_params
│   ├── simulate.py    point-target echo simulator
│   ├── quality.py     focus-quality / scene-contrast reporting
│   ├── hls_artifacts.py  artifact set of the ALOS-scale HLS runners
│   ├── plot.py        dB-scale image saving
│   └── alos.py        ALOS-1 raw loading and scene post-processing
├── data/              CEOS extraction tooling; local ALOS products are ignored
├── wka/               omega-K, plus a standalone hand-written HLS reference
├── rda/               Range-Doppler
├── csa/               Chirp Scaling (interpolation-free)
└── pfa/               Polar Format built from Python-defined operators
```

Every algorithm directory contains:

| File | Purpose |
| --- | --- |
| `README.md` | the algorithm's own notes and result images |
| `algorithm.py` | the imaging chain in the DSL: `build_kernel(n, params, name=..., dtype=...)`, `make_inputs(n, params)`; `dtype=sar.c64` builds the single-precision variant |
| `reference.py` | NumPy reference implementation (numerical ground truth) |
| `run_point_target_cpu.py` | **full flow on the cpu backend**: simulate, compile, focus, measure, save a PNG |
| `run_point_target_hls.py` | **full flow on the hls backend**: emit a Vitis HLS C++ design plus its validation package |

`pfa` additionally carries `geometry.py` (polar-grid setup). `wka/handwritten_hls/` is deliberately separate from this common layout: it is a self-contained Vitis HLS implementation that documents hardware optimization patterns, not another SAR-DSL backend or compiler dependency.

Runner scripts are named `run_<scene>_<backend>.py`: the scene is either `point_target` (a synthetic scene generated on the spot) or `alos` (the real dataset), and the backend is `cpu` or `hls`. Every runner takes `--output`; every runner whose raster is a free choice takes `--n`. Point-target HLS runners may omit golden data with `--no-testbench`; ALOS HLS runners always emit the complete matching validation package:

```bash
# from the repository root, after `make build`; PYTHONPATH must include python/
PYTHONPATH=python python3 examples/wka/run_point_target_cpu.py --n 512
PYTHONPATH=python python3 examples/wka/run_point_target_hls.py --n 256
PYTHONPATH=python python3 examples/rda/run_point_target_cpu.py
PYTHONPATH=python python3 examples/csa/run_point_target_hls.py
```

The CPU runners print where the brightest target landed and its impulse response; the HLS runners write a package that runs C-sim through Vitis HLS against golden data from the NumPy reference. Without Vitis HLS, run the distinctly named `<top>_portable_cpp_sim.sh` fallback in the same package.

## Point-target results

`--n 512` synthetic geometry for the stripmap chains, `--n 512` spotlight collection for PFA. Peak position error is `(0, 0)` for all four:

| Algorithm | Range IRW | Range PSLR | Azimuth IRW | Azimuth PSLR | C-sim max abs err |
| --- | --- | --- | --- | --- | --- |
| omega-K | 2.12 | -31.7 dB | 2.20 | -27.1 dB | 2.3e-10 |
| Range-Doppler | 2.12 | -31.9 dB | 2.20 | -27.2 dB | 0.0 (bit-exact) |
| Chirp Scaling | 2.12 | -31.7 dB | 2.20 | -27.2 dB | 2.3e-10 |
| PFA (uniform) | 1.80 | -13.3 dB | 1.80 | -13.3 dB | 1.9e-15 / 3.8e-15 |

The PSLRs here are measured on the runners' multi-target scene, where neighbouring targets' sidelobes overlap; on the single-target benchmark scene the same stripmap chains measure ~-38.4 dB ([benchmarks/README.md](../benchmarks/README.md#image-quality)).

Sidelobe control differs by design: the three stripmap chains apply the textbook band-matched Hann taper, while PFA demonstrates 2-D SVA -- per-pixel data-dependent weighting that nulls the sinc sidelobes of its uniformly weighted, 2x-oversampled image (PSLR -13.3 dB -> -23.3 dB with no mainlobe broadening).

Each `assets/<algo>_synthetic_512.png` regenerates with its chain's CPU runner, e.g. `python examples/wka/run_point_target_cpu.py --n 512`.

## Real data (ALOS-1)

The three stripmap algorithms also process the real ALOS-1 San Francisco dataset (16384 × 16384 raw echoes, extracted once via `data/extract_alos.py`; original data © JAXA/METI, obtained from JAXA -- the product itself is not part of this repository). The runners report wall time and `benchmarks/metrics.py:urban_contrast`; machine-specific real-data measurements are not kept as reference numbers.

<div align="center">
<table>
<tr>
<td align="center" valign="top" width="33%"><img src="wka/assets/san_francisco_wka.png" width="100%" alt="ALOS-1 scene focused by omega-K"/></td>
<td align="center" valign="top" width="33%"><img src="csa/assets/san_francisco_csa.png" width="100%" alt="ALOS-1 scene focused by Chirp Scaling"/></td>
<td align="center" valign="top" width="33%"><img src="rda/assets/san_francisco_rda.png" width="100%" alt="ALOS-1 scene focused by Range-Doppler"/></td>
</tr>
<tr>
<td align="center">omega-K</td>
<td align="center">Chirp Scaling</td>
<td align="center">Range-Doppler</td>
</tr>
</table>
</div>

Each also has a `run_alos_hls.py`, which emits one self-contained package under `hls_project/<algo>_alos/`. The design, testbench, binary input/golden planes, manifest, C-sim/C-synth/C-RTL scripts, and portable fallback all use the same `--n` and top-level ports. The interface is not chosen by the runner: it comes from `python/sar/backends/hls/hls_config.yaml`, `$SAR_DSL_HLS_CONFIG`, or Python compile options. The shipped default is `axi`, so full-size planes and compiler-owned scratch use memory-mapped masters.

Python callers can override the same backend options without adding a second runner-specific configuration surface:

```python
from wka.run_alos_hls import emit
emit(n=1024, options={"interface": "ap_memory"})
```

The validation checklist for reading Vitis reports is in [docs/backends.md](../docs/backends.md#validating-a-design-in-vitis-hls). At the packaged raster all three reproduce the NumPy reference exactly:

| Algorithm     | C-sim max abs err |
| ------------- | ----------------- |
| omega-K       | 0.0               |
| Range-Doppler | 0.0               |
| Chirp Scaling | 0.0               |

## Why PFA has no ALOS runner

PFA is a spotlight algorithm: it assumes the antenna stares at one fixed scene center, and the ALOS-1 product here is a stripmap collection with no common scene center to reformat around. The full argument -- and why a sub-aperture decomposition would be a different algorithm -- is in [pfa/README.md](pfa/README.md); the directory synthesizes its spotlight phase history in `geometry.py` instead.

All four algorithms are validated in `test/python/`: DSL vs reference equivalence, point-target focusing quality, cross-algorithm agreement on target positions, and the artifact set each HLS runner emits.
