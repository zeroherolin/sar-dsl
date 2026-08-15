# Imaging examples

Four complete SAR imaging algorithms, each following the same layout so
they can be compared side by side (`pfa` doubles as the showcase
for Python-defined operators):

```
examples/
├── common/            shared infrastructure
│   ├── params.py      RadarParams, ALOS_PARAMS, synthetic_params
│   ├── simulate.py    point-target echo simulator
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
| `algorithm.py` | the imaging chain in the DSL: `build_kernel(n, params)`, `make_inputs(n, params)` |
| `reference.py` | NumPy reference implementation (numerical ground truth) |
| `run_point_target_cpu.py` | **full flow on the cpu backend**: simulate, compile, focus, save a PNG |
| `run_point_target_hls.py` | **full flow on the hls backend**: emit a Vitis HLS C++ design plus its C-simulation testbench |

Runner scripts are named `run_<scene>_<backend>.py`: the scene is either
`point_target` (a synthetic scene generated on the spot) or `alos` (the
real dataset), and the backend is `cpu` or `hls`. Each one is
self-contained -- one file, one scene, one backend, whole flow:

```bash
# from the repository root, after `make build` (and `make build` for HLS)
python examples/wka/run_point_target_cpu.py --n 512
python examples/wka/run_point_target_hls.py --n 256
python examples/rda/run_point_target_cpu.py
python examples/csa/run_point_target_hls.py
```

The three stripmap algorithms also process the real ALOS-1 San
Francisco dataset
(16384 x 16384 raw echoes, extracted once via `data/extract_alos.py`):
each directory has a `run_alos_cpu.py`. Urban-area image contrast at full
size (`benchmarks/metrics.py:urban_contrast`, higher is sharper):
omega-K 0.810, Chirp Scaling 0.809, Range-Doppler 0.808 -- all focus
the scene in ~3.5 s on a large multi-core CPU.

`wka/run_alos_hls.py` emits a synthesizable design at that same
16384 x 16384 raster. It compiles with `axi_interface=True`, which puts
every full-size buffer -- including the FFT scratch planes -- behind an
AXI master backed by DRAM and leaves only the constant tables on chip;
see [docs/backends.md](../docs/backends.md) for how that decision is
made. It emits the design only: a csim package needs golden data for
every top-level port, and the promoted scratch ports have none, so the
arithmetic is validated by the point-target packages instead.

Sidelobe control differs by design: the three stripmap chains apply
the textbook band-matched Hann taper (first sidelobes at ~-38 dB, so
faint cross arms remain visible at a -60 dB display), while PFA
demonstrates 2-D SVA -- per-pixel data-dependent weighting that nulls
the sinc sidelobes outright on its 2x-oversampled baseband image.

All four algorithms are validated in `test/python/`: DSL vs reference
equivalence, point-target focusing quality, and cross-algorithm agreement
on target positions.
