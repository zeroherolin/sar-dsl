# Imaging examples

Three complete SAR imaging algorithms, each following the same layout so
they can be compared side by side:

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
└── csa/               Chirp Scaling (interpolation-free)
```

Every algorithm directory contains:

| File | Purpose |
|------|---------|
| `algorithm.py` | the imaging chain in the DSL: `build_kernel(n, params)`, `make_inputs(n, params)` |
| `reference.py` | NumPy reference implementation (numerical ground truth) |
| `run_cpu.py` | **full flow on the cpu backend**: simulate, compile, focus, save a PNG |
| `run_scalehls.py` | **full flow on the scalehls backend**: trace and emit a Vitis HLS C++ design |

Each `run_*.py` is self-contained -- one file, one backend, whole flow:

```bash
# from the repository root, after `make build` (and `make scalehls` for HLS)
python examples/wka/run_cpu.py --n 512
python examples/wka/run_scalehls.py --n 256
python examples/rda/run_cpu.py
python examples/csa/run_scalehls.py
```

All three algorithms also process the real ALOS-1 San Francisco dataset
(16384 x 16384 raw echoes, extracted once via `data/extract_alos.py`):
each directory has a `run_alos.py`. Urban-area image contrast at full
size: omega-K 131.6, Range-Doppler 130.3, Chirp Scaling 126.1 -- all
focus the scene in ~3.5 s on a large multi-core CPU.

All three algorithms are validated in `test/python/`: DSL vs reference
equivalence, point-target focusing quality, and cross-algorithm agreement
on target positions.
