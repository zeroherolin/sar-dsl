# omega-K (WKA) imaging with SAR-DSL

The complete wavenumber-domain (omega-K / WKA) SAR imaging algorithm as a
single SAR-DSL kernel, validated against a NumPy reference and
demonstrated on both synthetic point targets and a real ALOS-1 dataset.

| File | Purpose |
|------|---------|
| `algorithm.py` | The WKA chain in the DSL (`build_kernel`, `make_inputs`) |
| `reference.py` | NumPy reference implementation (`WKAProcessor`) |
| `run_point_target_cpu.py` | Full cpu-backend flow: simulate, focus, save a PNG |
| `run_point_target_hls.py` | Full hls-backend flow: HLS C++ design + csim package (`hls_project/`) |
| `run_alos_cpu.py` | Focus the real ALOS-1 San Francisco dataset (16384x16384) |
| `run_alos_hls.py` | Emit a synthesizable design at the full ALOS raster |
| `assets/` | Reference imagery |

## Processing chain

```
raw --> range FFT --> corner turn --> azimuth FFT --> corner turn
    --> bulk compression (reference phase multiply)
    --> Stolt interpolation (sar.stolt_interp)
    --> range window --> azimuth window
    --> range IFFT --> corner turn --> azimuth IFFT --> corner turn --> |.|
```

Everything runs inside one compiled kernel; the host only passes the
band-matched Hann tapers as inputs. Acquisition metadata -- scalar radar
parameters (`fc`, `Vr`, `R0`, `Kr`, ...) and the frequency axes -- bakes
into the IR at trace time.

## Synthetic scene (either backend)

```bash
# from the repository root, after `make build`
python examples/wka/run_point_target_cpu.py --n 512          # focus + PNG
python examples/wka/run_point_target_hls.py --n 256     # design + csim package
```

Each simulated point target focuses to roughly one resolution cell:

![synthetic point targets](assets/wka_synthetic_512.png)

## Real data (ALOS-1)

```bash
# from the repository root; the dataset is shared by all three algorithms
python examples/data/extract_alos.py   # CEOS L1.0 -> alos_raw_...bin (2 GiB)
python examples/wka/run_alos_cpu.py        # tens of GiB of RAM at full size
python examples/wka/run_alos_hls.py   # emit the design at the same size
```

The full 16384x16384 scene focuses in about 3.6 seconds on a large
multi-core machine (fused element-wise kernels under OpenMP; multithreaded
FFT/interpolation runtime). The effective radar velocity in `ALOS_PARAMS` is
autofocus-calibrated by image-contrast maximization over the urban area:

![San Francisco Bay](assets/san_francisco_wka.png)
