# Polar Format Algorithm + SVA from Python-defined operators

The spotlight-SAR imaging algorithm with spatially variant apodization. Every stage is a `@sar.op` operator composed of built-in ops, including two pieces that go beyond straight op chaining:

- `polar_positions` computes the regrid interpolation positions _inside the kernel_ from the collection axes (`1/cos(theta)` outer products, `sar.atan2` to invert the pulse angle);
- `sva2d` is a data-dependent per-pixel weight selection (Stankwitz SVA, separable 2-D form: a range pass, then an azimuth pass), written in Python with the `sar.where` / comparison primitives -- algorithm logic that no fixed weighting can reproduce.

| File | Purpose |
| --- | --- |
| `algorithm.py` | The PFA + SVA chain as Python-defined operators (`build_kernel`, `make_inputs`) |
| `geometry.py` | Polar collection grid, inscribed target grid, phase-history simulation |
| `reference.py` | NumPy reference implementation (`PFAProcessor`) |
| `assets/` | Reference imagery |
| `run_point_target_cpu.py` | Full cpu-backend flow: simulate, focus, measure, save a PNG |
| `run_point_target_hls.py` | Full hls-backend flow: HLS C++ design + validation package (`hls_project/`) |

## Collection model

The stripmap examples ([wka](../wka/), [rda](../rda/), [csa](../csa/)) each carry `run_alos_cpu.py` and `run_alos_hls.py`; PFA carries neither, because ALOS-1 is a stripmap sensor.

PFA assumes a **spotlight** collection: the antenna stares at one fixed scene center, so every pulse samples the 2-D wavenumber support of the _same_ patch along a radial arc, and the collection is a polar sector that two 1-D interpolations reformat onto a rectangular grid. The ALOS-1 PALSAR product described in [the examples guide](../README.md#alos-1-stripmap-data) is a Fine Beam Single stripmap acquisition -- a continuous strip of 35344 pulses (`Pdi_NoOfLines` in the product's `workreport`) with the beam fixed relative to the platform, so each pulse illuminates a different patch. There is no common scene center for the strip and therefore no polar sector to reformat.

Running PFA on it would first require cutting the strip into sub-apertures short enough for the beam to count as staring and processing each patch separately. That is a different algorithm (sub-aperture/spotlight-from-stripmap decomposition), not the chain in `algorithm.py`, so this directory synthesizes its spotlight phase history in `geometry.py` instead.

## Processing chain

```
polar phase history (one arc per pulse)
    --> polar_positions: in-kernel geometry math
            pass 1 index = (v_j / cos(theta_p) - k0) / dk
            pass 2 index = (atan2(u_i, v_j) - theta0) / dtheta
    --> polar_to_rect: sar.interp1d along range (axis=1),
                       then across pulses (axis=0)
    --> compress: centered zero-pad (2x oversample), baseband 2-D IFFT
    --> sva2d: per-pixel cosine-on-pedestal weight selection along
               range, then azimuth (sar.where + comparisons); uniform
               weighting kept for comparison
```

## Running

```bash
PYTHONPATH=python python examples/pfa/run_point_target_cpu.py --n 512
PYTHONPATH=python python examples/pfa/run_point_target_hls.py --n 64
```

`run_point_target_hls.py` writes `hls_project/pfa/`. The kernel returns uniform and SVA images, so the package carries two golden output planes. Package contents are listed in the [backend guide](../../docs/backends.md#generated-package); numerical results are maintained in the [benchmark report](../../benchmarks/README.md).

![CPU uniform vs SVA impulse response](../../benchmarks/assets/cpu_pfa_sva_response.png)

![synthetic point targets](assets/pfa_synthetic_512.png)

Tests (`test/python/test_pfa.py`) check numerical equivalence with the reference, point-target focusing, the SVA sidelobe gain and HLS C++ emission.
