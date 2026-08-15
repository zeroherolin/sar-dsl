# Polar Format Algorithm + SVA from Python-defined operators

The spotlight-SAR imaging algorithm with spatially variant apodization.
Every stage is a `@sar.op` operator composed of built-in ops, including
two pieces that go beyond straight op chaining:

- `polar_positions` computes the regrid interpolation positions
  *inside the kernel* from the collection axes (`1/cos(theta)` outer
  products, `sar.atan2` to invert the pulse angle);
- `sva2d` is a data-dependent per-pixel weight selection (Stankwitz
  SVA, separable 2-D form: a range pass, then an azimuth pass), written
  in Python with the `sar.where` / comparison primitives -- algorithm
  logic that no fixed weighting can reproduce.

| File | Purpose |
|------|---------|
| `algorithm.py` | The PFA + SVA chain as Python-defined operators (`build_kernel`, `make_inputs`) |
| `geometry.py` | Polar collection grid, inscribed target grid, phase-history simulation |
| `reference.py` | NumPy reference implementation (`PFAProcessor`) |
| `run_point_target_cpu.py` | Full cpu-backend flow: simulate, focus, measure, save a PNG |
| `run_point_target_scalehls.py` | Full scalehls-backend flow: HLS C++ design + csim package (`hls_project/`) |

PFA is a spotlight-mode algorithm, so unlike the stripmap examples
([wka](../wka/), [rda](../rda/), [csa](../csa/)) there is no
`run_alos_cpu.py`: the ALOS-1 product is stripmap data, which those three
process. The synthetic collection in `geometry.py` provides the
spotlight phase history instead.

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
# from the repository root, after `make build`
python examples/pfa/run_point_target_cpu.py --n 512       # focus + PNG
python examples/pfa/run_point_target_scalehls.py --n 128  # design + csim package
```

SVA removes the -13 dB sidelobes of the uniform weighting with *zero*
mainlobe broadening -- the property no amplitude window has:

```
uniform: range PSLR  -13.3 dB, IRW 0.91 cells
    SVA: range PSLR  -23.3 dB, IRW 0.91 cells
```

![uniform vs SVA impulse response](../../benchmarks/assets/sva_response.png)

![synthetic point targets](assets/pfa_synthetic_512.png)

Tests (`test/python/test_pfa.py`) check numerical equivalence with the
reference, point-target focusing, the SVA sidelobe gain and HLS C++
emission.
