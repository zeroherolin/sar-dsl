# Coming from MATLAB

SAR-DSL kernels are Python, and the vocabulary deliberately mirrors
MATLAB and Signal Processing Toolbox naming. The differences relevant to
porting are listed here.

## Indexing is 0-based, and slice ends are exclusive

This is the one difference that silently produces wrong images rather
than an error.

| MATLAB | SAR-DSL | Note |
|--------|---------|------|
| `x(1)` | `x[0:1]` | first element |
| `x(end)` | `x[-1:]` | last element |
| `x(1:8)` | `x[0:8]` | 8 elements; the end index is *excluded* |
| `x(2:2:end)` | `x[1::2]` | every other element from the second |
| `x(:, 3)` | `x[:, 2:3]` | third column |
| `size(x, 1)` | `x.shape[0]` | rows |
| `x.'` | `sar.transpose(x)` or `x.T` | transpose (no conjugation) |
| `x'` | `sar.conj(x.T)` | conjugate transpose |

Indexing is always by *slice*: `x[0]` is rejected rather than silently
dropping a dimension, so a single element or line is a length-1 slice
(`x[0:1]`) and every result keeps the tensor's rank. Negative bounds
count from the end as in NumPy.

Axis numbers follow the same rule: MATLAB's `fft(x, [], 2)` (along
columns) is `sar.fft(x, axis=1)`.

## Function names

Most names carry over unchanged: `fft`, `ifft`, `fft2`, `ifft2`,
`fftshift`, `ifftshift`, `circshift`, `abs`, `angle`, `conj`, `real`,
`imag`, `atan2`, `sqrt`, `exp`, `log`, `log2`, `log10`, `sinc`, `hypot`,
`sign`, `floor`, `ceil`, `round`, `sum`, `mean`, `std`, `var`, `max`,
`min`, `hamming`, `hanning`, `blackman`, `kaiser`, `taylorwin`,
`mag2db`, `db2mag`, `pow2db`, `db2pow`.

Where they differ:

| MATLAB | SAR-DSL | Note |
|--------|---------|------|
| `[~, i] = max(x)` | `sar.argmax(x, axis=...)` | index only |
| `[~, i] = min(x)` | `sar.argmin(x, axis=...)` | index only; composition over `argmax` |
| `interp1` | `sar.interp1d` | kernel selectable (`sinc`, `cubic`, ...); out of range, `boundary=` (`zero`/`edge`/`reflect`) applies, not MATLAB's NaN/`extrapval` |
| `cat(2, a, b)` | `sar.concat([a, b], axis=1)` | also spelled `concatenate` |
| `padarray` | `sar.pad` | |
| `fliplr` / `flipud` | `sar.flip(x, axis=1)` / `axis=0` | |
| `repmat` of a vector | `sar.broadcast(v, shape, axis=...)` | rank-1 to rank-2 |
| logical indexing | `sar.where(mask, a, b)` | see below |
| `db(x)` | `sar.db(x)` or `sar.mag2db(x)` | voltage convention |

`dim=` and `axis=` are accepted interchangeably everywhere (`dim` reads
like MATLAB, `axis` like NumPy) -- passing both is an error.

## Two convention differences

- **`round` matches MATLAB, not NumPy**: halves go away from zero, so
  `round(2.5) == 3`. NumPy would give 2.
- **`var` / `std` match NumPy, not MATLAB**: they normalize by `N`, not
  `N-1`.

## Compiled loops

A Python `for` unrolls at trace time. Use `sar.iterate(n, body, *carries)`
when the loop must remain one loop in the design. With `index=True` the
body can drive `dynamic_slice` / `dynamic_update_slice`.

## No in-place edits; masking replaces logical indexing

There is no `x(x < 0) = 0`. Selection is a value-producing operation:

```python
clean = sar.where(x < 0.0, 0.0, x)     # Matlab: x(x < 0) = 0
```

The same idiom covers thresholding, CFAR-style masks and spatially
variant weighting; `examples/pfa/` builds SVA out of it.

## Shapes are fixed at compile time

A kernel compiles for one geometry. Changing the scene size recompiles
(automatically -- the result is cached), but there is no dynamic
resizing inside a kernel, and no `end`-relative sizing at run time.

## Getting data in and out

Kernels take and return NumPy arrays. `.mat` files load through SciPy
(`pip install scipy` -- it is not a SAR-DSL dependency):

```python
import scipy.io
raw = scipy.io.loadmat("raw.mat")["echoes"]
image = focus(raw, replica)
scipy.io.savemat("image.mat", {"image": image})
```

Complex types map straight across: MATLAB `single` complex is
`sar.c64` (NumPy `complex64`), `double` complex is `sar.c128`
(`complex128`).

## Where to look next

- [defining-ops.md](defining-ops.md) -- writing your own operators and
  how NumPy arrays interact with kernels
- [dialect.md](dialect.md) -- the full operation reference
- [../examples/](../examples/) -- four complete imaging algorithms
