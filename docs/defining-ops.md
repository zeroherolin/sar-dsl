# Defining operators

One decorator, `@sar.op`, defines operators; built-in and user-defined ops are called identically inside kernels. Kernels themselves are `@sar.func` functions.

```python
@sar.op
def range_compress(data, replica):
    return sar.ifft(sar.fft(data, axis=1) * replica, axis=1)
```

The body composes existing ops:

- Inside a kernel it inlines at trace time -- no call overhead, no fusion barrier, and it compiles to **every backend**.
- Outside a kernel it runs directly on NumPy arrays (specialized and JIT-compiled per argument shapes, numba/Triton style).
- For emission backends (`hls`, and any backend whose `.compile()` produces an artifact rather than a callable), pin a signature first: `range_compress.specialize(sar.c64[512, 512], sar.c64[512, 512])`. Annotation-free `@sar.func` kernels and `@sar.op` operators use the same spelling; an annotated `@sar.func` is already pinned and compiles directly.

This form covers more than op chaining. With `sar.where` and tensor comparisons, data-dependent per-element logic is ordinary Python -- spatially variant apodization, thresholding, CFAR-style masking ([examples/pfa/](../examples/pfa/) implements SVA this way). The signal vocabulary (`sar.fft2`, `sar.matched_filter`, `sar.mag2db`, ...) is itself built from `@sar.op` compositions: user operators are not second-class.

Kernels are equally light: `@sar.func` works without type annotations (specializing per call), NumPy arrays mix freely with tensors (`spectrum * np.hanning(M)` lifts the window to a constant, promotes dtypes and broadcasts rank-1 operands along the last axis), and tensors carry the NumPy-style method, operator and slicing surface -- [python-api.md](python-api.md) lists it in full.

## NumPy interoperability

An operator has two call sites, and the same definition serves both:

```python
spectrum = sar.fft2(data_np)             # eager: NumPy in, NumPy out

@sar.func
def chain(data):                         # inlined: fuses with neighbours
    return abs(sar.fft2(data))
```

Outside a kernel the arguments split by role:

| Argument | Becomes | Example |
| --- | --- | --- |
| NumPy array | a parameter of the specialized kernel | `data` in `fft2(data)` |
| anything else | a compile-time constant baked into the trace | `axis=1`, `taps=8` |
| array named in `const=` | a baked constant, not a parameter | `fa`, `fr` below |
| _(no array at all)_ | nothing to specialize: the body runs directly | `sar.hanning(512)` |

Each distinct constant combination compiles its own variant, so `sar.mean(x, axis=0)` and `sar.mean(x, axis=1)` do not share a specialization.

`const` is for array arguments the body inspects with _host_ arithmetic -- acquisition axes, not data. They must stay NumPy arrays during tracing:

```python
@sar.op(const=("fa", "fr"))
def stolt_interp(data, fa, fr, *, c, fc, vr, t_shift):
    df = float(fr[1] - fr[0])            # host arithmetic on fr
    ...
```

Operators with no array arguments are _constructors_ rather than transforms. The window family is the built-in case: inside a kernel `sar.hanning(512)` bakes in as a `sar.constant`, outside one it hands back the plain NumPy samples, so it stays usable in host code (`np.fft.fft(x) * sar.hanning(n)`).

Two properties worth knowing about the eager path:

- **It compiles.** The first call with a given signature runs the toolchain (subprocess `sar-opt` + `clang`) and caches the result on disk; later calls with the same shapes and constants reuse it. For a handful of elements plain NumPy is faster -- the win is in `@sar.func` kernels, where a whole chain fuses and the intermediates disappear.
- **Semantics follow NumPy**, with the deliberate exceptions documented on each function: `sar.round` rounds half away from zero (MATLAB), and `sar.var` / `sar.std` normalize by N (NumPy) rather than N-1.

The eager path belongs to `@sar.op` operators. The thin wrappers that emit a single primitive (`sar.round`, `sar.where`, ...) exist only inside kernels: called on NumPy arrays outside one, they raise `TraceError` rather than falling back to NumPy.

## Why composition only

Operators decompose into IR primitives at trace time, so a user operator is exactly as portable as the built-in vocabulary: every backend that lowers the primitives runs it, and element-wise fusion optimizes across its boundary. Anything a primitive cannot express must be added as a backend-neutral primitive with verification and both lowerings -- not routed around with per-backend native code.
