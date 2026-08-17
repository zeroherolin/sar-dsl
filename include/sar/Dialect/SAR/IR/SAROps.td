//===- SAROps.td - SAR operation definitions ---------------*- tablegen -*-===//
//
// Part of the SAR-DSL Project. Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//

#ifndef SAR_DIALECT_SAR_IR_SAROPS_TD
#define SAR_DIALECT_SAR_IR_SAROPS_TD

include "mlir/IR/OpAsmInterface.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "sar/Dialect/SAR/IR/SARDialect.td"

//===----------------------------------------------------------------------===//
// Type constraints
//===----------------------------------------------------------------------===//

def SAR_Float : AnyTypeOf<[F32, F64]>;
def SAR_Int : AnyTypeOf<[I32, I64]>;
def SAR_Complex : AnyTypeOf<[Complex<F32>, Complex<F64>]>;
def SAR_Element : AnyTypeOf<[SAR_Float, SAR_Int, SAR_Complex]>;

// All SAR operations require ranked tensors with fully static shapes.
def SAR_Tensor : StaticShapeTensorOf<[SAR_Element]>;
def SAR_FloatTensor : StaticShapeTensorOf<[SAR_Float]>;
def SAR_ComplexTensor : StaticShapeTensorOf<[SAR_Complex]>;
def SAR_FloatOrComplexTensor : StaticShapeTensorOf<[SAR_Float, SAR_Complex]>;
def SAR_I64Tensor : StaticShapeTensorOf<[I64]>;

//===----------------------------------------------------------------------===//
// Base classes
//===----------------------------------------------------------------------===//

class SAR_Op<string mnemonic, list<Trait> traits = []>
    : Op<SAR_Dialect, mnemonic, traits>;

class SAR_ElementwiseBinaryOp<string mnemonic>
    : SAR_Op<mnemonic, [Pure, SameOperandsAndResultType]> {
  let arguments = (ins SAR_Tensor:$lhs, SAR_Tensor:$rhs);
  let results = (outs SAR_Tensor:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

class SAR_ScalarBinaryOp<string mnemonic,
                         Type tensorKind = SAR_FloatOrComplexTensor>
    : SAR_Op<mnemonic, [Pure, SameOperandsAndResultType]> {
  let arguments = (ins tensorKind:$input, F64Attr:$scalar);
  let results = (outs tensorKind:$result);
  let assemblyFormat = "$input `,` $scalar attr-dict `:` type($result)";
}

//===----------------------------------------------------------------------===//
// Constants
//===----------------------------------------------------------------------===//

def SAR_ConstantOp : SAR_Op<"constant", [ConstantLike, Pure,
                                         AllTypesMatch<["value", "result"]>]> {
  let summary = "SAR tensor constant";
  let description = [{
    Materializes a constant tensor from a dense elements attribute.

    Example:
    ```mlir
    %w = sar.constant dense<[1.0, 2.0]> : tensor<2xf32>
    ```
  }];
  let arguments = (ins ElementsAttr:$value);
  let results = (outs SAR_Tensor:$result);
  let assemblyFormat = "attr-dict $value";
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// Element-wise arithmetic
//===----------------------------------------------------------------------===//

def SAR_AddOp : SAR_ElementwiseBinaryOp<"add"> {
  let summary = "element-wise addition";
}
def SAR_SubOp : SAR_ElementwiseBinaryOp<"sub"> {
  let summary = "element-wise subtraction";
}
def SAR_MulOp : SAR_ElementwiseBinaryOp<"mul"> {
  let summary = "element-wise multiplication";
}
def SAR_DivOp : SAR_ElementwiseBinaryOp<"div"> {
  let summary = "element-wise division";
}

def SAR_AddScalarOp : SAR_ScalarBinaryOp<"add_scalar"> {
  let summary = "add a scalar to every element";
  let description = [{
    Adds `scalar` to every element. For complex tensors the scalar is added
    to the real part.
  }];
}

def SAR_MulScalarOp : SAR_ScalarBinaryOp<"mul_scalar"> {
  let summary = "multiply every element by a scalar";
  let hasFolder = 1;
}

class SAR_FloatUnaryOp<string mnemonic>
    : SAR_Op<mnemonic, [Pure, SameOperandsAndResultType]> {
  let arguments = (ins SAR_FloatTensor:$input);
  let results = (outs SAR_FloatTensor:$result);
  let assemblyFormat = "$input attr-dict `:` type($result)";
}

def SAR_SqrtOp : SAR_FloatUnaryOp<"sqrt"> {
  let summary = "element-wise square root";
}

def SAR_CosOp : SAR_FloatUnaryOp<"cos"> {
  let summary = "element-wise cosine";
}

def SAR_SinOp : SAR_FloatUnaryOp<"sin"> {
  let summary = "element-wise sine";
}

def SAR_ExpOp : SAR_FloatUnaryOp<"exp"> {
  let summary = "element-wise natural exponential";
}

def SAR_LogOp : SAR_FloatUnaryOp<"log"> {
  let summary = "element-wise natural logarithm";
}

def SAR_Atan2Op : SAR_Op<"atan2", [Pure, SameOperandsAndResultType]> {
  let summary = "element-wise two-argument arctangent";
  let description = [{
    Computes `atan2(y, x)` element-wise following the numpy `arctan2`
    argument order (`y` first).
  }];
  let arguments = (ins SAR_FloatTensor:$y, SAR_FloatTensor:$x);
  let results = (outs SAR_FloatTensor:$result);
  let assemblyFormat = "$y `,` $x attr-dict `:` type($result)";
}

def SAR_ConjOp : SAR_Op<"conj", [Pure, SameOperandsAndResultType]> {
  let summary = "element-wise complex conjugate";
  let arguments = (ins SAR_ComplexTensor:$input);
  let results = (outs SAR_ComplexTensor:$result);
  let assemblyFormat = "$input attr-dict `:` type($result)";
  let hasFolder = 1;
}

class SAR_ComplexToFloatOp<string mnemonic> : SAR_Op<mnemonic, [Pure]> {
  let arguments = (ins SAR_ComplexTensor:$input);
  let results = (outs SAR_FloatTensor:$result);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
  let hasVerifier = 1;
}

def SAR_RealOp : SAR_ComplexToFloatOp<"real"> {
  let summary = "element-wise real part";
  let hasFolder = 1;
}

def SAR_ImagOp : SAR_ComplexToFloatOp<"imag"> {
  let summary = "element-wise imaginary part";
  let hasFolder = 1;
}

def SAR_CmpOp : SAR_Op<"cmp", [Pure, SameOperandsAndResultType]> {
  let summary = "element-wise comparison producing a 0.0 / 1.0 mask";
  let description = [{
    Compares two float tensors element-wise (`predicate` is one of `eq`,
    `ne`, `lt`, `le`, `gt`, `ge`) and yields 1.0 where the predicate
    holds, 0.0 elsewhere. Masks feed `sar.where`, the numpy `np.where`
    idiom for data-dependent selection.
  }];
  let arguments = (ins SAR_FloatTensor:$lhs, SAR_FloatTensor:$rhs,
                       StrAttr:$predicate);
  let results = (outs SAR_FloatTensor:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
  let hasVerifier = 1;
}

def SAR_WhereOp : SAR_Op<"where", [Pure,
    AllTypesMatch<["lhs", "rhs", "result"]>]> {
  let summary = "element-wise selection by a mask";
  let description = [{
    `result[i] = mask[i] != 0.0 ? lhs[i] : rhs[i]` (numpy `where`). The
    mask is a float tensor of the same shape (typically from `sar.cmp`);
    the branches may be float or complex. Selection is exact: no
    arithmetic blending, so inf/NaN in the unselected branch does not
    leak.
  }];
  let arguments = (ins SAR_FloatTensor:$mask,
                       SAR_FloatOrComplexTensor:$lhs,
                       SAR_FloatOrComplexTensor:$rhs);
  let results = (outs SAR_FloatOrComplexTensor:$result);
  let assemblyFormat = [{
    $mask `,` $lhs `,` $rhs attr-dict `:` functional-type(operands, results)
  }];
  let hasVerifier = 1;
}

def SAR_ComplexOp : SAR_Op<"complex", [Pure, AllTypesMatch<["re", "im"]>]> {
  let summary = "assemble a complex tensor from real and imaginary planes";
  let arguments = (ins SAR_FloatTensor:$re, SAR_FloatTensor:$im);
  let results = (outs SAR_ComplexTensor:$result);
  let assemblyFormat = "$re `,` $im attr-dict `:` type($re) `->` type($result)";
  let hasVerifier = 1;
}

def SAR_AbsOp : SAR_Op<"abs", [Pure]> {
  let summary = "element-wise absolute value / complex magnitude";
  let description = [{
    Computes |x| element-wise. For a `complex<f32>` input the result element
    type is `f32` (respectively `f64` for `complex<f64>`); for float inputs
    the element type is preserved. Shapes are preserved.
  }];
  let arguments = (ins SAR_FloatOrComplexTensor:$input);
  let results = (outs SAR_FloatTensor:$result);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
  let hasVerifier = 1;
}

def SAR_CastOp : SAR_Op<"cast", [Pure]> {
  let summary = "element type cast";
  let description = [{
    Casts between element types while preserving the shape. Supported casts:
    float <-> float (width change), complex <-> complex (width change),
    float -> complex (imaginary part set to zero), int <-> int (width
    change), int -> float (exact for magnitudes below the mantissa
    limit), and float -> int (truncation toward zero, C semantics).
  }];
  let arguments = (ins SAR_Tensor:$input);
  let results = (outs SAR_Tensor:$result);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Reductions
//===----------------------------------------------------------------------===//

def SAR_ReduceOp : SAR_Op<"reduce", [Pure]> {
  let summary = "reduction along one axis of a rank-2 tensor";
  let description = [{
    Reduces a rank-2 tensor along axis `dim`, producing the rank-1 tensor
    of the remaining axis. `kind` selects the combinator:

    - `sum`: float or complex elements;
    - `max` / `min`: float elements only.

    Rank-1 reductions are expressed by broadcasting to shape `1 x n`
    first (the frontend does this), keeping every result rank >= 1.
  }];
  let arguments = (ins SAR_FloatOrComplexTensor:$input, StrAttr:$kind,
                       I64Attr:$dim);
  let results = (outs SAR_FloatOrComplexTensor:$result);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
  let hasVerifier = 1;
}

def SAR_ArgMaxOp : SAR_Op<"argmax", [Pure]> {
  let summary = "index of the maximum along one axis of a rank-2 tensor";
  let description = [{
    Returns, for every line of a rank-2 float tensor along axis `dim`, the
    `i64` index of its maximum element (first occurrence on ties, numpy
    `argmax` convention).
  }];
  let arguments = (ins SAR_FloatTensor:$input, I64Attr:$dim);
  let results = (outs SAR_I64Tensor:$result);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Data movement
//===----------------------------------------------------------------------===//

def SAR_TransposeOp : SAR_Op<"transpose", [Pure]> {
  let summary = "2-D transpose (corner turn)";
  let arguments = (ins SAR_Tensor:$input);
  let results = (outs SAR_Tensor:$result);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
  let hasVerifier = 1;
  let hasFolder = 1;
}

def SAR_BroadcastOp : SAR_Op<"broadcast", [Pure]> {
  let summary = "broadcast a 1-D tensor into a 2-D tensor along an axis";
  let description = [{
    Expands a 1-D tensor to a 2-D result. The vector lies along axis `dim`
    of the result and is replicated along the other axis:

    - `dim = 0`: `result[i, j] = input[i]`
    - `dim = 1`: `result[i, j] = input[j]`
  }];
  let arguments = (ins SAR_Tensor:$input, I64Attr:$dim);
  let results = (outs SAR_Tensor:$result);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
  let hasVerifier = 1;
}

def SAR_SliceOp : SAR_Op<"slice", [Pure]> {
  let summary = "extract a statically strided sub-tensor";
  let description = [{
    Copies `sizes` elements per axis starting at `offsets`, stepping by
    `strides` (numpy basic slicing: `x[o0 : o0+n0*s0 : s0, ...]`).
  }];
  let arguments = (ins SAR_Tensor:$input, DenseI64ArrayAttr:$offsets,
                       DenseI64ArrayAttr:$sizes,
                       DenseI64ArrayAttr:$strides);
  let results = (outs SAR_Tensor:$result);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
  let hasVerifier = 1;
}

def SAR_ConcatOp : SAR_Op<"concat", [Pure]> {
  let summary = "concatenate two tensors along an axis";
  let arguments = (ins SAR_Tensor:$lhs, SAR_Tensor:$rhs, I64Attr:$dim);
  let results = (outs SAR_Tensor:$result);
  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` functional-type(operands, results)
  }];
  let hasVerifier = 1;
}

def SAR_PadOp : SAR_Op<"pad", [Pure]> {
  let summary = "pad a tensor with a constant";
  let description = [{
    Grows the tensor by `low` leading and `high` trailing elements per
    axis, filled with `value` (imaginary part zero for complex elements).
  }];
  let arguments = (ins SAR_FloatOrComplexTensor:$input,
                       DenseI64ArrayAttr:$low, DenseI64ArrayAttr:$high,
                       F64Attr:$value);
  let results = (outs SAR_FloatOrComplexTensor:$result);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
  let hasVerifier = 1;
}

def SAR_ReverseOp : SAR_Op<"reverse", [Pure, SameOperandsAndResultType]> {
  let summary = "reverse the element order along one axis";
  let description = [{
    `result[..., i, ...] = input[..., n - 1 - i, ...]` along axis `dim`
    (numpy `flip`).
  }];
  let arguments = (ins SAR_Tensor:$input, I64Attr:$dim);
  let results = (outs SAR_Tensor:$result);
  let assemblyFormat = "$input attr-dict `:` type($result)";
  let hasVerifier = 1;
  let hasFolder = 1;
}

def SAR_FFTShiftOp : SAR_Op<"fftshift", [Pure, SameOperandsAndResultType]> {
  let summary = "shift the zero-frequency component to the center";
  let description = [{
    Circularly rotates the tensor along axis `dim` following the numpy
    `fftshift` convention. With the `inverse` flag set it computes
    `ifftshift` instead (the two differ for odd sizes).
  }];
  let arguments = (ins SAR_Tensor:$input, I64Attr:$dim, UnitAttr:$inverse);
  let results = (outs SAR_Tensor:$result);
  let assemblyFormat = "$input attr-dict `:` type($result)";
  let hasVerifier = 1;
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// Signal processing
//===----------------------------------------------------------------------===//

class SAR_FFTBaseOp<string mnemonic>
    : SAR_Op<mnemonic, [Pure, SameOperandsAndResultType]> {
  let arguments = (ins SAR_ComplexTensor:$input, I64Attr:$dim);
  let results = (outs SAR_ComplexTensor:$result);
  let assemblyFormat = "$input attr-dict `:` type($result)";
  let hasVerifier = 1;
}

def SAR_FFTOp : SAR_FFTBaseOp<"fft"> {
  let summary = "forward FFT along one axis";
  let description = [{
    Computes the unscaled forward DFT along axis `dim` (numpy convention).
    Any size >= 2 is accepted on both backends.
  }];
}

def SAR_IFFTOp : SAR_FFTBaseOp<"ifft"> {
  let summary = "inverse FFT along one axis";
  let description = [{
    Computes the inverse DFT along axis `dim`, scaled by `1/N` (numpy
    convention). Any size >= 2 is accepted on both backends.
  }];
}

def SAR_FFTSplitOp : SAR_Op<"fft_split", [Pure,
    AllTypesMatch<["re", "im", "out_re", "out_im"]>]> {
  let summary = "FFT along one axis on split real/imaginary planes";
  let description = [{
    The split-complex form of `sar.fft` / `sar.ifft`, produced by the
    `sar-decomplexify` pass for targets without complex arithmetic (HLS).
    Semantics match `sar.fft` (`sar.ifft` with the `inverse` flag) with the
    complex tensor represented as separate real and imaginary planes.

    Compiler-internal: the frontend never emits this op, and only the
    affine pipeline lowers it. Write `sar.fft` and let decomplexify
    produce it.
  }];
  let arguments = (ins SAR_FloatTensor:$re, SAR_FloatTensor:$im,
                       I64Attr:$dim, UnitAttr:$inverse);
  let results = (outs SAR_FloatTensor:$out_re, SAR_FloatTensor:$out_im);
  let assemblyFormat = "$re `,` $im attr-dict `:` type($re)";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// SAR-domain kernels
//===----------------------------------------------------------------------===//

def SAR_Interp1DOp
    : SAR_Op<"interp1d", [Pure, AllTypesMatch<["data", "result"]>]> {
  let summary = "kernel-based resampling along the last axis";
  let description = [{
    Resamples each row of a 2-D complex tensor at fractional sample
    positions with a selectable interpolation kernel. For the default
    windowed sinc:

    ```
    out[i, j] = sum_{k} data[i, k'] * w(positions[i, j] - k')
    k' = floor(positions[i, j]) - taps/2 + 1 + k,  k in [0, taps)
    w(d) = sinc(d) * window(d / (taps/2))
    ```

    `kernel` is one of:
    - `nearest`: one tap at `round(position)`
    - `linear`:  two taps, triangular weights
    - `cubic`:   four taps, Keys cubic convolution (a = -0.5)
    - `sinc`:    `taps` taps of windowed sinc (the default; `taps = 8`)

    For `sinc`, `window` selects the taper over the support: `rect`,
    `hann` (default), `hamming`, or `kaiser` with shape parameter
    `beta`. Taps that fall outside the row are resolved by `boundary`
    (below), which applies to every kernel.

    This is the orthogonal primitive underlying SAR interpolation
    stages: Stolt remapping (omega-K) and range cell migration
    correction (range-Doppler) are expressed as element-wise position
    computations feeding `sar.interp1d`.

    `dim` selects the resampled axis (default 1, i.e. per row);
    canonicalization normalizes `dim = 0` into transposes around the
    `dim = 1` form, which is the only one the lowerings implement.

    `boundary` controls what happens when a tap index falls outside `[0, N-1]`:
    - `zero` (default): out-of-range taps contribute zero.
    - `edge`: clamp the source index to the nearest valid sample.
    - `reflect`: mirror the index about the boundary, repeating the edge sample (`-1` reads sample 0): numpy's `pad(mode="symmetric")` and scipy.ndimage's `mode="reflect"`.
  }];
  let arguments = (ins SAR_ComplexTensor:$data,
                       SAR_FloatTensor:$positions,
                       DefaultValuedAttr<I64Attr, "1">:$dim,
                       DefaultValuedAttr<StrAttr, "\"sinc\"">:$kernel,
                       DefaultValuedAttr<I64Attr, "8">:$taps,
                       DefaultValuedAttr<StrAttr, "\"hann\"">:$window,
                       DefaultValuedAttr<F64Attr, "2.5">:$beta,
                       DefaultValuedAttr<StrAttr, "\"zero\"">:$boundary);
  let results = (outs SAR_ComplexTensor:$result);
  let assemblyFormat = [{
    $data `,` $positions attr-dict `:` functional-type(operands, results)
  }];
  let hasVerifier = 1;
  let hasCanonicalizer = 1;
}

def SAR_Gather2DOp
    : SAR_Op<"gather2d", [Pure, AllTypesMatch<["rows", "cols"]>]> {
  let summary = "2-D gather at data-dependent positions";
  let description = [{
    Samples a rank-2 tensor at fractional 2-D positions: for every output
    element, `rows` and `cols` name the source coordinates to read.

    ```
    out[i, j] = data[rows[i, j], cols[i, j]]
    ```

    The output takes the shape of the position tensors, which is
    independent of the data shape -- a backprojection gathers a pulse
    raster onto an image grid.

    `kernel` selects the sampling:
    - `nearest`: one tap at the rounded coordinates
    - `linear`:  bilinear over the four surrounding samples

    `boundary` resolves taps outside the data extent, per axis:
    - `zero` (default): out-of-range taps contribute zero
    - `edge`: indices clamp to the nearest valid sample

    This is the orthogonal primitive beyond per-line interpolation
    (`sar.interp1d`): both coordinates may be arbitrary functions of the
    output position, which is the access pattern of time-domain
    backprojection.
  }];
  let arguments = (ins SAR_ComplexTensor:$data,
                       SAR_FloatTensor:$rows,
                       SAR_FloatTensor:$cols,
                       DefaultValuedAttr<StrAttr, "\"linear\"">:$kernel,
                       DefaultValuedAttr<StrAttr, "\"zero\"">:$boundary);
  let results = (outs SAR_ComplexTensor:$result);
  let assemblyFormat = [{
    $data `,` $rows `,` $cols attr-dict `:`
    functional-type(operands, results)
  }];
  let hasVerifier = 1;
}

def SAR_Gather2DSplitOp : SAR_Op<"gather2d_split", [Pure,
    AllTypesMatch<["rows", "cols"]>,
    AllTypesMatch<["re", "im"]>,
    AllTypesMatch<["out_re", "out_im"]>]> {
  let summary = "split-complex 2-D gather at data-dependent positions";
  let description = [{
    The split-complex form of `sar.gather2d`, produced by
    `sar-decomplexify` for targets without complex arithmetic. Semantics
    match `sar.gather2d` with the complex tensor represented as separate
    real and imaginary planes.

    Compiler-internal: the frontend never emits this op. Write
    `sar.gather2d` and let decomplexify produce it.
  }];
  let arguments = (ins SAR_FloatTensor:$re, SAR_FloatTensor:$im,
                       SAR_FloatTensor:$rows, SAR_FloatTensor:$cols,
                       DefaultValuedAttr<StrAttr, "\"linear\"">:$kernel,
                       DefaultValuedAttr<StrAttr, "\"zero\"">:$boundary);
  let results = (outs SAR_FloatTensor:$out_re, SAR_FloatTensor:$out_im);
  let assemblyFormat = [{
    $re `,` $im `,` $rows `,` $cols attr-dict `:`
    functional-type(operands, results)
  }];
  let hasVerifier = 1;
}

def SAR_Interp1DSplitOp : SAR_Op<"interp1d_split", [Pure,
    AllTypesMatch<["re", "im", "out_re", "out_im"]>]> {
  let summary = "split-complex kernel-based resampling along the last axis";
  let description = [{
    The split-complex form of `sar.interp1d`, produced by
    `sar-decomplexify` for targets without complex arithmetic. Semantics
    match `sar.interp1d` with the complex tensor represented as separate
    real and imaginary planes.

    Compiler-internal: the frontend never emits this op, and only the
    affine pipeline lowers it. Write `sar.interp1d` and let decomplexify
    produce it.
  }];
  let arguments = (ins SAR_FloatTensor:$re, SAR_FloatTensor:$im,
                       SAR_FloatTensor:$positions,
                       DefaultValuedAttr<I64Attr, "1">:$dim,
                       DefaultValuedAttr<StrAttr, "\"sinc\"">:$kernel,
                       DefaultValuedAttr<I64Attr, "8">:$taps,
                       DefaultValuedAttr<StrAttr, "\"hann\"">:$window,
                       DefaultValuedAttr<F64Attr, "2.5">:$beta,
                       DefaultValuedAttr<StrAttr, "\"zero\"">:$boundary);
  let results = (outs SAR_FloatTensor:$out_re, SAR_FloatTensor:$out_im);
  let assemblyFormat = [{
    $re `,` $im `,` $positions attr-dict `:` functional-type(operands, results)
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Compiled loops
//===----------------------------------------------------------------------===//

def SAR_IterateOp : SAR_Op<"iterate", [RecursiveMemoryEffects,
    SingleBlockImplicitTerminator<"YieldOp">]> {
  let summary = "counted loop with tensor-carried state";
  let description = [{
    Applies the body `trips` times, feeding each iteration's yielded
    values to the next as the block arguments. Unlike trace-time Python
    loops, which unroll, this stays a single loop in the compiled design.

    The block argument types, the operand types, the result types and
    the yielded types all match, position by position. The trip count is
    a compile-time constant and the body cannot observe the iteration
    index.

    Lowers to `scf.for` with tensor-carried `iter_args`
    (`convert-sar-to-linalg`); `sar-decomplexify` splits complex carries
    into (re, im) plane pairs first on targets without complex support.
  }];
  let arguments = (ins Variadic<SAR_Tensor>:$inits, I64Attr:$trips);
  let results = (outs Variadic<SAR_Tensor>:$results);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = [{
    `(` $inits `)` attr-dict `:` functional-type($inits, $results) $body
  }];
  let hasVerifier = 1;

  // The default builders cannot infer the variadic result types (they
  // mirror the init types); the one builder every caller needs is spelled
  // out instead.
  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "::mlir::TypeRange":$results, "::mlir::ValueRange":$inits,
                   "::mlir::IntegerAttr":$trips), [{
      $_state.addOperands(inits);
      $_state.addTypes(results);
      $_state.addAttribute(getTripsAttrName($_state.name), trips);
      $_state.addRegion();
    }]>
  ];
}

def SAR_YieldOp : SAR_Op<"yield", [Pure, ReturnLike, Terminator,
    HasParent<"IterateOp">]> {
  let summary = "yields the carried values of one `sar.iterate` step";
  let arguments = (ins Variadic<SAR_Tensor>:$results);
  let assemblyFormat = "$results attr-dict `:` type($results)";
  let builders = [
    OpBuilder<(ins), "build($_builder, $_state, ::mlir::ValueRange());">
  ];
}

//===----------------------------------------------------------------------===//
// Prefix scan / order-statistic
//===----------------------------------------------------------------------===//

def SAR_CumsumOp : SAR_Op<"cumsum", [Pure, SameOperandsAndResultType]> {
  let summary = "prefix sum (cumulative sum) along one axis of a rank-2 tensor";
  let description = [{
    Computes the inclusive prefix sum along axis `dim` of a rank-2 tensor:

    - `dim = 0`: `result[i, j] = sum_{k=0}^{i} input[k, j]`
    - `dim = 1`: `result[i, j] = sum_{k=0}^{j} input[i, k]`

    Accepted element types: `f32`, `f64`, `complex<f32>`, `complex<f64>`.
    The scan is sequential; it is not decomposed into a parallel tree.
  }];
  let arguments = (ins SAR_FloatOrComplexTensor:$input, I64Attr:$dim);
  let results = (outs SAR_FloatOrComplexTensor:$result);
  let assemblyFormat = "$input attr-dict `:` type($result)";
  let hasVerifier = 1;
}

def SAR_RankFilterOp : SAR_Op<"rank_filter", [Pure, SameOperandsAndResultType]> {
  let summary = "windowed order-statistic filter along one axis of a rank-2 tensor";
  let description = [{
    Applies an order-statistic filter along axis `dim`: for every position a
    window of `window` elements (clamped at boundaries) is sorted and the
    element at index `rank` (0-based, ascending order) is written to the result.

    - `rank = 0` is the minimum, `rank = window - 1` is the maximum,
      `rank = window / 2` (floor) is the median.
    - `window` must be a positive odd integer.
    - `0 <= rank < window`.
    - Boundary: indices out of range are clamped to the nearest valid index.
    - Element type must be float (`f32` or `f64`); complex ordering is
      undefined, so complex inputs are rejected by the verifier.
  }];
  let arguments = (ins SAR_FloatTensor:$input, I64Attr:$window,
                       I64Attr:$rank, I64Attr:$dim);
  let results = (outs SAR_FloatTensor:$result);
  let assemblyFormat = "$input attr-dict `:` type($result)";
  let hasVerifier = 1;
}

def SAR_SortOp : SAR_Op<"sort", [Pure, SameOperandsAndResultType]> {
  let summary = "sorts each line of a rank-2 tensor along one axis";
  let description = [{
    Sorts every line along axis `dim` into ascending order (numpy `sort`):

    - `dim = 0`: each column is sorted independently.
    - `dim = 1`: each row is sorted independently.

    Element type must be float (`f32` or `f64`); complex ordering is
    undefined, so complex inputs are rejected by the verifier. The sort is
    a compare-exchange network over the static extent, so it carries no
    data-dependent control flow.
  }];
  let arguments = (ins SAR_FloatTensor:$input, I64Attr:$dim);
  let results = (outs SAR_FloatTensor:$result);
  let assemblyFormat = "$input attr-dict `:` type($result)";
  let hasVerifier = 1;
}

#endif // SAR_DIALECT_SAR_IR_SAROPS_TD
