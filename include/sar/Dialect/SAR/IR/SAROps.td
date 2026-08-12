//===- SAROps.td - SAR operation definitions ---------------*- tablegen -*-===//
//
// Part of the SAR-DSL Project. Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//

#ifndef SAR_DIALECT_SAR_IR_SAROPS_TD
#define SAR_DIALECT_SAR_IR_SAROPS_TD

include "mlir/IR/OpAsmInterface.td"
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

class SAR_ScalarBinaryOp<string mnemonic, Type tensorKind = SAR_FloatOrComplexTensor>
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

def SAR_MaxScalarOp : SAR_ScalarBinaryOp<"max_scalar", SAR_FloatTensor> {
  let summary = "element-wise maximum against a scalar";
}

def SAR_NegOp : SAR_Op<"neg", [Pure, SameOperandsAndResultType]> {
  let summary = "element-wise negation";
  let arguments = (ins SAR_FloatOrComplexTensor:$input);
  let results = (outs SAR_FloatOrComplexTensor:$result);
  let assemblyFormat = "$input attr-dict `:` type($result)";
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

def SAR_ExpJOp : SAR_Op<"expj", [Pure]> {
  let summary = "element-wise complex exponential of an imaginary argument";
  let description = [{
    Computes `exp(j * x) = cos(x) + j * sin(x)` element-wise. The input is a
    float tensor, the result is a complex tensor of matching precision and
    identical shape.
  }];
  let arguments = (ins SAR_FloatTensor:$input);
  let results = (outs SAR_ComplexTensor:$result);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
  let hasVerifier = 1;
}

def SAR_CastOp : SAR_Op<"cast", [Pure]> {
  let summary = "element type cast";
  let description = [{
    Casts between element types while preserving the shape. Supported casts:
    float <-> float (width change), complex <-> complex (width change), and
    float -> complex (imaginary part set to zero).
  }];
  let arguments = (ins SAR_FloatOrComplexTensor:$input);
  let results = (outs SAR_FloatOrComplexTensor:$result);
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

class SAR_FFTBaseOp<string mnemonic> : SAR_Op<mnemonic,
                                              [Pure, SameOperandsAndResultType]> {
  let arguments = (ins SAR_ComplexTensor:$input, I64Attr:$dim);
  let results = (outs SAR_ComplexTensor:$result);
  let assemblyFormat = "$input attr-dict `:` type($result)";
  let hasVerifier = 1;
}

def SAR_FFTOp : SAR_FFTBaseOp<"fft"> {
  let summary = "forward FFT along one axis";
  let description = [{
    Computes the unscaled forward DFT along axis `dim` (numpy convention).
    The size along `dim` must be a power of two.
  }];
}

def SAR_IFFTOp : SAR_FFTBaseOp<"ifft"> {
  let summary = "inverse FFT along one axis";
  let description = [{
    Computes the inverse DFT along axis `dim`, scaled by `1/N` (numpy
    convention). The size along `dim` must be a power of two.
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

def SAR_Interp1DOp : SAR_Op<"interp1d", [Pure, AllTypesMatch<["data", "result"]>]> {
  let summary = "windowed-sinc resampling along the last axis";
  let description = [{
    Resamples each row of a 2-D complex tensor at fractional sample
    positions using an 8-tap windowed-sinc kernel:

    ```
    out[i, j] = sum_{k=-3}^{4} data[i, k'] * w(positions[i, j] - k')
    k' = floor(positions[i, j]) + k
    w(d) = sinc(d) * (0.5 + 0.5 cos(pi d / 4))
    ```

    Out-of-range taps contribute zero. This is the orthogonal primitive
    underlying SAR interpolation stages: Stolt remapping (omega-K) and
    range cell migration correction (range-Doppler) are expressed as
    element-wise position computations feeding `sar.interp1d`.
  }];
  let arguments = (ins SAR_ComplexTensor:$data,
                       SAR_FloatTensor:$positions);
  let results = (outs SAR_ComplexTensor:$result);
  let assemblyFormat = [{
    $data `,` $positions attr-dict `:` functional-type(operands, results)
  }];
  let hasVerifier = 1;
}

def SAR_Interp1DSplitOp : SAR_Op<"interp1d_split", [Pure,
    AllTypesMatch<["re", "im", "out_re", "out_im"]>]> {
  let summary = "split-complex windowed-sinc resampling along the last axis";
  let description = [{
    The split-complex form of `sar.interp1d`, produced by
    `sar-decomplexify` for targets without complex arithmetic. Semantics
    match `sar.interp1d` with the complex tensor represented as separate
    real and imaginary planes.
  }];
  let arguments = (ins SAR_FloatTensor:$re, SAR_FloatTensor:$im,
                       SAR_FloatTensor:$positions);
  let results = (outs SAR_FloatTensor:$out_re, SAR_FloatTensor:$out_im);
  let assemblyFormat = [{
    $re `,` $im `,` $positions attr-dict `:` functional-type(operands, results)
  }];
  let hasVerifier = 1;
}

def SAR_StoltInterpSplitOp : SAR_Op<"stolt_interp_split", [Pure,
    AllTypesMatch<["re", "im", "out_re", "out_im"]>]> {
  let summary = "split-complex Stolt interpolation";
  let description = [{
    The split-complex form of `sar.stolt_interp`, produced by
    `sar-decomplexify` for targets without complex arithmetic.
  }];
  let arguments = (ins SAR_FloatTensor:$re, SAR_FloatTensor:$im,
                       SAR_FloatTensor:$fa, SAR_FloatTensor:$fr,
                       F64Attr:$c, F64Attr:$fc, F64Attr:$vr,
                       F64Attr:$t_shift);
  let results = (outs SAR_FloatTensor:$out_re, SAR_FloatTensor:$out_im);
  let assemblyFormat = [{
    $re `,` $im `,` $fa `,` $fr attr-dict `:` functional-type(operands, results)
  }];
  let hasVerifier = 1;
}

def SAR_StoltInterpOp : SAR_Op<"stolt_interp", [Pure]> {
  let summary = "Stolt interpolation (omega-K frequency remapping)";
  let description = [{
    Remaps the 2-D range-Doppler spectrum onto the Stolt grid using windowed
    sinc interpolation, as required by the omega-K (wavenumber-domain) SAR
    imaging algorithm.

    Operands:
    - `data`: 2-D complex spectrum of shape `A x R` (azimuth x range).
    - `fa`: azimuth frequency axis, 1-D float tensor of length `A`.
    - `fr`: range frequency axis, 1-D float tensor of length `R`.

    Attributes:
    - `c`: propagation speed (m/s).
    - `fc`: carrier frequency (Hz).
    - `vr`: effective platform velocity (m/s).
    - `t_shift`: reference time shift (s) applied before interpolation and
      compensated afterwards.

    For every output sample `(i, j)`:
    ```
    f_q  = sqrt(max((fr[j] + fc)^2 + (c * fa[i] / (2 vr))^2, 1e-10)) - fc
    out[i, j] = exp(-2 pi j f_q t_shift) *
                sum_{k=-3}^{4} data[i, k'] * exp(2 pi j fr[k'] t_shift) * w(d)
    ```
    where `k' = floor((f_q - fr[0]) / dfr) + k`, `d` is the fractional
    distance to tap `k'`, and `w(d) = sinc(d) * (0.5 + 0.5 cos(pi d / 4))`.
  }];
  let arguments = (ins SAR_ComplexTensor:$data,
                       SAR_FloatTensor:$fa,
                       SAR_FloatTensor:$fr,
                       F64Attr:$c,
                       F64Attr:$fc,
                       F64Attr:$vr,
                       F64Attr:$t_shift);
  let results = (outs SAR_ComplexTensor:$result);
  let assemblyFormat = [{
    $data `,` $fa `,` $fr attr-dict `:` functional-type(operands, results)
  }];
  let hasVerifier = 1;
}

#endif // SAR_DIALECT_SAR_IR_SAROPS_TD
