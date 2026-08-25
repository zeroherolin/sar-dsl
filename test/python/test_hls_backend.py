"""HLS backend tests: HLS C++ emission and subset diagnostics."""

import json
import re
from pathlib import Path

import numpy as np
import pytest

import sar

from conftest import REPO_ROOT, requires_hls, run_hls_csim

pytestmark = requires_hls

N = 16


def test_emit_elementwise_kernel():

    @sar.func
    def phase_mul(re: sar.f32[N, N], im: sar.f32[N, N], cosp: sar.f32[N, N],
                  sinp: sar.f32[N, N]) -> (sar.f32[N, N], sar.f32[N, N]):
        return re * cosp - im * sinp, re * sinp + im * cosp

    design = phase_mul.compile(backend="hls")
    source = design.source()
    assert "void phase_mul" in source
    assert "#pragma HLS" in source


def test_top_ports_carry_kernel_parameter_names():
    """The emitted top signature names its ports after the kernel's Python
    parameters; complex tensors keep the name on both planes and result
    planes stay `out<i>`."""

    @sar.func
    def named(raw: sar.c64[N, N], win_r: sar.f64[N]) -> sar.f32[N, N]:
        return sar.cast(sar.absolute(sar.fft(raw, axis=1) * win_r), sar.f32)

    source = named.compile(backend="hls").source()
    signature = source[source.index("void named("):]
    signature = signature[:signature.index(")")]
    for port in ("raw_re", "raw_im", "win_r", "out0"):
        assert port in signature, signature


def test_reserved_parameter_names_fall_back():
    """A parameter name the emitter cannot use verbatim (a C++ keyword, or
    the shape of a generated name) falls back to the role-based scheme
    instead of emitting C++ that does not compile."""

    @sar.func
    def kw(float: sar.f32[N, N],
           in0: sar.f32[N, N]) -> sar.f32[N, N]:  # noqa: A002
        return float + in0

    source = kw.compile(backend="hls").source()
    signature = source[source.index("void kw("):]
    signature = signature[:signature.index(")")]
    assert "in0" in signature and "in1" in signature
    assert "float float" not in signature


def test_hdl_reserved_parameter_names_fall_back():
    """Names that become illegal only in RTL still get a stable C++ ABI."""
    plane = sar.f32[N, N]

    @sar.func
    def hdl_names(signal: plane, shared: plane) -> plane:
        return signal + shared

    source = hdl_names.compile(backend="hls").source()
    signature = source[source.index("void hdl_names("):]
    signature = signature[:signature.index(")")]
    assert "in0" in signature and "in1" in signature
    assert " signal[" not in signature and " shared[" not in signature


def test_hls_design_is_not_executable():

    @sar.func
    def k(a: sar.f32[N, N]) -> sar.f32[N, N]:
        return a * 2.0

    design = k.compile(backend="hls")
    with pytest.raises(sar.errors.LaunchError, match="cannot be executed"):
        design()


def test_hls_design_survives_cache_artifact_eviction(tmp_path):
    n = 4

    @sar.func
    def scale(x: sar.f32[n]) -> sar.f32[n]:
        return x * 2.0

    design = scale.compile(backend="hls")
    source = design.source()
    assert "// SAR_DSL_INTERFACE: {" in source
    Path(design.cpp_path).unlink()
    assert design.source() == source

    synthesis_dir = tmp_path / "synthesis"
    design.write_synthesis_script(synthesis_dir)
    packaged = (synthesis_dir / "scale.cpp").read_text()
    assert '#include "scale.h"' in packaged
    assert "SAR_DSL_DECLARATIONS_BEGIN" not in packaged
    assert "Sub-function prototypes" not in packaged
    assert packaged.index("void scale(") < packaged.index(
        "static T sar_hls_maximum(T lhs, T rhs) {")
    assert "\n\n\n" not in packaged
    assert "void scale(" in (synthesis_dir / "scale.h").read_text()
    manifest = json.loads((synthesis_dir / "design_manifest.json").read_text())
    assert manifest["schema_version"] == 2
    assert manifest["top"] == "scale"
    assert manifest["generator"]["backend"] == "hls"
    assert manifest["kernel"]["arguments"][0]["dtype"] == "f32"
    assert manifest["config"]["interface"] == "axi"
    assert manifest["optimization_plan"]["clock_ns"] == 4.0
    assert manifest["optimization_plan"]["timing_budget_ns"] == 3.5
    assert "max_scratch_arenas" in manifest["optimization_plan"]["values"]
    assert manifest["interfaces"][0]["logical_elements"] == n
    assert manifest["interfaces"][0]["vector_lanes"] == 1

    x = np.arange(n, dtype=np.float32)
    testbench_dir = tmp_path / "testbench"
    design.write_testbench([x], [x * 2.0], testbench_dir)
    assert '#include "scale.h"' in (testbench_dir / "scale.cpp").read_text()
    assert (testbench_dir / "scale.h").is_file()
    # No constant tables in this kernel, so no ROM header is written.
    assert not design.has_tables()
    assert not (testbench_dir / "scale_tables.h").exists()


def test_constant_tables_are_packaged_as_their_own_header(tmp_path):
    """ROM tables leave the implementation file.

    A transform's twiddle tables routinely outweigh the logic that reads
    them, so they are packaged as `<top>_tables.h` and the implementation
    includes it. Both files together must still be the same translation
    unit the single-file translation output is.
    """
    n = 64

    @sar.func
    def transform(x: sar.c64[n]) -> sar.c64[n]:
        return sar.fft(x, dim=0)

    design = transform.compile(backend="hls")
    assert design.has_tables()
    out = tmp_path / "package"
    design.write_synthesis_script(out)

    tables = (out / "transform_tables.h").read_text()
    packaged = (out / "transform.cpp").read_text()
    assert '#include "transform_tables.h"' in packaged
    assert "SAR_DSL_TABLES_BEGIN" not in packaged
    assert "Constant tables" not in packaged
    # The tables themselves moved; the code that reads them did not.
    assert "kTwiddleSin_64S0" in tables
    assert "kTwiddleSin_64S0[" not in tables.split("static const")[0]
    assert "kTwiddleSin_64S0" in packaged
    assert "static const float kTwiddleSin_64S0" not in packaged


def test_manifest_records_the_physical_vector_interface(tmp_path):
    n = 64

    @sar.func
    def scale(x: sar.f64[n, n]) -> sar.f64[n, n]:
        return x * 2.0

    design = scale.compile(backend="hls", options={"interface": "axi"})
    design.write_synthesis_script(tmp_path)
    manifest = json.loads((tmp_path / "design_manifest.json").read_text())
    port = next(item for item in manifest["interfaces"] if item["name"] == "x")
    output = next(item for item in manifest["interfaces"]
                  if item["direction"] == "out")
    assert port["protocol"] == "m_axi"
    assert port["direction"] == "in"
    assert port["vector_lanes"] == 8
    assert port["data_bits"] == 512
    assert port["physical_shape"] == [n, n // 8]
    assert port["logical_elements"] == n * n
    assert output["vector_lanes"] == 8
    assert output["data_bits"] == 512
    assert output["physical_shape"] == [n, n // 8]
    assert output["logical_elements"] == n * n
    values = np.arange(n * n, dtype=np.float64).reshape(n, n)
    design.write_testbench([values], [values * 2.0], tmp_path)
    run_hls_csim(tmp_path, "scale")


@pytest.mark.parametrize("name,value", [
    ("rtol", -1.0),
    ("rtol", float("nan")),
    ("atol", float("inf")),
    ("atol", "small"),
])
def test_testbench_rejects_invalid_tolerances(tmp_path, name, value):

    @sar.func
    def scale(x: sar.f32[4]) -> sar.f32[4]:
        return x * 2.0

    values = np.ones(4, dtype=np.float32)
    kwargs = {name: value}
    design = scale.compile(backend="hls")
    with pytest.raises(sar.LaunchError, match="finite non-negative"):
        design.write_testbench([values], [values * 2.0], tmp_path, **kwargs)
    assert not list(tmp_path.iterdir())


def test_integer_ports_hls_csim_against_exact_golden_data(tmp_path):
    """Integer arguments and results reach the validation package.

    Integer planes are written and compared as exact binary values: a 64-bit
    value does not survive a round trip through the double oracle the
    floating planes use, and equality is the only meaningful tolerance.
    """
    n = 32

    @sar.func
    def mixed(counts: sar.i64[n],
              gain: sar.f64[n]) -> (sar.i64[n], sar.f64[n]):
        return counts + counts, gain * 2.0

    # Values above 2^53, where a double oracle would already have rounded.
    counts = (np.arange(n, dtype=np.int64) + 1) * 1234567890123
    gain = np.linspace(0.5, 3.0, n)
    design = mixed.compile(backend="hls")
    design.write_testbench([counts, gain], [counts + counts, gain * 2.0],
                           tmp_path)

    stored = np.fromfile(tmp_path / "mixed_tb_data" / "out0.bin",
                         dtype=np.int64)
    np.testing.assert_array_equal(stored, counts + counts)

    run_hls_csim(tmp_path, "mixed")


def test_integer_ports_hls_csim_over_axi_and_stream(tmp_path):
    """The exact-comparison path also covers the physical port ABIs."""
    n = 256

    @sar.func
    def twice(x: sar.i32[n]) -> sar.i32[n]:
        return x + x

    x = (np.arange(n, dtype=np.int32) % 200) - 100
    for interface in ("axi", "stream"):
        out = tmp_path / interface
        design = twice.compile(backend="hls", options={"interface": interface})
        design.write_testbench([x], [x + x], out)
        run_hls_csim(out, "twice")


def test_integer_golden_data_must_be_integral(tmp_path):
    """A float oracle for an integer result is a silently rounded
    comparison, so it is refused rather than truncated."""

    @sar.func
    def twice(x: sar.i32[4]) -> sar.i32[4]:
        return x + x

    x = np.arange(4, dtype=np.int32)
    design = twice.compile(backend="hls")
    with pytest.raises(sar.LaunchError, match="integer dtype"):
        design.write_testbench([x], [(x + x).astype(np.float64)], tmp_path)


def test_testbench_preserves_high_precision_golden_data(tmp_path):

    @sar.func
    def identity(x: sar.f32[2]) -> sar.f32[2]:
        return x

    values = np.ones(2, dtype=np.float32)
    reference = np.array([1.00000006, 0.99999997], dtype=np.float64)
    identity.compile(backend="hls").write_testbench([values], [reference],
                                                    tmp_path)
    stored = np.fromfile(tmp_path / "identity_tb_data" / "out0.bin",
                         dtype=np.float64)
    np.testing.assert_array_equal(stored, reference)


def test_testbench_rejects_unbounded_static_memory(tmp_path, monkeypatch):

    @sar.func
    def identity(x: sar.f32[16]) -> sar.f32[16]:
        return x

    values = np.ones(16, dtype=np.float32)
    monkeypatch.setenv("SAR_DSL_HLS_TESTBENCH_MAX_BYTES", "32")
    with pytest.raises(sar.LaunchError, match="static arrays"):
        identity.compile(backend="hls").write_testbench([values], [values],
                                                        tmp_path)
    assert not list(tmp_path.iterdir())


def test_explicit_production_testbench_disables_size_guard(
        tmp_path, monkeypatch):

    @sar.func
    def identity(x: sar.f32[16]) -> sar.f32[16]:
        return x

    values = np.ones(16, dtype=np.float32)
    monkeypatch.setenv("SAR_DSL_HLS_TESTBENCH_MAX_BYTES", "32")
    identity.compile(backend="hls").write_testbench([values], [values],
                                                    tmp_path,
                                                    max_bytes=0)
    assert (tmp_path / "identity_tb.cpp").is_file()


def test_fft_kernel_emits_via_affine_flow():
    """Complex kernels with FFTs go through decomplexify + Stockham affine
    lowering and the HLS pipeline."""

    @sar.func
    def spectrum(a: sar.c64[N, N], p: sar.f32[N, N]) -> sar.c64[N, N]:
        return sar.fftshift(sar.fft(a * sar.expj(p), dim=1), dim=1)

    design = spectrum.compile(backend="hls")
    source = design.source()
    assert "void spectrum" in source
    assert "#pragma HLS" in source
    # The Stockham twiddle tables become constant arrays.
    assert "twiddle" in source or "float v" in source


def test_interp_kernel_emits_via_affine_flow():

    @sar.func
    def k(d: sar.c128[N, N], p: sar.f64[N, N]) -> sar.c128[N, N]:
        return sar.interp1d(d, p)

    design = k.compile(backend="hls")
    assert "void k" in design.source()


def test_composed_vocabulary_emits_hls():
    """Trace-time compositions the symmetry gate does not carry -- the
    statistics vocabulary and the Stolt remapping -- must reach HLS C++
    like the primitives they decompose into."""
    n = 16

    @sar.func
    def stats(x):
        m = sar.multilook(x * x, (2, 2))
        col = sar.std(x, axis=1) + sar.mean(x, axis=1)
        return m + sar.broadcast(col[:n // 2], (n // 2, n // 2), dim=0)

    fa = np.fft.fftshift(np.fft.fftfreq(n, d=1 / 2000.0))
    fr = np.fft.fftshift(np.fft.fftfreq(n, d=1 / 3.2e7))

    @sar.func
    def stolt(z):
        return sar.stolt_interp(z,
                                fa,
                                fr,
                                c=3e8,
                                fc=1.27e9,
                                vr=7100.0,
                                t_shift=1.5e-4)

    cases = [
        (stats, (sar.f64[n, n], )),
        (stolt, (sar.c128[n, n], )),
    ]
    for fn, types in cases:
        design = fn.specialize(*types).compile(backend="hls")
        assert "#pragma HLS" in design.source(), fn


def test_testbench_hls_csim_passes(tmp_path):
    """The emitted design reproduces golden data in C-sim through Vitis HLS."""
    import numpy as np

    n = 16

    @sar.func
    def phase(z: sar.c128[n, n], g: sar.f64[n, n]) -> sar.c128[n, n]:
        return z * sar.expj(g)

    design = phase.compile(backend="hls")

    rng = np.random.default_rng(2)
    z = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    g = rng.uniform(-3.0, 3.0, (n, n))
    design.write_testbench([z, g], [z * np.exp(1j * g)], tmp_path, rtol=1e-12)

    run_hls_csim(tmp_path, "phase")
    assert (tmp_path / "phase_hls_csim.tcl").exists()
    assert (tmp_path / "phase_portable_cpp_sim.sh").exists()
    assert (tmp_path / "phase_csynth.tcl").exists()


def test_stream_testbench_has_a_numerical_oracle(tmp_path):

    @sar.func
    def stream_scale(x: sar.f32[16]) -> sar.f32[16]:
        return x * 3.0

    values = np.arange(16, dtype=np.float32)
    design = stream_scale.compile(backend="hls",
                                  options={"interface": "stream"})
    design.write_testbench([values], [values * 3.0], tmp_path)
    run_hls_csim(tmp_path, "stream_scale")
    manifest = json.loads((tmp_path / "design_manifest.json").read_text())
    assert {port["protocol"] for port in manifest["interfaces"]} == {"axis"}


def test_nan_semantics_match_numpy_in_emitted_cpp(tmp_path):

    @sar.func
    def nan_ops(
        x: sar.f64[4]
    ) -> (sar.f64[4], sar.f64[4], sar.f64[4], sar.f64[1], sar.f64[1]):
        return (x != 0.0, sar.maximum(x, 1.0), sar.minimum(x, 1.0), sar.max(x),
                sar.min(x))

    x = np.array([0.0, np.nan, 2.0, -2.0])
    expected = [
        (x != 0.0).astype(np.float64),
        np.maximum(x, 1.0),
        np.minimum(x, 1.0),
        np.asarray([np.max(x)]),
        np.asarray([np.min(x)]),
    ]
    design = nan_ops.compile(backend="hls")
    design.write_testbench([x], expected, tmp_path)
    run_hls_csim(tmp_path, "nan_ops")


def test_maximum_minimum_preserve_nan_and_signed_zero(tmp_path):
    import subprocess

    from sar.compiler.toolchain import find_tool

    @sar.func
    def fp_edges(x: sar.f64[4]) -> (sar.f64[4], sar.f64[4]):
        return sar.maximum(x, -0.0), sar.minimum(x, 0.0)

    design = fp_edges.compile(backend="hls")
    (tmp_path / "fp_edges.cpp").write_text(design.source())
    (tmp_path / "edge_tb.cpp").write_text(r"""\
#include <cmath>
#include "fp_edges.cpp"
int main() {
  double x[4] = {-0.0, 0.0, NAN, 2.0};
  double maximum[4] = {};
  double minimum[4] = {};
  fp_edges(x, maximum, minimum);
  if (!std::signbit(maximum[0]) || std::signbit(maximum[1]))
    return 1;
  if (!std::signbit(minimum[0]) || std::signbit(minimum[1]))
    return 2;
  if (!std::isnan(maximum[2]) || !std::isnan(minimum[2]))
    return 3;
  return 0;
}
""")
    result = subprocess.run([
        find_tool("clang") + "++", "-O2", "-Wno-unknown-pragmas",
        "edge_tb.cpp", "-o", "edge_tb"
    ],
                            cwd=tmp_path,
                            capture_output=True,
                            text=True)
    assert result.returncode == 0, result.stderr
    result = subprocess.run(["./edge_tb"], cwd=tmp_path)
    assert result.returncode == 0


def test_testbench_rejects_nan_mismatch(tmp_path):

    @sar.func
    def identity(x: sar.f64[4]) -> sar.f64[4]:
        return x

    x = np.arange(4, dtype=np.float64)
    expected = x.copy()
    expected[1] = np.nan
    identity.compile(backend="hls").write_testbench([x], [expected], tmp_path)
    _, result = run_hls_csim(tmp_path, "identity", expect_success=False)
    assert "FAIL" in result.stdout


def test_f32_chain_hls_csim_matches_the_cpu_backend(tmp_path):
    """The HLS x f32 leg of the precision matrix.

    A chain built with `dtype=sar.c64` must emit, simulate, and agree with
    the same build on the cpu backend to single-precision rounding --
    the cpu FFT computes its butterflies in f64 while the HLS FFT
    computes in the declared width, so this is a real cross-backend
    check, not a self-comparison.
    """
    import numpy as np

    from common.params import synthetic_params
    from common.simulate import single_target_scene
    from wka.algorithm import build_kernel, make_inputs

    n = 32
    params = synthetic_params(n)
    kernel = build_kernel(n, params, name="wka_f32", dtype=sar.c64)
    raw = single_target_scene(n, params).astype(np.complex64)
    inputs = [
        np.asarray(a, dtype=t.dtype.to_numpy())
        for a, t in zip([raw, *make_inputs(n, params)], kernel.arg_types)
    ]

    golden = kernel.compile("cpu")(*inputs)
    peak = float(golden.max())
    assert peak > 1.0  # a focused target, so the tolerances below bite

    design = kernel.compile(backend="hls")
    design.write_testbench(inputs, [golden],
                           tmp_path,
                           rtol=1e-3,
                           atol=peak * 1e-3)
    run_hls_csim(tmp_path, "wka_f32")


def test_mixed_precision_axi_uses_typed_scratch_arenas():
    """Spilled planes cost one port per element type, not one per plane.

    A port is a platform resource an integrator wires to a memory channel,
    so the signature is the algorithm's own I/O plus the scratch the design
    genuinely needs. Two element types spill here, and a typed C++ signature
    cannot express both through one pointer. Each type may need several
    conflict-colored ping-pong arenas, but never more than the derived cap.
    """
    n = 64

    @sar.func
    def kernel(data: sar.c64[n, n], geometry: sar.f64[n, n]) -> sar.c64[n, n]:
        narrow = sar.fft(data, axis=1)
        wide_input = sar.make_complex(geometry, geometry * 0.0)
        wide = sar.fft(wide_input, axis=1)
        return narrow + sar.cast(wide, sar.c64)

    kernel.name = "mixed_scratch"
    design = kernel.compile(backend="hls",
                            options={
                                "interface": "axi",
                                "external_buffer_threshold": n * n
                            })
    source = design.source()
    signature = source[source.index(f"void {design.name}("):]
    signature = signature[:signature.index(") {")]
    args = [line.strip().rstrip(",") for line in signature.splitlines()[1:]]
    io_ports = sum(2 if t.dtype.is_complex else 1
                   for t in kernel.arg_types + kernel.declared_result_types)
    scratch = args[io_ports:]

    def scalar_type(declaration):
        vector = re.match(r"hls::vector<(float|double),", declaration)
        return vector.group(1) if vector else declaration.split(" ", 1)[0]

    scratch_types = [scalar_type(arg) for arg in scratch]
    assert set(scratch_types) == {"float", "double"}
    assert all(
        scratch_types.count(kind) <= design.config.max_scratch_arenas
        for kind in set(scratch_types))
    assert all(re.search(r"\[\d+\]$", arg) for arg in scratch)


def test_twiddle_tables_are_shared_across_same_size_ffts():
    """Transforms of one size share one cos/sin table per stage.

    The emitter lifts twiddle tables to file-scope `const` arrays so the
    tool can map them to ROM; a design whose three same-size transforms
    each carried their own copy would spend BRAM primitives per
    transform. One definition per table is the static half of the ROM
    budget; whether Vitis then shares the primitives is read from the
    synthesis report (docs/backends.md).
    """
    n = 32

    @sar.func
    def three(a: sar.c128[n, n], b: sar.c128[n, n]) -> sar.c128[n, n]:
        return (sar.fft(a, axis=1) * sar.fft(b, axis=1) * sar.fft(a, axis=1))

    source = three.compile(backend="hls").source()
    # The final four-point stage has only 1/0 twiddles; canonicalization folds
    # those constants away, so only the two non-trivial stage tables remain.
    for stage in range(2):
        cos = f"kTwiddleCos_{n}S{stage}"
        sin = f"kTwiddleSin_{n}S{stage}"
        assert source.count(cos + "[") == 1, source
        assert source.count(sin + "[") == 1, source
        # The uniquing suffix appearing would mean a duplicate slipped in
        # under another name.
        assert cos + "_2" not in source
    assert f"kTwiddleCos_{n}S2" not in source


def test_synthesis_script_covers_every_interface(tmp_path):
    """`write_synthesis_script` exists for designs C simulation cannot cover:
    synthesis needs no golden data, so the AXI/stream designs get their
    script from here. The scripts name the configured part and clock."""
    n = 16

    @sar.func
    def scale(z: sar.c128[n, n]) -> sar.c128[n, n]:
        return z * 2.0

    for interface in ("ap_memory", "axi", "stream"):
        design = scale.compile(backend="hls",
                               options={
                                   "interface": interface,
                                   "part": "xczu9eg-ffvb1156-2-e",
                                   "clock_ns": 4,
                               })
        out = tmp_path / interface
        script = design.write_synthesis_script(out)
        assert script == out / "scale_csynth.tcl"
        assert '#include "scale.h"' in (out / "scale.cpp").read_text()
        assert "void scale(" in (out / "scale.h").read_text()
        tcl = script.read_text()
        assert "set_top scale" in tcl
        assert "csynth_design" in tcl
        assert "set_part xczu9eg-ffvb1156-2-e" in tcl
        assert "create_clock -period 4" in tcl
        assert "set_clock_uncertainty 12.5%" in tcl
        assert "config_interface -m_axi_alignment_byte_size=64" in tcl
        assert "config_interface -m_axi_max_read_burst_length=64" in tcl
        assert "scale_csynth_elapsed_s.txt" in tcl
        assert "synthesis report is missing resource estimates" in tcl
        # Timing is a goal and resources are budgets, so the script treats
        # them differently: a missed clock is reported and synthesis
        # continues, an over-budget resource aborts it.
        assert "SAR-DSL WARNING: estimated clock" in tcl
        assert 'format "%.3f" $_sar_estimated_clock' in tcl
        assert 'error "SAR-DSL: $_sar_name usage' in tcl
        assert "BRAM_18K" in tcl
        assert "URAM" in tcl
        assert "DSP" in tcl


def test_scratch_is_one_line_wide():
    """The FFT scratch must not scale with the scene.

    Stockham works a line at a time, so the scratch buffers are one
    transform long whatever the raster height. That is what keeps the
    working set on chip -- a full-raster scratch would be 1 GiB at the ALOS
    size -- and it is why the top function stays down to the kernel's own
    arguments instead of growing a port per scratch plane.
    """
    import re

    for n in (64, 256):

        @sar.func
        def spectrum(x: sar.c64[n, n]) -> sar.c64[n, n]:
            return sar.fft(x, axis=1)

        spectrum.name = f"scratch_{n}"
        design = spectrum.compile(backend="hls", options={"interface": "axi"})
        source = design.source()

        # No full-raster array may be declared anywhere.
        assert not re.findall(rf"float v\d+\[{n}\]\[{n}\];", source), n
        # The top function takes the kernel's planes, not a port per buffer.
        signature = re.search(r"^void scratch_\d+\((.*?)\n\) \{", source,
                              re.S | re.M)
        assert signature, source[:200]
        assert signature.group(1).count("float v") <= 8, signature.group(1)


def test_stages_become_a_dataflow_region():
    """The transform stages must reach the backend as a pipelineable chain.

    Each Stockham stage writes its own scratch line, so the stages form a
    chain rather than a cycle and the backend can schedule them -- which is
    what earns the `#pragma HLS dataflow` on the line loop, letting one
    line's late stages overlap the next line's early ones. Sharing two
    scratch buffers in ping-pong would be smaller but unschedulable, and the
    whole transform would run one stage at a time.
    """
    N = 64

    @sar.func
    def spectrum(x: sar.c64[N, N]) -> sar.c64[N, N]:
        return sar.fft(x, axis=1)

    source = spectrum.compile(backend="hls").source()
    assert "#pragma HLS dataflow" in source


def test_partition_factors_stay_synthesizable():
    """Array partitioning must not degenerate into per-element registers.

    A factor equal to the transform length would split one block RAM into
    N registers behind a crossbar, which no device can place. Banks only
    have to cover the accesses in flight.
    """
    import re

    N = 256

    @sar.func
    def spectrum(x: sar.c64[N, N]) -> sar.c64[N, N]:
        return sar.fft(x, axis=1)

    source = spectrum.compile(backend="hls").source()
    factors = [
        int(f)
        for f in re.findall(r"array_partition \S+ \w+ factor=(\d+)", source)
    ]
    assert factors, "expected the backend to partition the scratch lines"
    assert max(factors) <= 32, sorted(set(factors))


def test_axi_testbench_drives_output_and_emits_cosim(tmp_path):
    """An AXI package drives its named output and carries an RTL co-sim
    script."""
    N = 32

    @sar.func
    def spectrum(x: sar.c64[N, N]) -> sar.c64[N, N]:
        return sar.fft(x, axis=1)

    rng = np.random.default_rng(8)
    values = (rng.standard_normal((N, N)) + 1j * rng.standard_normal(
        (N, N))).astype(np.complex64)
    golden = spectrum.compile("cpu")(values)
    design = spectrum.compile(backend="hls", options={"interface": "axi"})
    design.write_testbench([values], [golden], tmp_path)
    assert (tmp_path / "spectrum_cosim.tcl").is_file()
    testbench = (tmp_path / "spectrum_tb.cpp").read_text()
    assert "out0" in testbench


def test_streamed_planes_are_shared_across_lifetimes():
    """A streamed chain must not hold one plane per intermediate.

    An imaging chain is a sequence of whole-raster passes, so a plane goes
    dead as soon as the next pass has consumed it. Giving every intermediate
    its own allocation is what puts an ALOS-size design past any board's
    DRAM; sharing the ones whose lifetimes do not overlap is what brings it
    back. The count has to stay flat as the chain grows, not track its
    length.
    """
    n = 64

    @sar.func
    def short(x: sar.c64[n, n]) -> sar.c64[n, n]:
        return sar.ifft(sar.fft(x * 2.0, axis=1) + 1.0, axis=1)

    @sar.func
    def long(x: sar.c64[n, n]) -> sar.c64[n, n]:
        z = x
        for _ in range(4):
            z = sar.ifft(sar.fft(z * 2.0, axis=1) + 1.0, axis=1)
        return z

    def ports(fn, name):
        fn.name = name
        source = fn.compile(backend="hls",
                            options={
                                "interface": "axi",
                                "bram_bytes": 1 << 22,
                                "uram_bytes": 0,
                                "lutram_bytes": 0
                            }).source()
        return source.count("#pragma HLS interface m_axi")

    short_ports = ports(short, "share_short")
    long_ports = ports(long, "share_long")
    # Four times the passes must not cost four times the planes.
    assert long_ports < 2 * short_ports, (short_ports, long_ports)


def test_broadcast_does_not_materialize_a_plane():
    """Broadcasting a frequency axis must not cost a full raster.

    Every SAR chain multiplies a raster by something that varies along one
    axis only. Materializing that as a plane costs a full-size buffer
    holding one row of distinct values, plus a pass over it -- so the
    broadcast has to fold into the consumer that reads it.
    """
    import re

    import numpy as np

    N = 256
    axis = np.linspace(-1.0, 1.0, N)

    @sar.func
    def phased(x: sar.f64[N, N]) -> sar.f64[N, N]:
        return x * sar.broadcast(axis, (N, N), dim=1)

    source = phased.compile(backend="hls",
                            options={
                                "interface": "axi",
                                "bram_bytes": 1 << 20,
                                "uram_bytes": 0,
                                "lutram_bytes": 0
                            }).source()
    # One plane in, one out: a materialized broadcast would be a third DRAM
    # buffer, and with no port of its own to hide in it would have to open a
    # scratch arena. So the check is that the design's masters are exactly
    # the kernel's two planes -- no arena, hence nothing was materialized.
    maxi_ports = re.findall(r"m_axi.*port=(\w+)", source)
    assert len(maxi_ports) == 2, maxi_ports


def test_streaming_passes_keep_full_rows():
    """A pass that streams DRAM must sweep whole rows.

    Tiling exists to capture reuse; an elementwise pass over a raster has
    none, and splitting its contiguous dimension turns one long AXI burst
    per row into a burst per tile. Burst length is not recoverable later, so
    the row has to survive tiling intact.
    """
    import re

    N = 512

    @sar.func
    def scale(x: sar.f64[N, N]) -> sar.f64[N, N]:
        return x * 2.0 + 1.0

    source = scale.compile(backend="hls",
                           options={
                               "interface": "axi",
                               "bram_bytes": 1 << 20,
                               "uram_bytes": 0,
                               "lutram_bytes": 0
                           }).source()
    trips = [
        int(t) for t in re.findall(r"for \(int(?:64_t)? \w+ = 0; \w+ < (\d+);",
                                   source)
    ]
    assert N in trips, sorted(set(trips))


def test_axi_ports_are_shaped_for_bandwidth():
    """AXI masters must be told how wide a beat and how long a burst is.

    Left at the defaults a port moves one element per beat, which on a
    512-bit bus wastes seven eighths of it, and issues a fresh 16-beat burst
    where the row supports a full-length one.
    """
    import re

    N = 512

    @sar.func
    def scale(x: sar.f64[N, N]) -> sar.f64[N, N]:
        return x * 2.0

    source = scale.compile(backend="hls",
                           options={
                               "interface": "axi",
                               "bram_bytes": 1 << 20,
                               "uram_bytes": 0,
                               "lutram_bytes": 0
                           }).source()
    widen = {int(w) for w in re.findall(r"max_widen_bitwidth=(\d+)", source)}
    burst = {
        int(b)
        for b in re.findall(r"max_read_burst_length=(\d+)", source)
    }
    # A beat is the full bus; the burst is however many beats a row spans,
    # here 512 doubles over a 512-bit bus.
    assert widen == {512}, widen
    assert burst == {N * 64 // 512}, burst


def test_scratch_lands_in_block_ram():
    """Transform scratch must not be bound to distributed RAM.

    Distributed RAM is built out of LUTs, so a bank's cost is its bit count.
    A 512-deep bank of doubles is 32 Kbit -- more than a block RAM primitive
    holds -- and a chain carrying a hundred of them would ask for several
    times the distributed RAM any device has, which fails synthesis rather
    than merely running slowly.
    """
    N = 4096

    @sar.func
    def spectrum(x: sar.c128[N, N]) -> sar.c128[N, N]:
        return sar.fft(x, axis=1)

    source = spectrum.compile(backend="hls", options={
        "interface": "axi"
    }).source()
    assert "bind_storage" in source
    assert "impl=lutram" not in source


def test_axi_interface_is_a_contract_not_a_placement_result():
    """`interface` says how the host binds the kernel, so it cannot depend
    on where the compiler put a buffer. An AXI design is AXI whether or not
    its planes happened to fit on chip -- otherwise the host would have to
    bind a different interface per budget, and `bundle=` would land on a
    `bram` port, which Vitis does not accept."""

    def emit(n, **options):

        @sar.func
        def spectrum(x: sar.c64[n, n]) -> sar.c64[n, n]:
            return sar.fft(x, axis=1)

        spectrum.name = f"iface_{n}_{options.get('bram_bytes', 'def')}"
        return spectrum.compile(backend="hls",
                                options={
                                    "interface": "axi",
                                    **options
                                }).source()

    # Comfortably resident, forced to stream, and a larger raster on the
    # shipped caps.
    for source in (emit(64),
                   emit(64, bram_bytes=1 << 19, uram_bytes=0,
                        lutram_bytes=0), emit(256)):
        modes = set(re.findall(r"#pragma HLS interface (\w+)", source))
        assert modes == {"m_axi", "s_axilite"}, modes
        # `bundle=` belongs to m_axi / axis / s_axilite, never to a memory
        # port; emitting it on one is the malformed pragma this guards.
        assert not [
            line for line in source.splitlines()
            if "interface bram" in line and "bundle=" in line
        ]


def test_local_interface_emits_memory_ports():
    """An explicit ap_memory interface emits local block-memory arrays."""
    n = 64

    @sar.func
    def spectrum(x: sar.c64[n, n]) -> sar.c64[n, n]:
        return sar.fft(x, axis=1)

    spectrum.name = "iface_local"
    source = spectrum.compile(backend="hls",
                              options={
                                  "interface": "ap_memory"
                              }).source()
    modes = set(re.findall(r"#pragma HLS interface (\w+)", source))
    # `bram` is Vitis's spelling of the ap_memory protocol in the pragma.
    assert modes == {"bram", "s_axilite"}, modes
    assert "m_axi" not in source


def test_local_interface_rejects_off_chip_spill():
    """ap_memory has no external master for an intermediate or argument."""
    n = 64

    @sar.func
    def scale(x: sar.f32[n, n]) -> sar.f32[n, n]:
        return x * 2.0

    scale.name = "iface_local_overflow"
    with pytest.raises(sar.CompilationError, match="DRAM spill is disabled"):
        scale.compile(backend="hls",
                      options={
                          "interface": "ap_memory",
                          "bram_bytes": 4608,
                          "uram_bytes": 0,
                          "lutram_bytes": 0,
                      })


def test_stream_interface_emits_axis_pragmas():
    """A proven single row-major sweep reaches AXI4-Stream pragmas."""
    n = 64

    @sar.func
    def spectrum(x: sar.f32[n, n]) -> sar.f32[n, n]:
        return x * 2.0

    spectrum.name = "iface_stream"
    source = spectrum.compile(backend="hls", options={
        "interface": "stream"
    }).source()
    # The kernel's inputs and outputs stream, and nothing spilled here, so
    # there is no scratch arena to keep memory-mapped: a design that needs
    # no DRAM declares no AXI master at all.
    axis_ports = set(re.findall(r"interface axis port=(\w+)", source))
    assert "x" in axis_ports and len(axis_ports) == 2, axis_ports
    assert "hls::stream<float> &x" in source
    assert ".read()" in source and ".write(" in source
    assert not re.search(r"interface axis port=\w+ bundle=", source)
    maxi_ports = re.findall(r"m_axi.*port=(\w+)", source)
    assert not maxi_ports, maxi_ports
    # A stream port carries no burst shaping: that describes addressed
    # access to DRAM, which a FIFO does not perform.
    assert "max_read_burst_length" not in source


def test_stream_interface_rejects_nonsequential_access():
    """FFT traffic is addressed and cannot silently become an AXI stream."""
    n = 32

    @sar.func
    def two_ffts(x: sar.c64[n, n]) -> sar.c64[n, n]:
        y = sar.fft(x, axis=1)
        return sar.fft(y, axis=0)

    two_ffts.name = "iface_stream_scratch"
    with pytest.raises(sar.CompilationError,
                       match="complete monotonic row-major access sweep"):
        two_ffts.compile(backend="hls", options={"interface": "stream"})


def test_stream_interface_rejects_square_transpose():
    """Equal extents do not make a column-major sweep row-major."""
    n = 32

    @sar.func
    def transpose(x: sar.f32[n, n]) -> sar.f32[n, n]:
        return sar.transpose(x)

    transpose.name = "iface_stream_transpose"
    with pytest.raises(sar.CompilationError,
                       match="complete monotonic row-major access sweep"):
        transpose.compile(backend="hls", options={"interface": "stream"})


@pytest.mark.parametrize("interface", ["ap_memory", "axi", "stream"])
def test_no_memory_port_ever_carries_a_bundle(interface):
    """`bundle=` belongs to m_axi / axis / s_axilite and is invalid on a
    memory-mode port."""
    n = 32

    @sar.func
    def spectrum(x: sar.f32[n, n]) -> sar.f32[n, n]:
        return x * 2.0

    spectrum.name = f"iface_nobundle_{interface}"
    source = spectrum.compile(backend="hls",
                              options={
                                  "interface": interface,
                                  "top_func": "nobundle_top"
                              }).source()
    offenders = [
        line for line in source.splitlines()
        if re.search(r"interface\s+(bram|ap_memory)\b", line)
        and "bundle=" in line
    ]
    assert not offenders, offenders


def test_stream_fft_rejection_is_actionable(tmp_path):
    """An addressed FFT cannot be mislabeled as a stream."""
    n = 16

    @sar.func
    def spectrum(x: sar.c64[n, n]) -> sar.c64[n, n]:
        return sar.fft(x, axis=1)

    spectrum.name = "iface_stream_tb"
    with pytest.raises(sar.CompilationError,
                       match="complete monotonic row-major access sweep"):
        spectrum.compile(backend="hls", options={"interface": "stream"})


def test_full_alos_raster_emits():
    """The 16384 x 16384 ALOS scene emits as one design.

    This is the size the CPU runner processes, and it only fits because
    the full-size planes -- the scene and the FFT scratch -- go behind AXI
    masters under the shipped 80% resource budgets. Guards the whole chain:
    budget decision, buffer placement, shared constant tables, and the
    master count.
    """
    import re
    import sys

    sys.path.insert(0, str(REPO_ROOT / "examples"))
    from common.params import ALOS_PARAMS
    from wka.algorithm import build_kernel

    design = build_kernel(16384,
                          ALOS_PARAMS).compile(backend="hls",
                                               options={"interface": "axi"})
    source = design.source()

    assert not re.findall(r"^\s+(?:static )?float v\d+\[\d+\]\[\d+\];", source,
                          re.M), "no full-size array may stay on chip"
    ports = {
        line.split("port=")[1].split()[0]
        for line in source.splitlines()
        if "#pragma HLS interface m_axi" in line
    }
    bundles = {
        line.split("bundle=")[1].split()[0]
        for line in source.splitlines()
        if "#pragma HLS interface m_axi" in line
    }
    assert bundles, "expected AXI master ports"
    # Ports are fixed by the algorithm's I/O; masters are not. A full-size
    # plane takes its own -- sharing would serialize two whole-pass sweeps,
    # and six f64 ports on one master measured II 13-57 on the gather loops.
    # Small read-only tables share, which is what a hand-written design
    # does: synthesizing this chain both ways puts the shared form within
    # 0.07% on latency and level on BRAM, URAM and DSP, for one channel
    # fewer. So the masters are bounded by the ports, not equal to them.
    assert len(bundles) <= len(ports), (sorted(bundles), sorted(ports))
    assert len(bundles) >= len(ports) - 2, (sorted(bundles), sorted(ports))


def test_build_kernel_name_reaches_the_top_function():
    """`name` lets two designs from one chain coexist in a directory."""
    from common.params import alos_params
    from wka.algorithm import build_kernel

    design = build_kernel(16, alos_params(16),
                          name="wka_alos").compile(backend="hls")
    assert design.name == "wka_alos"
    assert "void wka_alos(" in design.source()


@pytest.mark.parametrize("algorithm", ["wka", "rda", "csa"])
@pytest.mark.parametrize("interface", ["axi", "ap_memory"])
def test_alos_runner_emits_one_matching_artifact_set(algorithm, interface,
                                                     tmp_path):
    """One selected size and interface drive every emitted artifact."""
    import importlib

    from common.hls_artifacts import emit_alos_artifacts

    algo = importlib.import_module(f"{algorithm}.algorithm")
    ref = importlib.import_module(f"{algorithm}.reference")
    processor = getattr(ref, f"{algorithm.upper()}Processor")
    stale = tmp_path / f"{algorithm}_alos_axi.cpp"
    stale.write_text("stale design")

    emit_alos_artifacts(algorithm,
                        algo.build_kernel,
                        algo.make_inputs,
                        processor,
                        n=64,
                        out=tmp_path,
                        options={"interface": interface})

    top = f"{algorithm}_alos"
    for artifact in (f"{top}.h", f"{top}.cpp", f"{top}_tb.cpp",
                     f"{top}_hls_csim.tcl", f"{top}_portable_cpp_sim.sh",
                     f"{top}_csynth.tcl", f"{top}_tb_data", f"{top}_cosim.tcl",
                     "design_manifest.json", "stubs"):
        assert (tmp_path / artifact).exists(), artifact
    assert list((tmp_path / f"{top}_tb_data").glob("*.bin"))

    assert f"void {top}(" in (tmp_path / f"{top}.cpp").read_text()
    tcl = (tmp_path / f"{top}_hls_csim.tcl").read_text()
    assert f"#include \"{top}.h\"" in (tmp_path / f"{top}.cpp").read_text()
    assert f"set_top {top}" in tcl
    manifest = json.loads((tmp_path / "design_manifest.json").read_text())
    assert manifest["config"]["interface"] == interface
    assert manifest["kernel"]["arguments"][0]["shape"] == [64, 64]
    assert not list(tmp_path.glob(f"{top}_axi*"))
    assert not stale.exists()
