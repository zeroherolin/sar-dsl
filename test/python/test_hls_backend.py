"""HLS backend tests: HLS C++ emission and subset diagnostics."""

import re

import numpy as np
import pytest

import sar

from conftest import REPO_ROOT, requires_hls

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


def test_hls_design_is_not_executable():

    @sar.func
    def k(a: sar.f32[N, N]) -> sar.f32[N, N]:
        return a * 2.0

    design = k.compile(backend="hls")
    with pytest.raises(RuntimeError, match="cannot be executed"):
        design()


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


def test_every_construct_emits_hls():
    """Backend-symmetry gate: one kernel per DSL construct group must
    emit HLS C++ (transpose/broadcast/strided slices route to the
    HLS pipeline)."""
    n = 16

    @sar.func
    def layout(x, v):
        a = sar.concatenate((x[::2, :], x[1::2, :]), dim=0)
        b = sar.flip(a, axis=1).T + sar.broadcast(v, (n, n), dim=0)
        return sar.circshift(b, 3, axis=1)

    @sar.func
    def reductions(x):
        idx = sar.cast(sar.argmax(x, axis=1), sar.f64)
        return x.sum(axis=0) + x.max(axis=1) + idx * 0.5

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
        (layout, (sar.f64[n, n], sar.f64[n])),
        (reductions, (sar.f64[n, n], )),
        (stats, (sar.f64[n, n], )),
        (stolt, (sar.c128[n, n], )),
    ]
    for fn, types in cases:
        design = fn.specialize(*types).compile(backend="hls")
        assert "#pragma HLS" in design.source(), fn


def test_testbench_csim_passes(tmp_path):
    """The emitted design + generated testbench form a self-contained
    C-simulation package: compiled with plain clang++ (Vitis header
    stubs included in the package), the design must reproduce the
    golden data."""
    import subprocess

    import numpy as np

    from sar.compiler.toolchain import find_tool

    n = 16

    @sar.func
    def phase(z: sar.c128[n, n], g: sar.f64[n, n]) -> sar.c128[n, n]:
        return z * sar.expj(g)

    design = phase.compile(backend="hls")

    rng = np.random.default_rng(2)
    z = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    g = rng.uniform(-3.0, 3.0, (n, n))
    design.write_testbench([z, g], [z * np.exp(1j * g)], tmp_path, rtol=1e-12)

    clang = find_tool("clang")
    subprocess.run([
        clang + "++", "-O2", "-Wno-unknown-pragmas", "-I", "stubs",
        "phase.cpp", "phase_tb.cpp", "-o", "csim", "-pthread"
    ],
                   cwd=tmp_path,
                   check=True,
                   capture_output=True)
    result = subprocess.run(["./csim"],
                            cwd=tmp_path,
                            capture_output=True,
                            text=True)
    assert result.returncode == 0, result.stdout
    assert "PASS" in result.stdout
    assert (tmp_path / "phase_csim.tcl").exists()
    assert (tmp_path / "phase_csynth.tcl").exists()


def test_f32_chain_csim_matches_the_cpu_backend(tmp_path):
    """The HLS x f32 leg of the precision matrix.

    A chain built with `dtype=sar.c64` must emit, csim, and agree with
    the same build on the cpu backend to single-precision rounding --
    the cpu FFT computes its butterflies in f64 while the HLS FFT
    computes in the declared width, so this is a real cross-backend
    check, not a self-comparison.
    """
    import subprocess

    import numpy as np

    from common.params import synthetic_params
    from common.simulate import single_target_scene
    from wka.algorithm import build_kernel, make_inputs
    from sar.compiler.toolchain import find_tool

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
    clang = find_tool("clang")
    subprocess.run([
        clang + "++", "-O2", "-Wno-unknown-pragmas", "-I", "stubs",
        "wka_f32.cpp", "wka_f32_tb.cpp", "-o", "csim", "-pthread"
    ],
                   cwd=tmp_path,
                   check=True,
                   capture_output=True)
    result = subprocess.run(["./csim"],
                            cwd=tmp_path,
                            capture_output=True,
                            text=True)
    assert result.returncode == 0, result.stdout
    assert "PASS" in result.stdout


def test_twiddle_tables_are_shared_across_same_size_ffts():
    """Transforms of one size share one cos and one sin table.

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
        return sar.ifft(sar.fft(a, axis=1), axis=1) * sar.fft(b, axis=1)

    source = three.compile(backend="hls").source()
    assert source.count(f"kTwiddleCos_{n}[") == 1, source
    assert source.count(f"kTwiddleSin_{n}[") == 1, source
    # The uniquing suffix appearing would mean a duplicate slipped in
    # under another name.
    assert f"kTwiddleCos_{n}_2" not in source


def test_synthesis_script_covers_every_interface(tmp_path):
    """`write_synthesis_script` exists for the designs csim cannot cover:
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
        assert (out / "scale.cpp").read_text() == design.source()
        tcl = script.read_text()
        assert "set_top scale" in tcl
        assert "csynth_design" in tcl
        assert "set_part xczu9eg-ffvb1156-2-e" in tcl
        assert "create_clock -period 4" in tcl


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
        design = spectrum.compile(backend="hls",
                                  options={"axi_interface": True})
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


def test_testbench_rejects_axi_with_a_reason():
    """The csim testbench cannot drive the promoted scratch ports, and the
    error has to say so rather than just refusing."""
    N = 32

    @sar.func
    def spectrum(x: sar.c64[N, N]) -> sar.c64[N, N]:
        return sar.fft(x, axis=1)

    design = spectrum.compile(backend="hls", options={"axi_interface": True})
    with pytest.raises(sar.LaunchError, match="AXI port"):
        design.write_testbench([], [])


def test_streamed_planes_are_shared_across_lifetimes():
    """A streamed chain must not hold one plane per intermediate.

    An imaging chain is a sequence of whole-raster passes, so a plane goes
    dead as soon as the next pass has consumed it. Giving every intermediate
    its own allocation is what puts an ALOS-size design past any board's
    DRAM; sharing the ones whose lifetimes do not overlap is what brings it
    back. The count has to stay flat as the chain grows, not track its
    length.
    """
    import re

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
                                "axi_interface": True,
                                "on_chip_budget": 1
                            }).source()
        return len(re.findall(r"m_axi", source))

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
                                "axi_interface": True,
                                "on_chip_budget": 1
                            }).source()
    # One plane in, one out: a materialized broadcast would be a third
    # DRAM buffer. It would land in the scratch allocation, so the check
    # is that the scratch stayed at its 1-element placeholder size while
    # exactly two full planes hold pragmas.
    maxi_ports = re.findall(r"m_axi.*port=(\w+)", source)
    placeholder = {
        name
        for name in maxi_ports if re.search(rf"double {name}\[1\]", source)
    }
    assert len(placeholder) == 1, source
    assert len(set(maxi_ports) - placeholder) == 2, maxi_ports


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
                               "axi_interface": True,
                               "on_chip_budget": 1
                           }).source()
    trips = [
        int(t) for t in re.findall(r"for \(int \w+ = 0; \w+ < (\d+);", source)
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
                               "axi_interface": True,
                               "on_chip_budget": 1
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
        "axi_interface": True
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

        spectrum.name = f"iface_{n}_{options.get('on_chip_budget', 'def')}"
        return spectrum.compile(backend="hls",
                                options={
                                    "interface": "axi",
                                    **options
                                }).source()

    # Comfortably resident, forced to stream, and pinned on chip.
    for source in (emit(64), emit(64, on_chip_budget=32 * 1024),
                   emit(256, on_chip_budget=0)):
        modes = set(re.findall(r"#pragma HLS interface (\w+)", source))
        assert modes == {"m_axi", "s_axilite"}, modes
        # `bundle=` belongs to m_axi / axis / s_axilite, never to a memory
        # port; emitting it on one is the malformed pragma this guards.
        assert not [
            line for line in source.splitlines()
            if "interface bram" in line and "bundle=" in line
        ]


def test_local_interface_emits_memory_ports():
    """The default interface hands the kernel plain arrays, which is the
    signature the generated csim testbench drives."""
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


def test_bram_alias_emits_the_same_pragmas_as_ap_memory():
    """`bram` is the deprecated spelling of the same protocol, so the two
    must produce identical interface pragmas -- the only reason the alias
    exists is to keep old configs working."""
    n = 32

    def pragmas(iface):

        @sar.func
        def spectrum(x: sar.c64[n, n]) -> sar.c64[n, n]:
            return sar.fft(x, axis=1)

        spectrum.name = f"iface_alias_{iface}"
        src = spectrum.compile(backend="hls", options={
            "interface": iface
        }).source()
        return sorted([
            line.strip() for line in src.splitlines()
            if line.strip().startswith("#pragma HLS interface")
        ])

    assert pragmas("bram") == pragmas("ap_memory")


def test_stream_interface_emits_axis_pragmas():
    """A streaming radar front end wants its ports as AXI4-Stream. The
    emitter already knew how to write `axis`; this is the path from the
    config down to it."""
    n = 64

    @sar.func
    def spectrum(x: sar.c64[n, n]) -> sar.c64[n, n]:
        return sar.fft(x, axis=1)

    spectrum.name = "iface_stream"
    source = spectrum.compile(backend="hls", options={
        "interface": "stream"
    }).source()
    # The kernel's inputs and outputs stream. The only memory-mapped port
    # is the fixed DRAM scratch, sitting at its 1-element placeholder size
    # since nothing spilled here.
    axis_ports = set(re.findall(r"interface axis port=(\w+)", source))
    assert {"x_re", "x_im"} <= axis_ports, axis_ports
    maxi_ports = re.findall(r"m_axi.*port=(\w+)", source)
    assert len(maxi_ports) == 1, maxi_ports
    assert re.search(rf"\w+ {maxi_ports[0]}\[1\]", source), maxi_ports
    # A stream port carries no burst shaping: that describes addressed
    # access to DRAM, which a FIFO does not perform. The placeholder moves
    # no bulk data, so it is not shaped either.
    assert "max_read_burst_length" not in source


def test_stream_interface_keeps_spilled_scratch_memory_mapped():
    """A buffer that spills to DRAM is read and written by the design, and
    an AXI4-Stream is unidirectional and consumed once -- `axis` on such a
    port cannot synthesize. In stream mode only the pure inputs and outputs
    stream; the scratch stays an AXI master (PFA hit this for real)."""
    n = 32

    @sar.func
    def two_ffts(x: sar.c64[n, n]) -> sar.c64[n, n]:
        y = sar.fft(x, axis=1)
        return sar.fft(y, axis=0)

    two_ffts.name = "iface_stream_scratch"
    source = two_ffts.compile(backend="hls",
                              options={
                                  "interface": "stream",
                                  "on_chip_budget": 4096,
                              }).source()
    lines = [
        line.strip() for line in source.splitlines()
        if line.strip().startswith("#pragma HLS interface")
    ]
    maxi = [line for line in lines if " m_axi " in line]
    axis = [line for line in lines if " axis " in line]

    # The budget forces a spill, so the design has a scratch port -- and it
    # must be memory-mapped, with burst shaping, despite the stream mode.
    assert maxi, lines
    assert all("max_read_burst_length" in line for line in maxi), maxi

    # The kernel's own ports still stream, without burst shaping.
    axis_ports = {line.split("port=")[1].split()[0] for line in axis}
    assert {"x_re", "x_im"} <= axis_ports, axis_ports
    assert all("max_read_burst_length" not in line for line in axis), axis


@pytest.mark.parametrize("interface", ["ap_memory", "bram", "axi", "stream"])
def test_no_memory_port_ever_carries_a_bundle(interface):
    """`bundle=` belongs to m_axi / axis / s_axilite and is invalid on a
    memory-mode port. Vitis rejects that combination outright, and it was a
    real bug once -- so every interface keeps a regression test for it."""
    n = 32

    @sar.func
    def spectrum(x: sar.c64[n, n]) -> sar.c64[n, n]:
        return sar.fft(x, axis=1)

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


def test_stream_testbench_raises_an_actionable_error(tmp_path):
    """csim for AXI-Stream needs a FIFO-feeding harness that is not
    implemented. Saying so beats emitting a testbench that cannot compile
    against the design's signature."""
    n = 16

    @sar.func
    def spectrum(x: sar.c64[n, n]) -> sar.c64[n, n]:
        return sar.fft(x, axis=1)

    spectrum.name = "iface_stream_tb"
    design = spectrum.compile(backend="hls", options={"interface": "stream"})
    data = np.zeros((n, n), dtype=np.complex64)
    with pytest.raises(sar.errors.LaunchError) as excinfo:
        design.write_testbench([data], [data], output_dir=tmp_path)
    message = str(excinfo.value)
    # The error has to name the cause and the way forward, not just fail.
    assert "stream" in message
    assert "ap_memory" in message
    assert "FIFO" in message
    # Nothing half-written is left behind.
    assert not list(tmp_path.glob("*_tb.cpp"))


def test_full_alos_raster_emits():
    """The 16384 x 16384 ALOS scene emits as one design.

    This is the size the CPU runner processes, and it only fits because
    the full-size planes -- the scene and the FFT scratch -- go behind AXI
    masters. Guards the whole chain: budget decision, buffer placement,
    bundle sharing.
    """
    import re
    import sys

    sys.path.insert(0, str(REPO_ROOT / "examples"))
    from common.params import ALOS_PARAMS
    from wka.algorithm import build_kernel

    design = build_kernel(16384,
                          ALOS_PARAMS).compile(backend="hls",
                                               options={"axi_interface": True})
    source = design.source()

    assert not re.findall(r"^\s+(?:static )?float v\d+\[\d+\]\[\d+\];", source,
                          re.M), "no full-size array may stay on chip"
    bundles = {
        line.split("bundle=")[1].split()[0]
        for line in source.splitlines() if "m_axi" in line
    }
    assert bundles, "expected AXI master ports"
    # Ports share a bundle per element type rather than taking one each.
    assert len(bundles) <= 4, sorted(bundles)


def test_cumsum_emits_hls():
    """The scan lowers to a loop nest the C++ emitter can carry."""

    @sar.func
    def scan(x: sar.f32[N, N]) -> sar.f32[N, N]:
        return sar.cumsum(x, axis=1)

    source = scan.compile(backend="hls").source()
    assert "void scan" in source
    assert "#pragma HLS" in source


def test_cumsum_complex_emits_hls():
    """A complex scan splits into two float scans through decomplexify."""

    @sar.func
    def cscan(x: sar.c64[N, N]) -> sar.c64[N, N]:
        return sar.cumsum(x, axis=0)

    source = cscan.compile(backend="hls").source()
    assert "void cscan" in source
    assert "#pragma HLS" in source


def test_rank_filter_emits_hls():
    """The window sort is a straight-line compare-exchange network."""

    @sar.func
    def rf(x: sar.f32[N, N]) -> sar.f32[N, N]:
        return sar.rank_filter(x, window=5, rank=2, axis=1)

    source = rf.compile(backend="hls").source()
    assert "void rf" in source
    assert "#pragma HLS" in source


def test_median_filter_emits_hls():

    @sar.func
    def mf(x: sar.f32[N, N]) -> sar.f32[N, N]:
        return sar.median_filter(x, window=3, axis=0)

    source = mf.compile(backend="hls").source()
    assert "void mf" in source
    assert "#pragma HLS" in source


# --------------------------------------------------------------------- #
# Example artifact sets (examples/common/hls_artifacts.py)
# --------------------------------------------------------------------- #


def test_build_kernel_name_reaches_the_top_function():
    """`name` lets two designs from one chain coexist in a directory."""
    from common.params import alos_params
    from wka.algorithm import build_kernel

    design = build_kernel(16, alos_params(16),
                          name="wka_alos").compile(backend="hls")
    assert design.name == "wka_alos"
    assert "void wka_alos(" in design.source()


@pytest.mark.parametrize("algorithm", ["wka", "rda", "csa"])
def test_alos_runner_emits_the_full_artifact_set(algorithm, tmp_path):
    """Every ALOS HLS runner produces a synthesizable design *and* a
    complete csim package, not just the design."""
    import importlib

    from common.hls_artifacts import emit_alos_artifacts

    algo = importlib.import_module(f"{algorithm}.algorithm")
    ref = importlib.import_module(f"{algorithm}.reference")
    processor = getattr(ref, f"{algorithm.upper()}Processor")

    emit_alos_artifacts(algorithm,
                        algo.build_kernel,
                        algo.make_inputs,
                        processor,
                        n=64,
                        csim_n=32,
                        out=tmp_path)

    top = f"{algorithm}_alos"
    for artifact in (f"{top}_axi.cpp", f"{top}_axi_csynth.tcl", f"{top}.cpp",
                     f"{top}_tb.cpp", f"{top}_csim.tcl", f"{top}_csynth.tcl",
                     f"{top}_tb_data", "stubs"):
        assert (tmp_path / artifact).exists(), artifact
    assert list((tmp_path / f"{top}_tb_data").glob("*.dat"))

    # The two designs are separate symbols, so both can sit in one project.
    assert f"void {top}(" in (tmp_path / f"{top}.cpp").read_text()
    assert f"void {top}_axi(" in (tmp_path / f"{top}_axi.cpp").read_text()
    # The csim script says which raster it simulates and why.
    tcl = (tmp_path / f"{top}_csim.tcl").read_text()
    assert "32 x 32" in tcl and f"{top}_axi.cpp" in tcl
    assert f"set_top {top}" in tcl
