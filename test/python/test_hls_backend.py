"""ScaleHLS backend tests: HLS C++ emission and subset diagnostics."""

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


def test_hls_design_is_not_executable():

    @sar.func
    def k(a: sar.f32[N, N]) -> sar.f32[N, N]:
        return a * 2.0

    design = k.compile(backend="hls")
    with pytest.raises(RuntimeError, match="cannot be executed"):
        design()


def test_fft_kernel_emits_via_affine_flow():
    """Complex kernels with FFTs go through decomplexify + Stockham affine
    lowering and the HIDA C++ entry point."""

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
    affine flow; HIDA's PyTorch entry cannot schedule them)."""
    import numpy as np

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

    design = spectrum.compile(backend="hls",
                              options={"axi_interface": True})
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
    # One plane in, one out: a materialized broadcast would be a third.
    assert len(re.findall(r"m_axi", source)) == 2, source.count("m_axi")


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

    source = spectrum.compile(backend="hls",
                              options={
                                  "axi_interface": True
                              }).source()
    assert "bind_storage" in source
    assert "impl=lutram" not in source


def test_on_chip_budget_decides_placement():
    """Placement is the compiler's call: a working set that fits the budget
    stays resident, one that does not is streamed. The user picks the
    interface, not the split."""

    def emit(n, **options):

        @sar.func
        def spectrum(x: sar.c64[n, n]) -> sar.c64[n, n]:
            return sar.fft(x, axis=1)

        spectrum.name = f"budget_{n}_{options.get('on_chip_budget', 'def')}"
        return spectrum.compile(backend="hls",
                                options={
                                    "axi_interface": True,
                                    **options
                                })

    def is_streamed(design):
        return "m_axi" in design.source()

    # The working set is the kernel's own planes plus whatever full-size
    # intermediates survive buffer sharing; the transform scratch is one line
    # wide and does not enter it. Here that is just the input and the output,
    # 64 * 64 * 8 B each, so 64 KiB total: inside the 4 MiB default, outside
    # a 32 KiB budget.
    assert not is_streamed(emit(64))
    assert is_streamed(emit(64, on_chip_budget=32 * 1024))
    # A budget of zero pins everything on chip regardless of size.
    assert not is_streamed(emit(256, on_chip_budget=0))


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
