"""Generality gate: the optimization passes must not be tuned to one shape.

Every pass in the HLS path decides from what it measures -- reuse, layout,
element width, budget -- rather than from a pattern it recognizes. This
sweeps precision, scale and operator shape against the properties those
decisions are supposed to guarantee, so a pass that starts special-casing
one of them shows up here rather than in a design that fails to synthesize.
"""

import re

import numpy as np
import pytest

import sar

from conftest import requires_hls
from sar.backends.hls.devices import DEVICES

pytestmark = requires_hls

#: A bank of at most this many bits belongs in distributed RAM; anything
#: larger asks for more LUTs than a device can spare. One bus beat at the
#: widest supported AXI bus, matching the inclusive banking threshold in
#: `hls-array-partition`.
_LUTRAM_MAX_BITS = 1024

_ELEMENT_BITS = {"float": 32, "double": 64}

_DEVICE_PARTS = {
    "xcvu13p": "xcvu13p-fhgb2104-2-i",
    "xcvu9p": "xcvu9p-flga2104-2-i",
    "xcvu11p": "xcvu11p-flga2577-2-e",
    "xcu250": "xcu250-figd2104-2L-e",
    "xcu280": "xcu280-fsvh2892-2L-e",
    "xczu9eg": "xczu9eg-ffvb1156-2-e",
    "xczu28dr": "xczu28dr-ffvg1517-2-e",
    "xc7z020": "xc7z020-clg400-1",
    "xc7k325t": "xc7k325t-ffg900-2",
}


def _worst_lutram_bank_bits(source: str) -> int:
    """Widest bank the design asks distributed RAM to hold."""
    worst = 0
    for match in re.finditer(
            r"bind_storage variable=(v\d+) \w+=\w+ impl=lutram", source):
        name = match.group(1)
        decl = re.search(rf"(float|double) {name}((?:\[\d+\])+);", source)
        if not decl:
            raise AssertionError(f"missing declaration for LUTRAM bank {name}")
        factor = re.search(
            rf"array_partition variable={name} \w+ factor=(\d+)", source)
        elements = 1
        for dim in re.findall(r"\[(\d+)\]", decl.group(2)):
            elements *= int(dim)
        banks = int(factor.group(1)) if factor else 1
        bank_elements = (elements + banks - 1) // banks
        worst = max(worst, bank_elements * _ELEMENT_BITS[decl.group(1)])
    return worst


def _emit(name, spec, body):

    @sar.func
    def kernel(x):
        return body(x)

    kernel.name = name
    return kernel.specialize(spec).compile(backend="hls",
                                           options={
                                               "interface": "axi",
                                               "bram_bytes": 1 << 24,
                                               "uram_bytes": 1 << 25,
                                               "lutram_bytes": 0
                                           }).source()


def _shapes(n):
    return [
        ("elemwise", lambda x: x * 2.0 + 1.0),
        ("transpose", sar.transpose),
        ("shift", lambda x: sar.fftshift(x, axis=1)),
        ("reduce", lambda x: sar.broadcast(sar.sum(sar.absolute(x), axis=1),
                                           (n, n),
                                           dim=1)),
        ("corner", lambda x: sar.transpose(sar.transpose(x) * 2.0) + 1.0),
        ("cumsum", lambda x: sar.cumsum(x, axis=1)),
        ("rank_filter",
         lambda x: sar.rank_filter(sar.absolute(x), window=3, rank=1, axis=1)),
        ("median", lambda x: sar.median_filter(sar.absolute(x), 3, axis=1)),
        ("sort", lambda x: sar.sort(sar.absolute(x), axis=1)),
    ]


@pytest.mark.parametrize("dtype,spec", [("f32", sar.f32), ("f64", sar.f64),
                                        ("c64", sar.c64), ("c128", sar.c128)])
@pytest.mark.parametrize("n", [8, 64, 300, 512])
def test_every_shape_emits(dtype, spec, n):
    """Any precision, any scale, any operator shape: the design must emit
    and must not ask distributed RAM to hold a block RAM's worth of bits."""
    shapes = list(_shapes(n))
    if spec in (sar.c64, sar.c128):
        shapes.append(("fft", lambda x: sar.fft(x, axis=1)))

    for name, body in shapes:
        source = _emit(f"gen_{name}_{dtype}_{n}", spec[n, n], body)
        assert "#pragma HLS" in source, name
        bits = _worst_lutram_bank_bits(source)
        assert bits <= _LUTRAM_MAX_BITS, (name, bits)


@pytest.mark.parametrize("n", [64, 512])
def test_streaming_survives_every_precision(n):
    """The contiguous sweep of a streaming pass is kept whole whatever the
    element width -- the decision follows the access pattern, not the type."""
    for dtype, spec in (("f32", sar.f32), ("f64", sar.f64)):
        source = _emit(f"stream_{dtype}_{n}", spec[n, n],
                       lambda x: x * 2.0 + 1.0)
        trips = [
            int(t) for t in re.findall(
                r"for \(int(?:64_t)? \w+ = 0; \w+ < (\d+);", source)
        ]
        assert n in trips, (dtype, sorted(set(trips)))


def test_every_tabulated_device_drives_a_complete_design():
    """Device-derived budgets work across families, including no-URAM parts."""
    n = 16
    positions = np.linspace(0.0, n - 1, n)

    @sar.func
    def probe(x: sar.c64[n, n]) -> sar.c64[n, n]:
        grid = sar.broadcast(positions, (n, n), dim=1)
        return sar.interp1d(sar.fft(x, axis=1),
                            grid,
                            kernel="linear",
                            boundary="edge")

    assert set(_DEVICE_PARTS) == set(DEVICES)
    for token, part in _DEVICE_PARTS.items():
        probe.name = f"device_probe_{token}"
        probe._compiled.clear()
        design = probe.compile("hls",
                               options={
                                   "part": part,
                                   "interface": "axi",
                               })
        assert design.config.part == part
        assert design.config.bram_bytes > 0
        assert "#pragma HLS" in design.source()
        if DEVICES[token].uram == 0:
            assert design.config.uram_bytes == 0
            assert "impl=uram" not in design.source()


@pytest.mark.parametrize("n", [64, 128])
def test_port_count_does_not_track_chain_length(n):
    """Buffer sharing has to hold for any chain: a longer one reuses the
    same planes rather than adding a port per stage.

    The count saturates rather than staying flat from the first stage. A
    split-complex chain that ping-pongs needs a read and a write arena for
    each of its two planes, and a one-stage chain has not yet formed every
    one of those conflicts. What matters is that the ceiling is a property
    of the data layout, not of chain length: past it, more stages reuse the
    arenas they have however long the chain grows.

    Storage is left at the shipped device budgets, and the sizes here are
    ones the reference device holds: a longer chain does cost more
    storage, because every transform stage reads its own copy of the
    twiddle ROM and a copy claims whole memory primitives. That is a
    storage cost, not a port cost, and separating the two is the point of
    this test.
    """

    def ports(passes):

        @sar.func
        def kernel(x: sar.c64[n, n]) -> sar.c64[n, n]:
            z = x
            for _ in range(passes):
                z = sar.ifft(sar.fft(z * 2.0, axis=1) + 1.0, axis=1)
            return z

        kernel.name = f"len_{n}_{passes}"
        return kernel.compile(backend="hls", options={
            "interface": "axi"
        }).source().count("#pragma HLS interface m_axi")

    counts = [ports(passes) for passes in (1, 8, 16, 32)]
    # Non-decreasing, and bounded well below one port per stage: the
    # ceiling is set by the split-complex read/write arenas, so it stays a
    # small constant however many stages the chain grows.
    assert counts == sorted(counts)
    assert counts[-1] <= 12
    assert counts[-1] == max(counts)


@pytest.mark.parametrize("n", [64, 128])
def test_oversampled_grid_is_streamed(n):
    """Placement follows the planes the kernel works on, not the ones it
    is handed.

    A chain may resample onto a grid larger than any argument -- polar
    format oversamples by two -- and those planes are what decide the
    working set. Sizing the decision from the signature alone leaves them
    resident, which is how a design ends up asking for more on-chip
    memory than a device has.
    """
    import re

    budget = 256 * 1024

    @sar.func
    def upsample(x: sar.c64[n, n]) -> sar.c64[2 * n, 2 * n]:
        wide = sar.pad(x, ((0, n), (0, n)))
        return sar.fft(wide * 2.0, axis=1)

    upsample.name = f"oversampled_{n}"
    source = upsample.compile(backend="hls",
                              options={
                                  "interface": "axi",
                                  "bram_bytes": budget,
                                  "uram_bytes": 0,
                                  "lutram_bytes": 0
                              }).source()

    on_chip = 0
    for decl in re.finditer(r"^\s+(float|double) \w+((?:\[\d+\])+);", source,
                            re.M):
        elements = 1
        for dim in re.findall(r"\[(\d+)\]", decl.group(2)):
            elements *= int(dim)
        on_chip += elements * _ELEMENT_BITS[decl.group(1)] // 8
    assert on_chip <= budget, (
        f"{on_chip} B on chip against a {budget} B budget")
