"""The AXI top-level signature is the algorithm's own I/O.

A deliverable design's ports are its contract with the host, and each one
is a platform resource: an integrator wires it to a physical memory
channel, so a master the compiler minted for its own convenience spends a
channel the algorithm cannot use. Buffers the compiler keeps off chip are
internal and must not surface as ports of their own; they are carved out
of scratch arenas.

What sets the arena count is aliasing, not convenience. A typed C++
signature needs one pointer per element type, since an arena is `float *`
or `double *` and one pointer cannot be both. Beyond that, buffers a
single node reads and writes need separate pointers: one pointer both
loaded from and stored to forces HLS to assume the loads alias the
in-flight stores, and that node's bus traffic serializes. The arenas are
a coloring of exactly those conflicts -- the same ping-pong a hand-written
design allocates by hand, and no more.

What stays variable is the arena itself: a design that spills nothing
declares no arena, and the scratch extent tracks what went off chip. The
algorithm's own ports do not move, and no placeholder is invented to pad
the list. These tests pin that on the four imaging chains -- across scene
sizes, interfaces, and memory budgets.
"""

import importlib
import re

import pytest
import sar
from conftest import requires_hls

CHAINS = ["wka", "rda", "csa", "pfa"]
SIZES = [256, 512]
#: On-chip tiers small enough that every chain has to push planes off chip
#: -- yet large enough to hold the tables that cannot stream, since the
#: caps are hard and an over-budget design fails compilation -- and
#: generous enough that none of them spills. What the compiler decides in
#: between must not reach the signature either, so both are tested.
SPILLING = {
    "bram_bytes": 1 << 22,
    "uram_bytes": 1 << 22,
    "lutram_bytes": 1 << 15
}
RESIDENT = {
    "bram_bytes": 1 << 28,
    "uram_bytes": 1 << 28,
    "lutram_bytes": 1 << 20
}
PLACEMENTS = [SPILLING, RESIDENT]
CTYPE = r"(?:float|double|hls::vector<(?:float|double),\s*\d+>)"


def build_kernel(chain: str, n: int):
    algorithm = importlib.import_module(f"{chain}.algorithm")
    if chain == "pfa":
        from pfa.geometry import Geometry
        return algorithm.build_kernel(n, Geometry(n))
    from common.params import synthetic_params
    return algorithm.build_kernel(n, synthetic_params(n))


def signature(design) -> list:
    """The top function's parameter declarations, in order.

    Matched on the top function's own name so the test reads the interface
    rather than however the emitter happens to annotate it.
    """
    match = re.search(rf"^void {re.escape(design.name)}\(\n(.*?)\n\) \{{",
                      design.source(), re.S | re.M)
    assert match, f"no definition of {design.name} in the emitted design"
    return [arg.strip() for arg in match.group(1).split(",\n")]


def io_port_count(kernel) -> int:
    """Ports the algorithm's own I/O needs: a complex tensor is carried as
    an (re, im) pair of float planes."""
    planes = list(kernel.arg_types) + list(kernel.declared_result_types)
    return sum(2 if t.dtype.is_complex else 1 for t in planes)


def axi_design(kernel, placement):
    kernel._compiled.clear()
    return kernel.compile(backend="hls",
                          options={
                              "interface": "axi",
                              **placement
                          })


def scratch_bytes(args, io_ports: int) -> int:
    widths = {"float": 4, "double": 8}
    total = 0
    for arg in args[io_ports:]:
        declaration = re.fullmatch(
            rf"(?P<ctype>{CTYPE})\s+\w+\[(?P<extent>\d+)\]", arg)
        assert declaration, arg
        ctype = declaration.group("ctype")
        vector = re.fullmatch(r"hls::vector<(float|double),\s*(\d+)>", ctype)
        element_bytes = (widths[vector.group(1)] *
                         int(vector.group(2)) if vector else widths[ctype])
        total += element_bytes * int(declaration.group("extent"))
    return total


def scratch_types(args, io_ports: int) -> set:
    """Element types of the trailing scratch arenas."""
    return {
        re.fullmatch(rf"(?P<ctype>{CTYPE})\s+\w+\[\d+\]", arg).group("ctype")
        for arg in args[io_ports:]
    }


@requires_hls
@pytest.mark.parametrize("chain", CHAINS)
@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("placement", PLACEMENTS, ids=["spilling", "resident"])
def test_port_count_is_the_algorithm_io(chain, n, placement):
    """Ports are the algorithm's I/O plus the arenas aliasing forces.

    The upper bound is what makes this a real check: the compiler may spill
    any number of planes, and each one has to land in an arena some other
    plane already conflicts it out of, rather than in a port of its own. A
    design that spills nothing adds nothing.
    """
    kernel = build_kernel(chain, n)
    io_ports = io_port_count(kernel)
    design = axi_design(kernel, placement)
    args = signature(design)
    source = design.source()
    scratch = args[io_ports:]
    # `max_scratch_arenas` caps the masters the compiler may open per
    # spilling scalar type, and a chain spills at most its data planes and
    # its geometry planes. That product is the contract; what it rules out
    # is an arena per spilled plane, which is the failure this guards --
    # the chains spill far more planes than they declare arenas.
    arenas = int(design.config.max_scratch_arenas)
    assert len(scratch) <= arenas * len(scratch_types(args, io_ports)), args
    assert len(scratch) <= 2 * arenas, args
    if "Scratch ABI  : conflict graph compacted" in source:
        penalty = re.search(
            r"Scratch cost : predicted serialization penalty (\d+)", source)
        assert penalty and int(penalty.group(1)) > 0


def port_shapes(args, io_ports: int) -> list:
    """(element type, extents) per port, dropping the emitter's value names
    and the scratch extent -- the one thing sizing legitimately moves."""
    shapes = []
    for index, arg in enumerate(args):
        declaration = re.fullmatch(
            rf"(?P<ctype>{CTYPE})\s+\w+(?P<dims>(?:\[\d+\])+)", arg)
        assert declaration, arg
        ctype = declaration.group("ctype")
        dims = re.findall(r"\[(\d+)\]", declaration.group("dims"))
        if index >= io_ports:
            dims = dims[:-1] + ["scratch"]
        shapes.append((ctype, tuple(dims)))
    return shapes


@requires_hls
@pytest.mark.parametrize("chain", CHAINS)
@pytest.mark.parametrize("n", SIZES)
def test_public_ports_do_not_follow_the_placement(chain, n):
    """The invariant being established: the algorithm's own ports are a
    function of the algorithm, not of what the optimizer decided to spill.

    Only the scratch arena answers to placement, and only by existing at
    all: squeezed onto a small device the chain gains one, and no port the
    kernel declared changes type, rank or position to pay for it.
    """
    kernel = build_kernel(chain, n)
    signatures = [signature(axi_design(kernel, p)) for p in PLACEMENTS]
    io_ports = io_port_count(kernel)
    assert port_shapes(signatures[0][:io_ports],
                       io_ports) == port_shapes(signatures[1][:io_ports],
                                                io_ports), signatures


@requires_hls
@pytest.mark.parametrize("chain", CHAINS)
def test_scratch_arenas_are_trailing_and_flat(chain):
    """Typed scratch arenas trail the public I/O and are flat arrays."""
    kernel = build_kernel(chain, 256)
    args = signature(axi_design(kernel, SPILLING))
    io_ports = io_port_count(kernel)
    scratch = args[io_ports:]
    assert scratch
    for arg in scratch:
        assert re.fullmatch(rf"{CTYPE} \w+\[\d+\]", arg), arg
    # Every other port is one of the algorithm's own planes: 1-D arrays are
    # its vectors, 2-D its rasters, and none of them is the flat scratch.
    for arg in args[:io_ports]:
        assert re.fullmatch(rf"{CTYPE} \w+(?:\[\d+\])+", arg), arg


@requires_hls
@pytest.mark.parametrize("interface", ["ap_memory", "axi", "stream"])
def test_every_port_carries_one_data_interface(interface):
    """A port with no data pragma falls into whatever protocol the Vitis flow
    defaults to, which differs between the IP and the kernel flows -- the
    signature stops being a contract. Every argument, the scratch arena
    included, therefore needs exactly one explicit data pragma. Addressed AXI
    masters additionally carry one AXI-Lite base-address register."""
    if interface == "stream":

        @sar.func
        def stream_scale(x: sar.f32[64]) -> sar.f32[64]:
            return x * 2.0

        kernel = stream_scale
    else:
        kernel = build_kernel("wka", 64)
    kernel._compiled.clear()
    design = kernel.compile(backend="hls", options={"interface": interface})
    ports = [
        re.search(r"(\w+)(?:\[|$)",
                  arg.split(" ", 1)[1]).group(1) for arg in signature(design)
    ]
    data_pragmas = re.findall(
        r"#pragma HLS interface (?:m_axi|bram|axis) .*?port=(\w+)",
        design.source())
    control_pragmas = re.findall(
        r"#pragma HLS interface s_axilite port=(\w+) bundle=ctrl",
        design.source())
    for port in ports:
        assert data_pragmas.count(port) == 1, (port, data_pragmas)
        # An addressed AXI master also needs one AXI-Lite base-address
        # register. Plain memories and streams carry no such register.
        assert control_pragmas.count(port) == (1 if interface == "axi" else 0)


@requires_hls
def test_axi_write_only_result_uses_an_output_name():

    @sar.func
    def scale(x: sar.f32[64]) -> sar.f32[64]:
        return x * 2.0

    design = scale.compile(backend="hls", options={"interface": "axi"})
    args = signature(design)
    assert any(
        re.fullmatch(r"(?:float|hls::vector<float,\s*\d+>) out0\[\d+\]", arg)
        for arg in args), args


@requires_hls
def test_a_resident_design_declares_no_scratch_port():
    """The other half of the contract: no spill, no port.

    A placeholder arena would keep the signature constant across placements,
    but it would also ask an integrator to wire a master the design never
    drives -- so a chain that fits on chip must come out with the kernel's
    own ports and nothing appended.
    """
    kernel = build_kernel("wka", 256)
    args = signature(axi_design(kernel, RESIDENT))
    assert len(args) == io_port_count(kernel), args


@requires_hls
@pytest.mark.parametrize("chain", CHAINS)
def test_scratch_holds_what_went_off_chip(chain):
    """The carving has to be real. Squeezed onto a small device every chain
    pushes whole planes off chip, and with the ports gone the only place
    left for them is the scratch, which must be sized to match."""
    kernel = build_kernel(chain, 256)
    io_ports = io_port_count(kernel)
    spilled = scratch_bytes(signature(axi_design(kernel, SPILLING)), io_ports)
    resident = scratch_bytes(signature(axi_design(kernel, RESIDENT)), io_ports)
    assert spilled >= 256 * 256 * 4, spilled
    assert resident < spilled, (resident, spilled)


@requires_hls
@pytest.mark.parametrize("chain", ("wka", "rda", "csa"))
def test_small_read_only_inputs_share_one_master(chain):
    """A port is a platform resource, so one is spent only where needed.

    A full-size plane earns its own AXI master: it is swept for a whole pass
    and sharing would serialize two such sweeps. A small read-only vector --
    an axis, a window, a reference chirp -- does not. Those are read a row at
    a time out of a line-sized table, so several share a master the way a
    hand-written design would, and the design asks the board for fewer
    physical channels without changing its port list.
    """
    import json

    design = axi_design(build_kernel(chain, 256), PLACEMENTS[0])
    ports = [
        json.loads(line.split("SAR_DSL_INTERFACE: ")[1])
        for line in design.source().splitlines() if "SAR_DSL_INTERFACE" in line
    ]
    bundles = {port["bundle"] for port in ports}
    assert len(bundles) < len(ports), (
        "no master is shared; every port still claims its own channel")

    # Sharing is confined to read-only inputs: a writer needs exclusive
    # access, and a plane is large enough that contention would cost more
    # than the channel saves.
    shared = {
        bundle
        for bundle in bundles
        if sum(port["bundle"] == bundle for port in ports) > 1
    }
    for port in ports:
        if port["bundle"] in shared:
            assert port["direction"] == "in", port["name"]
            assert port["kind"] == "public", port["name"]


@requires_hls
def test_read_only_master_sharing_stays_within_one_dataflow_process():
    """Vitis 2022.2 rejects a bundle read by different dataflow processes.

    Omega-K's real/imaginary input planes are consumed together by the first
    FFT and may share; its two windows are consumed together by the windowing
    phase and may share.  Those two process groups must not be merged into one
    master even when every array is small enough for the size heuristic.
    """
    import json

    design = axi_design(build_kernel("wka", 32), RESIDENT)
    ports = {
        port["name"]: port
        for port in (json.loads(line.split("SAR_DSL_INTERFACE: ")[1])
                     for line in design.source().splitlines()
                     if "SAR_DSL_INTERFACE" in line)
    }
    assert ports["raw_re"]["bundle"] == ports["raw_im"]["bundle"]
    assert ports["win_r"]["bundle"] == ports["win_a"]["bundle"]
    assert ports["raw_re"]["bundle"] != ports["win_r"]["bundle"]
