"""The AXI top-level signature is fixed by the algorithm.

A deliverable design's ports are its contract with the host. Buffers the
compiler decides to keep off chip are internal, so they must not surface as
ports: otherwise a host binds a different number of AXI interfaces every
time a budget, a scene size or a fusion decision moves a buffer across the
on-chip line. These tests pin that invariant on the four imaging chains --
across scene sizes, interfaces, and memory budgets.
"""

import importlib
import re

import pytest
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
        ctype = arg.split(" ", 1)[0]
        total += widths[ctype] * int(re.search(r"\[(\d+)\]$", arg).group(1))
    return total


@requires_hls
@pytest.mark.parametrize("chain", CHAINS)
@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("placement", PLACEMENTS, ids=["spilling", "resident"])
def test_port_count_is_the_algorithm_io(chain, n, placement):
    """Ports are algorithm I/O plus stable typed scratch arenas."""
    kernel = build_kernel(chain, n)
    args = signature(axi_design(kernel, placement))
    scratch_count = len(args) - io_port_count(kernel)
    assert 1 <= scratch_count <= 2, args


def port_shapes(args, io_ports: int) -> list:
    """(element type, extents) per port, dropping the emitter's value names
    and the scratch extent -- the one thing sizing legitimately moves."""
    shapes = []
    for index, arg in enumerate(args):
        ctype, rest = arg.split(" ", 1)
        dims = re.findall(r"\[(\d+)\]", rest)
        if index >= io_ports:
            dims = dims[:-1] + ["scratch"]
        shapes.append((ctype, tuple(dims)))
    return shapes


@requires_hls
@pytest.mark.parametrize("chain", CHAINS)
@pytest.mark.parametrize("n", SIZES)
def test_port_count_does_not_follow_the_placement(chain, n):
    """The invariant being established: the interface is a function of the
    algorithm, not of what the optimizer decided to spill this time."""
    kernel = build_kernel(chain, n)
    signatures = [signature(axi_design(kernel, p)) for p in PLACEMENTS]
    io_ports = io_port_count(kernel)
    assert port_shapes(signatures[0],
                       io_ports) == port_shapes(signatures[1],
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
        assert re.fullmatch(r"(?:float|double) \w+\[\d+\]", arg), arg
    # Every other port is one of the algorithm's own planes: 1-D arrays are
    # its vectors, 2-D its rasters, and none of them is the flat scratch.
    for arg in args[:io_ports]:
        assert re.fullmatch(r"(?:float|double) \w+(?:\[\d+\])+", arg), arg


@requires_hls
@pytest.mark.parametrize("interface", ["ap_memory", "axi", "stream"])
def test_every_port_carries_exactly_one_interface_pragma(interface):
    """A port with no pragma falls into whatever protocol the Vitis flow
    defaults to, which differs between the IP and the kernel flows -- the
    signature stops being a contract. Every argument, including an unused
    placeholder scratch arena, therefore needs one explicit pragma."""
    kernel = build_kernel("wka", 256)
    kernel._compiled.clear()
    design = kernel.compile(backend="hls", options={"interface": interface})
    ports = [
        re.search(r"(\w+)(?:\[|$)",
                  arg.split(" ", 1)[1]).group(1) for arg in signature(design)
    ]
    pragmas = re.findall(r"#pragma HLS interface \S+ .*?port=(\w+)",
                         design.source())
    for port in ports:
        assert pragmas.count(port) == 1, (port, pragmas)


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
