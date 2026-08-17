"""The AXI top-level signature is fixed by the algorithm.

A deliverable design's ports are its contract with the host. Buffers the
compiler decides to keep off chip are internal, so they must not surface as
ports: otherwise a host binds a different number of AXI interfaces every
time a budget, a scene size or a fusion decision moves a buffer across the
on-chip line. These tests pin that invariant on the four imaging chains --
the same designs whose port counts used to range from 8 to 31.
"""

import importlib
import re

import pytest
from conftest import requires_hls

CHAINS = ["wka", "rda", "csa", "pfa"]
SIZES = [256, 512]
#: On-chip tiers small enough that every chain has to push planes off chip,
#: and generous enough that none of them does. What the compiler decides in
#: between must not reach the signature either, so both are tested.
SPILLING = {
    "bram_bytes": 1 << 18,
    "uram_bytes": 1 << 18,
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
                              "axi_interface": True,
                              **placement
                          })


def scratch_elements(args) -> int:
    return int(re.search(r"\[(\d+)\]$", args[-1]).group(1))


@requires_hls
@pytest.mark.parametrize("chain", CHAINS)
@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("placement", PLACEMENTS, ids=["spilling", "resident"])
def test_port_count_is_the_algorithm_io(chain, n, placement):
    """Ports == the kernel's own I/O planes, plus the one scratch pointer."""
    kernel = build_kernel(chain, n)
    args = signature(axi_design(kernel, placement))
    assert len(args) == io_port_count(kernel) + 1, args


def port_shapes(args) -> list:
    """(element type, extents) per port, dropping the emitter's value names
    and the scratch extent -- the one thing sizing legitimately moves."""
    shapes = []
    for index, arg in enumerate(args):
        ctype, rest = arg.split(" ", 1)
        dims = re.findall(r"\[(\d+)\]", rest)
        if index == len(args) - 1:
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
    assert port_shapes(signatures[0]) == port_shapes(signatures[1]), signatures


@requires_hls
@pytest.mark.parametrize("chain", CHAINS)
def test_scratch_is_the_last_port_and_is_flat(chain):
    """The scratch is one trailing flat array, so a host allocates a single
    buffer and the design carves it: the size it must allocate is the port's
    own extent."""
    kernel = build_kernel(chain, 256)
    args = signature(axi_design(kernel, SPILLING))
    assert re.fullmatch(r"(?:float|double) \w+\[\d+\]", args[-1]), args[-1]
    # Every other port is one of the algorithm's own planes: 1-D arrays are
    # its vectors, 2-D its rasters, and none of them is the flat scratch.
    for arg in args[:-1]:
        assert re.fullmatch(r"(?:float|double) \w+(?:\[\d+\])+", arg), arg


@requires_hls
@pytest.mark.parametrize("interface", ["ap_memory", "axi", "stream"])
def test_every_port_carries_exactly_one_interface_pragma(interface):
    """A port with no pragma falls into whatever protocol the Vitis flow
    defaults to, which differs between the IP and the kernel flows -- the
    signature stops being a contract. The placeholder scratch used to reach
    the signature exactly like that: its port op looked dead and was DCE'd,
    leaving a bare argument."""
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
    spilled = scratch_elements(signature(axi_design(kernel, SPILLING)))
    resident = scratch_elements(signature(axi_design(kernel, RESIDENT)))
    assert spilled >= 256 * 256, spilled
    assert resident < spilled, (resident, spilled)
