"""Strategy the HLS backend decides for itself.

The schema in `config.py` splits into *constraints* (device and
deliverable facts only the user knows) and *strategy* (everything the
compiler can work out from those constraints and from measured kernel
properties -- plane sizes, transform lengths, element widths, buffer
lifetimes). This module is the strategy half: `derive()` is the whole
policy, one function per decision. The user can still pin any strategy
key (`options={"fft_stage_group": 2}`); `derive` only fills what was
left at null.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from .config import OPTIONS
from ...compiler.toolchain import find_tool, run_tool

__all__ = [
    "AUTO_OPTIONS", "KernelFacts", "array_partition_max_factor",
    "buffer_extents", "derive", "external_buffer_threshold", "fft_stage_group",
    "fft_io_unroll", "fft_parallel_rows", "interp_banded_gather",
    "kernel_facts_from_json",
    "loop_tile_size", "lutram_max_bytes", "measure_kernel",
    "storage_min_elements", "streaming_threshold", "transpose_block_bytes"
]

#: The options this module decides -- exactly the schema's `advanced`
#: keys, so a new derived key cannot be forgotten here. Everything else
#: is a constraint the user has to state.
AUTO_OPTIONS = tuple(key for key, spec in OPTIONS.items() if spec.advanced)

#: Share of the on-chip budget one working buffer may claim: several are
#: in flight while the planes they serve still need the rest. The same
#: eighth rations the loop-tiling staging blocks inside `-hls-pipeline`.
_WORKING_SHARE = 8

#: Threshold that leaves every buffer resident (no buffer has this many
#: elements). Mirrors the ceiling of the pass option it feeds.
KEEP_ON_CHIP = 2**32 - 1

#: A buffer within this factor of the largest plane counts as one of them.
#: Slicing an edge off a raster leaves a buffer a hair under full size --
#: it is still a plane, and streaming its siblings while keeping it
#: resident would be the worst of both.
_PLANE_TOLERANCE = 4

#: Tiling is off at 1, so a tile is at least a pair; past 64 elements a
#: side the per-band local buffer has stopped being a bank and become a
#: plane, which is what `external_buffer_threshold` is for.
_MIN_TILE, _MAX_TILE = 2, 64

# ---------------------------------------------------------------------- #
# What the compiler can measure about a kernel
# ---------------------------------------------------------------------- #


@dataclass(frozen=True)
class KernelFacts:
    """Properties of a kernel the strategy is derived from.

    Measured from the `sar` module the frontend traced, not from the
    signature: a chain may work on a grid larger than anything it is
    handed -- polar format resamples onto an oversampled raster -- and it
    is those planes, and those transform lengths, that decide the cost.
    """

    #: Elements in the largest tensor the kernel touches anywhere.
    plane_elements: int
    #: Narrowest real element on the data path, in bytes. A tile sized in
    #: these is a whole number of beats for every wider plane too.
    element_bytes: int
    #: (transform length, element bytes) per FFT the kernel performs.
    transforms: Tuple[Tuple[int, int], ...]
    #: Corner turns (`sar.transpose`) the kernel performs. Each staged
    #: transpose holds a block resident, so the staging budget is split
    #: between them.
    transposes: int = 0
    #: (elements, bytes) for every explicit allocation in lowered IR.
    buffers: Tuple[Tuple[int, int], ...] = ()


def kernel_facts_from_json(text: str) -> KernelFacts:
    """Builds immutable strategy facts from `sar-translate` JSON."""
    values = json.loads(text)
    return KernelFacts(
        plane_elements=int(values["plane_elements"]),
        element_bytes=int(values["element_bytes"]),
        transforms=tuple((int(length), int(width))
                         for length, width in values["transforms"]),
        transposes=int(values["transposes"]),
        buffers=tuple((int(elements), int(size))
                      for elements, size in values["buffers"]),
    )


def measure_kernel(module_text: str, metadata=None) -> KernelFacts:
    """Parses MLIR with the compiler and returns structured strategy facts."""
    output = run_tool(
        "sar-kernel-facts",
        [find_tool("sar-translate"), "--sar-emit-kernel-facts", "-"],
        input_text=module_text)
    return kernel_facts_from_json(output)


# ---------------------------------------------------------------------- #
# What the lowered kernel costs, once there is one
# ---------------------------------------------------------------------- #


def buffer_extents(lowered: str):
    """(elements, bytes) of every explicit allocation in lowered IR."""
    return iter(measure_kernel(lowered).buffers)


def _plane_elements(metadata, lowered_facts: KernelFacts) -> int:
    """Element count above which a buffer counts as a full-scene plane."""
    planes = list(metadata.arg_types) + list(metadata.result_types)
    largest = max([int(np.prod(t.shape)) for t in planes] +
                  [elements for elements, _ in lowered_facts.buffers])
    return largest // _PLANE_TOLERANCE


def _resident_bytes(metadata, lowered_facts: KernelFacts,
                    min_elements: int) -> int:
    """On-chip memory the lowered kernel would need for its full-size planes.

    The kernel's own planes come from the signature; the intermediates are
    counted in the IR, after buffers with disjoint lifetimes have been
    shared. Counting rather than estimating is what makes the placement
    decision hold for any algorithm: the same four-stage chain needs six
    live planes under range-Doppler and ten under chirp-scaling, and nothing
    in the signature says which.
    """
    planes = list(metadata.arg_types) + list(metadata.result_types)
    total = sum(
        int(np.prod(t.shape)) * t.dtype.to_numpy().itemsize for t in planes)
    total += sum(size for elements, size in lowered_facts.buffers
                 if elements >= min_elements)
    return total


# ---------------------------------------------------------------------- #
# One decision each
# ---------------------------------------------------------------------- #


def _scratch_slots(stages: int, group: int) -> int:
    """Scratch lines a grouped Stockham transform holds live.

    Mirrors `scratchSlots` in SARFFTToAffine.cpp: the cost model and the
    lowering have to agree, or the budget is spent against a design that
    was never emitted.
    """
    intermediates = stages - 1
    if intermediates <= 0:
        return 0
    if group == 0:
        return intermediates
    slots = -(-stages // group) - 1
    if slots < 2:
        slots = min(intermediates, 2)
    return min(slots, intermediates)


def _transform_stages(length: int) -> Tuple[int, int, bool]:
    """(mixed-radix stages, line length, whether Bluestein carries it)."""
    if length > 0 and not length & (length - 1):
        exponent = length.bit_length() - 1
        return (exponent + 1) // 2, length, False
    padded = 1 << (2 * length - 1).bit_length()
    exponent = padded.bit_length() - 1
    return (exponent + 1) // 2, padded, True


def _fft_scratch_bytes(transforms, group: int, lanes: int = 1,
                       io: int = 1) -> int:
    """Primitive-aware bytes for transform storage at `group`/`lanes`/`io`.

    Each bank occupies at least one BRAM/URAM primitive, so logical payload
    bytes alone understate banked designs by an order of magnitude. A
    lane-parallel engine banks every buffer completely over the lane
    dimension; the prefetch and write-back blocks additionally bank the
    element dimension by the io unroll. Serial engines leave banking to
    the automatic search, whose factor caps at one bus beat.
    """

    def physical(line_bytes: int, banks: int) -> int:
        bank_bytes = -(-line_bytes // banks)
        primitive = 36864 if line_bytes >= 36864 else 4608
        return banks * (-(-bank_bytes // primitive)) * primitive

    total = 0
    for length, width in transforms:
        stages, line, bluestein = _transform_stages(length)
        slots = _scratch_slots(stages, group)
        line_bytes = line * width * max(1, lanes)
        if bluestein:
            # Chirp-z runs two transforms over shared stage/spectrum/
            # product/convolution lines, so eight lines stand whatever the
            # grouping and each transform draws its own slots. It stays
            # line-serial, so lanes and io do not enter.
            total += (8 + 4 * slots) * physical(line * width, min(32, line))
        elif lanes > 1:
            total += slots * 2 * physical(line_bytes, lanes)
            total += 2 * 2 * physical(line_bytes, lanes * max(1, io))
        else:
            total += (slots + 2) * 2 * physical(line_bytes, min(32, line))
    return total


def _fft_groups(transforms) -> Tuple[int, ...]:
    """Groupings worth considering, most throughput first.

    0 is the full unroll. 1 is the same slot count under another name, so
    it is not a distinct point. Past the longest transform's stage count
    the slot pool has bottomed out at its floor of two.
    """
    longest = max((_transform_stages(length)[0] for length, _ in transforms),
                  default=0)
    return (0, ) + tuple(range(2, max(longest, 2) + 1))


def fft_stage_group(facts: KernelFacts, budget: int, lanes: int = 1,
                    io: int = 1) -> int:
    """Stockham stages per scratch slot: the least grouping whose scratch
    fits the working share of the budget (the tightest available when none
    does), because full unroll is the throughput point and grouping buys
    area back with pipeline depth. Charged at the lane count the engine
    will actually run."""
    if not facts.transforms:
        return 0
    groups = _fft_groups(facts.transforms)
    if budget <= 0:
        return min(groups,
                   key=lambda g: _fft_scratch_bytes(facts.transforms, g,
                                                    lanes, io))
    share = max(1, budget // _WORKING_SHARE)
    for group in groups:
        if _fft_scratch_bytes(facts.transforms, group, lanes, io) <= share:
            return group
    return min(groups,
               key=lambda g: _fft_scratch_bytes(facts.transforms, g,
                                                lanes, io))


def loop_tile_size(facts: KernelFacts, axi_bus_bits: int) -> int:
    """Elements a tiled loop dimension carries: one bus beat in the
    kernel's narrowest element, because a tile is the contiguous run of an
    external access and a fraction of a beat wastes the rest of it."""
    beat_bytes = max(1, axi_bus_bits // 8)
    tile = beat_bytes // max(1, facts.element_bytes)
    return max(_MIN_TILE, min(_MAX_TILE, tile))


def fft_parallel_rows(facts: KernelFacts, dsp_budget: int, budget: int,
                      io: int = 1) -> int:
    """Power-of-two FFT lane count the arithmetic and memory afford.

    Lines are prefetched in blocks, so lane parallelism no longer multiplies
    external traffic -- the transfers stay unit-stride whatever the count.
    What lanes do consume is DSP slices (one butterfly datapath each, ~512
    slices with the f64 position arithmetic around it) and line-buffer
    banks, whose block RAM cost the scratch model charges per lane. The
    lane count is the largest power of two both budgets accept, capped at
    16: past that the banked line buffers outgrow what placement can pay.
    A result of one is represented as zero because the pass uses zero to
    mean "serial lines".
    """
    if not facts.transforms:
        return 0
    dsp_lanes = max(1, dsp_budget // 512)
    lanes = 1 << (min(dsp_lanes, 16).bit_length() - 1)
    # Line buffers live in the block tiers and pay a whole primitive per
    # bank, so the check is against BRAM+URAM, not the aggregate that
    # includes distributed RAM.
    while lanes > 1 and _fft_scratch_bytes(facts.transforms, 0, lanes,
                                           io) > budget // 2:
        lanes >>= 1
    return lanes if lanes > 1 else 0


def fft_io_unroll(facts: KernelFacts, axi_bus_bits: int) -> int:
    """Elements one FFT transfer access moves. Half a beat per plane: the
    real and imaginary sweeps run in one loop, so together they fill the
    bus; wider unrolls only multiply the line-buffer banks each lane pays
    a memory primitive for."""
    if not facts.transforms:
        return 1
    element_bytes = min(width for _, width in facts.transforms)
    beat = max(1, axi_bus_bits // 16) // max(1, element_bytes)
    return max(1, 1 << (min(beat, 8).bit_length() - 1))


def lutram_max_bytes(axi_bus_bits: int) -> int:
    """Bank size below which distributed RAM is the right tier: one bus
    beat, since a bank that cannot fill one transfer is not worth a
    dedicated block RAM primitive."""
    return max(1, axi_bus_bits // 8)


def array_partition_max_factor(facts: KernelFacts, axi_bus_bits: int) -> int:
    """Maximum useful bank count before it exceeds one external bus beat."""
    lanes = max(1, axi_bus_bits // max(8, facts.element_bytes * 8))
    factor = 1 << (lanes.bit_length() - 1)
    return min(32, factor)


def interp_banded_gather() -> bool:
    """Whether interpolation may gather through an on-chip band: always
    on, because the pass falls back to the full-plane gather on its own
    whenever it cannot prove a bounded displacement."""
    return True


def storage_min_elements(facts: KernelFacts, budget: int) -> int:
    """Element count at which a buffer stops being a dataflow channel and
    becomes storage (shared or recomputed), because a full-scene plane
    kept as a private channel would cost its own bytes on chip; a tighter
    budget pulls the line down to one working buffer's share."""
    if budget <= 0:
        return 1
    share = max(1, budget // _WORKING_SHARE)
    return max(1, min(facts.plane_elements, share // facts.element_bytes))


def transpose_block_bytes(facts: KernelFacts, bram_bytes: int) -> int:
    """Bytes one transpose staging block may occupy.

    The blocks live in block RAM, every corner turn holds one per plane
    (a complex transpose stages re and im), and Vitis ping-pong doubles
    each -- so the block RAM cap is split across all of them, with half
    of the tier left for everything else. The 4 KiB floor keeps a block
    at least one burst long on any budget.
    """
    if bram_bytes <= 0:
        return 0
    slots = max(1, 2 * facts.transposes)
    return max(4096, bram_bytes // (4 * slots))


def external_buffer_threshold(facts: KernelFacts, budget: int, metadata,
                              lowered_facts: KernelFacts) -> int:
    """Element count above which a buffer is moved off chip: planes stay
    resident while the whole set fits the budget and stream once it does
    not, because full-scene planes dominate the working set and streaming
    them is what lets scene size grow past the device.

    The resident set is charged at twice its single-instance size, since
    Vitis ping-pong double-buffers every dataflow channel.
    """
    elements = _plane_elements(metadata, lowered_facts)
    if not elements:
        return KEEP_ON_CHIP
    if budget <= 0:
        return elements
    if 2 * _resident_bytes(metadata, lowered_facts, elements) <= budget:
        return KEEP_ON_CHIP
    return elements


def streaming_threshold(facts: KernelFacts) -> int:
    """The threshold that streams every full-size plane: what the resident
    decision falls back to when the placed design still overruns the
    budgets (the fork/balance copies are not visible to the resident
    estimate, so the first answer can be too optimistic)."""
    return max(1, facts.plane_elements // _PLANE_TOLERANCE)


# ---------------------------------------------------------------------- #
# The policy, in one call
# ---------------------------------------------------------------------- #


def derive(config,
           facts: KernelFacts,
           metadata=None,
           lowered_facts: Optional[KernelFacts] = None) -> Dict[str, object]:
    """Every strategy value the constraints and the kernel imply.

    Called twice: once on the traced module, where everything but the
    placement threshold is decidable, and again once the kernel is lowered
    and its buffers can be counted. Keys the user pinned are still
    returned; `HLSConfig.adopt` keeps the pinned value.
    """
    assert (metadata is None) == (lowered_facts is None), \
        "the placement decision needs the metadata and the lowered IR together"
    budget = config.on_chip_bytes()
    io = fft_io_unroll(facts, int(config.axi_bus_bits))
    lanes = fft_parallel_rows(
        facts, int(config.dsp),
        int(config.bram_bytes) + int(config.uram_bytes), io)
    values = {
        "fft_stage_group":
        fft_stage_group(facts, budget, max(1, lanes), io),
        "loop_tile_size":
        loop_tile_size(facts, int(config.axi_bus_bits)),
        "fft_parallel_rows":
        lanes,
        "fft_io_unroll":
        io,
        "interp_banded_gather":
        interp_banded_gather(),
        "reuse_buffer_min_elements":
        storage_min_elements(facts, budget),
        "recompute_min_elements":
        storage_min_elements(facts, budget),
        "lutram_max_bytes":
        lutram_max_bytes(int(config.axi_bus_bits)),
        "array_partition_max_factor":
        array_partition_max_factor(facts, int(config.axi_bus_bits)),
    }
    if lowered_facts is not None:
        values["external_buffer_threshold"] = external_buffer_threshold(
            facts, budget, metadata, lowered_facts)
    return values
