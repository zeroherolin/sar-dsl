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

import re
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

__all__ = [
    "AUTO_OPTIONS", "KernelFacts", "derive", "lutram_max_bytes",
    "measure_kernel"
]

#: The options this module decides. Everything else in the schema is a
#: constraint the user has to state.
AUTO_OPTIONS = ("fft_stage_group", "loop_tile_size", "interp_banded_gather",
                "reuse_buffer_min_elements", "recompute_min_elements",
                "external_buffer_threshold", "lutram_max_bytes")

#: Share of the on-chip budget one working buffer may claim. `-hls-pipeline`
#: already rations a staged transpose block this way (`getStagingBudget`),
#: and transform scratch and private dataflow channels are the same kind of
#: per-region working set: several are in flight while the planes they serve
#: still need the rest of the budget.
_WORKING_SHARE = 8

#: Threshold that leaves every buffer resident (no buffer has this many
#: elements). Mirrors the ceiling of the pass option it feeds.
KEEP_ON_CHIP = 2**32 - 1

#: A buffer within this factor of the largest plane counts as one of them.
#: Slicing an edge off a raster leaves a buffer a hair under full size --
#: it is still a plane, and streaming its siblings while keeping it
#: resident would be the worst of both.
_PLANE_TOLERANCE = 4

#: Byte width of the element types the affine flow emits. A complex tensor
#: is decomplexified into planes of its component float, so the data path
#: only ever sees these two.
_ELEMENT_BYTES = {"f32": 4, "f64": 8}

#: Tiling is off at 1, so a tile is at least a pair; past 64 elements a
#: side the per-band local buffer has stopped being a bank and become a
#: plane, which is what `external_buffer_threshold` is for.
_MIN_TILE, _MAX_TILE = 2, 64

#: Element width assumed when a kernel names no float plane at all: the
#: widest, which keeps the derived tile the shortest one that still fills
#: a beat.
_WIDEST_ELEMENT = 8

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


_TENSOR_RE = re.compile(r"tensor<([0-9x]*?)((?:complex<)?[a-z]\w*>?)>")
_FFT_RE = re.compile(r"sar\.(?:i?fft|fft_split)\b")
_DIM_RE = re.compile(r"dim = (\d+)")


def _element_bytes(element: str) -> Optional[int]:
    """Bytes one plane element takes, or None for a type the data path
    never carries."""
    for name, width in _ELEMENT_BYTES.items():
        if name in element:
            return width
    return None


def _tensors(text: str):
    """(shape, element bytes or None) of every static tensor named."""
    for match in _TENSOR_RE.finditer(text):
        dims = [int(d) for d in match.group(1).split("x") if d]
        yield dims, _element_bytes(match.group(2))


def _transforms(module_text: str):
    """(length, element bytes) of every FFT the module performs.

    The length is the extent the transform runs along, so a corner-turned
    chain and a straight one are told apart by the axis they name rather
    than by how the planes happen to be laid out.
    """
    for line in module_text.splitlines():
        if not _FFT_RE.search(line):
            continue
        dim = _DIM_RE.search(line)
        shapes = [(d, w) for d, w in _tensors(line) if w is not None and d]
        if dim is None or not shapes:
            continue
        dims, width = max(shapes, key=lambda s: len(s[0]))
        axis = int(dim.group(1))
        if 0 <= axis < len(dims):
            yield dims[axis], width


def measure_kernel(module_text: str, metadata) -> KernelFacts:
    """Reads the strategy inputs out of a traced kernel."""
    planes = [(list(t.shape), t.dtype.to_numpy().itemsize)
              for t in list(metadata.arg_types) + list(metadata.result_types)]
    tensors = list(_tensors(module_text)) + planes
    widths = [width for _, width in tensors if width]
    return KernelFacts(
        plane_elements=max([int(np.prod(dims)) for dims, _ in tensors] + [0]),
        element_bytes=min(widths) if widths else _WIDEST_ELEMENT,
        transforms=tuple(_transforms(module_text)))


# ---------------------------------------------------------------------- #
# What the lowered kernel costs, once there is one
# ---------------------------------------------------------------------- #


def buffer_extents(lowered: str):
    """(elements, bytes) of every buffer the lowered kernel allocates."""
    for match in re.finditer(r"memref\.alloc\(\)[^:]*: memref<([^>]+)>",
                             lowered):
        parts = match.group(1).split("x")
        width = _ELEMENT_BYTES.get(parts[-1])
        if width is None:
            continue
        elements = int(np.prod([int(d) for d in parts[:-1]]))
        yield elements, elements * width


def _plane_elements(metadata, lowered: str) -> int:
    """Element count above which a buffer counts as a full-scene plane."""
    planes = list(metadata.arg_types) + list(metadata.result_types)
    largest = max([int(np.prod(t.shape)) for t in planes] +
                  [elements for elements, _ in buffer_extents(lowered)])
    return largest // _PLANE_TOLERANCE


def _resident_bytes(metadata, lowered: str, min_elements: int) -> int:
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
    total += sum(size for elements, size in buffer_extents(lowered)
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
    """(butterfly stages, line length, whether Bluestein carries it)."""
    if length > 0 and not length & (length - 1):
        return length.bit_length() - 1, length, False
    padded = 1 << (2 * length - 1).bit_length()
    return padded.bit_length() - 1, padded, True


def _fft_scratch_bytes(transforms, group: int) -> int:
    """On-chip bytes the transforms' scratch lines hold live at `group`."""
    total = 0
    for length, width in transforms:
        stages, line, bluestein = _transform_stages(length)
        slots = _scratch_slots(stages, group)
        if bluestein:
            # Chirp-z runs two transforms over shared stage/spectrum/
            # product/convolution lines, so eight lines stand whatever the
            # grouping and each transform draws its own slots.
            total += (8 + 4 * slots) * line * width
        else:
            total += slots * 2 * line * width
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


def fft_stage_group(facts: KernelFacts, budget: int) -> int:
    """Stockham stages per scratch slot: the least grouping whose scratch
    fits the working share of the budget (the tightest available when none
    does), because full unroll is the throughput point and grouping buys
    area back with pipeline depth."""
    if not facts.transforms:
        return 0
    if budget <= 0:  # an unbounded budget can afford the full unroll
        return 0
    share = max(1, budget // _WORKING_SHARE)
    groups = _fft_groups(facts.transforms)
    for group in groups:
        if _fft_scratch_bytes(facts.transforms, group) <= share:
            return group
    return min(groups, key=lambda g: _fft_scratch_bytes(facts.transforms, g))


def loop_tile_size(facts: KernelFacts, axi_bus_bits: int) -> int:
    """Elements a tiled loop dimension carries: one bus beat in the
    kernel's narrowest element, because a tile is the contiguous run of an
    external access and a fraction of a beat wastes the rest of it."""
    beat_bytes = max(1, axi_bus_bits // 8)
    tile = beat_bytes // max(1, facts.element_bytes)
    return max(_MIN_TILE, min(_MAX_TILE, tile))


def lutram_max_bytes(axi_bus_bits: int) -> int:
    """Bank size below which distributed RAM is the right tier: one bus
    beat.

    A bank holding less than a beat cannot fill one transfer, so spending
    a dedicated block RAM primitive on it wastes the primitive; such a
    bank belongs in the SLICEM LUTs it can be built from directly.
    """
    return max(1, axi_bus_bits // 8)


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
        return max(1, facts.plane_elements)
    share = max(1, budget // _WORKING_SHARE)
    return max(1, min(facts.plane_elements, share // facts.element_bytes))


def external_buffer_threshold(facts: KernelFacts, budget: int, metadata,
                              lowered: str) -> int:
    """Element count above which a buffer is moved off chip: planes stay
    resident while the whole set fits the budget and stream once it does
    not, because full-scene planes dominate the working set and streaming
    them is what lets scene size grow past the device."""
    elements = _plane_elements(metadata, lowered)
    if budget <= 0 or not elements:
        return KEEP_ON_CHIP
    if _resident_bytes(metadata, lowered, elements) <= budget:
        return KEEP_ON_CHIP
    return elements


# ---------------------------------------------------------------------- #
# The policy, in one call
# ---------------------------------------------------------------------- #


def derive(config,
           facts: KernelFacts,
           metadata=None,
           lowered: Optional[str] = None) -> Dict[str, object]:
    """Every strategy value the constraints and the kernel imply.

    Called twice: once on the traced module, where everything but the
    placement threshold is decidable, and again once the kernel is lowered
    and its buffers can be counted. Keys the user pinned are still
    returned -- `HLSConfig.derive` keeps the pinned value and reports this
    one as the road not taken.
    """
    budget = int(config.on_chip_budget)
    values = {
        "fft_stage_group": fft_stage_group(facts, budget),
        "loop_tile_size": loop_tile_size(facts, int(config.axi_bus_bits)),
        "interp_banded_gather": interp_banded_gather(),
        "reuse_buffer_min_elements": storage_min_elements(facts, budget),
        "recompute_min_elements": storage_min_elements(facts, budget),
        "lutram_max_bytes": lutram_max_bytes(int(config.axi_bus_bits)),
    }
    if lowered is not None:
        values["external_buffer_threshold"] = external_buffer_threshold(
            facts, budget, metadata, lowered)
    return values
