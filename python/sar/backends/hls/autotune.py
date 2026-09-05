"""Strategy the HLS backend decides for itself.

The schema in `config.py` splits into *constraints* (device and
deliverable facts only the user knows) and *strategy* (everything the
compiler can work out from those constraints and from measured kernel
properties -- plane sizes, transform lengths, element widths, buffer
lifetimes). This module is the strategy half: `plan()` is the whole
policy, one function per decision. The user can still pin any strategy
key (`options={"fft_stage_group": 2}`); `HLSConfig.adopt` keeps the pinned
value and fills only what was left at null.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from .config import FIXED, OPTIONS, TUNED
from .devices import storage_primitives
from ...compiler.toolchain import find_tool, run_tool

__all__ = [
    "AUTO_OPTIONS", "FIXED_OPTIONS", "TUNED_OPTIONS", "KernelFacts",
    "PerformancePlan", "array_partition_max_factor",
    "external_buffer_threshold", "fft_stage_group",
    "external_vector_max_lanes", "external_vector_min_elements",
    "external_vector_pack_outputs", "external_vector_compute_lanes",
    "fft_io_unroll", "fft_parallel_rows", "interp_banded_gather",
    "interp_cache_copies", "interp_complete_bank_max_elements",
    "interp_full_row_max_bytes", "fuse_sibling_sweeps", "loop_unroll_budget",
    "cached_kernel_facts", "cached_performance_plan", "kernel_facts_from_json",
    "kernel_facts_to_dict", "loop_tile_size", "lutram_max_bytes", "plan",
    "max_scratch_arenas", "measure_kernel", "reuse_min_elements",
    "should_share_buffers", "storage_min_elements", "streaming_threshold",
    "transpose_block_bytes", "transform_lane_storage_ceiling"
]

#: The options this module decides -- exactly the schema's `advanced`
#: keys, so a new derived key cannot be forgotten here. Everything else
#: is a constraint the user has to state.
AUTO_OPTIONS = tuple(key for key, spec in OPTIONS.items() if spec.advanced)

#: The subset that is genuinely tuned: modelled per kernel against the
#: resource budgets, and therefore the search space where the latency/area
#: trade-off is made.
TUNED_OPTIONS = tuple(key for key, spec in OPTIONS.items()
                      if spec.strategy == TUNED)

#: The subset that is the same for every kernel. These encode a property of
#: the lowering or of the synthesis tool rather than a per-kernel choice, so
#: they are constants with a rationale, not decisions with a cost model.
FIXED_OPTIONS = tuple(key for key, spec in OPTIONS.items()
                      if spec.strategy == FIXED)

#: Share of the on-chip budget one working buffer may claim: several are
#: in flight while the planes they serve still need the rest. The same
#: eighth rations the loop-tiling staging blocks inside `-hls-pipeline`.
_WORKING_SHARE = 8

#: Share of the block-memory tiers the transform engine may hold at once.
#: Stage grouping changes local line scratch but not external plane traffic,
#: so it receives half the tiers and leaves the remainder for resident planes
#: and lane parallelism, which does reduce complete line iterations.
_TRANSFORM_SHARE = 1
_TRANSFORM_SHARE_DENOMINATOR = 2

#: Lane-parallel line buffers may use a little more of the block-memory tiers
#: than stage unrolling.  The distinction is intentional: another stage slot
#: only shortens a local call chain, while another line lane removes complete
#: row iterations from every transform.  The final banking pass can rebalance
#: BRAM and URAM after their physical bank sizes are known, so lane feasibility
#: is charged against the aggregate tier headroom instead of rejecting a lane
#: merely because its first placement prefers the wrong primitive.
_TRANSFORM_LANE_SHARE = 5
_TRANSFORM_LANE_SHARE_DENOMINATOR = 8


def transform_storage_ceiling(budget: int) -> int:
    """Bytes the transform engine may hold on chip, out of `budget`.

    `budget` is the block-RAM tier total (`bram_bytes + uram_bytes`), the
    tiers a banked line buffer can actually land in.
    """
    return max(1, budget * _TRANSFORM_SHARE // _TRANSFORM_SHARE_DENOMINATOR)


def transform_lane_storage_ceiling(budget: int) -> int:
    """Bytes lane-parallel transform storage may claim from ``budget``.

    Stage grouping keeps the half-tier ceiling above. Lanes get
    five eighths because they reduce the number of complete line iterations,
    and the bank-aware placement pass can trade BRAM against URAM to fit the
    resulting physical banks without exceeding either hard cap.
    """
    return max(
        1, budget * _TRANSFORM_LANE_SHARE // _TRANSFORM_LANE_SHARE_DENOMINATOR)


#: Threshold that leaves every buffer resident (no buffer has this many
#: elements). Mirrors the ceiling of the pass option it feeds.
KEEP_ON_CHIP = 2**32 - 1

#: Threshold that shares no allocation at all. The reuse option is a 64-bit
#: element count, so this is above any buffer a kernel can declare.
NEVER_SHARE = 2**63 - 1

#: A buffer within this factor of the largest plane counts as one of them.
#: Slicing an edge off a raster leaves a buffer a hair under full size --
#: it is still a plane, and streaming its siblings while keeping it
#: resident would be the worst of both.
_PLANE_TOLERANCE = 4

#: Tiling is off at 1, so a tile is at least a pair; past 64 elements a
#: side the per-band local buffer has stopped being a bank and become a
#: plane, which is what `external_buffer_threshold` is for.
_MIN_TILE, _MAX_TILE = 2, 64

# Vitis 2022.2 synthesis keeps replicated FFT transfers at eight f32 lanes;
# contiguous external words may use all 16 f32 lanes of a 512-bit beat when
# the lowered arithmetic and fanout model says that packing stays affordable.
_MAX_VECTOR_LANES = 8
_MAX_EXTERNAL_VECTOR_LANES = 16
_MAX_VECTOR_BINDING_SCORE = 512

# Complete partition turns a dynamic band lookup into one mux per read. Only a
# quarter of either logic budget may be spent on this access structure, and a
# separate clock-dependent cap bounds its high-fanout routing.
_COMPLETE_BANK_LUT_PER_ELEMENT_LANE = 384
_COMPLETE_BANK_FF_PER_ELEMENT_LANE = 576
_MAX_COMPLETE_BANK_ELEMENTS = 128

#: `(block RAM, UltraRAM)` block size used when the target is not one of the
#: tabulated parts. UltraScale+ geometry, which is what the shipped
#: constraints describe.
_DEFAULT_PRIMITIVES = storage_primitives(None)


def _transfer_banks(io: int) -> int:
    """Minimum cyclic element banks for an ``io``-wide transfer.

    A dual-port memory bank can serve two scalar accesses per cycle, so a
    packed sweep needs ``ceil(io / 2)`` element banks.  Keep one bank as the
    representation for a scalar transfer: the C++ lowering uses factor zero
    there to select ``none`` partitioning, but the cost model still has to
    charge the complete lane banking once.
    """
    return max(1, (io + 1) // 2)


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
    #: Every scalar plane width present in the kernel, in bytes.
    element_bytes_set: Tuple[int, ...] = ()
    #: Data-dependent interpolation/gather operations in the traced graph.
    gather_ops: int = 0
    #: On-chip sliding gather buffers proven by affine lowering.
    banded_gathers: int = 0
    full_row_gathers: int = 0
    direct_gathers: int = 0
    #: Total operations and memory/arithmetic pressure in the measured IR.
    operations: int = 0
    loads: int = 0
    stores: int = 0
    expensive_ops: int = 0
    calls: int = 0
    max_fanout: int = 0
    #: Whether each transform axis is not the fastest-varying dimension.
    #: Such an engine widens transfers over adjacent lines rather than over
    #: adjacent elements of one line.
    transform_strided: Tuple[bool, ...] = ()


@dataclass(frozen=True)
class PerformancePlan:
    """One coherent set of HLS strategy decisions and its measured basis."""

    values: Dict[str, object]
    clock_ns: float
    timing_budget_ns: float
    on_chip_bytes: int
    memory_accesses: int
    operation_count: int

    def to_dict(self) -> Dict[str, object]:
        """The one JSON shape plan caches and decision records use."""
        return {
            "values": dict(self.values),
            "clock_ns": self.clock_ns,
            "timing_budget_ns": self.timing_budget_ns,
            "on_chip_bytes": self.on_chip_bytes,
            "memory_accesses": self.memory_accesses,
            "operation_count": self.operation_count,
        }

    @classmethod
    def from_dict(cls, value: Dict[str, object]) -> "PerformancePlan":
        return cls(values=dict(value["values"]),
                   clock_ns=float(value["clock_ns"]),
                   timing_budget_ns=float(value["timing_budget_ns"]),
                   on_chip_bytes=int(value["on_chip_bytes"]),
                   memory_accesses=int(value["memory_accesses"]),
                   operation_count=int(value["operation_count"]))


def kernel_facts_from_json(text: str) -> KernelFacts:
    """Builds immutable strategy facts from `sar-translate` JSON."""
    values = json.loads(text)
    transforms = tuple(
        (int(length), int(width)) for length, width in values["transforms"])
    strided = tuple(
        bool(value) for value in values.get("transform_strided", ()))
    if not strided:
        strided = (False, ) * len(transforms)
    if len(strided) != len(transforms):
        raise ValueError("kernel facts contain a mismatched "
                         "transform_strided list")
    return KernelFacts(
        plane_elements=int(values["plane_elements"]),
        element_bytes=int(values["element_bytes"]),
        transforms=transforms,
        transposes=int(values["transposes"]),
        buffers=tuple((int(elements), int(size))
                      for elements, size in values["buffers"]),
        element_bytes_set=tuple(
            int(width) for width in values.get("element_bytes_set",
                                               [values["element_bytes"]])),
        gather_ops=int(values.get("gathers", 0)),
        banded_gathers=int(values.get("banded_gathers", 0)),
        full_row_gathers=int(values.get("full_row_gathers", 0)),
        direct_gathers=int(values.get("direct_gathers", 0)),
        operations=int(values.get("operations", 0)),
        loads=int(values.get("loads", 0)),
        stores=int(values.get("stores", 0)),
        expensive_ops=int(values.get("expensive_ops", 0)),
        calls=int(values.get("calls", 0)),
        max_fanout=int(values.get("max_fanout", 0)),
        transform_strided=strided,
    )


def kernel_facts_to_dict(facts: KernelFacts) -> Dict[str, object]:
    """The one cache-file shape `kernel_facts_from_json` reads back."""
    return {
        "plane_elements":
        facts.plane_elements,
        "element_bytes":
        facts.element_bytes,
        "transforms":
        facts.transforms,
        "transposes":
        facts.transposes,
        "buffers":
        facts.buffers,
        "element_bytes_set":
        facts.element_bytes_set,
        "gathers":
        facts.gather_ops,
        "banded_gathers":
        facts.banded_gathers,
        "full_row_gathers":
        facts.full_row_gathers,
        "direct_gathers":
        facts.direct_gathers,
        "operations":
        facts.operations,
        "loads":
        facts.loads,
        "stores":
        facts.stores,
        "expensive_ops":
        facts.expensive_ops,
        "calls":
        facts.calls,
        "max_fanout":
        facts.max_fanout,
        "transform_strided": (facts.transform_strided
                              or (False, ) * len(facts.transforms)),
    }


def measure_kernel(module_text: str) -> KernelFacts:
    """Parses MLIR with the compiler and returns structured strategy facts."""
    output = run_tool(
        "sar-kernel-facts",
        [find_tool("sar-translate"), "--sar-emit-kernel-facts", "-"],
        input_text=module_text)
    return kernel_facts_from_json(output)


def cached_kernel_facts(cache, filename: str, module_text: str) -> KernelFacts:
    """Structured kernel facts through the compilation artifact cache.

    The stored JSON names a digest of the IR it was measured from, so a
    cache entry whose IR was rewritten (a retry rung, an interrupted
    compile) recomputes instead of serving facts for different IR."""
    digest = hashlib.sha256(module_text.encode()).hexdigest()
    cached = cache.read_if_cached(filename)
    if cached is not None:
        try:
            if json.loads(cached).get("module_sha256") == digest:
                return kernel_facts_from_json(cached)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            pass
        cache.path(filename).unlink(missing_ok=True)
    facts = measure_kernel(module_text)
    payload = kernel_facts_to_dict(facts)
    payload["module_sha256"] = digest
    text = json.dumps(payload, sort_keys=True)
    cache.write_text(filename, text)
    # Parse what was stored: warm and cold compiles then agree exactly.
    return kernel_facts_from_json(text)


def cached_performance_plan(
        cache,
        filename: str,
        config,
        facts: KernelFacts,
        metadata=None,
        lowered_facts: Optional[KernelFacts] = None) -> PerformancePlan:
    """Performance plan through the same content-addressed artifact cache."""
    cached = cache.read_if_cached(filename)
    if cached is not None:
        try:
            return PerformancePlan.from_dict(json.loads(cached))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            cache.path(filename).unlink(missing_ok=True)
    result = plan(config, facts, metadata, lowered_facts)
    cache.write_text(filename, json.dumps(result.to_dict(), sort_keys=True))
    return result


# ---------------------------------------------------------------------- #
# What the lowered kernel costs, once there is one
# ---------------------------------------------------------------------- #


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


def _fft_scratch_bytes(
        transforms,
        group: int,
        lanes: int = 1,
        io: int = 1,
        transform_strided=(),
        primitives: Tuple[int, int] = _DEFAULT_PRIMITIVES) -> int:
    """Primitive-aware bytes for transform storage at `group`/`lanes`/`io`.

    Each bank occupies at least one BRAM/URAM primitive, so logical payload
    bytes alone understate banked designs by an order of magnitude. A
    lane-parallel engine banks every buffer completely over the lane
    dimension; the prefetch and write-back blocks additionally bank the
    element dimension by the io unroll. Serial engines leave banking to
    the automatic search, whose factor caps at one bus beat.

    `primitives` is the target's `(block RAM, UltraRAM)` block size; a
    family without UltraRAM reports zero for it and rounds everything to
    block RAM, since that is the only tier a bank can land in.
    """
    bram_bytes, uram_bytes = primitives

    def physical(line_bytes: int, banks: int) -> int:
        bank_bytes = -(-line_bytes // banks)
        primitive = (uram_bytes if uram_bytes and line_bytes >= uram_bytes else
                     bram_bytes)
        return banks * (-(-bank_bytes // primitive)) * primitive

    total = 0
    for index, (length, width) in enumerate(transforms):
        stages, line, bluestein = _transform_stages(length)
        slots = _scratch_slots(stages, group)
        line_bytes = line * width * max(1, lanes)
        strided = (index < len(transform_strided) and transform_strided[index])
        if bluestein:
            # Chirp-z runs two transforms over shared stage/spectrum/
            # product/convolution lines, so eight lines stand whatever the
            # grouping and each transform draws its own slots. It stays
            # line-serial, so lanes and io do not enter.
            total += (8 + 4 * slots) * physical(line * width, min(32, line))
        elif strided and (lanes > 1 or io > 1):
            # The staged transfer block covers a complete packed word of
            # adjacent lines while a narrower compute engine visits it in
            # sub-blocks. Only source and destination pay the transfer width;
            # the Stockham intermediates retain the compute-lane width.
            if lanes > 1:
                total += slots * 2 * physical(line_bytes, lanes)
            else:
                total += slots * 2 * physical(line * width, min(32, line))
            transfer_lanes = max(1, lanes, io)
            transfer_bytes = line * width * transfer_lanes
            total += 2 * 2 * physical(transfer_bytes, transfer_lanes)
        elif lanes > 1:
            total += slots * 2 * physical(line_bytes, lanes)
            total += 2 * 2 * physical(line_bytes, lanes * _transfer_banks(io))
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


def fft_stage_group(facts: KernelFacts,
                    budget: int,
                    lanes: int = 1,
                    io: int = 1,
                    primitives: Tuple[int, int] = _DEFAULT_PRIMITIVES) -> int:
    """Stockham stages per scratch slot: the least grouping whose scratch
    fits, because full unroll is the throughput point and grouping buys area
    back with pipeline depth. Charged at the lane count the engine will
    actually run.

    Two ceilings, tried in order. The working share is what one buffer may
    claim while the rest of the chain is in flight; when a production
    transform fits none of the groupings under it, the transform's own
    storage ceiling applies instead -- half the on-chip budget, the same
    bound `fft_parallel_rows` checks lane feasibility against. Skipping
    straight to the smallest scratch would discard the ordering these
    groupings are in: it deepens the stage chain past what the memory
    actually demands, and a longer chain is a longer critical path at no
    latency benefit.
    """
    if not facts.transforms:
        return 0
    groups = _fft_groups(facts.transforms)

    def scratch(group: int) -> int:
        return _fft_scratch_bytes(facts.transforms, group, lanes, io,
                                  facts.transform_strided, primitives)

    if budget <= 0:
        return min(groups, key=scratch)
    for ceiling in (max(1, budget // _WORKING_SHARE),
                    transform_storage_ceiling(budget)):
        for group in groups:
            if scratch(group) <= ceiling:
                return group
    return min(groups, key=scratch)


def loop_tile_size(facts: KernelFacts, axi_bus_bits: int) -> int:
    """Elements a tiled loop dimension carries: one bus beat in the
    kernel's narrowest element, because a tile is the contiguous run of an
    external access and a fraction of a beat wastes the rest of it."""
    beat_bytes = max(1, axi_bus_bits // 8)
    tile = beat_bytes // max(1, facts.element_bytes)
    return max(_MIN_TILE, min(_MAX_TILE, tile))


def fft_parallel_rows(
        facts: KernelFacts,
        dsp_budget: int,
        budget: int,
        io: int = 1,
        clock_ns: float = 4.0,
        primitives: Tuple[int, int] = _DEFAULT_PRIMITIVES) -> int:
    """Power-of-two FFT lane count the arithmetic and memory afford.

    Lines are prefetched in blocks, so lane parallelism no longer multiplies
    external traffic -- the transfers stay unit-stride whatever the count.
    What lanes do consume is DSP slices and line-buffer banks. Every
    transform site instantiates its own stage datapaths, so the arithmetic
    bound charges lanes across all sites and stages; the memory bound charges
    the banked line blocks against the block-memory tiers. The lane count
    is the largest power of two both accept, capped at 16. A result of one
    is represented as zero because the pass uses zero to mean "serial
    lines".
    """
    if not facts.transforms:
        return 0
    # Multiple data-dependent regrids already dominate the schedule and
    # memory-conflict graph; replicating their surrounding transforms spends
    # area and synthesis complexity without addressing that bottleneck.
    if facts.gather_ops > 1:
        return 0
    stage_lanes = sum(6 * _transform_stages(length)[0]
                      for length, _ in facts.transforms)
    dsp_lanes = max(1, dsp_budget // max(1, stage_lanes))
    lanes = 1 << (min(dsp_lanes, 16).bit_length() - 1)
    # Wider replicated datapaths make routing and operator chaining harder.
    # A tighter clock therefore lowers the architectural cap before Vitis,
    # rather than emitting the same design for every timing contract.
    timing_cap = 16 if clock_ns >= 4.0 else (8 if clock_ns >= 3.0 else 4)
    lanes = min(lanes, timing_cap)
    longest = max(length for length, _ in facts.transforms)
    independent_lines = max(1, facts.plane_elements // max(1, longest))
    # A lane is only worthwhile when it has several line blocks to process;
    # otherwise it multiplies RTL and synthesis work to shave a handful of
    # iterations from validation-size designs.
    amortized = max(1, independent_lines // 8)
    lanes = min(lanes, 1 << (min(amortized, 16).bit_length() - 1))
    # Slow-axis transfers stage a packed word of adjacent lines independently
    # of this compute-lane count. The memory model therefore charges their
    # wider source/destination blocks without replicating the Stockham
    # intermediates. Compute parallelism remains an arithmetic decision.
    # Feasibility is checked at the saturated stage grouping (the two-slot
    # floor), because that is what the grouping decision itself falls back
    # to under the same pressure; checking at full unroll would give up
    # lanes to scratch the design will never allocate.
    ceiling = transform_lane_storage_ceiling(budget)
    while lanes > 1 and _fft_scratch_bytes(facts.transforms, 64, lanes, io,
                                           facts.transform_strided,
                                           primitives) > ceiling:
        lanes >>= 1
    return lanes if lanes > 1 else 0


def fft_io_unroll(facts: KernelFacts, axi_bus_bits: int) -> int:
    """Elements one FFT transfer access moves.

    Each real/imaginary plane owns an independent AXI bundle, so the physical
    limit is a full beat per plane, and synthesis feedback caps the replicated
    transfer at eight lanes whatever the beat holds. What a wider transfer
    costs in staged block bytes is charged where the engine is sized:
    `_transform_engine` halves it against the same ceiling `fft_parallel_rows`
    checks lanes against.
    """
    if not facts.transforms:
        return 1
    element_bytes = min(width for _, width in facts.transforms)
    # A bus beat narrower than one element carries no whole element, so the
    # transfer moves one per access and the port widens instead. Flooring at
    # one keeps that case a valid unroll rather than a zero-lane transfer.
    beat = max(1, max(1, axi_bus_bits // 8) // max(1, element_bytes))
    lanes = min(beat, _MAX_VECTOR_LANES)
    return max(1, 1 << (lanes.bit_length() - 1))


def _element_widths(facts: KernelFacts) -> Tuple[int, ...]:
    return facts.element_bytes_set or (facts.element_bytes, )


def external_vector_max_lanes(
        facts: KernelFacts,
        axi_bus_bits: int,
        lowered_facts: Optional[KernelFacts] = None) -> int:
    """Common packed lane count under bus and synthesis-complexity limits.

    Multiple data-dependent regrids use a bounded four-lane packed scratch
    representation. Public output ABIs remain scalar in those graphs, while
    compiler-owned scratch and read-only inputs still benefit from complete
    word transfers.
    """
    widest = max(_element_widths(facts))
    lanes = max(1, axi_bus_bits // max(8, widest * 8))
    lanes = min(lanes, _MAX_EXTERNAL_VECTOR_LANES)
    if facts.gather_ops > 1:
        lanes = min(lanes, 4)
    if lowered_facts is not None:
        while lanes > 1 and (lowered_facts.expensive_ops * lanes +
                             lowered_facts.max_fanout
                             > _MAX_VECTOR_BINDING_SCORE):
            lanes >>= 1
    return 1 << (lanes.bit_length() - 1)


def external_vector_min_elements(facts: KernelFacts, lanes: int) -> int:
    """Amortization floor for changing a public array to a packed ABI."""
    return max(4096, lanes * lanes * 8)


def external_vector_pack_outputs(facts: KernelFacts) -> bool:
    """Keep public output ABI stable around multi-gather workspace graphs."""
    return facts.gather_ops <= 1


def external_vector_compute_lanes(facts: KernelFacts, lanes: int) -> int:
    """Compute unroll lanes inside a packed transfer word."""
    if facts.gather_ops:
        return min(lanes, 4)
    return lanes


def lutram_max_bytes(axi_bus_bits: int) -> int:
    """Bank size below which distributed RAM is the right tier: one bus
    beat, since a bank that cannot fill one transfer is not worth a
    dedicated block RAM primitive."""
    return max(1, axi_bus_bits // 8)


def max_scratch_arenas(facts: KernelFacts) -> int:
    """Bound physical scratch masters independently of graph depth.

    Four masters cover read/write ping-pong for split scalar planes. Additional
    logical lifetimes are carved into those arenas; growing a chain must not
    grow the platform interface.
    """
    return 4


def array_partition_max_factor(facts: KernelFacts, axi_bus_bits: int) -> int:
    """Maximum useful bank count before it exceeds one external bus beat."""
    widest = max(_element_widths(facts))
    lanes = max(1, axi_bus_bits // max(8, widest * 8))
    factor = 1 << (lanes.bit_length() - 1)
    return min(32, factor)


def interp_banded_gather() -> bool:
    """Whether interpolation may gather through an on-chip band: always
    on, because the pass selects a bounded band, a budgeted complete-row
    cache, or the direct full-plane gather without changing semantics."""
    return True


def interp_full_row_max_bytes(facts: KernelFacts, budget: int) -> int:
    """Block-memory share available to complete interpolation row caches.

    Each gather site owns its own split-complex row pair, so the common
    working-buffer share is divided across the sites.  The lowering checks
    the concrete row size against this cap and retains the direct gather when
    it does not fit.
    """
    if facts.gather_ops == 0 or budget <= 0:
        return 0
    return budget // (_WORKING_SHARE * facts.gather_ops)


def interp_cache_copies(facts: KernelFacts, lanes: int) -> int:
    """Replicas that isolate packed lanes reading an interpolation cache."""
    if facts.gather_ops == 0:
        return 1
    return min(4, max(1, lanes))


def interp_complete_bank_max_elements(facts: KernelFacts,
                                      lut_budget: int,
                                      ff_budget: int,
                                      lanes: int,
                                      clock_ns: float = 4.0) -> int:
    """Power-of-two band width affordable as fully partitioned storage."""
    if facts.gather_ops == 0 or lut_budget <= 0 or ff_budget <= 0:
        return 0
    parallel = max(1, lanes)
    sites = max(1, facts.gather_ops)
    lut_limit = lut_budget // (4 * sites * parallel *
                               _COMPLETE_BANK_LUT_PER_ELEMENT_LANE)
    ff_limit = ff_budget // (4 * sites * parallel *
                             _COMPLETE_BANK_FF_PER_ELEMENT_LANE)
    # Area alone does not bound the high-fanout mux after integration, so the
    # complete-bank width also follows a clock-dependent routing ceiling.
    routing_limit = (16 if clock_ns <= 4.0 else
                     32 if clock_ns <= 5.0 else _MAX_COMPLETE_BANK_ELEMENTS)
    limit = min(routing_limit, lut_limit, ff_limit)
    return 0 if limit < 1 else 1 << (limit.bit_length() - 1)


def fuse_sibling_sweeps(facts: KernelFacts) -> bool:
    """Whether whole affine sweeps fuse before allocation reuse.

    Fusion removes duplicate phase and corner-turn scans in ordinary signal
    graphs. With several data-dependent regrids it instead widens live ranges
    around gather stages that already dominate the schedule, so the compact
    lifetime graph is worth more than the shared scan. The later HLS fusion
    pass still takes the opportunities that stay safe after reuse.
    """
    return facts.gather_ops <= 1


def loop_unroll_budget(facts: KernelFacts) -> tuple[int, int]:
    """Bound source growth in gather-heavy pipelined loops.

    A gather's tap loop is intentionally left as a compact lane loop once its
    body would otherwise materialize hundreds of operations per outer sample.
    Element-wise and FFT-only graphs retain the wider default unroll budget.
    """
    if facts.gather_ops:
        return 512, 8
    return 4096, 32


def storage_min_elements(facts: KernelFacts, budget: int) -> int:
    """Element count at which a buffer stops being a dataflow channel and
    becomes storage (shared or recomputed), because a full-scene plane
    kept as a private channel would cost its own bytes on chip; a tighter
    budget pulls the line down to one working buffer's share."""
    if budget <= 0:
        return 1
    share = max(1, budget // _WORKING_SHARE)
    return max(1, min(facts.plane_elements, share // facts.element_bytes))


def reuse_min_elements(facts: KernelFacts) -> int:
    """Element count at which giving two lifetimes one allocation pays.

    Inside a dataflow region every node runs concurrently, so two lifetimes a
    sequential scan calls disjoint are not: sharing their allocation gives the
    buffer a second producer, which the region cannot express. Legalization
    then undoes the sharing -- the later producer gets a private buffer and a
    copy process carrying the earlier contents forward -- leaving one more
    plane and one more process than never sharing would have. An external
    buffer is exempt, because DRAM is memory rather than a channel and several
    nodes may write it in turn. So sharing pays exactly for the buffers that
    leave the die, which are the full-size planes.
    """
    return streaming_threshold(facts)


def should_share_buffers(config, facts: KernelFacts, metadata,
                         lowered_facts: KernelFacts) -> bool:
    """Whether sharing allocations pays, measured against the unshared form.

    Sharing survives only where a buffer may have several writers, which in
    this hierarchy means DRAM: a shared on-chip buffer gets a second producer,
    which a dataflow region cannot express, so legalization privatizes it
    again and adds a copy process doing it -- one more plane and one more
    process than never sharing. So the question is whether this design streams
    planes off chip at all: with no off-chip tier the answer is no whatever
    the budget says, and with one it follows the placement rule.
    """
    if config.interface == "ap_memory":
        return False
    return external_buffer_threshold(facts, config.on_chip_bytes(), metadata,
                                     lowered_facts) != KEEP_ON_CHIP


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
    block = bram_bytes // (4 * slots)
    return block if block >= 4096 else 0


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


def _transform_engine(facts: KernelFacts, config, budget: int, clock_ns: float,
                      primitives: Tuple[int, int]) -> Tuple[int, int]:
    """(transfer width, compute lanes) the transform storage can afford.

    The two are not independent: lanes replicate the line buffers a transfer
    fills, so the widest transfer is only affordable at the lane count it
    leaves room for. The transfer is sized first and the lanes take what is
    left, because the two are worth very different things.

    For the production range-Doppler chain, trading its eight-element transfer
    for twice the lanes -- the pair fits the same storage ceiling either way --
    costs 4.4x the latency (3.49 to 15.3
    billion cycles) and raises block RAM, UltraRAM, DSP and LUT with it. A
    plane crosses the external bus once per pass whatever the lane count, so
    a narrower transfer lengthens every one of those crossings, while a lane
    only shortens the on-chip work between them. Bandwidth is bought first;
    parallelism spends what remains.
    """
    io = fft_io_unroll(facts, int(config.axi_bus_bits))
    while True:
        lanes = fft_parallel_rows(facts, int(config.dsp), budget, io, clock_ns,
                                  primitives)
        if io == 1 or budget <= 0:
            return io, lanes
        scratch = _fft_scratch_bytes(facts.transforms, 64, max(1, lanes), io,
                                     facts.transform_strided, primitives)
        if scratch <= transform_lane_storage_ceiling(budget):
            return io, lanes
        io >>= 1


def plan(config,
         facts: KernelFacts,
         metadata=None,
         lowered_facts: Optional[KernelFacts] = None) -> PerformancePlan:
    """Every strategy value the constraints and the kernel imply.

    Called twice: once on the traced module, where everything but the
    placement threshold is decidable, and again once the kernel is lowered
    and its buffers can be counted. Keys the user pinned are still
    returned; `HLSConfig.adopt` keeps the pinned value.
    """
    assert (metadata is None) == (lowered_facts is None), \
        "the placement decision needs the metadata and the lowered IR together"
    budget = config.on_chip_bytes()
    transform_budget = int(config.bram_bytes) + int(config.uram_bytes)
    primitives = (int(config.bram_block_bytes), int(config.uram_block_bytes))
    timing_budget = config.effective_clock_ns()
    model_io, model_lanes = _transform_engine(facts, config, transform_budget,
                                              timing_budget, primitives)

    # Correlated decisions must use values the caller pinned.  ``adopt``
    # correctly preserves a pin, but deriving the neighboring settings from
    # the model's discarded candidate still produces an incoherent design --
    # for example, eight FFT lanes with a stage grouping sized for four.
    io = (int(config.fft_io_unroll)
          if config.fft_io_unroll is not None else model_io)
    if config.fft_parallel_rows is not None:
        lanes = int(config.fft_parallel_rows)
    elif io != model_io:
        lanes = fft_parallel_rows(facts, int(config.dsp), transform_budget, io,
                                  timing_budget, primitives)
    else:
        lanes = model_lanes

    model_vector_lanes = external_vector_max_lanes(facts,
                                                   int(config.axi_bus_bits),
                                                   lowered_facts)
    vector_lanes = (int(config.external_vector_max_lanes)
                    if config.external_vector_max_lanes is not None else
                    model_vector_lanes)
    model_compute_lanes = external_vector_compute_lanes(facts, vector_lanes)
    compute_lanes = (int(config.external_vector_compute_lanes)
                     if config.external_vector_compute_lanes is not None else
                     model_compute_lanes)
    cache_copies = (int(config.interp_cache_copies)
                    if config.interp_cache_copies is not None else
                    interp_cache_copies(facts, compute_lanes))
    values = {
        "fft_stage_group":
        fft_stage_group(facts, budget, max(1, lanes), io, primitives),
        "loop_tile_size":
        loop_tile_size(facts, int(config.axi_bus_bits)),
        "fft_parallel_rows":
        lanes,
        "fft_io_unroll":
        io,
        "external_vector_max_lanes":
        vector_lanes,
        "external_vector_min_elements":
        external_vector_min_elements(facts, vector_lanes),
        "external_vector_pack_outputs":
        external_vector_pack_outputs(facts),
        "external_vector_compute_lanes":
        compute_lanes,
        "interp_banded_gather":
        interp_banded_gather(),
        "interp_full_row_max_bytes":
        interp_full_row_max_bytes(facts, transform_budget),
        "interp_cache_copies":
        cache_copies,
        "interp_complete_bank_max_elements":
        interp_complete_bank_max_elements(facts,
                                          int(config.lut), int(config.ff),
                                          max(1,
                                              compute_lanes), timing_budget),
        "fuse_sibling_sweeps":
        fuse_sibling_sweeps(facts),
        "max_unrolled_ops":
        loop_unroll_budget(facts)[0],
        "max_unroll_factor":
        loop_unroll_budget(facts)[1],
        "reuse_buffer_min_elements":
        NEVER_SHARE,
        "recompute_min_elements":
        storage_min_elements(facts, budget),
        "lutram_max_bytes":
        lutram_max_bytes(int(config.axi_bus_bits)),
        "max_scratch_arenas":
        max_scratch_arenas(facts),
        "array_partition_max_factor":
        array_partition_max_factor(facts, int(config.axi_bus_bits)),
    }
    if lowered_facts is not None:
        values["external_buffer_threshold"] = external_buffer_threshold(
            facts, budget, metadata, lowered_facts)
    measured = lowered_facts or facts
    return PerformancePlan(values=values,
                           clock_ns=float(config.clock_ns),
                           timing_budget_ns=timing_budget,
                           on_chip_bytes=budget,
                           memory_accesses=measured.loads + measured.stores,
                           operation_count=measured.operations)
