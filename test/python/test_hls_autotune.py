"""Tests for `sar.backends.hls.autotune`.

Three kinds: unit tests for each derivation (no toolchain needed),
integration tests that confirm the derived values reach the passes when
the strategy keys are left at null, and a generality check that a tiny
kernel and a 16384-scale kernel both get sensible choices.
"""

import sar
import pytest
from sar.backends.hls.autotune import (
    AUTO_OPTIONS, KernelFacts, array_partition_max_factor,
    external_vector_max_lanes, external_vector_min_elements, fft_parallel_rows,
    external_vector_compute_lanes, external_vector_pack_outputs,
    fft_stage_group, fuse_sibling_sweeps, interp_banded_gather,
    interp_cache_copies, interp_complete_bank_max_elements,
    interp_full_row_max_bytes, loop_unroll_budget, kernel_facts_from_json,
    loop_tile_size, FIXED_OPTIONS, TUNED_OPTIONS, lutram_max_bytes,
    measure_kernel, NEVER_SHARE, plan, _fft_groups, reuse_min_elements,
    storage_min_elements, streaming_threshold, transform_lane_storage_ceiling,
    transform_storage_ceiling, _transform_engine, transpose_block_bytes,
    _fft_scratch_bytes, _scratch_slots, _transform_stages, _transfer_banks)
from sar.backends.hls.config import HLSConfig
from sar.backends.hls.devices import storage_primitives

from conftest import requires_hls

# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #


def _facts(plane_elements=65536, element_bytes=4, transforms=()):
    return KernelFacts(plane_elements=plane_elements,
                       element_bytes=element_bytes,
                       transforms=transforms)


def _cfg(**opts):
    return HLSConfig.resolve(opts)


def test_structured_kernel_facts_parser():
    facts = kernel_facts_from_json(
        '{"plane_elements":64,"element_bytes":4,"transposes":2,'
        '"element_bytes_set":[4,8],'
        '"transforms":[[8,4]],"transform_strided":[true],'
        '"banded_gathers":2,"full_row_gathers":1,"direct_gathers":2,'
        '"buffers":[[64,256]]}')
    assert facts == KernelFacts(64,
                                4, ((8, 4), ),
                                2, ((64, 256), ), (4, 8),
                                0,
                                2,
                                1,
                                2,
                                transform_strided=(True, ))


def test_structured_kernel_facts_reject_mismatched_transform_layouts():
    with pytest.raises(ValueError, match="transform_strided"):
        kernel_facts_from_json(
            '{"plane_elements":64,"element_bytes":4,'
            '"transforms":[[8,4]],"transform_strided":[true,false],'
            '"buffers":[]}')


def test_kernel_facts_cache_hits_and_recovers_from_corruption(
        tmp_path, monkeypatch):
    from sar.compiler.cache import KernelCache
    from sar.backends.hls import autotune

    monkeypatch.setenv("SAR_DSL_CACHE_DIR", str(tmp_path))
    cache = KernelCache("facts", "hls", {})
    calls = []

    def measure(text):
        calls.append(text)
        return _facts(plane_elements=64, transforms=((8, 4), ))

    monkeypatch.setattr(autotune, "measure_kernel", measure)
    first = autotune.cached_kernel_facts(cache, "facts.json", "module")
    second = autotune.cached_kernel_facts(cache, "facts.json", "module")
    assert first == second
    assert calls == ["module"]

    cache.path("facts.json").write_text("not json")
    third = autotune.cached_kernel_facts(cache, "facts.json", "module")
    assert third == first
    assert calls == ["module", "module"]


def test_performance_plan_cache_hits_and_recovers_from_corruption(
        tmp_path, monkeypatch):
    from sar.compiler.cache import KernelCache
    from sar.backends.hls import autotune

    monkeypatch.setenv("SAR_DSL_CACHE_DIR", str(tmp_path))
    cache = KernelCache("plan", "hls", {})
    facts = _facts()
    config = _cfg()
    real_plan = autotune.plan
    calls = []

    def measured(*args, **kwargs):
        calls.append(1)
        return real_plan(*args, **kwargs)

    monkeypatch.setattr(autotune, "plan", measured)
    first = autotune.cached_performance_plan(cache, "plan.json", config, facts)
    second = autotune.cached_performance_plan(cache, "plan.json", config,
                                              facts)
    assert first == second
    assert calls == [1]

    cache.path("plan.json").write_text("not json")
    third = autotune.cached_performance_plan(cache, "plan.json", config, facts)
    assert third == first
    assert calls == [1, 1]


def test_tighter_clock_reduces_fft_routing_width():
    facts = _facts(plane_elements=1 << 20, transforms=((1024, 4), ))
    plenty = 1 << 40
    relaxed = fft_parallel_rows(facts, 9830, plenty, clock_ns=4.0)
    tight = fft_parallel_rows(facts, 9830, plenty, clock_ns=2.5)
    assert relaxed >= tight
    assert tight <= 4


# ---------------------------------------------------------------------- #
# fft_stage_group
# ---------------------------------------------------------------------- #


class TestFftStageGroup:

    def test_no_transforms_gives_0(self):
        assert fft_stage_group(_facts(), budget=1) == 0

    def test_zero_budget_uses_the_smallest_scratch(self):
        f = _facts(transforms=((512, 8), ))
        assert fft_stage_group(f, budget=0) == 2

    @pytest.mark.parametrize("budget,expected", [
        (1 << 20, 2),
        (1 << 19, 2),
        (1 << 18, 2),
        (1 << 17, 2),
        (1 << 10, 2),
    ])
    def test_budget_picks_the_grouping(self, budget, expected):
        """Partitioned lines are charged in whole memory primitives, so this
        budget range selects the two-slot floor rather than a logically small
        but physically fragmented full unroll."""
        f = _facts(transforms=((512, 8), ))
        assert fft_stage_group(f, budget) == expected

    def test_grouping_is_monotone_in_the_budget(self):
        f = _facts(transforms=((1024, 8), (1024, 8)))
        scratch = [(b, fft_stage_group(f, b))
                   for b in (1 << 22, 1 << 20, 1 << 18, 1 << 16)]
        # Full unroll is 0 and sorts before every grouping, so compare the
        # slot counts the choices actually buy rather than the labels.
        slots = [_scratch_slots(10, g) for _, g in scratch]
        assert slots == sorted(slots, reverse=True)

    def test_scratch_model_is_self_consistent(self):
        """The slot count the policy charges for, at the documented points.

        These are the Python side alone; `test_scratch_model_matches_cpp`
        is what ties them to the design the compiler actually emits.
        """
        # Full unroll: one slot per intermediate stage.
        assert _scratch_slots(9, 0) == 8
        # k=1 is the full unroll's slot count under another name.
        assert _scratch_slots(9, 1) == 8
        # k=2: ceil(9/2) = 5 groups, 4 slots between them.
        assert _scratch_slots(9, 2) == 4
        # k=4: ceil(9/4) = 3 groups, 2 slots -- already the floor.
        assert _scratch_slots(9, 4) == 2
        # Past the stage count the pool saturates at the floor of two.
        assert _scratch_slots(9, 64) == 2
        # A single-stage transform writes straight to the destination.
        assert _scratch_slots(1, 0) == 0
        assert _scratch_slots(1, 8) == 0

    def test_reduced_transfer_banking_frees_full_stage_unroll(self):
        """The smaller transfer bank count frees the full stage chain.

        Full unroll still exceeds the working-set share, so it is not chosen
        until the transform ceiling is considered. It fits that second ceiling
        because the transfer blocks use ``ceil(io / 2)`` banks.
        """
        f = _facts(plane_elements=1 << 28, transforms=((16384, 4), ) * 4)
        budget = 9_907_200 + 37_748_736 + 2_883_584
        assert _fft_scratch_bytes(f.transforms, 0, 4, 8) > budget // 8
        assert _fft_scratch_bytes(f.transforms, 0, 4, 8) <= budget // 2
        assert fft_stage_group(f, budget, lanes=4, io=8) == 0

    def test_transform_storage_ceiling_stays_at_half_the_block_tiers(self):
        """Grouping cannot spend the tier reserved for planes and lanes.

        Stage grouping only changes local line scratch; it does not reduce
        external plane traffic. Half the block-memory tiers therefore bounds
        this area-only trade while lane parallelism has separate headroom.
        """
        tiers = 9_907_200 + 37_748_736
        assert transform_storage_ceiling(tiers) == tiers // 2
        # Full unroll is selected when it fits the transform ceiling.
        f = _facts(plane_elements=1 << 28, transforms=((16384, 4), ) * 4)
        assert fft_stage_group(f, tiers + 2_883_584, lanes=4, io=8) == 0

    def test_lane_storage_uses_bank_rebalancing_headroom(self):
        """Lanes may spend more than stage unrolling without relaxing caps.

        Banking is rebalanced between BRAM and URAM before the hard final
        check.  Five eighths admits the production-size eight-lane engine,
        while the half-tier stage ceiling still prevents area-only unrolling.
        """
        tiers = 9_907_200 + 37_748_736
        assert transform_lane_storage_ceiling(tiers) == tiers * 5 // 8
        assert transform_lane_storage_ceiling(
            tiers) > transform_storage_ceiling(tiers)

    def test_no_grouping_fits_the_share_falls_back_to_the_half_budget(self):
        """A production f64 transform fits no grouping under the working
        share. The fallback must still take the least grouping the transform
        storage ceiling allows, not the smallest scratch: over-grouping
        deepens the stage chain, which costs timing and buys no latency."""
        f = _facts(plane_elements=1 << 28, transforms=((16384, 8), ) * 4)
        budget = 9_907_200 + 37_748_736 + 2_883_584
        chosen = fft_stage_group(f, budget, lanes=4, io=4)
        scratch = _fft_scratch_bytes(f.transforms, chosen, 4, 4)
        assert scratch > budget // 8, "the share would have selected it"
        assert scratch <= budget // 2
        # Smaller groupings exist and are cheaper in scratch; the point is
        # that the least one the ceiling admits is what gets picked.
        assert all(
            _fft_scratch_bytes(f.transforms, g, 4, 4) > budget // 2
            for g in _fft_groups(f.transforms)
            [:_fft_groups(f.transforms).index(chosen)])

    def test_slow_axis_transfer_width_does_not_replicate_all_scratch(self):
        transforms = ((16384, 4), )
        coupled = _fft_scratch_bytes(transforms, 3, lanes=8, io=8)
        staged = _fft_scratch_bytes(transforms,
                                    3,
                                    lanes=4,
                                    io=8,
                                    transform_strided=(True, ))
        assert staged < coupled


@pytest.mark.parametrize("io,expected", [
    (0, 1),
    (1, 1),
    (2, 1),
    (3, 2),
    (8, 4),
])
def test_transfer_banks_match_dual_port_demand(io, expected):
    assert _transfer_banks(io) == expected


# ---------------------------------------------------------------------- #
# loop_tile_size
# ---------------------------------------------------------------------- #


class TestLoopTileSize:

    def test_512bit_bus_4byte_element(self):
        # 512/8 = 64 bytes / 4 = 16; clamped to _MAX_TILE=64
        assert loop_tile_size(_facts(element_bytes=4), 512) == 16

    def test_512bit_bus_8byte_element(self):
        # 64 / 8 = 8
        assert loop_tile_size(_facts(element_bytes=8), 512) == 8

    def test_256bit_bus_4byte_element(self):
        # 32 / 4 = 8
        assert loop_tile_size(_facts(element_bytes=4), 256) == 8

    def test_result_is_whole_number_of_beats(self):
        # tile * element_bytes must be a multiple of (bus/8).
        for bus in (128, 256, 512, 1024):
            for elem in (4, 8):
                t = loop_tile_size(_facts(element_bytes=elem), bus)
                beat = bus // 8
                assert (t * elem) % beat == 0 or t * elem < beat

    def test_minimum_is_two(self):
        # Even a 32-bit bus with 4-byte elements gives 1 element/beat,
        # which would be 1; the minimum clips it to 2.
        assert loop_tile_size(_facts(element_bytes=4), 32) == 2

    def test_maximum_is_64(self):
        # A very wide bus should not produce an oversized tile.
        assert loop_tile_size(_facts(element_bytes=1), 1024) <= 64


# ---------------------------------------------------------------------- #
# fft_parallel_rows
# ---------------------------------------------------------------------- #


def test_parallelism_needs_a_transform():
    # Lanes belong to the transform engine; a kernel without one has no
    # engine to widen.
    assert fft_parallel_rows(_facts(), 9830, 1 << 30) == 0


def test_multiple_data_dependent_regrids_keep_one_fft_engine():
    facts = KernelFacts(1 << 28, 4, ((16384, 4), (16384, 4)), gather_ops=2)
    assert fft_parallel_rows(facts, 9830, 1 << 30) == 0


def test_parallelism_respects_the_dsp_budget():
    facts = _facts(transforms=((1024, 4), ))
    plenty = 1 << 30
    # A 1024-point transform runs five stages; each lane costs about six
    # slices per stage. One lane's worth means serial lines.
    assert fft_parallel_rows(facts, 30, plenty) == 0
    assert fft_parallel_rows(facts, 60, plenty) == 2
    # Sixty-four independent lines amortize eight engines; wider hardware
    # would mostly increase synthesis work at this validation scale.
    assert fft_parallel_rows(facts, 9830, plenty) == 8
    four_sites = _facts(transforms=((1024, 4), ) * 4)
    assert fft_parallel_rows(four_sites, 9830, plenty) == 8


def test_parallelism_respects_the_memory_budget():
    facts = _facts(transforms=((1024, 4), ))
    # A tight on-chip budget pulls the lane count down before the DSP
    # budget would.
    assert fft_parallel_rows(facts, 9830, 1 << 20) < 16


def test_transfer_width_is_preferred_over_lane_count():
    """Bandwidth is bought before parallelism, and the order is measured.

    At the production range-Doppler geometry the storage ceiling admits
    either a full eight-element transfer with four lanes or a halved
    transfer with eight. Synthesizing both puts the second at 4.4x the
    latency (3.49 -> 15.3 billion cycles) and more of every resource: a
    plane crosses the external bus once per pass whatever the lane count,
    so narrowing the transfer lengthens every crossing while a lane only
    shortens the on-chip work between them.
    """
    facts = _facts(plane_elements=1 << 28, transforms=((16384, 4), ) * 4)
    config = _cfg(interface="axi")
    tiers = int(config.bram_bytes) + int(config.uram_bytes)
    io, lanes = _transform_engine(facts, config, tiers, 4.0,
                                  storage_primitives(config.part))
    assert io == 8, "the full per-plane beat must survive"
    assert lanes == 8
    # A further doubling still yields before the full-beat transfer does.
    wider_lanes = _fft_scratch_bytes(facts.transforms, 64, lanes * 2, io, (),
                                     storage_primitives(config.part))
    assert wider_lanes > transform_lane_storage_ceiling(tiers)


def test_io_unroll_is_a_bus_beat_under_the_synthesis_cap():
    """A transfer moves a whole beat, up to the eight lanes synthesis
    sustains. The wider element reaches the cap with the narrower beat, so
    both widths land on the same transfer width -- what a beat costs in
    line buffers is charged by the storage decisions, not capped twice."""
    from sar.backends.hls.autotune import fft_io_unroll
    assert fft_io_unroll(_facts(transforms=((1024, 4), )), 512) == 8
    assert fft_io_unroll(_facts(transforms=((1024, 8), )), 512) == 8
    # A narrow bus cannot carry eight f64 lanes, so the beat binds instead.
    assert fft_io_unroll(_facts(transforms=((1024, 8), )), 256) == 4
    assert fft_io_unroll(_facts(), 512) == 1


def test_partition_factor_is_bounded_by_one_bus_beat():
    assert array_partition_max_factor(_facts(element_bytes=4), 512) == 16
    assert array_partition_max_factor(_facts(element_bytes=8), 512) == 8
    assert array_partition_max_factor(_facts(element_bytes=4), 1024) == 32


def test_mixed_width_strategy_uses_a_common_physical_lane_count():
    facts = KernelFacts(65536, 4, (), element_bytes_set=(4, 8))
    assert external_vector_max_lanes(facts, 512) == 8
    assert external_vector_min_elements(facts, 8) == 4096
    assert array_partition_max_factor(facts, 512) == 8


def test_multiple_gathers_use_bounded_scratch_packing():
    facts = KernelFacts(65536, 4, (), gather_ops=2)
    assert external_vector_max_lanes(facts, 512) == 4
    assert external_vector_pack_outputs(facts) is False
    assert external_vector_compute_lanes(facts, 8) == 4
    assert fuse_sibling_sweeps(facts) is False


def test_single_regrid_keeps_early_sweep_fusion():
    facts = KernelFacts(65536, 4, (), gather_ops=1)
    assert fuse_sibling_sweeps(facts) is True


def test_gather_unroll_budget_is_bounded_but_regular_graph_keeps_default():
    assert loop_unroll_budget(KernelFacts(1, 4, (), gather_ops=1)) == (512, 8)
    assert loop_unroll_budget(KernelFacts(1, 4, (),
                                          gather_ops=0)) == (4096, 32)


def test_vector_width_respects_lowered_binding_complexity():
    facts = KernelFacts(1 << 20, 4, (), gather_ops=1)
    light = KernelFacts(1 << 20, 4, (), expensive_ops=8, max_fanout=16)
    heavy = KernelFacts(1 << 20, 4, (), expensive_ops=40, max_fanout=84)
    assert external_vector_max_lanes(facts, 512, light) == 16
    assert external_vector_max_lanes(facts, 512, heavy) == 8


def test_banded_gather_retains_physical_packing_width():
    facts = KernelFacts(1 << 28, 4, (), gather_ops=1)
    lowered = KernelFacts(1 << 28,
                          4, (),
                          banded_gathers=2,
                          expensive_ops=40,
                          max_fanout=84)
    assert external_vector_max_lanes(facts, 512, lowered) == 8


# ---------------------------------------------------------------------- #
# lutram_max_bytes
# ---------------------------------------------------------------------- #


class TestLutramMaxBytes:
    """The tier boundary is one bus beat: a bank that cannot fill a single
    transfer does not earn a dedicated block RAM primitive."""

    @pytest.mark.parametrize("bus_bits,expected", [
        (512, 64),
        (256, 32),
        (128, 16),
        (1024, 128),
    ])
    def test_is_one_bus_beat(self, bus_bits, expected):
        assert lutram_max_bytes(bus_bits) == expected

    def test_scales_with_the_bus(self):
        """A wider bus makes a wider bank worth keeping out of block RAM,
        so the threshold has to move with it rather than stay pinned."""
        widths = [128, 256, 512, 1024]
        values = [lutram_max_bytes(w) for w in widths]
        assert values == sorted(values)
        assert len(set(values)) == len(values)

    def test_never_zero(self):
        # 0 would mean "nothing is ever distributed RAM" by accident; the
        # floor keeps the tier reachable at any legal bus width.
        assert lutram_max_bytes(1) >= 1
        assert lutram_max_bytes(0) >= 1

    def test_derive_fills_it_from_the_bus(self):
        cfg = _cfg(axi_bus_bits=256)
        d = plan(cfg, _facts()).values
        assert d["lutram_max_bytes"] == 32

    def test_is_an_auto_option(self):
        # It has to be derived like every other strategy value, not frozen
        # as a pass default.
        assert "lutram_max_bytes" in AUTO_OPTIONS

    def test_pinning_it_wins_over_the_derivation(self):
        cfg = _cfg(lutram_max_bytes=256)
        cfg.adopt(plan(cfg, _facts()).values)
        assert cfg["lutram_max_bytes"] == 256
        assert cfg.provenance["lutram_max_bytes"] == HLSConfig.FROM_OPTIONS


def test_dsp_budget_is_a_device_constraint():
    assert _cfg().dsp == 9830
    assert "dsp" not in AUTO_OPTIONS
    # Retargeting to a smaller part is stating a different fact.
    assert _cfg(dsp=1824).dsp == 1824


# ---------------------------------------------------------------------- #
# interp_banded_gather
# ---------------------------------------------------------------------- #


def test_interp_banded_gather_always_on():
    # The pass chooses a proven band, a budgeted row cache, or the direct
    # gather per op; enabling the capability does not force one shape.
    assert interp_banded_gather() is True


def test_full_row_staging_divides_the_working_share_across_gathers():
    budget = 48 << 20
    one = KernelFacts(1 << 28, 4, (), gather_ops=1)
    two = KernelFacts(1 << 28, 4, (), gather_ops=2)
    assert interp_full_row_max_bytes(one, budget) == budget // 8
    assert interp_full_row_max_bytes(two, budget) == budget // 16
    assert interp_full_row_max_bytes(_facts(), budget) == 0


def test_complete_band_banking_tracks_logic_budget_and_lanes():
    facts = KernelFacts(1 << 28, 4, (), gather_ops=1)
    assert interp_complete_bank_max_elements(facts, 1_382_400, 2_764_800, 4,
                                             4.0) == 16
    assert interp_complete_bank_max_elements(facts, 1_382_400, 2_764_800, 4,
                                             6.0) == 128
    assert interp_complete_bank_max_elements(facts, 200_000, 400_000, 4) < 128
    assert interp_complete_bank_max_elements(_facts(), 1_000_000, 1_000_000,
                                             4) == 0


def test_full_row_cache_replication_tracks_compute_lanes():
    facts = KernelFacts(1 << 28, 4, (), gather_ops=1)
    assert interp_cache_copies(facts, 1) == 1
    assert interp_cache_copies(facts, 2) == 2
    assert interp_cache_copies(facts, 8) == 4
    assert interp_cache_copies(_facts(), 8) == 1


# ---------------------------------------------------------------------- #
# storage_min_elements
# ---------------------------------------------------------------------- #


class TestStorageMinElements:

    def test_zero_budget_shares_every_buffer(self):
        f = _facts(plane_elements=65536, element_bytes=4)
        assert storage_min_elements(f, 0) == 1

    @pytest.mark.parametrize("budget,expected", [
        (4 << 20, 65536),
        (1 << 20, 32768),
        (1 << 18, 8192),
    ])
    def test_budget_pulls_the_line_below_a_plane(self, budget, expected):
        """An eighth of the budget is what one private buffer may claim;
        the plane size is the ceiling, because a plane is always storage."""
        f = _facts(plane_elements=65536, element_bytes=4)
        assert storage_min_elements(f, budget) == expected

    def test_never_exceeds_the_plane_size(self):
        f = _facts(plane_elements=1024, element_bytes=4)
        for budget in (0, 1 << 16, 1 << 30):
            assert storage_min_elements(f, budget) <= 1024

    def test_stays_positive_at_any_budget(self):
        # 0 is a legal pass value meaning "share everything", but it is not
        # a decision the policy should reach by rounding.
        f = _facts(plane_elements=65536, element_bytes=8)
        assert storage_min_elements(f, 1) >= 1


# ---------------------------------------------------------------------- #
# reuse_min_elements
# ---------------------------------------------------------------------- #


class TestReuseMinElements:

    def test_only_full_planes_share(self):
        """Sharing survives dataflow legalization only for buffers that leave
        the die, which are the ones at full-plane scale."""
        f = _facts(plane_elements=65536, element_bytes=4)
        threshold = reuse_min_elements(f)
        assert threshold == streaming_threshold(f)
        assert 0 < threshold <= f.plane_elements

    def test_the_plan_shares_nothing_until_placement_is_known(self):
        """Which buffers go off chip needs the lowered allocations, so the
        plan starts from the safe answer and the lowering revisits it."""
        cfg = _cfg()
        assert plan(
            cfg, _facts()).values["reuse_buffer_min_elements"] == NEVER_SHARE


# ---------------------------------------------------------------------- #
# derive: picks null keys, honours pinned keys
# ---------------------------------------------------------------------- #


class TestDerive:

    def test_all_auto_options_are_filled(self):
        f = _facts(plane_elements=65536,
                   element_bytes=4,
                   transforms=((256, 8), ))
        cfg = _cfg()
        d = plan(cfg, f).values
        for key in AUTO_OPTIONS:
            if key == "external_buffer_threshold":
                continue  # needs lowered IR
            assert key in d
            assert d[key] is not None

    def test_pinned_value_is_not_overwritten_by_adopt(self):
        # When the user pins fft_stage_group=3, adopt must not touch it.
        cfg = _cfg(fft_stage_group=3)
        assert cfg["fft_stage_group"] == 3
        f = _facts(transforms=((512, 8), ))
        cfg.adopt(plan(cfg, f).values)
        assert cfg["fft_stage_group"] == 3

    def test_pinned_fft_lanes_size_the_derived_stage_group(self):
        facts = _facts(plane_elements=1 << 28, transforms=((16384, 4), ) * 4)
        cfg = _cfg(interface="axi", fft_parallel_rows=8)
        values = plan(cfg, facts).values
        assert values["fft_stage_group"] == 3
        assert values["fft_io_unroll"] == 8

    def test_pinned_transfer_width_resizes_derived_lanes(self):
        facts = _facts(plane_elements=1 << 28, transforms=((16384, 4), ) * 4)
        cfg = _cfg(interface="axi", fft_io_unroll=4)
        values = plan(cfg, facts).values
        expected = fft_parallel_rows(facts, cfg.dsp,
                                     cfg.bram_bytes + cfg.uram_bytes,
                                     4, cfg.clock_ns,
                                     storage_primitives(cfg.part))
        assert values["fft_parallel_rows"] == expected

    def test_pinned_vector_width_sizes_dependent_thresholds(self):
        cfg = _cfg(external_vector_max_lanes=4)
        values = plan(cfg, _facts()).values
        assert values["external_vector_min_elements"] == \
            external_vector_min_elements(_facts(), 4)
        assert values["external_vector_compute_lanes"] == \
            external_vector_compute_lanes(_facts(), 4)

    def test_pinned_compute_lanes_size_complete_band_banking(self):
        facts = KernelFacts(1 << 28, 4, (), gather_ops=1)
        four = plan(_cfg(external_vector_compute_lanes=4), facts).values
        eight = plan(_cfg(external_vector_compute_lanes=8), facts).values
        assert four["interp_complete_bank_max_elements"] == 16
        assert eight["interp_complete_bank_max_elements"] == 16
        assert four["interp_cache_copies"] == 4
        assert eight["interp_cache_copies"] == 4

    def test_provenance_marks_derived_keys(self):
        cfg = _cfg()
        f = _facts(transforms=((256, 8), ))
        cfg.adopt(plan(cfg, f).values)
        assert cfg.provenance["fft_stage_group"] == HLSConfig.DERIVED

    def test_provenance_marks_user_keys(self):
        cfg = _cfg(fft_stage_group=2)
        f = _facts(transforms=((256, 8), ))
        cfg.adopt(plan(cfg, f).values)
        assert cfg.provenance["fft_stage_group"] == HLSConfig.FROM_OPTIONS

    def test_smaller_budget_picks_higher_group(self):
        # A budget that cannot afford the full FFT scratch should push the
        # grouping above 0 (full unroll).  Budget=1 cannot afford any scratch.
        f = _facts(transforms=((512, 8), ))
        tight = _cfg(bram_bytes=1, uram_bytes=0, lutram_bytes=0)
        loose = _cfg()
        g_tight = fft_stage_group(f, tight.on_chip_bytes())
        g_loose = fft_stage_group(f, loose.on_chip_bytes())
        assert g_tight > g_loose or g_tight >= 2


# ---------------------------------------------------------------------- #
# measure_kernel: reads facts from a traced SAR module
# ---------------------------------------------------------------------- #


@requires_hls
class TestMeasureKernel:

    def test_reads_plane_size_from_signature(self):
        n = 128

        @sar.func
        def scale(x: sar.f32[n, n]) -> sar.f32[n, n]:
            return x * 2.0

        f = measure_kernel(scale.to_mlir())
        assert f.plane_elements >= n * n

    def test_finds_fft_transforms(self):
        n = 64

        @sar.func
        def with_fft(x: sar.c64[n, n]) -> sar.c64[n, n]:
            return sar.fft(x, axis=1)

        f = measure_kernel(with_fft.to_mlir())
        assert len(f.transforms) > 0

    def test_element_bytes_is_narrowest_plane(self):
        n = 16

        @sar.func
        def mixed(x: sar.f32[n, n], y: sar.f64[n, n]) -> sar.f32[n, n]:
            return x * sar.cast(y, sar.f32)

        f = measure_kernel(mixed.to_mlir())
        assert f.element_bytes == 4
        assert f.element_bytes_set == (4, 8)


@requires_hls
@pytest.mark.parametrize("length,width,mlir_type", [
    (1024, 8, "f64"),
    (16384, 4, "f32"),
])
def test_scratch_model_matches_cpp(length, width, mlir_type):
    """The Python scratch model must equal what the C++ lowering allocates.

    `_scratch_slots` mirrors `scratchSlots` in `SARFFTToAffine.cpp`, and the
    policy spends the on-chip budget against it. Comparing the two by hand
    lets them drift with both suites green, so this counts the scratch lines
    the compiler actually emits at each grouping and holds the model to it.

    A transform allocates one (re, im) pair per scratch slot plus the
    prefetch and write-back block pairs -- four lines that are not slots.
    """
    import re
    import subprocess

    from sar.compiler.toolchain import find_tool

    module = (f"func.func @t(%re: tensor<2x{length}x{mlir_type}>, "
              f"%im: tensor<2x{length}x{mlir_type}>) -> "
              f"(tensor<2x{length}x{mlir_type}>, "
              f"tensor<2x{length}x{mlir_type}>) {{\n"
              f"  %r, %i = sar.fft_split %re, %im {{dim = 1 : i64}} : "
              f"tensor<2x{length}x{mlir_type}>\n"
              f"  return %r, %i : tensor<2x{length}x{mlir_type}>, "
              f"tensor<2x{length}x{mlir_type}>\n}}\n")
    line = re.compile(rf"memref\.alloc\(\) : memref<{length}x{mlir_type}>")
    stages = _transform_stages(length)[0]

    for group in (0, 2, 3, 4, stages, 64):
        lowered = subprocess.run([
            find_tool("sar-opt"),
            f"--convert-sar-fft-to-affine=fft-stage-group={group}", "-"
        ],
                                 input=module,
                                 capture_output=True,
                                 text=True,
                                 check=True).stdout
        emitted = (len(line.findall(lowered)) - 4) // 2
        assert emitted == _scratch_slots(stages, group), (
            f"grouping {group}: the compiler allocates {emitted} scratch "
            f"slots, the cost model charges "
            f"{_scratch_slots(stages, group)}")


# ---------------------------------------------------------------------- #
# Generality: tiny and very large kernels both get sensible choices
# ---------------------------------------------------------------------- #


@requires_hls
class TestGenerality:

    def test_tiny_kernel(self):
        # A 4x4 kernel has no FFTs and a trivial plane; derived values must
        # be valid, non-zero, and not reference any per-algorithm constant.
        n = 4

        @sar.func
        def tiny(x: sar.f32[n, n]) -> sar.f32[n, n]:
            return x * 2.0

        f = measure_kernel(tiny.to_mlir())
        cfg = _cfg()
        d = plan(cfg, f).values
        assert d["loop_tile_size"] >= 2
        assert d["fft_stage_group"] == 0  # no transforms: full unroll is fine
        assert d["interp_banded_gather"] is True

    def test_large_kernel(self):
        # A 16384-scale kernel: derived values must remain sensible.
        n = 16384

        @sar.func
        def large(x: sar.f32[n, n]) -> sar.f32[n, n]:
            return x * 2.0

        f = measure_kernel(large.to_mlir())
        cfg = _cfg()
        d = plan(cfg, f).values
        assert 2 <= d["loop_tile_size"] <= 64
        # The recompute threshold must not exceed the actual plane size,
        # or nothing in the kernel would ever be recomputed.
        assert d["recompute_min_elements"] <= f.plane_elements
        # This one-operation graph has no reusable producer chain.
        assert d["reuse_buffer_min_elements"] > f.plane_elements


# ---------------------------------------------------------------------- #
# Integration: derived values reach the passes when keys are null
# ---------------------------------------------------------------------- #


@requires_hls
def test_null_strategy_keys_are_derived_and_reported():
    """When all strategy keys are null, `derive` fills them; the reported
    config must have no null strategy key after compilation."""

    n = 32

    @sar.func
    def scale(x: sar.f32[n, n]) -> sar.f32[n, n]:
        return x * 2.0

    scale.name = "autotune_null_filled"
    design = scale.compile(backend="hls")
    for key in AUTO_OPTIONS:
        assert design.config[key] is not None, f"{key} is still null"
    assert HLSConfig.DERIVED in design.config.sources


def test_every_strategy_key_is_classified_tuned_or_fixed():
    """The two kinds of strategy value must stay told apart.

    A `tuned` key is modelled per kernel against the resource budgets and is
    where the latency/area trade-off is made; a `fixed` key is the same for
    every design and encodes a property of the lowering or of the synthesis
    tool. A new key that lands in neither set would silently read as one of
    them, so the partition is exact rather than approximate.
    """
    assert set(TUNED_OPTIONS) | set(FIXED_OPTIONS) == set(AUTO_OPTIONS)
    assert not set(TUNED_OPTIONS) & set(FIXED_OPTIONS)
    assert TUNED_OPTIONS and FIXED_OPTIONS
    # Allocation reuse is revisited after lowering, once the compiler can
    # measure whether the unshared full-size planes leave the device.
    assert "reuse_buffer_min_elements" in TUNED_OPTIONS


def test_fixed_strategy_keys_do_not_vary_with_the_kernel():
    """What makes a key `fixed` is that no kernel moves it.

    Two kernels chosen to differ in the properties the policy reads --
    transforms, gathers, plane size -- must still agree on every fixed key,
    or it belongs in the tuned set with a cost model behind it.
    """
    plain = _facts(plane_elements=256, element_bytes=4)
    heavy = _facts(plane_elements=1 << 24,
                   element_bytes=8,
                   transforms=((4096, 8), ) * 3)
    heavy = KernelFacts(**{**heavy.__dict__, "gather_ops": 3})
    config = _cfg()
    first = plan(config, plain).values
    second = plan(config, heavy).values
    for key in FIXED_OPTIONS:
        assert first[key] == second[key], (
            f"{key} is marked fixed but changed with the kernel")


@requires_hls
def test_pinned_key_overrides_derived():
    """A user-pinned strategy key reaches the design unchanged."""

    n = 64

    @sar.func
    def scale(x: sar.f32[n, n]) -> sar.f32[n, n]:
        return x * 2.0

    scale.name = "autotune_pinned"
    design = scale.compile(backend="hls",
                           options={
                               "fft_stage_group": 3,
                               "loop_tile_size": 4
                           })
    assert design.config.fft_stage_group == 3
    assert design.config.loop_tile_size == 4
    assert design.config.provenance[
        "fft_stage_group"] == HLSConfig.FROM_OPTIONS
    assert design.config.provenance["loop_tile_size"] == HLSConfig.FROM_OPTIONS


@requires_hls
def test_tighter_budget_moves_storage_threshold():
    """A smaller budget should lower the storage threshold so fewer buffers
    stay private, confirming the constraint propagates through to the pass."""

    n = 64

    @sar.func
    def scale(x: sar.f32[n, n]) -> sar.f32[n, n]:
        return x * 2.0

    scale.name = "autotune_budget_threshold"
    big = scale.compile(backend="hls", options={"interface": "axi"}).config
    small = scale.compile(backend="hls",
                          options={
                              "interface": "axi",
                              "bram_bytes": 65536,
                              "uram_bytes": 0,
                              "lutram_bytes": 0
                          }).config
    # With a near-zero budget the threshold should be at or below the
    # full budget threshold.
    assert small.recompute_min_elements <= big.recompute_min_elements


@requires_hls
def test_over_budget_retries_with_streaming(monkeypatch):
    """When the scheduled design overruns the resource caps (the
    placement pass fails), the backend retries once with every plane
    streamed and repins the threshold; when the retry fits, its design
    is the one that ships."""
    import sar.backends.hls.compiler as hls_compiler
    from sar.errors import CompilationError

    # A cache hit would skip the tools, and with them the retry under test.
    monkeypatch.setenv("SAR_DSL_DISABLE_CACHE", "1")

    n = 64

    @sar.func
    def chain(x: sar.c128[n, n], w: sar.c128[n, n]) -> sar.c128[n, n]:
        return sar.ifft(sar.fft(x, axis=1) * w, axis=1)

    chain.name = "autotune_overbudget"

    calls = []
    real_run_tool = hls_compiler.run_tool

    def spy(stage, command, input_text=None):
        if stage == "sar-hls":
            threshold = next(c for c in command[1].split()
                             if c.startswith("external-buffer-threshold"))
            calls.append(int(threshold.split("=")[1]))
            # First call: pretend placement refused the working set; the
            # retry runs for real, so the second call decides the outcome.
            if len(calls) == 1:
                raise CompilationError(
                    stage, command,
                    "error: SAR_HLS_RETRYABLE_MEMORY_OVERFLOW: placement "
                    "needs 999 additional bytes")
        return real_run_tool(stage, command, input_text=input_text)

    monkeypatch.setattr(hls_compiler, "run_tool", spy)
    design = chain.compile(backend="hls")
    assert len(calls) == 2, calls
    assert calls[1] < calls[0]
    assert design.config.external_buffer_threshold == calls[1]
    assert design._metadata.extra["hls_retry_trace"] == [{
        "reason": "memory",
        "action": "stream_full_planes",
        "before": calls[0],
        "after": calls[1],
    }]


@requires_hls
def test_over_budget_narrows_the_transform_transfer(monkeypatch):
    """Streaming is not the last rung of the ladder.

    `fft_io_unroll` is sized against the transform's own storage ceiling,
    which a full beat always clears; the blocks it widens still compete with
    every other resident plane. So a design that overruns after streaming
    narrows the transfer and lowers again, and the design that ships reports
    the width it was actually built with.
    """
    import sar.backends.hls.compiler as hls_compiler
    from sar.errors import CompilationError

    monkeypatch.setenv("SAR_DSL_DISABLE_CACHE", "1")

    n = 64

    @sar.func
    def chain(x: sar.c128[n, n], w: sar.c128[n, n]) -> sar.c128[n, n]:
        return sar.ifft(sar.fft(x, axis=1) * w, axis=1)

    chain.name = "autotune_narrow_transfer"

    widths = []
    real_run_tool = hls_compiler.run_tool

    def spy(stage, command, input_text=None):
        if stage == "sar-lower":
            option = next(c for c in command[-2].split()
                          if c.startswith("fft-io-unroll"))
            widths.append(int(option.split("=")[1]))
        if stage == "sar-hls" and widths and widths[-1] > 1:
            # Refuse every schedule until the transfer has been narrowed.
            raise CompilationError(
                stage, command,
                "error: SAR_HLS_RETRYABLE_MEMORY_OVERFLOW: placement "
                "needs 999 additional bytes")
        return real_run_tool(stage, command, input_text=input_text)

    monkeypatch.setattr(hls_compiler, "run_tool", spy)
    design = chain.compile(backend="hls")
    assert widths[0] > 1, "the kernel must start at a wide transfer"
    assert widths[-1] == 1, widths
    # Halved one step at a time, never skipped straight to the floor.
    assert widths == sorted(widths, reverse=True)
    assert design.config.fft_io_unroll == 1
    assert any(item["action"] == "halve_fft_io_unroll"
               for item in design._metadata.extra["hls_retry_trace"])


@requires_hls
def test_over_budget_after_streaming_refuses_the_design(monkeypatch):
    """The caps are hard: when even the fully-streamed design overruns
    them, compilation fails rather than emit a design that cannot fit
    the device."""
    import pytest as _pytest

    import sar.backends.hls.compiler as hls_compiler
    from sar.backends.hls.config import HLSConfigError
    from sar.errors import CompilationError

    monkeypatch.setenv("SAR_DSL_DISABLE_CACHE", "1")

    n = 64

    @sar.func
    def chain(x: sar.c128[n, n]) -> sar.c128[n, n]:
        return sar.fft(chain_body(x), axis=1)

    @sar.op
    def chain_body(x):
        return x * 2.0

    chain.name = "autotune_overbudget_hard"

    real_run_tool = hls_compiler.run_tool

    def spy(stage, command, input_text=None):
        if stage == "sar-hls":
            raise CompilationError(
                stage, command, "error: SAR_HLS_RETRYABLE_MEMORY_OVERFLOW: "
                "on-chip working set exceeds the memory budgets")
        return real_run_tool(stage, command, input_text=input_text)

    monkeypatch.setattr(hls_compiler, "run_tool", spy)
    with _pytest.raises(HLSConfigError, match="exceeds the resource caps"):
        chain.compile(backend="hls")


def test_repin_rejects_user_pinned_values():
    """`repin` may only revise derived values; a user-pinned key is a
    contract the compiler cannot override."""
    import pytest as _pytest

    from sar.backends.hls.config import HLSConfig

    config = HLSConfig.resolve({"fft_stage_group": 2})
    with _pytest.raises(ValueError, match="not a derived value"):
        config.repin("fft_stage_group", 4)


@requires_hls
def test_axis0_interp_counts_its_hidden_transposes():
    """`interp1d(dim=0)` canonicalizes into transposes around the row-wise
    form; the staging budget must see those corner turns even though the
    traced module names none."""
    import sar

    n = 64

    @sar.func
    def rcmc(z: sar.c128[n, n], p: sar.f64[n, n]) -> sar.c128[n, n]:
        return sar.interp1d(z, p, dim=0)

    facts = measure_kernel(rcmc.to_mlir())
    assert facts.transposes >= 3
    # The staging budget shrinks accordingly instead of promising each
    # of the hidden corner turns the whole allowance.
    whole = transpose_block_bytes(_facts(), bram_bytes=6193152)
    split = transpose_block_bytes(facts, bram_bytes=6193152)
    assert split < whole


def test_zero_bram_disables_transpose_staging():
    assert transpose_block_bytes(_facts(), bram_bytes=0) == 0


def test_sub_block_bram_budget_disables_transpose_staging():
    assert transpose_block_bytes(_facts(), bram_bytes=4095) == 0
