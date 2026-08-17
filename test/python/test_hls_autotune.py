"""Tests for `sar.backends.hls.autotune`.

Three kinds: unit tests for each derivation (no toolchain needed),
integration tests that confirm the derived values reach the passes when
the strategy keys are left at null, and a generality check that a tiny
kernel and a 16384-scale kernel both get sensible choices.
"""

import sar
import pytest
from sar.backends.base import KernelMetadata
from sar.backends.hls.autotune import (AUTO_OPTIONS, KernelFacts, derive,
                                       fft_stage_group, interp_banded_gather,
                                       loop_tile_size, lutram_max_bytes,
                                       measure_kernel, storage_min_elements,
                                       _scratch_slots)
from sar.backends.hls.config import HLSConfig

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


# ---------------------------------------------------------------------- #
# fft_stage_group
# ---------------------------------------------------------------------- #


class TestFftStageGroup:

    def test_no_transforms_gives_0(self):
        assert fft_stage_group(_facts(), budget=1) == 0

    def test_unbounded_budget_chooses_full_unroll(self):
        # 0 means "keep everything resident", so nothing has to be grouped.
        f = _facts(transforms=((512, 8), ))
        assert fft_stage_group(f, budget=0) == 0

    @pytest.mark.parametrize("budget,expected", [
        (1 << 20, 0),
        (1 << 19, 0),
        (1 << 18, 2),
        (1 << 17, 3),
        (1 << 10, 3),
    ])
    def test_budget_picks_the_grouping(self, budget, expected):
        """One transform of N=512 costs 64 KiB of scratch at full unroll,
        32 KiB at k=2 and 16 KiB from k=3 on, where the pool has bottomed
        out. An eighth of each budget is what those have to fit, and the
        least grouping that fits is the one to take -- down to the tightest
        available once none of them does."""
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

    def test_scratch_model_matches_cpp(self):
        """The Python scratch model must agree with the C++ `scratchSlots`.

        The policy derives a grouping whose scratch fits the budget and the
        C++ builds the design from that grouping; if the two disagree the
        budget was spent against a design that was never emitted.
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
        # tile × element_bytes must be a multiple of (bus/8).
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
        d = derive(cfg, _facts())
        assert d["lutram_max_bytes"] == 32

    def test_is_an_auto_option(self):
        # It has to be derived like every other strategy value, not frozen
        # as a pass default.
        assert "lutram_max_bytes" in AUTO_OPTIONS

    def test_pinning_it_wins_over_the_derivation(self):
        cfg = _cfg(lutram_max_bytes=256)
        cfg.adopt(derive(cfg, _facts()))
        assert cfg["lutram_max_bytes"] == 256
        assert cfg.provenance["lutram_max_bytes"] == HLSConfig.FROM_OPTIONS


def test_uram_min_bytes_is_a_device_fact():
    """36864 bytes is one 288 Kb UltraRAM block. It belongs to the device
    description, so it comes from the config file rather than a TableGen
    constant, and it is a constraint the user states -- not strategy."""
    assert _cfg().uram_min_bytes == 36864
    assert "uram_min_bytes" not in AUTO_OPTIONS
    # Retargeting to a device without URAM is stating a different fact.
    assert _cfg(uram_min_bytes=0).uram_min_bytes == 0


# ---------------------------------------------------------------------- #
# interp_banded_gather
# ---------------------------------------------------------------------- #


def test_interp_banded_gather_always_on():
    # The pass proves displacement bounds per-op and falls back to the plain
    # gather when it cannot; enabling it globally costs nothing.
    assert interp_banded_gather() is True


# ---------------------------------------------------------------------- #
# storage_min_elements
# ---------------------------------------------------------------------- #


class TestStorageMinElements:

    def test_unbounded_budget_returns_plane_size(self):
        # Nothing has to be shared until a buffer is a full-scene plane.
        f = _facts(plane_elements=65536, element_bytes=4)
        assert storage_min_elements(f, 0) == 65536

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
# derive: picks null keys, honours pinned keys
# ---------------------------------------------------------------------- #


class TestDerive:

    def test_all_auto_options_are_filled(self):
        f = _facts(plane_elements=65536,
                   element_bytes=4,
                   transforms=((256, 8), ))
        cfg = _cfg()
        d = derive(cfg, f)
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
        cfg.adopt(derive(cfg, f))
        assert cfg["fft_stage_group"] == 3

    def test_provenance_marks_derived_keys(self):
        cfg = _cfg()
        f = _facts(transforms=((256, 8), ))
        cfg.adopt(derive(cfg, f))
        assert cfg.provenance["fft_stage_group"] == HLSConfig.DERIVED

    def test_provenance_marks_user_keys(self):
        cfg = _cfg(fft_stage_group=2)
        f = _facts(transforms=((256, 8), ))
        cfg.adopt(derive(cfg, f))
        assert cfg.provenance["fft_stage_group"] == HLSConfig.FROM_OPTIONS

    def test_smaller_budget_picks_higher_group(self):
        # A budget that cannot afford the full FFT scratch should push the
        # grouping above 0 (full unroll).  Budget=1 cannot afford any scratch.
        f = _facts(transforms=((512, 8), ))
        tight = _cfg(on_chip_budget=1)
        loose = _cfg(on_chip_budget=0)
        g_tight = fft_stage_group(f, int(tight["on_chip_budget"]))
        g_loose = fft_stage_group(f, int(loose["on_chip_budget"]))
        assert g_tight > g_loose or g_tight >= 2


# ---------------------------------------------------------------------- #
# measure_kernel: reads facts from a traced SAR module
# ---------------------------------------------------------------------- #


class TestMeasureKernel:

    def test_reads_plane_size_from_signature(self):
        n = 128

        @sar.func
        def scale(x: sar.f32[n, n]) -> sar.f32[n, n]:
            return x * 2.0

        md = KernelMetadata("scale", list(scale.arg_types),
                            list(scale.declared_result_types))
        f = measure_kernel(scale.to_mlir(), md)
        assert f.plane_elements >= n * n

    def test_finds_fft_transforms(self):
        n = 64

        @sar.func
        def with_fft(x: sar.c64[n, n]) -> sar.c64[n, n]:
            return sar.fft(x, axis=1)

        md = KernelMetadata("with_fft", list(with_fft.arg_types),
                            list(with_fft.declared_result_types))
        f = measure_kernel(with_fft.to_mlir(), md)
        assert len(f.transforms) > 0

    def test_element_bytes_is_narrowest_plane(self):
        n = 16

        @sar.func
        def mixed(x: sar.f32[n, n]) -> sar.f32[n, n]:
            return x * 2.0

        md = KernelMetadata("mixed", list(mixed.arg_types),
                            list(mixed.declared_result_types))
        f = measure_kernel(mixed.to_mlir(), md)
        assert f.element_bytes == 4


# ---------------------------------------------------------------------- #
# Generality: tiny and very large kernels both get sensible choices
# ---------------------------------------------------------------------- #


class TestGenerality:

    def test_tiny_kernel(self):
        # A 4×4 kernel has no FFTs and a trivial plane; derived values must
        # be valid, non-zero, and not reference any per-algorithm constant.
        n = 4

        @sar.func
        def tiny(x: sar.f32[n, n]) -> sar.f32[n, n]:
            return x * 2.0

        md = KernelMetadata("tiny", list(tiny.arg_types),
                            list(tiny.declared_result_types))
        f = measure_kernel(tiny.to_mlir(), md)
        cfg = _cfg()
        d = derive(cfg, f)
        assert d["loop_tile_size"] >= 2
        assert d["fft_stage_group"] == 0  # no transforms: full unroll is fine
        assert d["interp_banded_gather"] is True

    def test_large_kernel(self):
        # A 16384-scale kernel: derived values must remain sensible.
        n = 16384

        @sar.func
        def large(x: sar.f32[n, n]) -> sar.f32[n, n]:
            return x * 2.0

        md = KernelMetadata("large", list(large.arg_types),
                            list(large.declared_result_types))
        f = measure_kernel(large.to_mlir(), md)
        cfg = _cfg()
        d = derive(cfg, f)
        assert 2 <= d["loop_tile_size"] <= 64
        # storage threshold must not exceed the actual plane size
        assert d["reuse_buffer_min_elements"] <= f.plane_elements
        assert d["recompute_min_elements"] <= f.plane_elements


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
    big = scale.compile(backend="hls", options={"on_chip_budget": 0}).config
    small = scale.compile(backend="hls", options={"on_chip_budget": 1}).config
    # With a near-zero budget the threshold should be at or below the
    # full budget threshold.
    assert small.reuse_buffer_min_elements <= big.reuse_buffer_min_elements
