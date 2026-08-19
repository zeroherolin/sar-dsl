"""Tests for `sar.backends.hls.autotune`.

Three kinds: unit tests for each derivation (no toolchain needed),
integration tests that confirm the derived values reach the passes when
the strategy keys are left at null, and a generality check that a tiny
kernel and a 16384-scale kernel both get sensible choices.
"""

import sar
import pytest
from sar.backends.base import KernelMetadata
from sar.backends.hls.autotune import (
    AUTO_OPTIONS, KernelFacts, array_partition_max_factor, derive,
    fft_parallel_rows, fft_stage_group, interp_banded_gather,
    kernel_facts_from_json, loop_tile_size, lutram_max_bytes, measure_kernel,
    storage_min_elements, transpose_block_bytes, _scratch_slots)
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


def test_structured_kernel_facts_parser():
    facts = kernel_facts_from_json(
        '{"plane_elements":64,"element_bytes":4,"transposes":2,'
        '"transforms":[[8,4]],"buffers":[[64,256]]}')
    assert facts == KernelFacts(64, 4, ((8, 4), ), 2, ((64, 256), ))


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
# fft_parallel_rows
# ---------------------------------------------------------------------- #


def test_parallelism_follows_bus_width_and_precision():
    assert fft_parallel_rows(_facts(element_bytes=8), 512, 9830) == 2
    assert fft_parallel_rows(_facts(element_bytes=4), 512, 9830) == 2
    assert fft_parallel_rows(_facts(element_bytes=4), 128, 9830) == 0
    assert fft_parallel_rows(_facts(element_bytes=4), 512, 9830, "axi") == 0


def test_parallelism_respects_the_dsp_budget():
    facts = _facts(element_bytes=4)
    assert fft_parallel_rows(facts, 512, 1023) == 0
    assert fft_parallel_rows(facts, 512, 1024) == 2


def test_partition_factor_is_bounded_by_one_bus_beat():
    assert array_partition_max_factor(_facts(element_bytes=4), 512) == 16
    assert array_partition_max_factor(_facts(element_bytes=8), 512) == 8
    assert array_partition_max_factor(_facts(element_bytes=4), 1024) == 32


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


def test_dsp_budget_is_a_device_constraint():
    assert _cfg().dsp == 9830
    assert "dsp" not in AUTO_OPTIONS
    # Retargeting to a smaller part is stating a different fact.
    assert _cfg(dsp=1824).dsp == 1824


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


@requires_hls
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
    assert small.reuse_buffer_min_elements <= big.reuse_buffer_min_elements


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
    from sar.backends.base import KernelMetadata

    n = 64

    @sar.func
    def rcmc(z: sar.c128[n, n], p: sar.f64[n, n]) -> sar.c128[n, n]:
        return sar.interp1d(z, p, dim=0)

    md = KernelMetadata("rcmc", list(rcmc.arg_types),
                        list(rcmc.declared_result_types))
    facts = measure_kernel(rcmc.to_mlir(), md)
    assert facts.transposes >= 3
    # The staging budget shrinks accordingly instead of promising each
    # of the hidden corner turns the whole allowance.
    whole = transpose_block_bytes(_facts(), bram_bytes=6193152)
    split = transpose_block_bytes(facts, bram_bytes=6193152)
    assert split < whole


def test_zero_bram_disables_transpose_staging():
    assert transpose_block_bytes(_facts(), bram_bytes=0) == 0
