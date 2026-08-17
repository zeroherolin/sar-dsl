"""Configuration surface of the HLS backend.

Two halves: the schema (defaults, overriding, validation), which needs no
toolchain, and the wiring, which compiles kernels and checks that the
resolved values reach the passes.
"""

import re

import pytest

import sar
from sar.backends.hls.config import (CONFIG_ENV_VAR, OPTIONS, HLSConfig,
                                     HLSConfigError, shipped_config_path)

from conftest import requires_hls


@pytest.fixture(autouse=True)
def _no_ambient_config(monkeypatch):
    """A config file in the environment must not decide these results."""
    monkeypatch.delenv(CONFIG_ENV_VAR, raising=False)


# --------------------------------------------------------------------- #
# The shipped defaults
# --------------------------------------------------------------------- #


def test_shipped_file_carries_the_constraints_and_only_those():
    """The file is the surface a user configures, so it holds every
    constraint and no strategy: a constraint without a default would
    resolve differently per code path, and a strategy key present here
    would invite a guess the compiler makes better."""
    from sar.backends.hls.config import _parse_yaml

    path = shipped_config_path()
    in_file = set(_parse_yaml(path.read_text(), str(path)))
    constraints = {k for k, spec in OPTIONS.items() if not spec.advanced}
    assert in_file == constraints


def test_derived_options_start_unset_and_are_marked():
    config = HLSConfig.resolve()
    for key, spec in OPTIONS.items():
        if spec.advanced:
            assert config[key] is None, key
            assert config.provenance[key] == HLSConfig.DERIVED, key


def test_pinning_a_derived_option_still_wins():
    config = HLSConfig.resolve({"fft_stage_group": 3})
    assert config.fft_stage_group == 3
    assert config.provenance["fft_stage_group"] == HLSConfig.FROM_OPTIONS


def test_defaults_describe_a_vu13p_half():
    """Half of the device (2688 x 36 Kb block RAM, 1280 x 288 Kb UltraRAM,
    12288 DSP slices), which is the headroom a pre-synthesis estimate
    needs."""
    config = HLSConfig.resolve()
    assert config.bram_bytes == 2688 * 36 * 1024 // 8 // 2
    assert config.uram_bytes == 1280 * 288 * 1024 // 8 // 2
    # Distributed RAM is rationed harder than the dedicated tiers: it comes
    # out of the SLICEM LUTs the datapath itself is built from, so a quarter
    # of the device's 27.5 Mb rather than a half.
    assert config.lutram_bytes == int(27.5 * 1024 * 1024 / 8 / 4)
    # A null total is the sum of the tiers.
    assert config.on_chip_budget == (config.bram_bytes + config.uram_bytes +
                                     config.lutram_bytes)


def test_resolved_config_reads_as_a_mapping():
    config = HLSConfig.resolve({"axi_bus_bits": 256})
    assert config["axi_bus_bits"] == 256 == config.axi_bus_bits
    assert dict(config) == config.as_dict()
    assert set(config) == set(OPTIONS)
    assert str(shipped_config_path()) in config.sources


# --------------------------------------------------------------------- #
# Overriding
# --------------------------------------------------------------------- #


def test_option_overrides_the_file():
    assert HLSConfig.resolve({"loop_tile_size": 32}).loop_tile_size == 32
    assert HLSConfig.resolve({"on_chip_budget": 0}).on_chip_budget == 0


def test_config_file_overrides_the_shipped_defaults(tmp_path):
    path = tmp_path / "device.yaml"
    path.write_text("# a smaller part\n"
                    "bram_bytes: 1024\n"
                    "axi_bus_bits: 128  # narrow bus\n")

    config = HLSConfig.resolve({"config": str(path)})
    assert config.bram_bytes == 1024
    assert config.axi_bus_bits == 128
    # Keys the file leaves out keep the shipped default.
    assert config.uram_bytes == 23592960
    assert str(path) in config.sources


def test_env_var_names_a_config_file(tmp_path, monkeypatch):
    path = tmp_path / "device.yaml"
    path.write_text("loop_tile_size: 4\n")
    monkeypatch.setenv(CONFIG_ENV_VAR, str(path))

    assert HLSConfig.resolve().loop_tile_size == 4
    # An option still wins over the file the environment names.
    assert HLSConfig.resolve({"loop_tile_size": 16}).loop_tile_size == 16


def test_config_option_wins_over_the_env_var(tmp_path, monkeypatch):
    from_env = tmp_path / "env.yaml"
    from_env.write_text("loop_tile_size: 4\n")
    given = tmp_path / "given.yaml"
    given.write_text("loop_tile_size: 2\n")
    monkeypatch.setenv(CONFIG_ENV_VAR, str(from_env))

    assert HLSConfig.resolve({"config": str(given)}).loop_tile_size == 2


def test_missing_config_file_names_where_it_came_from(tmp_path):
    with pytest.raises(HLSConfigError, match="cannot be read"):
        HLSConfig.resolve({"config": str(tmp_path / "absent.yaml")})


def test_axi_interface_flag_still_selects_the_interface():
    """`axi_interface` predates `interface` and is what the examples and
    benchmarks pass."""
    assert HLSConfig.resolve({"axi_interface": True}).interface == "axi"
    # False selects the memory protocol, stored under its canonical name.
    assert HLSConfig.resolve({"axi_interface": False}).interface == "ap_memory"
    with pytest.raises(HLSConfigError, match="conflicting interface"):
        HLSConfig.resolve({"axi_interface": True, "interface": "bram"})
    # `bram` is the deprecated spelling of `ap_memory`, so pairing it with
    # axi_interface=False agrees rather than conflicts.
    assert HLSConfig.resolve({
        "axi_interface": False,
        "interface": "bram"
    }).interface == "ap_memory"


# --------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------- #


def test_unknown_option_lists_the_valid_ones():
    with pytest.raises(HLSConfigError) as excinfo:
        HLSConfig.resolve({"on_chip_bugdet": 1024})
    message = str(excinfo.value)
    assert "unknown HLS option 'on_chip_bugdet'" in message
    assert "did you mean 'on_chip_budget'?" in message
    # The whole schema, so the error is also the documentation.
    for name in OPTIONS:
        assert name in message


def test_unknown_key_in_a_config_file_names_the_file(tmp_path):
    path = tmp_path / "device.yaml"
    path.write_text("bram_byte: 1024\n")
    with pytest.raises(HLSConfigError, match=str(path)):
        HLSConfig.resolve({"config": str(path)})


@pytest.mark.parametrize("options,expected", [
    ({
        "axi_bus_bits": 7
    }, "must be at least 32"),
    ({
        "axi_bus_bits": 4096
    }, "must be at most 1024"),
    ({
        "axi_bus_bits": 384
    }, "must be a power of two"),
    ({
        "axi_max_burst_length": 512
    }, "must be at most 256"),
    ({
        "axi_max_outstanding": 0
    }, "must be at least 1"),
    ({
        "loop_tile_size": 0
    }, "must be at least 1"),
    ({
        "fft_stage_group": -1
    }, "must be at least 0"),
    ({
        "on_chip_budget": 2**40
    }, "must be at most"),
    ({
        "bram_bytes": -1
    }, "must be at least 0"),
    ({
        "clock_ns": 0.5
    }, "must be at least 1"),
])
def test_out_of_range_values_are_rejected(options, expected):
    with pytest.raises(HLSConfigError, match=expected):
        HLSConfig.resolve(options)


@pytest.mark.parametrize("options,expected", [
    ({
        "on_chip_budget": "8 MiB"
    }, "expected an integer"),
    ({
        "loop_tile_size": True
    }, "expected an integer"),
    ({
        "interp_banded_gather": 1
    }, "expected true or false"),
    ({
        "interface": "pcie"
    }, r"expected one of \['ap_memory', 'axi', 'stream'\]"),
    ({
        "precision": "half"
    }, r"expected one of"),
    ({
        "top_func": "2fast"
    }, "expected a C identifier"),
    ({
        "axi_bus_bits": None
    }, "null is not allowed"),
    ({
        "part": "vu13p with spaces"
    }, "expected a Vitis part name"),
    ({
        "clock_ns": "fast"
    }, "expected a number"),
])
def test_ill_typed_values_are_rejected(options, expected):
    with pytest.raises(HLSConfigError, match=expected):
        HLSConfig.resolve(options)


def test_bram_is_accepted_as_a_deprecated_alias():
    """`bram` was the old name for the memory protocol. Existing configs
    and benchmarks still say it, so it keeps working -- and normalizes to
    the canonical name so nothing downstream sees two spellings."""
    assert HLSConfig.resolve({"interface": "bram"}).interface == "ap_memory"
    assert HLSConfig.resolve({
        "interface": "ap_memory"
    }).interface == "ap_memory"


def test_bram_alias_works_from_a_config_file(tmp_path):
    """The alias has to survive the YAML path too, not just compile
    options: that is where a retargeting project states it."""
    path = tmp_path / "legacy.yaml"
    path.write_text("interface: bram\n")
    assert HLSConfig.resolve({"config": str(path)}).interface == "ap_memory"


def test_stream_interface_is_accepted():
    """AXI4-Stream is a top-level port protocol a streaming radar front
    end asks for; it used to be rejected on the grounds that streams are
    the compiler's call, which is true only of internal dataflow
    channels."""
    assert HLSConfig.resolve({"interface": "stream"}).interface == "stream"


def test_a_bad_option_is_also_a_value_error():
    """`compile(options=...)` is a call, and a bad option is a bad
    argument to it."""
    with pytest.raises(ValueError):
        HLSConfig.resolve({"nonesuch": 1})
    with pytest.raises(sar.SARError):
        HLSConfig.resolve({"nonesuch": 1})


# --------------------------------------------------------------------- #
# The YAML subset
# --------------------------------------------------------------------- #


def test_yaml_subset_reads_scalars_and_comments():
    from sar.backends.hls.config import _parse_yaml

    values = _parse_yaml(
        "# leading comment\n"
        "\n"
        "count: 4_096   # inline comment\n"
        "mask: 0x20\n"
        "flag: true\n"
        "empty:\n"
        "nothing: null\n"
        "name: 'hash # not a comment'\n"
        "bare: some_name\n"
        "ratio: 0.5\n", "<test>")
    assert values == {
        "count": 4096,
        "mask": 0x20,
        "flag": True,
        "empty": None,
        "nothing": None,
        "name": "hash # not a comment",
        "bare": "some_name",
        "ratio": 0.5,
    }


@pytest.mark.parametrize("text,expected", [
    ("bram_bytes 1024\n", "expected 'key: value'"),
    ("device:\n  bram_bytes: 1024\n", "flat 'key: value' mapping"),
    ("bram_bytes: 1\nbram_bytes: 2\n", "duplicate key"),
])
def test_yaml_subset_rejects_what_it_cannot_read(text, expected):
    from sar.backends.hls.config import _parse_yaml

    with pytest.raises(HLSConfigError, match=expected):
        _parse_yaml(text, "<test>")


# --------------------------------------------------------------------- #
# Precision policy
# --------------------------------------------------------------------- #


def test_precision_policy_rejects_the_other_width():
    """The passes have no precision knob, so the policy is a gate: an f64
    plane in a design budgeted for f32 costs several times the DSPs."""
    n = 8

    @sar.func
    def wide(x: sar.c128[n, n]) -> sar.c128[n, n]:
        return x * 2.0

    with pytest.raises(HLSConfigError, match="whose data path is f64"):
        wide.compile(backend="hls", options={"precision": "f32"})


def test_precision_policy_accepts_a_matching_kernel():
    from sar.backends.hls.config import check_precision
    from sar.ir import C64, F32, TensorType

    config = HLSConfig.resolve({"precision": "f32"})
    check_precision(config, [TensorType((4, 4), C64)],
                    [TensorType((4, ), F32)])


# --------------------------------------------------------------------- #
# Reaching the passes
# --------------------------------------------------------------------- #


@pytest.fixture
def recorded_commands(monkeypatch):
    """The tool command lines a compilation builds."""
    import sys

    # The registry loads a backend from its file, so the module the stages
    # actually run in is not `sar.backends.hls.compiler`.
    module = sys.modules[sar.get_backend("hls").__module__]

    # A cache hit skips the tools, and with them the point of the test.
    monkeypatch.setenv("SAR_DSL_DISABLE_CACHE", "1")
    commands = []
    real = module.run_tool

    def spy(stage, command, input_text=None):
        commands.append(" ".join(str(c) for c in command))
        return real(stage, command, input_text=input_text)

    monkeypatch.setattr(module, "run_tool", spy)
    return commands


@requires_hls
def test_configuration_reaches_the_tool_options(recorded_commands):
    n = 8

    @sar.func
    def scale(x: sar.f32[n, n]) -> sar.f32[n, n]:
        return x * 2.0

    scale.name = "cfg_options"
    scale.compile(backend="hls",
                  options={
                      "top_func": "renamed_top",
                      "loop_tile_size": 4,
                      "fft_stage_group": 2,
                      "interp_banded_gather": False,
                      "reuse_buffer_min_elements": 7,
                      "recompute_min_elements": 9,
                      "external_buffer_threshold": 11,
                      "bram_bytes": 2048,
                      "uram_bytes": 4096,
                      "lutram_bytes": 0,
                  })
    lower, hls, emit = recorded_commands

    assert "reuse-buffer-min-elements=7" in lower
    assert "recompute-min-elements=9" in lower
    assert "interp-enable-banded-gather=false" in lower
    assert "fft-stage-group=2" in lower
    # The tiers with no total sum to one, and an eighth of it is what a
    # staged transpose block may occupy.
    assert "transpose-block-bytes=768" in lower

    assert "top-func=renamed_top" in hls
    assert "loop-tile-size=4" in hls
    assert "on-chip-bytes=6144" in hls
    assert "axi-interface=false" in hls
    assert "external-buffer-threshold=11" in hls
    # Both tier thresholds are derived rather than frozen as pass
    # defaults, and both have to arrive in bytes: `lutram-max-bytes` is
    # one bus beat (512/8), `uram-min-bytes` is one physical URAM block.
    assert "lutram-max-bytes=64" in hls
    assert "uram-min-bytes=36864" in hls

    assert "-axi-bus-bits=512" in emit
    # Buffering is both directions' worth of full-length bursts.
    assert f"-axi-buffer-bits={2 * 256 * 16 * 512}" in emit


@requires_hls
def test_design_reports_the_configuration_it_used():
    n = 8

    @sar.func
    def scale(x: sar.f32[n, n]) -> sar.f32[n, n]:
        return x * 2.0

    scale.name = "cfg_reported"
    design = scale.compile(backend="hls", options={"loop_tile_size": 4})
    assert design.config.loop_tile_size == 4
    assert design.config.axi_bus_bits == 512
    assert set(design.config) == set(OPTIONS)


@requires_hls
def test_top_func_names_the_emitted_function():
    """The pipeline selects its top function by name, so the option has to
    rename the kernel rather than just ask for a name nothing carries."""
    n = 8

    @sar.func
    def scale(x: sar.f32[n, n]) -> sar.f32[n, n]:
        return x * 2.0

    scale.name = "cfg_renamed"
    design = scale.compile(backend="hls", options={"top_func": "custom_top"})
    assert design.name == "custom_top"
    assert "void custom_top" in design.source()


def _axi_shape(source):
    """(beat width, burst beats, outstanding bursts) of the AXI ports."""

    def numbers(pragma):
        return {int(v) for v in re.findall(rf"{pragma}=(\d+)", source)}

    return (numbers("max_widen_bitwidth"), numbers("max_read_burst_length"),
            numbers("num_read_outstanding"))


@requires_hls
@pytest.mark.parametrize("bus_bits", [512, 256])
def test_configured_bus_width_reaches_the_pragmas(bus_bits):
    """The headline wiring check: what the config says a beat is, is what
    the emitted interface pragmas say."""
    n = 512

    @sar.func
    def scale(x: sar.f64[n, n]) -> sar.f64[n, n]:
        return x * 2.0

    scale.name = f"cfg_bus_{bus_bits}"
    source = scale.compile(backend="hls",
                           options={
                               "interface": "axi",
                               "on_chip_budget": 1,
                               "axi_bus_bits": bus_bits
                           }).source()
    widen, burst, _ = _axi_shape(source)
    assert widen == {bus_bits}
    # A row is 512 doubles, so the burst is however many beats that spans.
    assert burst == {n * 64 // bus_bits}


@requires_hls
def test_configured_buffering_bounds_the_outstanding_bursts():
    """Outstanding bursts are area: every one in flight has to be buffered.
    The configured count is what a port gets at full burst length, and a
    port whose bursts come out shorter gets proportionally more."""
    n = 512

    def outstanding(**options):

        @sar.func
        def scale(x: sar.f64[n, n]) -> sar.f64[n, n]:
            return x * 2.0

        scale.name = f"cfg_out_{options.get('axi_max_outstanding', 'def')}"
        source = scale.compile(backend="hls",
                               options={
                                   "interface": "axi",
                                   "on_chip_budget": 1,
                                   **options
                               }).source()
        return max(_axi_shape(source)[2])

    assert outstanding(axi_max_outstanding=1) < outstanding()


@requires_hls
def test_env_var_config_reaches_the_emitted_design(tmp_path, monkeypatch):
    """A project-wide config file is worth nothing if the artifact cache
    hands back a design compiled against another one."""
    n = 512
    path = tmp_path / "narrow.yaml"
    path.write_text("axi_bus_bits: 128\n")

    def emit():

        @sar.func
        def scale(x: sar.f64[n, n]) -> sar.f64[n, n]:
            return x * 2.0

        scale.name = "cfg_env"
        return scale.compile(backend="hls",
                             options={
                                 "interface": "axi",
                                 "on_chip_budget": 1
                             }).source()

    assert _axi_shape(emit())[0] == {512}
    monkeypatch.setenv(CONFIG_ENV_VAR, str(path))
    assert _axi_shape(emit())[0] == {128}


@requires_hls
def test_tier_budgets_steer_placement():
    """A starved tier pushes its buffers down to the next one.

    Tiering is driven by measured buffer size, so the probe has to own a
    buffer large enough to prefer the coarse tier: the transform scratch of
    a plain FFT is one line wide and belongs in distributed RAM whatever the
    budget says.
    """
    n = 256

    @sar.func
    def corner_turn(x: sar.c64[n, n]) -> sar.c64[n, n]:
        # A transpose stages a whole plane, which is what reaches URAM.
        return sar.transpose(sar.fft(x, axis=1))

    corner_turn.name = "tier_budget_probe"
    common = {"external_buffer_threshold": n * n + 1}
    plentiful = corner_turn.compile(backend="hls",
                                    options={
                                        **common, "uram_bytes": 23592960
                                    }).source()
    starved = corner_turn.compile(backend="hls",
                                  options={
                                      **common, "uram_bytes": 1
                                  }).source()

    import re
    from collections import Counter
    a = Counter(re.findall(r"impl=(\w+)", plentiful))
    b = Counter(re.findall(r"impl=(\w+)", starved))
    # Whatever the plane lands in with URAM available, starving URAM must
    # not leave it there, and must not lose it either.
    assert a["uram"] > 0, f"no URAM-tier buffer to test with: {dict(a)}"
    assert b["uram"] == 0
    assert b["bram"] + b["lutram"] > a["bram"] + a["lutram"]
