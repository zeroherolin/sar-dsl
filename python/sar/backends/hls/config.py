"""Configuration surface of the HLS backend.

Every knob the backend hands to the passes has a default in
`hls_config.yaml`, which ships with the package. Resolution order,
weakest first::

    hls_config.yaml  <  user config file  <  compile options

where the user file comes from ``options={"config": path}`` or, failing
that, ``$SAR_DSL_HLS_CONFIG``; it overrides the keys it names and leaves
the rest at the shipped default.

Every value is validated against `OPTIONS`. An unknown key, a value of
the wrong type and a value outside the range the passes accept all raise
`HLSConfigError` -- an option the user believed in and the backend
ignored is the failure mode this module exists to prevent.

The resolved mapping is what the backend passes down and what a design
reports as `HLSDesign.config`.
"""

from __future__ import annotations

import difflib
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

from sar.errors import SARError

__all__ = [
    "CONFIG_ENV_VAR", "HLSConfig", "HLSConfigError", "OPTIONS",
    "shipped_config_path"
]

#: Names a YAML file replacing the shipped defaults for a whole project.
CONFIG_ENV_VAR = "SAR_DSL_HLS_CONFIG"

_CONFIG_FILENAME = "hls_config.yaml"

#: Ceilings of the pass options these values reach: the byte budgets and
#: the external-buffer threshold land in `unsigned` pass options, the
#: element counts in `uint64_t` ones.
_UINT32_MAX = 2**32 - 1
_UINT64_MAX = 2**64 - 1


class HLSConfigError(SARError, ValueError):
    """An HLS option, or a file supplying one, is invalid.

    Also a `ValueError`: a bad option is a bad argument to `compile`.
    """


@dataclass(frozen=True)
class _Spec:
    kind: str  # "int" | "float" | "bool" | "choice" | "identifier" | "part"
    doc: str
    minimum: Optional[int] = None
    maximum: Optional[int] = None
    choices: Tuple[str, ...] = ()
    #: Deprecated spellings of a choice, mapped to the canonical one. An
    #: alias is accepted wherever the choice is and stored normalized, so
    #: everything downstream sees exactly one name per value.
    aliases: Optional[Dict[str, str]] = None
    power_of_two: bool = False
    #: None is accepted and means "the backend decides".
    nullable: bool = False
    #: Values that parse but cannot be honoured, mapped to the reason.
    rejected: Optional[Dict[str, str]] = None
    #: Optimization strategy rather than a constraint. The compiler derives
    #: these, so they carry no default in the shipped file and are not part
    #: of the surface a user configures; pinning one is a diagnostic act.
    advanced: bool = False


#: The complete option schema; the shipped YAML carries a default for
#: every key, and the docs table is generated from these one-liners.
OPTIONS: Dict[str, _Spec] = {
    "bram_bytes":
    _Spec("int",
          "Block RAM the design may occupy, in bytes (0 = unbounded)",
          minimum=0,
          maximum=_UINT32_MAX),
    "uram_bytes":
    _Spec("int",
          "UltraRAM the design may occupy, in bytes (0 = unbounded)",
          minimum=0,
          maximum=_UINT32_MAX),
    "lutram_bytes":
    _Spec("int",
          "Distributed RAM the design may occupy, in bytes (0 = unbounded)",
          minimum=0,
          maximum=_UINT32_MAX),
    "on_chip_budget":
    _Spec("int", "Total on-chip bytes before planes are streamed "
          "(null = sum of the tiers, 0 = keep everything resident)",
          minimum=0,
          maximum=_UINT32_MAX,
          nullable=True),
    "uram_min_bytes":
    _Spec("int", "Buffer size, in bytes, at or above which ultra RAM is used "
          "(device fact: 36864 = one 288 Kb UltraRAM block on VU13P)",
          minimum=0,
          maximum=_UINT32_MAX),
    "interface":
    _Spec("choice",
          "Protocol the top function's ports speak: 'ap_memory' (plain "
          "arrays behind the Vitis ap_memory handshake), 'axi' (AXI4 "
          "masters) or 'stream' (AXI4-Stream)",
          choices=("ap_memory", "axi", "stream"),
          aliases={"bram": "ap_memory"}),
    "axi_bus_bits":
    _Spec("int",
          "Data width of the AXI masters, in bits",
          minimum=32,
          maximum=1024,
          power_of_two=True),
    "axi_max_burst_length":
    _Spec("int",
          "Beats per AXI burst at full bus width (AXI4 caps it at 256)",
          minimum=1,
          maximum=256),
    "axi_max_outstanding":
    _Spec("int",
          "Full-length bursts in flight per direction (Vitis caps it at 32)",
          minimum=1,
          maximum=32),
    "precision":
    _Spec("choice",
          "Data-path precision the kernel must carry: 'native', 'f32', "
          "'f64'",
          choices=("native", "f32", "f64")),
    "part":
    _Spec(
        "part", "Device part the generated Vitis scripts name (the memory "
        "budgets above must describe the same device)"),
    "clock_ns":
    _Spec("float",
          "Target clock period of the generated Vitis scripts, in ns",
          minimum=1,
          maximum=1000),
    "fft_stage_group":
    _Spec("int",
          "Stockham stages sharing one scratch slot (0 = a slot per stage, "
          "null = the compiler decides)",
          minimum=0,
          maximum=64,
          nullable=True,
          advanced=True),
    "loop_tile_size":
    _Spec("int", "Loop tiling factor (1 disables tiling, null = the compiler "
          "decides)",
          minimum=1,
          maximum=1024,
          nullable=True,
          advanced=True),
    "interp_banded_gather":
    _Spec("bool", "Gather interpolation taps from an on-chip band (null = the "
          "compiler decides)",
          nullable=True,
          advanced=True),
    "reuse_buffer_min_elements":
    _Spec("int",
          "Buffers of at least this many elements may share an allocation "
          "(null = the compiler decides)",
          minimum=0,
          maximum=_UINT64_MAX,
          nullable=True,
          advanced=True),
    "recompute_min_elements":
    _Spec("int",
          "Producers of at least this many elements are recomputed rather "
          "than stored (null = the compiler decides)",
          minimum=0,
          maximum=_UINT64_MAX,
          nullable=True,
          advanced=True),
    "external_buffer_threshold":
    _Spec("int", "Buffers of at least this many elements go off chip "
          "(null = the compiler decides)",
          minimum=0,
          maximum=_UINT32_MAX,
          nullable=True,
          advanced=True),
    "lutram_max_bytes":
    _Spec("int", "Bank size, in bytes, below which distributed RAM is used "
          "(null = the compiler decides)",
          minimum=0,
          maximum=_UINT32_MAX,
          nullable=True,
          advanced=True),
    "top_func":
    _Spec("identifier",
          "Name of the emitted top function (null = the kernel's name)",
          nullable=True),
}

#: Deprecated boolean spellings accepted alongside the schema keys:
#: alias -> (target key, value per flag). The single source for both the
#: rewriting in `_apply_aliases` and the unknown-option suggestions.
_ALIASES: Dict[str, Tuple[str, Dict[bool, str]]] = {
    "axi_interface": ("interface", {
        True: "axi",
        False: "bram"
    }),
}

#: Not a setting but a pointer to one more file of them.
_CONFIG_KEY = "config"


def shipped_config_path() -> Path:
    """The `hls_config.yaml` installed next to this module."""
    return Path(__file__).with_name(_CONFIG_FILENAME)


# ---------------------------------------------------------------------- #
# The YAML subset the config file is written in
# ---------------------------------------------------------------------- #

_NULL_TOKENS = {"", "null", "~"}
_TRUE_TOKENS = {"true", "yes", "on"}
_FALSE_TOKENS = {"false", "no", "off"}

_KEY_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_PART_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-]*")
_INT_RE = re.compile(r"[+-]?\d[\d_]*")
_BASED_INT_RE = re.compile(r"[+-]?0[xXbBoO][0-9a-fA-F_]+")
_FLOAT_RE = re.compile(r"[+-]?(\d[\d_]*\.?\d*|\.\d+)([eE][+-]?\d+)?")


def _strip_comment(line: str) -> str:
    """Drops a `#` comment that starts outside a quoted scalar."""
    out = []
    quote = None
    for i, char in enumerate(line):
        if quote is not None:
            out.append(char)
            if char == quote:
                quote = None
        elif char in "\"'":
            quote = char
            out.append(char)
        elif char == "#" and (i == 0 or line[i - 1].isspace()):
            break
        else:
            out.append(char)
    return "".join(out)


def _parse_scalar(token: str):
    token = token.strip()
    if len(token) >= 2 and token[0] in "\"'" and token[-1] == token[0]:
        return token[1:-1]
    lowered = token.lower()
    if lowered in _NULL_TOKENS:
        return None
    if lowered in _TRUE_TOKENS:
        return True
    if lowered in _FALSE_TOKENS:
        return False
    if _INT_RE.fullmatch(token):
        return int(token.replace("_", ""))
    if _BASED_INT_RE.fullmatch(token):
        return int(token.replace("_", ""), 0)
    if _FLOAT_RE.fullmatch(token):
        return float(token.replace("_", ""))
    return token


def _parse_yaml(text: str, source: str) -> Dict[str, object]:
    """Reads the flat `key: scalar` mapping the HLS config is written in.

    Kept in-tree rather than depending on PyYAML: nothing else in the
    repository parses YAML, and the schema is one level of scalars.
    """
    values: Dict[str, object] = {}
    for lineno, raw in enumerate(text.splitlines(), 1):
        line = _strip_comment(raw).rstrip()
        if not line.strip():
            continue
        where = f"{source}:{lineno}"
        if line[0].isspace():
            raise HLSConfigError(
                f"{where}: indented line {raw.strip()!r}. The HLS "
                "configuration is a flat 'key: value' mapping -- nested "
                "mappings, lists and anchors are not supported")
        key, separator, value = line.partition(":")
        key = key.strip()
        if not separator or not _KEY_RE.fullmatch(key):
            raise HLSConfigError(
                f"{where}: expected 'key: value', got {raw.strip()!r}")
        if key in values:
            raise HLSConfigError(f"{where}: duplicate key {key!r}")
        values[key] = _parse_scalar(value)
    return values


# ---------------------------------------------------------------------- #
# Validation
# ---------------------------------------------------------------------- #


def _known_options() -> str:
    lines = [
        f"  {name:<26}{spec.doc}" for name, spec in OPTIONS.items()
        if not spec.advanced
    ]
    lines.append(f"  {_CONFIG_KEY:<26}"
                 "Path to a YAML file overriding the shipped defaults")
    advanced = [name for name, spec in OPTIONS.items() if spec.advanced]
    if advanced:
        lines.append("")
        lines.append("The compiler derives these; pin one only to diagnose "
                     "its effect:")
        lines.append("  " + ", ".join(advanced))
    return "\n".join(lines)


def _unknown_option(key: str, origin: str) -> HLSConfigError:
    close = difflib.get_close_matches(key, list(OPTIONS) + list(_ALIASES), 1)
    hint = f" (did you mean {close[0]!r}?)" if close else ""
    return HLSConfigError(f"unknown HLS option {key!r} in {origin}{hint}. "
                          f"Valid options are:\n{_known_options()}")


def _bad_value(key: str, origin: str, detail: str, value) -> HLSConfigError:
    return HLSConfigError(
        f"HLS option {key!r} in {origin}: {detail}, got {value!r}")


def _validate(key: str, value, origin: str):
    """Checks one option against the schema, returning the stored value."""
    spec = OPTIONS.get(key)
    if spec is None:
        raise _unknown_option(key, origin)

    if value is None:
        if spec.nullable:
            return None
        raise _bad_value(key, origin, "null is not allowed here", value)

    if spec.kind == "bool":
        if not isinstance(value, bool):
            raise _bad_value(key, origin, "expected true or false", value)
        return value

    if spec.kind == "choice":
        rejected = spec.rejected or {}
        if isinstance(value, str) and value in rejected:
            raise _bad_value(key, origin, rejected[value], value)
        # A deprecated spelling is accepted and stored canonically, so the
        # rest of the backend only ever sees one name per value.
        if isinstance(value, str) and value in (spec.aliases or {}):
            value = spec.aliases[value]
        if value not in spec.choices:
            raise _bad_value(key, origin,
                             f"expected one of {list(spec.choices)}", value)
        return value

    if spec.kind == "identifier":
        if not isinstance(value, str) or not _KEY_RE.fullmatch(value):
            raise _bad_value(
                key, origin, "expected a C identifier (the name goes into "
                "the emitted C++)", value)
        return value

    if spec.kind == "part":
        if not isinstance(value, str) or not _PART_RE.fullmatch(value):
            raise _bad_value(
                key, origin, "expected a Vitis part name such as "
                "'xcvu13p-fhgb2104-2-e' (the name goes into a Tcl "
                "set_part command)", value)
        return value

    if spec.kind == "float":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise _bad_value(key, origin, "expected a number", value)
        value = float(value)
        if spec.minimum is not None and value < spec.minimum:
            raise _bad_value(key, origin, f"must be at least {spec.minimum}",
                             value)
        if spec.maximum is not None and value > spec.maximum:
            raise _bad_value(key, origin, f"must be at most {spec.maximum}",
                             value)
        return value

    # bool is an int in Python, and a bool here is always a mistake.
    if isinstance(value, bool) or not isinstance(value, int):
        raise _bad_value(key, origin, "expected an integer", value)
    if spec.minimum is not None and value < spec.minimum:
        raise _bad_value(key, origin, f"must be at least {spec.minimum}",
                         value)
    if spec.maximum is not None and value > spec.maximum:
        raise _bad_value(key, origin, f"must be at most {spec.maximum}", value)
    if spec.power_of_two and value & (value - 1):
        raise _bad_value(key, origin, "must be a power of two", value)
    return value


def _read_file(path: Path, origin: str) -> Dict[str, object]:
    try:
        text = path.read_text()
    except OSError as exc:
        raise HLSConfigError(
            f"HLS configuration file {str(path)!r} ({origin}) cannot be "
            f"read: {exc}") from None
    parsed = _parse_yaml(text, str(path))
    return {
        key: _validate(key, value, str(path))
        for key, value in parsed.items()
    }


def _user_config(options: Dict[str, object]) -> Tuple[Optional[Path], str]:
    """The user's own config file, from the option or the environment."""
    given = options.pop(_CONFIG_KEY, None)
    if given is not None:
        if not isinstance(given, (str, Path)):
            raise _bad_value(_CONFIG_KEY, "compile options", "expected a path",
                             given)
        return Path(given), "options={'config': ...}"
    from_env = os.environ.get(CONFIG_ENV_VAR)
    if from_env:
        return Path(from_env), f"${CONFIG_ENV_VAR}"
    return None, ""


def _apply_aliases(options: Dict[str, object]) -> None:
    for alias, (target, by_flag) in _ALIASES.items():
        if alias not in options:
            continue
        flag = options.pop(alias)
        if not isinstance(flag, bool):
            raise _bad_value(alias, "compile options",
                             "expected true or false", flag)
        kind = by_flag[flag]
        existing = options.setdefault(target, kind)
        # After alias normalisation `bram` and `ap_memory` are the same
        # interface, so compare the canonical forms rather than the raw
        # strings a user might have written in either spelling.
        canonical = OPTIONS[target].aliases or {}
        if (canonical.get(str(existing), str(existing))
                != canonical.get(kind, kind)):
            raise HLSConfigError(
                f"conflicting interface options: {alias}={flag} says "
                f"{kind!r} but {target}={existing!r} was also given")


# ---------------------------------------------------------------------- #
# The resolved configuration
# ---------------------------------------------------------------------- #


class HLSConfig(Mapping):
    """The options a design was compiled with, after defaults and files.

    A read-only mapping (`config["axi_bus_bits"]`, `dict(config)`) that
    also answers to attribute access (`config.axi_bus_bits`).

    The strategy keys (`autotune.AUTO_OPTIONS`) come out of `resolve` as
    None -- the compiler has not seen the kernel yet -- and are filled by
    `adopt` while it compiles. `provenance` says which of the four
    origins each value came from, so a design can be asked not just what
    it was built with but who decided it.
    """

    #: Provenance labels. The file origins are the paths themselves.
    FROM_OPTIONS = "compile options"
    DERIVED = "derived"

    def __init__(self,
                 values: Dict[str, object],
                 sources: Sequence[str],
                 provenance: Optional[Dict[str, str]] = None):
        self._values = dict(values)
        self._provenance = dict(provenance or {})
        self.sources = tuple(sources)

    @classmethod
    def resolve(cls, options=None) -> "HLSConfig":
        options = dict(options or {})
        path, origin = _user_config(options)
        _apply_aliases(options)

        shipped = shipped_config_path()
        values = _read_file(shipped, "shipped defaults")
        provenance = {key: str(shipped) for key in values}
        sources = [str(shipped)]
        missing = {
            key
            for key in set(OPTIONS) - set(values) if not OPTIONS[key].advanced
        }
        if missing:
            raise HLSConfigError(f"{shipped} has no default for "
                                 f"{sorted(missing)}")
        # Strategy keys are absent from the file by design: unset is what
        # tells the backend to derive them.
        for key, spec in OPTIONS.items():
            if spec.advanced and key not in values:
                values[key] = None
                provenance[key] = cls.DERIVED

        if path is not None:
            from_file = _read_file(path, origin)
            values.update(from_file)
            provenance.update({key: str(path) for key in from_file})
            sources.append(str(path))
        for key, value in options.items():
            values[key] = _validate(key, value, "compile options")
            provenance[key] = cls.FROM_OPTIONS
        if options:
            sources.append(cls.FROM_OPTIONS)

        if values["on_chip_budget"] is None:
            values["on_chip_budget"] = (values["bram_bytes"] +
                                        values["uram_bytes"] +
                                        values["lutram_bytes"])
            provenance["on_chip_budget"] = cls.DERIVED
        return cls(values, sources, provenance)

    # -- Derived values ------------------------------------------------ #

    def adopt(self, derived: Mapping) -> None:
        """Fills the keys still at null with what the compiler decided.

        A value the user pinned wins: `derive` computes what it would have
        chosen for every key, and this is where the user's choice is
        allowed to stand instead.
        """
        for key, value in derived.items():
            if self._values.get(key) is not None:
                continue
            self._values[key] = _validate(key, value, self.DERIVED)
            self._provenance[key] = self.DERIVED
        if self.DERIVED in self._provenance.values():
            if self.DERIVED not in self.sources:
                self.sources = self.sources + (self.DERIVED, )

    @property
    def provenance(self) -> Dict[str, str]:
        """Where each value came from: a config path, `compile options`
        or `derived`."""
        return dict(self._provenance)

    # -- Mapping ------------------------------------------------------- #

    def __getitem__(self, key):
        return self._values[key]

    def __iter__(self):
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __getattr__(self, name):
        values = self.__dict__.get("_values", {})
        if name in values:
            return values[name]
        raise AttributeError(name)

    def as_dict(self) -> Dict[str, object]:
        return dict(self._values)

    def __repr__(self) -> str:
        body = ", ".join(f"{k}={v!r}" for k, v in self._values.items())
        return f"HLSConfig({body})"


def check_precision(config: HLSConfig, arg_types, result_types) -> None:
    """Enforces the `precision` policy on a kernel's planes.

    The passes have no precision knob -- the declared dtypes are the data
    path -- so the policy is a gate rather than a transformation: an f64
    plane reaching a design budgeted for f32 costs several times the DSPs
    and block RAM per operator, and nothing downstream reports it.
    """
    policy = config.precision
    if policy == "native":
        return
    for kind, types in (("argument", arg_types), ("result", result_types)):
        for index, ty in enumerate(types):
            if not (ty.dtype.is_float or ty.dtype.is_complex):
                continue
            width = "f64" if ty.dtype.name in ("f64", "c128") else "f32"
            if width != policy:
                raise HLSConfigError(
                    f"precision={policy!r} but {kind} {index} is "
                    f"sar.{ty.dtype.name}, whose data path is {width}; "
                    f"cast the kernel to {policy} or set "
                    "precision='native'")
