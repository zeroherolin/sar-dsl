"""HLS backend: emits Vitis HLS C++ for FPGA synthesis.

A kernel is decomplexified (complex tensors become re/im float planes),
FFTs become Stockham loop nests, interpolation becomes windowed-sinc
gather loops (`sar-to-affine-pipeline`), and the resulting affine IR
enters the HLS pipeline (`-hls-pipeline`), which builds the dataflow
hierarchy, places buffers on or off chip, and shapes the interfaces.
`sar-translate` then writes the C++.

The compilation result is an `HLSDesign` handle pointing at the emitted
C++ source, not an executable kernel. The generated top function takes
each complex tensor as two adjacent float arrays (re, im), inputs
first, then one output array per result plane;
`HLSDesign.write_testbench` emits a matching simulation testbench
with golden data.

Every SAR operation has an HLS lowering, so the omega-K, range-Doppler,
chirp-scaling, and polar-format chains emit as single designs.

Options are validated against the schema in `config.py` and default from
the shipped `hls_config.yaml` (see `docs/backends.md`). The strategy half
of the schema is derived rather than defaulted -- see `autotune.py` -- and
the resolved set, with where each value came from, is reported by
`HLSDesign.config`.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from ..base import BaseBackend, KernelMetadata, cached_stage
from . import autotune
from .config import (HLSConfig, HLSConfigError, check_precision,
                     normalize_hls_top_identifier)
from .design import HLSDesign
from .devices import bram18_count, uram_count
from ...compiler.toolchain import find_tool, run_tool
from ...errors import CompilationError, LaunchError, ToolchainError

_C_TYPES = {
    "f32": "float",
    "f64": "double",
    "i32": "ap_int<32>",
    "i64": "ap_int<64>"
}
_RETRYABLE_MEMORY_OVERFLOW = "SAR_HLS_RETRYABLE_MEMORY_OVERFLOW"
_RETRYABLE_PARTITION_OVERFLOW = "SAR_HLS_RETRYABLE_PARTITION_OVERFLOW"


def _top_func(config, metadata: KernelMetadata) -> str:
    """Name the emitted top function carries."""
    return normalize_hls_top_identifier(config.top_func or metadata.name)


def _rename_symbol(module_text: str, old: str, new: str) -> str:
    if old == new:
        return module_text
    return re.sub(rf"@{re.escape(old)}\b", f"@{new}", module_text)


def _split_planes(kind: str, types) -> list:
    """(name, shape, dtype-name) per top-function port: complex tensors
    contribute adjacent (re, im) float planes."""
    ports = []
    for i, t in enumerate(types):
        if t.dtype.is_complex:
            plane = "f32" if t.dtype.name == "c64" else "f64"
            ports.append((f"{kind}{i}_re", t.shape, plane))
            ports.append((f"{kind}{i}_im", t.shape, plane))
        else:
            ports.append((f"{kind}{i}", t.shape, t.dtype.name))
    return ports


def _plane_data(arrays, types):
    """Float planes (in port order) for numpy arrays of the given types.

    The arrays must carry the kernel's declared dtypes exactly: a silent
    cast here would write golden data at a precision the design never
    computes, and the testbench would then diff against it."""
    if len(arrays) != len(types):
        raise LaunchError(f"expected {len(types)} array(s), got {len(arrays)}")
    for i, (arr, t) in enumerate(zip(arrays, types)):
        arr = np.asarray(arr)
        if tuple(arr.shape) != t.shape:
            raise LaunchError(
                f"array shape {arr.shape} does not match {t.shape}")
        expected = np.dtype(t.dtype.to_numpy())
        if arr.dtype != expected:
            raise LaunchError(
                f"array #{i} has dtype {arr.dtype}, but the kernel declares "
                f"sar.{t.dtype.name} ({expected}); cast the array rather "
                "than let the testbench data silently change precision")
        if t.dtype.is_complex:
            plane = np.float32 if t.dtype.name == "c64" else np.float64
            yield np.ascontiguousarray(arr.real, dtype=plane)
            yield np.ascontiguousarray(arr.imag, dtype=plane)
        else:
            yield np.ascontiguousarray(arr)


def _golden_plane_data(arrays, types):
    """Reference planes kept at their supplied precision.

    C-simulation compares generated float or double outputs in double
    precision. Keeping an f64 oracle intact is what lets an f32 design report
    its actual quantization error rather than its error against an already
    rounded reference. Integer planes stay integral instead: they are checked
    for equality, and a 64-bit value does not survive a trip through double.
    """
    if len(arrays) != len(types):
        raise LaunchError(f"expected {len(types)} array(s), got {len(arrays)}")
    for i, (arr, t) in enumerate(zip(arrays, types)):
        arr = np.asarray(arr)
        if tuple(arr.shape) != t.shape:
            raise LaunchError(
                f"array shape {arr.shape} does not match {t.shape}")
        if t.dtype.is_int:
            if not np.issubdtype(arr.dtype, np.integer):
                raise LaunchError(
                    f"golden array #{i} must have an integer dtype for "
                    f"sar.{t.dtype.name}, got {arr.dtype}")
            yield np.ascontiguousarray(arr, dtype=np.int64)
            continue
        if not (np.issubdtype(arr.dtype, np.floating)
                or np.issubdtype(arr.dtype, np.complexfloating)):
            raise LaunchError(
                f"golden array #{i} must have a real or complex floating "
                f"dtype, got {arr.dtype}")
        if t.dtype.is_complex:
            if not np.issubdtype(arr.dtype, np.complexfloating):
                raise LaunchError(
                    f"golden array #{i} must be complex for sar.{t.dtype.name}"
                )
            yield np.ascontiguousarray(arr.real, dtype=np.float64)
            yield np.ascontiguousarray(arr.imag, dtype=np.float64)
        else:
            if np.issubdtype(arr.dtype, np.complexfloating):
                raise LaunchError(
                    f"golden array #{i} must be real for sar.{t.dtype.name}")
            yield np.ascontiguousarray(arr, dtype=np.float64)


def _is_int_port(port) -> bool:
    """Whether a logical `(name, shape, dtype)` port carries integers."""
    return port[2] in ("i32", "i64")


def _storage_type(port) -> str:
    """C++ scalar type stored in one binary testbench data file."""
    return {
        "f32": "float",
        "f64": "double",
        "i32": "int32_t",
        "i64": "int64_t",
    }[port[2]]


def _c_decl(port) -> str:
    name, shape, dtype = port
    dims = "".join(f"[{d}]" for d in shape)
    return f"{_C_TYPES[dtype]} {name}{dims}"


def _testbench_helpers(stage_elements: int, float_gold: int, int_gold: int,
                       rtol, atol) -> str:
    """Loader and comparator templates the generated testbench needs.

    Floating planes are compared against a double oracle within `rtol`/`atol`;
    integer planes are loaded and compared as `int64_t`, because a 64-bit
    value neither survives a trip through double nor has a meaningful
    tolerance. Only the comparators a design's ports actually call are
    emitted -- the gold buffers are static arrays, and an empty one is not
    valid C++.
    """
    parts = [
        f"""\
template <typename Storage, typename T>
bool load(const std::string &dir, const std::string &name, T *dst,
          long count) {{
  std::string path = dir + "/" + name;
  std::ifstream f(path, std::ios::binary);
  if (!f) {{
    path = name;
    f.clear();
    f.open(path, std::ios::binary);
  }}
  if (!f) {{
    std::fprintf(stderr, "cannot open %s/%s or ./%s\\n", dir.c_str(),
                 name.c_str(), name.c_str());
    return false;
  }}
  constexpr long chunk_size = 65536;
  Storage buffer[chunk_size];
  for (long base = 0; base < count; base += chunk_size) {{
    long chunk = count - base < chunk_size ? count - base : chunk_size;
    f.read(reinterpret_cast<char *>(buffer), chunk * sizeof(Storage));
    if (f.gcount() != chunk * static_cast<long>(sizeof(Storage))) {{
      std::fprintf(stderr, "short read in %s\\n", path.c_str());
      return false;
    }}
    for (long i = 0; i < chunk; ++i)
      dst[base + i] = static_cast<T>(buffer[i]);
  }}
  return true;
}}

template <typename Storage, typename T, std::size_t N>
bool load_vector(const std::string &dir, const std::string &name,
                 hls::vector<T, N> *dst, long count) {{
  static T scalars[{stage_elements}];
  if (!load<Storage>(dir, name, scalars, count))
    return false;
  for (long i = 0; i < count; ++i)
    dst[i / N][i % N] = scalars[i];
  return true;
}}

template <typename Storage, typename T>
bool load_stream(const std::string &dir, const std::string &name,
                 hls::stream<T> &dst, long count) {{
  static T scalars[{stage_elements}];
  if (!load<Storage>(dir, name, scalars, count))
    return false;
  for (long i = 0; i < count; ++i)
    dst.write(scalars[i]);
  return true;
}}
"""
    ]
    if float_gold:
        parts.append(f"""\
template <typename T>
int check(const std::string &dir, const std::string &name, const char *port,
          const T *got, long count) {{
  static double gold[{float_gold}];
  if (!load<double>(dir, name, gold, count))
    return 2;
  double max_err = 0.0;
  long bad = 0;
  for (long i = 0; i < count; ++i) {{
    double actual = double(got[i]);
    bool actual_nan = std::isnan(actual);
    bool gold_nan = std::isnan(gold[i]);
    if (actual_nan || gold_nan) {{
      if (actual_nan != gold_nan) {{
        ++bad;
        max_err = std::numeric_limits<double>::infinity();
      }}
      continue;
    }}
    if (std::isinf(actual) || std::isinf(gold[i])) {{
      if (actual != gold[i]) {{
        ++bad;
        max_err = std::numeric_limits<double>::infinity();
      }}
      continue;
    }}
    double err = std::abs(actual - gold[i]);
    max_err = std::fmax(max_err, err);
    if (err > {atol} + {rtol} * std::abs(gold[i]))
      ++bad;
  }}
  std::printf("%-12s max |err| = %.3e  (%ld/%ld out of tolerance)\\n",
              port, max_err, bad, count);
  return bad != 0;
}}

template <typename T, std::size_t N>
int check_vector(const std::string &dir, const std::string &name,
                 const char *port, const hls::vector<T, N> *got, long count) {{
  static double gold[{float_gold}];
  if (!load<double>(dir, name, gold, count))
    return 2;
  double max_err = 0.0;
  long bad = 0;
  for (long i = 0; i < count; ++i) {{
    double actual = double(got[i / N][i % N]);
    double err = std::abs(actual - gold[i]);
    max_err = std::fmax(max_err, err);
    if ((std::isnan(actual) != std::isnan(gold[i])) ||
        (!std::isnan(actual) &&
         err > {atol} + {rtol} * std::abs(gold[i])))
      ++bad;
  }}
  std::printf("%-12s max |err| = %.3e  (%ld/%ld out of tolerance)\\n",
              port, max_err, bad, count);
  return bad != 0;
}}

template <typename T>
int check_stream(const std::string &dir, const std::string &name,
                 const char *port, hls::stream<T> &got, long count) {{
  static T values[{float_gold}];
  for (long i = 0; i < count; ++i)
    values[i] = got.read();
  return check(dir, name, port, values, count);
}}
""")
    if int_gold:
        parts.append(f"""\
template <typename T>
int check_int(const std::string &dir, const std::string &name,
              const char *port, const T *got, long count) {{
  static long long gold[{int_gold}];
  if (!load<int64_t>(dir, name, gold, count))
    return 2;
  long bad = 0;
  for (long i = 0; i < count; ++i)
    if (static_cast<long long>(got[i]) != gold[i])
      ++bad;
  std::printf("%-12s %ld/%ld mismatched\\n", port, bad, count);
  return bad != 0;
}}

template <typename T, std::size_t N>
int check_vector_int(const std::string &dir, const std::string &name,
                     const char *port, const hls::vector<T, N> *got,
                     long count) {{
  static long long gold[{int_gold}];
  if (!load<int64_t>(dir, name, gold, count))
    return 2;
  long bad = 0;
  for (long i = 0; i < count; ++i)
    if (static_cast<long long>(got[i / N][i % N]) != gold[i])
      ++bad;
  std::printf("%-12s %ld/%ld mismatched\\n", port, bad, count);
  return bad != 0;
}}

template <typename T>
int check_stream_int(const std::string &dir, const std::string &name,
                     const char *port, hls::stream<T> &got, long count) {{
  static T values[{int_gold}];
  for (long i = 0; i < count; ++i)
    values[i] = got.read();
  return check_int(dir, name, port, values, count);
}}
""")
    return "\n".join(parts)


def _testbench_source(top,
                      in_ports,
                      out_ports,
                      rtol,
                      atol,
                      physical_ports=None) -> str:
    """C-simulation testbench: loads binary data, runs the design,
    compares each output plane against the golden data."""
    if physical_ports is None:
        proto = ",\n    ".join(_c_decl(p) for p in in_ports + out_ports)
        decls = "\n".join(f"  static {_c_decl(p)};"
                          for p in in_ports + out_ports)
        loads = "\n".join(f'  if (!load<{_storage_type(p)}>(dir, '
                          f'"{p[0]}.bin", '
                          f'&{p[0]}{"[0]" * len(p[1])}, '
                          f"{int(np.prod(p[1]))})) return 2;"
                          for p in in_ports)
        call_args = ", ".join(p[0] for p in in_ports + out_ports)
        checks = "\n".join(
            f'  fail |= {"check_int" if _is_int_port(p) else "check"}'
            f'(dir, "{p[0]}.bin", "{p[0]}", '
            f'&{p[0]}{"[0]" * len(p[1])}, {int(np.prod(p[1]))});'
            for p in out_ports)
    else:

        def physical_decl(port, parameter=True):
            if port.get("protocol") == "axis":
                suffix = " &" if parameter else ""
                return (f'hls::stream<{port["c_type"]}>{suffix} '
                        f'{port["name"]}')
            dims = "".join(f"[{extent}]" for extent in port["physical_shape"])
            return f'{port["c_type"]} {port["name"]}{dims}'

        public_count = len(in_ports) + len(out_ports)
        if len(physical_ports) < public_count:
            raise LaunchError(
                f"generated top {top!r} has fewer ports than its public ABI")
        proto = ",\n    ".join(physical_decl(p) for p in physical_ports)
        decls = "\n".join(f"  static {physical_decl(p, parameter=False)};"
                          for p in physical_ports)
        input_physical = physical_ports[:len(in_ports)]
        output_physical = physical_ports[len(in_ports):public_count]

        def load_call(logical, physical):
            storage = _storage_type(logical)
            if physical.get("protocol") == "axis":
                return (f'  if (!load_stream<{storage}>(dir, '
                        f'"{logical[0]}.bin", '
                        f'{physical["name"]}, '
                        f"{int(np.prod(logical[1]))})) return 2;")
            loader = ("load_vector"
                      if physical["vector_lanes"] != 1 else "load")
            zeros = "[0]" * len(physical["physical_shape"])
            return (f'  if (!{loader}<{storage}>(dir, '
                    f'"{logical[0]}.bin", '
                    f'&{physical["name"]}{zeros}, '
                    f"{int(np.prod(logical[1]))})) return 2;")

        loads = "\n".join(
            load_call(logical, physical)
            for logical, physical in zip(in_ports, input_physical))
        call_args = ", ".join(p["name"] for p in physical_ports)

        def check_call(logical, physical):
            suffix = "_int" if _is_int_port(logical) else ""
            if physical.get("protocol") == "axis":
                return (f'  fail |= check_stream{suffix}'
                        f'(dir, "{logical[0]}.bin", '
                        f'"{physical["name"]}", {physical["name"]}, '
                        f"{int(np.prod(logical[1]))});")
            checker = ("check_vector"
                       if physical["vector_lanes"] != 1 else "check")
            return (f'  fail |= {checker}{suffix}(dir, "{logical[0]}.bin", '
                    f'"{physical["name"]}", &{physical["name"]}'
                    f'{"[0]" * len(physical["physical_shape"])}, '
                    f"{int(np.prod(logical[1]))});")

        checks = "\n".join(
            check_call(logical, physical)
            for logical, physical in zip(out_ports, output_physical))

    # Staging buffers are sized per kind, and the integer helpers are only
    # emitted when a port needs them: an empty static array is not valid
    # C++, and a design without integer ports should not carry the
    # exact-comparison machinery.
    stage_elements = max((int(np.prod(p[1])) for p in in_ports), default=1)
    float_gold = max(
        (int(np.prod(p[1])) for p in out_ports if not _is_int_port(p)),
        default=0)
    int_gold = max((int(np.prod(p[1])) for p in out_ports if _is_int_port(p)),
                   default=0)
    helpers = _testbench_helpers(max(1, stage_elements), float_gold, int_gold,
                                 rtol, atol)
    return f"""\
//===- SAR-DSL generated C-sim testbench ---------------*- C++ -*-===//
//
// Generated by SAR-DSL HLS artifact generation.
// Regenerate this file instead of editing it.
//
//===----------------------------------------------------------------------===//
//
// C-sim through Vitis HLS: vitis_hls -f {top}_hls_csim.tcl
// Portable C++ fallback:  sh {top}_portable_cpp_sim.sh
//
// The design runs on a dedicated 1 GiB stack: during simulation the on-chip
// buffers are stack arrays, far beyond the default thread stack.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <pthread.h>
#include <string>
#include <ap_int.h>
#include <hls_vector.h>
#include <hls_stream.h>

void {top}(
    {proto});

namespace {{

// Vitis HLS may copy testbench data flat into its simulation working
// directory; fall back from `dir`/`name` to the bare name.
{helpers}
int run(const std::string &dir) {{
{decls}

{loads}

  {top}({call_args});

  int fail = 0;
{checks}

  std::printf("%s\\n", fail ? "FAIL" : "PASS");
  return fail;
}}

}} // namespace

int main(int argc, char **argv) {{
  const std::string dir = argc > 1 ? argv[1] : "{top}_tb_data";
  auto trampoline = [](void *arg) -> void * {{
    auto rc = new int(run(*static_cast<const std::string *>(arg)));
    return rc;
  }};
  pthread_attr_t attr;
  if (pthread_attr_init(&attr) != 0 ||
      pthread_attr_setstacksize(&attr, 1ull << 30) != 0) {{
    std::fprintf(stderr, "cannot configure the simulation worker stack\\n");
    return 3;
  }}
  pthread_t worker;
  if (pthread_create(&worker, &attr, trampoline, (void *)&dir) != 0) {{
    pthread_attr_destroy(&attr);
    std::fprintf(stderr, "cannot start the simulation worker thread\\n");
    return 3;
  }}
  pthread_attr_destroy(&attr);
  void *result = nullptr;
  if (pthread_join(worker, &result) != 0 || result == nullptr) {{
    std::fprintf(stderr, "cannot join the simulation worker thread\\n");
    return 3;
  }}
  int fail = *static_cast<int *>(result);
  delete static_cast<int *>(result);
  return fail;
}}
"""


_STUB_HEADER = """\
//===- SAR-DSL generated Vitis HLS compatibility stub ----------*- C++ -*-===//
//
// Generated by SAR-DSL HLS artifact generation.
// Regenerate this file instead of editing it.
//
//===----------------------------------------------------------------------===//

#pragma once
"""

_AP_INT_STUB = _STUB_HEADER + """\
// Stand-ins for the integer widths emitted by SAR-DSL. They preserve
// truncation and sign extension for widths up to the host storage type.
template <int W> struct ap_int {
  long long v;
  ap_int() : v(0) {}
  ap_int(long long x) : v(normalize(x)) {}
  operator long long() const { return v; }
private:
  static long long normalize(unsigned long long x) {
    static_assert(W > 0 && W <= 64, "stub supports widths 1..64");
    if constexpr (W == 64) {
      return static_cast<long long>(x);
    } else {
      const unsigned long long mask = (1ull << W) - 1;
      x &= mask;
      if (x & (1ull << (W - 1)))
        x |= ~mask;
      return static_cast<long long>(x);
    }
  }
};
template <int W> struct ap_uint {
  unsigned long long v;
  ap_uint() : v(0) {}
  ap_uint(unsigned long long x) : v(normalize(x)) {}
  operator unsigned long long() const { return v; }
private:
  static unsigned long long normalize(unsigned long long x) {
    static_assert(W > 0 && W <= 64, "stub supports widths 1..64");
    if constexpr (W == 64) {
      return x;
    } else {
      return x & ((1ull << W) - 1);
    }
  }
};
"""

#: `hls::stream` stand-in. A design with dataflow channels includes this
#: header, so an empty stub would make the package uncompilable outside
#: Vitis -- which is the whole point of shipping stubs. Sequential simulation
#: never blocks, so an unbounded queue is behaviourally exact here.
_HLS_STREAM_STUB = _STUB_HEADER + """\
#include <cassert>
#include <cstddef>
#include <queue>

namespace hls {
template <typename T> class stream {
public:
  stream() = default;
  explicit stream(const char *) {}

  void write(const T &value) { q.push(value); }
  bool write_nb(const T &value) { q.push(value); return true; }

  T read() {
    assert(!q.empty() && "read from an empty hls::stream");
    T value = q.front();
    q.pop();
    return value;
  }
  bool read_nb(T &value) {
    if (q.empty())
      return false;
    value = read();
    return true;
  }
  void read(T &value) { value = read(); }

  bool empty() const { return q.empty(); }
  bool full() const { return false; }
  std::size_t size() const { return q.size(); }

  stream &operator<<(const T &value) { write(value); return *this; }
  stream &operator>>(T &value) { value = read(); return *this; }

private:
  std::queue<T> q;
};
} // namespace hls
"""

_HLS_VECTOR_STUB = _STUB_HEADER + """\
#include <cstddef>

namespace hls {
template <typename T, std::size_t N> struct vector {
  T data[N] = {};
  T &operator[](std::size_t index) { return data[index]; }
  const T &operator[](std::size_t index) const { return data[index]; }
};
} // namespace hls
"""


def _write_header_stubs(stub_dir: Path) -> None:
    """Vitis header stand-ins so the package csims with any compiler."""
    stub_dir.mkdir(parents=True, exist_ok=True)
    (stub_dir / "ap_int.h").write_text(_AP_INT_STUB)
    (stub_dir / "hls_stream.h").write_text(_HLS_STREAM_STUB)
    (stub_dir / "hls_vector.h").write_text(_HLS_VECTOR_STUB)
    for header in ("ap_axi_sdata.h", "ap_fixed.h", "hls_math.h"):
        (stub_dir / header).write_text(_STUB_HEADER + "#include <cmath>\n")


def _part_and_clock(config) -> tuple:
    """(part, clock period) the generated Vitis scripts name."""
    if config is None:
        return "xcvu13p-fhgb2104-2-i", 4.0
    return config.part, float(config.clock_ns)


def _hls_csim_script(top, part, clock_ns) -> str:
    """C-sim script for Vitis HLS 2022.2."""
    return f"""\
# C-sim for `{top}` through Vitis HLS:
#   vitis_hls -f {top}_hls_csim.tcl
open_project -reset {top}_hls_csim_proj
set_top {top}
add_files {top}.cpp
add_files -tb {top}_tb.cpp
add_files -tb {top}_tb_data
open_solution -reset sol1 -flow_target vivado
set_part {part}
create_clock -period {clock_ns:g}
csim_design -ldflags {{-lpthread}}
exit
"""


def _portable_cpp_sim_script(top) -> str:
    """Portable C++ fallback used only when Vitis HLS is unavailable."""
    return f"""\
#!/usr/bin/env sh
set -eu

# Portable functional simulation; this does not invoke Vitis HLS.
"${{CXX:-c++}}" -O2 -Wno-unknown-pragmas -I stubs \
  {top}.cpp {top}_tb.cpp -o portable_cpp_sim -pthread
./portable_cpp_sim {top}_tb_data
"""


def _csynth_script(top, part, clock_ns, config=None) -> str:
    """Vitis HLS synthesis script (written against Vitis HLS 2022.2).

    Synthesis supplies what C simulation cannot: achieved II, timing at the
    target clock, and measured BRAM/URAM/DSP against the config
    budgets. The checklist for reading the reports is in
    docs/backends.md ("Validating a design in Vitis HLS").

    The two kinds of constraint are enforced differently, because they fail
    differently. A design over its resource budget cannot be placed on the
    device at all, so the script exits nonzero. A design that misses the
    clock still exists: the estimate is pre-route and the period is a
    target the compiler optimizes toward, so a miss is reported and left
    for the user to judge against their own margin.
    """
    bus_bits = int(config.axi_bus_bits) if config is not None else 512
    burst = int(config.axi_max_burst_length) if config is not None else 64
    outstanding = int(config.axi_max_outstanding) if config is not None else 16
    uncertainty = (float(config.clock_uncertainty_percent)
                   if config is not None else 12.5)
    timing_budget = clock_ns * (1.0 - uncertainty / 100.0)
    beat_bytes = max(1, bus_bits // 8)
    burst = min(256, burst, max(1, 4096 // beat_bytes))
    if config is None:
        limits = {}
    else:
        limits = {
            "BRAM_18K": bram18_count(config.bram_bytes),
            "URAM": uram_count(config.uram_bytes, config.part),
            "DSP": int(config.dsp),
            "FF": int(config.ff),
            "LUT": int(config.lut),
        }
    resource_limits = " ".join(f"{name} {limit}"
                               for name, limit in limits.items())
    return f"""\
# Vitis HLS synthesis for `{top}`: vitis_hls -f {top}_csynth.tcl
#
# Reports land in {top}_csynth_proj/sol1/syn/report/; start with
# {top}_csynth.rpt (timing, latency, utilization). See
# docs/backends.md for the validation checklist.
open_project -reset {top}_csynth_proj
set_top {top}
add_files {top}.cpp
open_solution -reset sol1 -flow_target vivado
set_part {part}
create_clock -period {clock_ns:g}
set_clock_uncertainty {uncertainty:g}%
config_interface -m_axi_addr64=true
config_interface -m_axi_alignment_byte_size={beat_bytes}
config_interface -m_axi_max_widen_bitwidth={bus_bits}
config_interface -m_axi_max_read_burst_length={burst}
config_interface -m_axi_max_write_burst_length={burst}
config_interface -m_axi_num_read_outstanding={outstanding}
config_interface -m_axi_num_write_outstanding={outstanding}
set _sar_started_ms [clock milliseconds]
csynth_design
set _sar_finished_ms [clock milliseconds]
set _sar_elapsed_s [expr {{($_sar_finished_ms - $_sar_started_ms) / 1000.0}}]
set _sar_elapsed_file [open "{top}_csynth_elapsed_s.txt" w]
puts $_sar_elapsed_file $_sar_elapsed_s
close $_sar_elapsed_file
set _sar_report_path "{top}_csynth_proj/sol1/syn/report/{top}_csynth.xml"
if {{![file exists $_sar_report_path]}} {{
  error "SAR-DSL: synthesis did not produce $_sar_report_path"
}}
set _sar_report_file [open $_sar_report_path r]
set _sar_report_xml [read $_sar_report_file]
close $_sar_report_file
proc _sar_xml_value {{xml tag}} {{
  set pattern [format {{<%s>([^<]+)</%s>}} $tag $tag]
  if {{![regexp -- $pattern $xml _ value]}} {{
    error "SAR-DSL: synthesis report is missing $tag"
  }}
  return $value
}}
set _sar_estimated_clock [expr {{double([_sar_xml_value \
    $_sar_report_xml EstimatedClockPeriod])}}]
if {{$_sar_estimated_clock > {timing_budget:g}}} {{
  set _sar_estimated_clock_text [format "%.3f" $_sar_estimated_clock]
  puts "SAR-DSL WARNING: estimated clock $_sar_estimated_clock_text ns misses \
the effective {timing_budget:g} ns scheduling budget ({clock_ns:g} ns target, \
{uncertainty:g}% uncertainty); this is a pre-route timing estimate, not a \
resource overflow, so synthesis continues"
}}
if {{![regexp {{(?s)<Resources>(.*?)</Resources>}} \
          $_sar_report_xml _ _sar_resources]}} {{
  error "SAR-DSL: synthesis report is missing resource estimates"
}}
foreach {{_sar_name _sar_limit}} {{{resource_limits}}} {{
  set _sar_used [expr {{int([_sar_xml_value $_sar_resources $_sar_name])}}]
  if {{$_sar_used > $_sar_limit}} {{
    error "SAR-DSL: $_sar_name usage $_sar_used exceeds budget $_sar_limit"
  }}
}}
exit
"""


def _cosim_script(top, part, clock_ns, config=None) -> str:
    """Vitis HLS Verilog co-simulation for a generated testbench package."""
    bus_bits = int(config.axi_bus_bits) if config is not None else 512
    burst = int(config.axi_max_burst_length) if config is not None else 64
    outstanding = int(config.axi_max_outstanding) if config is not None else 16
    uncertainty = (float(config.clock_uncertainty_percent)
                   if config is not None else 12.5)
    beat_bytes = max(1, bus_bits // 8)
    burst = min(256, burst, max(1, 4096 // beat_bytes))
    return f"""\
# Vitis HLS RTL co-simulation for `{top}`:
#   vitis_hls -f {top}_cosim.tcl
open_project -reset {top}_cosim_proj
set_top {top}
add_files {top}.cpp
add_files -tb {top}_tb.cpp
add_files -tb {top}_tb_data
open_solution -reset sol1 -flow_target vivado
set_part {part}
create_clock -period {clock_ns:g}
set_clock_uncertainty {uncertainty:g}%
config_interface -m_axi_addr64=true
config_interface -m_axi_alignment_byte_size={beat_bytes}
config_interface -m_axi_max_widen_bitwidth={bus_bits}
config_interface -m_axi_max_read_burst_length={burst}
config_interface -m_axi_max_write_burst_length={burst}
config_interface -m_axi_num_read_outstanding={outstanding}
config_interface -m_axi_num_write_outstanding={outstanding}
csynth_design
cosim_design -rtl verilog -trace_level none -ldflags {{-lpthread}}
exit
"""


class Backend(BaseBackend):
    """Vitis HLS backend: MLIR to a synthesizable C++ design package.

    The compilation result is a design handle rather than a callable, so
    `make_launcher` returns an `HLSDesign` instead of an executable kernel.
    """

    name = "hls"

    @classmethod
    def is_available(cls) -> bool:
        """Whether the compiler tools this backend drives can be found."""
        try:
            find_tool("sar-opt")
            find_tool("sar-translate")
            return True
        except ToolchainError:
            return False

    # ------------------------------------------------------------------ #
    # Stages
    # ------------------------------------------------------------------ #

    def add_stages(self, stages, metadata: KernelMetadata) -> None:
        """Resolves the configuration and registers the lower/emit stages."""
        requested_options = dict(metadata.options)
        config = HLSConfig.resolve(metadata.options)
        check_precision(config, metadata.arg_types, metadata.result_types)
        metadata.extra["hls_config"] = config
        metadata.extra["hls_requested_options"] = requested_options
        # The artifact cache keys on `options`, so the resolved values --
        # not the handful the user happened to pass -- are what has to be
        # in it: a design compiled against a different config file is a
        # different design.
        metadata.options.clear()
        metadata.options.update(config.as_dict())

        stages["lower"] = self._stage_lower
        stages["hls"] = self._stage_hls_cpp

    @staticmethod
    def _stage_lower(module_text: str, metadata: KernelMetadata, cache) -> str:
        config = metadata.extra["hls_config"]
        # Measured before the cache is consulted: a design served from the
        # cache still has to report the settings it was built with.
        facts = autotune.measure_kernel(module_text)
        metadata.extra["hls_facts"] = facts
        performance_plan = autotune.plan(config, facts)
        metadata.extra["hls_performance_plan"] = performance_plan
        config.adopt(performance_plan.values)

        def lower(reuse_min_elements: int, io_unroll: int) -> str:
            # `top_func` renames the kernel's symbol here rather than at
            # the pipeline's `top-func` option alone: that option selects
            # a function by name, so a name the module does not carry
            # finds nothing.
            renamed = _rename_symbol(module_text, metadata.name,
                                     _top_func(config, metadata))
            staging = autotune.transpose_block_bytes(facts,
                                                     int(config.bram_bytes))
            banded = "true" if config.interp_banded_gather else "false"
            fuse_sweeps = "true" if config.fuse_sibling_sweeps else "false"
            pipeline = ("--sar-to-affine-pipeline="
                        f"reuse-buffer-min-elements={reuse_min_elements} "
                        f"recompute-min-elements="
                        f"{int(config.recompute_min_elements)} "
                        f"transpose-block-bytes={staging} "
                        f"interp-enable-banded-gather={banded} "
                        f"interp-full-row-max-bytes="
                        f"{int(config.interp_full_row_max_bytes)} "
                        f"interp-cache-copies="
                        f"{int(config.interp_cache_copies)} "
                        f"interp-complete-bank-max-elements="
                        f"{int(config.interp_complete_bank_max_elements)} "
                        f"fft-stage-group={int(config.fft_stage_group)} "
                        f"fft-parallel-rows={int(config.fft_parallel_rows)} "
                        f"fft-io-unroll={io_unroll} "
                        f"fuse-sibling-sweeps={fuse_sweeps}")
            command = [find_tool("sar-opt")]
            if config.precision != "native":
                command.append(
                    f"--sar-verify-precision=precision={config.precision}")
            command.extend([pipeline, "-"])
            return run_tool("sar-lower", command, input_text=renamed)

        def build() -> str:
            cache.write_text("kernel.sar.mlir", module_text)
            io_unroll = int(config.fft_io_unroll)
            # A pinned threshold is the user's decision; lower once with it.
            if config.provenance.get(
                    "reuse_buffer_min_elements") != config.DERIVED:
                return lower(int(config.reuse_buffer_min_elements), io_unroll)
            # Whether sharing pays depends on what the unshared design
            # costs, and that needs the lowered allocations. Lower once
            # sharing nothing, ask the placement rule whether that form keeps
            # its planes on chip, and lower again only when it does not.
            private = lower(autotune.NEVER_SHARE, io_unroll)
            if not autotune.should_share_buffers(
                    config, facts, metadata, autotune.measure_kernel(private)):
                return private
            decided = autotune.reuse_min_elements(facts)
            cache.write_text("kernel.reuse.txt", str(decided))
            return lower(decided, io_unroll)

        affine = cached_stage(cache, "kernel.affine.mlir", build)
        # The threshold this artifact was lowered with, replayed on a cache
        # hit as well: sharing changes which buffers survive to the placement
        # stage, so re-deriving it there from the shared IR would report a
        # decision the emitted design was not built with.
        decided = cache.read_if_cached("kernel.reuse.txt")
        if decided is not None and config.provenance.get(
                "reuse_buffer_min_elements") == config.DERIVED:
            config.repin("reuse_buffer_min_elements", int(decided))
        metadata.extra["hls_reuse_min_elements"] = int(
            config.reuse_buffer_min_elements)
        # A narrower transfer is the last thing placement can trade for room,
        # and only re-lowering can produce one: the width is baked into the
        # transform's line blocks. Hand the stage the means to do it.
        metadata.extra["hls_relower"] = lambda io: lower(
            int(config.reuse_buffer_min_elements), io)
        return affine

    @staticmethod
    def _stage_hls_cpp(lowered: str, metadata: KernelMetadata, cache) -> str:
        config = metadata.extra["hls_config"]
        # Placement is the one decision that needs the lowered buffers, so
        # it is derived here rather than with the rest -- and, like them,
        # before the cache short-circuits the stage.
        lowered_facts = autotune.measure_kernel(lowered)
        performance_plan = autotune.plan(config, metadata.extra["hls_facts"],
                                         metadata, lowered_facts)
        # Sharing was settled while lowering, against the unshared IR; the
        # shared IR this stage measures holds fewer buffers and would answer
        # the same question differently.
        performance_plan.values["reuse_buffer_min_elements"] = (
            metadata.extra["hls_reuse_min_elements"])
        metadata.extra["hls_performance_plan"] = performance_plan
        config.adopt(performance_plan.values)
        for key, value in performance_plan.values.items():
            if (config.provenance.get(key) == config.DERIVED
                    and config[key] != value):
                config.repin(key, value)

        def build() -> str:
            top_func = _top_func(config, metadata)

            # `interface` decides how off-chip buffers reach the top
            # function: as AXI master ports (SoC integration), AXI-Stream
            # (streaming front-end), or as plain arrays (what the generated
            # simulation testbench drives). Which buffers go off chip is the
            # compiler's call -- see `autotune.external_buffer_threshold`.
            #
            # `stream` implies axi-interface=true so the pass creates the
            # port wrappers; stream-interface=true routes them into axis
            # bundles instead of m_axi ones.
            iface = config.interface
            axi = "axi-interface=" + ("true" if iface in ("axi", "stream") else
                                      "false")
            stream_flag = ("stream-interface=true"
                           if iface == "stream" else "")
            # The tier caps are the hard resource contract; placement
            # charges each tier in whole primitives and fails the design
            # rather than exceed one. `lutram-max-bytes` (one bus beat) is
            # derived from the config.
            lutram_max = int(config.lutram_max_bytes)

            def schedule(threshold: int, partition_factor: int,
                         module: str) -> tuple:
                """Runs the HLS pipeline, returning (IR, retry reason)."""
                pack_outputs = ("true" if config.external_vector_pack_outputs
                                else "false")
                hls_arg = (f"-hls-pipeline=top-func={top_func} "
                           f"loop-tile-size={int(config.loop_tile_size)} "
                           f"{axi} " +
                           (f"{stream_flag} " if stream_flag else "") +
                           f"axi-bus-bits={int(config.axi_bus_bits)} "
                           f"external-vector-max-lanes="
                           f"{int(config.external_vector_max_lanes)} "
                           f"external-vector-min-elements="
                           f"{int(config.external_vector_min_elements)} "
                           f"external-vector-pack-outputs="
                           f"{pack_outputs} "
                           f"external-vector-compute-lanes="
                           f"{int(config.external_vector_compute_lanes)} "
                           f"max-scratch-arenas="
                           f"{int(config.max_scratch_arenas)} "
                           f"max-unrolled-ops={int(config.max_unrolled_ops)} "
                           f"max-unroll-factor="
                           f"{int(config.max_unroll_factor)} "
                           f"bram-bytes={int(config.bram_bytes)} "
                           f"uram-bytes={int(config.uram_bytes)} "
                           f"lutram-bytes={int(config.lutram_bytes)} "
                           f"lutram-max-bytes={lutram_max} "
                           f"array-partition-max-factor={partition_factor} "
                           f"external-buffer-threshold={threshold}")
                try:
                    out = run_tool("sar-hls",
                                   [find_tool("sar-opt"), hls_arg, "-"],
                                   input_text=module)
                except CompilationError as err:
                    if _RETRYABLE_MEMORY_OVERFLOW in str(err):
                        return None, "memory"
                    if _RETRYABLE_PARTITION_OVERFLOW in str(err):
                        return None, "partition"
                    raise
                return out, None

            def schedule_with_banking(threshold: int, module: str) -> tuple:
                factor = int(config.array_partition_max_factor)
                derived = config.provenance.get(
                    "array_partition_max_factor") == config.DERIVED
                while True:
                    out, reason = schedule(threshold, factor, module)
                    if reason != "partition" or not derived or factor == 1:
                        return out, reason
                    factor //= 2
                    config.repin("array_partition_max_factor", factor)

            # When the resident working set cannot fit the caps, retry once
            # with every full-size plane streamed: the threshold drops to
            # the plane size, which is the same decision
            # `external_buffer_threshold` takes for a scene one size up. A
            # user-pinned threshold is respected -- the failure then names
            # it.
            threshold = int(config.external_buffer_threshold)
            module = lowered
            scheduled, retry_reason = schedule_with_banking(threshold, module)
            if retry_reason and config.provenance.get(
                    "external_buffer_threshold") == config.DERIVED:
                facts = metadata.extra["hls_facts"]
                streamed = autotune.streaming_threshold(facts)
                if streamed < threshold:
                    # Keep the streamed placement for every later rung even
                    # when this first retry still overflows.  Reverting to
                    # the resident threshold while narrowing transfers (or
                    # lanes) makes the retry ladder spend time re-lowering a
                    # design that is known to exceed the device by whole
                    # planes.
                    threshold = streamed
                    retried, retry_reason = schedule_with_banking(
                        streamed, module)
                    if not retry_reason:
                        scheduled = retried
                        config.repin("external_buffer_threshold", streamed)

            # Still over budget: narrow the FFT transfer and lower again.
            # `fft_io_unroll` is sized against the transform's own storage
            # ceiling, which a whole-beat transfer always clears -- but the
            # blocks it widens are resident alongside every other plane the
            # chain holds live, and it is that total the caps apply to. The
            # transform cannot see the total from its own share, so the
            # design as a whole reports it here. A narrower transfer costs
            # bandwidth on every line, so it is spent one halving at a time
            # and only after streaming has already been tried.
            relower = metadata.extra.get("hls_relower")
            io_unroll = int(config.fft_io_unroll)
            while (retry_reason and relower is not None and io_unroll > 1 and
                   config.provenance.get("fft_io_unroll") == config.DERIVED):
                io_unroll //= 2
                module = relower(io_unroll)
                retried, retry_reason = schedule_with_banking(
                    threshold, module)
                if not retry_reason:
                    scheduled = retried
                    config.repin("fft_io_unroll", io_unroll)
                    cache.write_text("kernel.affine.mlir", module)

            # A deep transform chain can still overflow after its planes have
            # spilled and its transfer has reached scalar width: every FFT
            # engine then owns the same minimum scratch arenas, and the
            # aggregate binding cost is not visible in the pre-lowering
            # estimate.  Reduce compiler-derived compute lanes as the final
            # reversible trade, preserving the user's pin if one was given.
            lanes = int(config.fft_parallel_rows)
            while (retry_reason and relower is not None and lanes > 1
                   and config.provenance.get("fft_parallel_rows")
                   == config.DERIVED):
                lanes //= 2
                config.repin("fft_parallel_rows", lanes)
                module = relower(io_unroll)
                retried, retry_reason = schedule_with_banking(
                    threshold, module)
                if not retry_reason:
                    scheduled = retried
                    cache.write_text("kernel.affine.mlir", module)

            # Nothing left to trade: the constraints and the kernel are
            # irreconcilable, and an over-budget design would be permanently
            # invalid on the device, so none is emitted.
            if retry_reason:
                raise HLSConfigError(
                    f"{metadata.name}: the on-chip working set exceeds the "
                    "resource caps after reducing automatic banking, "
                    "narrowing the transform transfers, and streaming "
                    "full-size planes; "
                    "raise bram_bytes/uram_bytes to match a larger device, "
                    "or shrink the kernel's resident tables")
            cache.write_text("kernel.hls.mlir", scheduled)
            # The threshold the shipped design was actually built with
            # (the retry above may have repinned it), persisted beside the
            # artifact: a later compile served from the cache must report
            # the configuration this build decided on, not its own
            # pre-retry estimate.
            cache.write_text(
                "kernel.hls.decisions.json",
                json.dumps({
                    "external_buffer_threshold":
                    int(config.external_buffer_threshold),
                    "array_partition_max_factor":
                    int(config.array_partition_max_factor),
                    "fft_io_unroll":
                    int(config.fft_io_unroll),
                    "fft_parallel_rows":
                    int(config.fft_parallel_rows),
                    "performance_plan": {
                        "clock_ns": performance_plan.clock_ns,
                        "timing_budget_ns": performance_plan.timing_budget_ns,
                        "on_chip_bytes": performance_plan.on_chip_bytes,
                        "memory_accesses": performance_plan.memory_accesses,
                        "operation_count": performance_plan.operation_count,
                        "values": performance_plan.values,
                    },
                }))

            return run_tool("sar-emit-hls", [
                find_tool("sar-translate"), "-hls-emit-hlscpp",
                "-emit-vitis-directives",
                f"-axi-bus-bits={config.axi_bus_bits}",
                f"-axi-max-burst-length={config.axi_max_burst_length}",
                f"-axi-max-outstanding={config.axi_max_outstanding}", "-"
            ],
                            input_text=scheduled)

        decisions = cache.read_if_cached("kernel.hls.decisions.json")
        if decisions is not None:
            try:
                json.loads(decisions)
            except (json.JSONDecodeError, TypeError):
                cache.path("kernel.hls.cpp").unlink(missing_ok=True)
                cache.path("kernel.hls.decisions.json").unlink(missing_ok=True)
        cached_stage(cache, "kernel.hls.cpp", build)
        # A cache hit skips `build` and with it the retry ladder; replay
        # the persisted decision so cold and warm compiles report the
        # same configuration.
        decisions = cache.read_if_cached("kernel.hls.decisions.json")
        if decisions is not None:
            values = json.loads(decisions)
            threshold = values.get("external_buffer_threshold")
            if (threshold is not None
                    and int(config.external_buffer_threshold) != threshold
                    and config.provenance.get("external_buffer_threshold")
                    == config.DERIVED):
                config.repin("external_buffer_threshold", threshold)
            factor = values.get("array_partition_max_factor")
            if (factor is not None
                    and int(config.array_partition_max_factor) != factor
                    and config.provenance.get("array_partition_max_factor")
                    == config.DERIVED):
                config.repin("array_partition_max_factor", factor)
            io_unroll = values.get("fft_io_unroll")
            if (io_unroll is not None
                    and int(config.fft_io_unroll) != io_unroll and
                    config.provenance.get("fft_io_unroll") == config.DERIVED):
                config.repin("fft_io_unroll", io_unroll)
            lanes = values.get("fft_parallel_rows")
            if (lanes is not None and int(config.fft_parallel_rows) != lanes
                    and config.provenance.get("fft_parallel_rows")
                    == config.DERIVED):
                config.repin("fft_parallel_rows", lanes)
        return str(cache.path("kernel.hls.cpp"))

    # ------------------------------------------------------------------ #
    # Launcher
    # ------------------------------------------------------------------ #

    def make_launcher(self, artifact: str, metadata: KernelMetadata):
        """Wraps the emitted C++ in the handle that writes the packages."""
        config = metadata.extra["hls_config"]
        # The design is named after the function it emits, which is what
        # the testbench has to call.
        return HLSDesign(artifact,
                         _top_func(config, metadata),
                         metadata=metadata,
                         config=config)
