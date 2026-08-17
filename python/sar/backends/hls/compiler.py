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
`HLSDesign.write_testbench` emits a matching C-simulation testbench
with golden data.

Every SAR operation has an HLS lowering, so complete imaging chains
(omega-K, range-Doppler, chirp scaling) emit as single designs.

Options are validated against the schema in `config.py` and default from
the shipped `hls_config.yaml` (see `docs/backends.md`). The strategy half
of the schema is derived rather than defaulted -- see `autotune.py` -- and
the resolved set, with where each value came from, is reported by
`HLSDesign.config`.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from sar.backends.base import BaseBackend, KernelMetadata, cached_stage
from sar.backends.hls import autotune
from sar.backends.hls.config import HLSConfig, check_precision
from sar.compiler.toolchain import find_tool, run_tool
from sar.errors import LaunchError, ToolchainError

_C_TYPES = {"f32": "float", "f64": "double"}


def _top_func(config, metadata: KernelMetadata) -> str:
    """Name the emitted top function carries."""
    return config.top_func or metadata.name


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


class HLSDesign:
    """Handle to an emitted HLS C++ design."""

    def __init__(self, cpp_path: str, name: str, metadata=None, config=None):
        self.cpp_path = str(cpp_path)
        self.name = name
        #: The `HLSConfig` the design was compiled with: shipped defaults,
        #: the user's config file and the compile options, resolved.
        self.config = config
        self._metadata = metadata

    def source(self) -> str:
        return Path(self.cpp_path).read_text()

    def write_testbench(self,
                        inputs,
                        expected,
                        output_dir=None,
                        rtol: float = 1e-4,
                        atol: float = 1e-5) -> Path:
        """Writes a self-contained C-simulation package into `output_dir`:
        the design, a testbench comparing against golden data, the data
        files, and a Vitis HLS csim script.

        `inputs` / `expected` are numpy arrays matching the kernel's
        original argument/result types (complex arrays for complex
        tensors); golden outputs typically come from the NumPy reference
        or the cpu backend. `output_dir` defaults to
        `hls_project/<top>`. Returns the testbench path.
        """
        meta = self._metadata
        if self.config is not None and self.config.interface == "axi":
            raise LaunchError(
                "testbench generation needs the plain-array top signature: "
                "with interface='axi' every DRAM buffer is promoted to its "
                "own AXI port, which the testbench has no data to drive. "
                "Emit the csim package with the default "
                "interface='ap_memory', and hand the interface='axi' "
                "design to Vitis.")
        if self.config is not None and self.config.interface == "stream":
            raise LaunchError(
                "testbench generation does not support interface='stream' "
                "yet: an AXI4-Stream design takes hls::stream<> ports, and "
                "driving them needs a FIFO-feeding harness the array "
                "testbench is not. Emit the csim package with "
                "interface='ap_memory' (the numerics are identical), and "
                "hand the interface='stream' design to Vitis.")
        for t in list(meta.arg_types) + list(meta.result_types):
            if t.dtype.is_int:
                raise LaunchError(
                    "testbench generation does not support integer "
                    "kernel arguments/results yet")

        in_ports = _split_planes("in", meta.arg_types)
        out_ports = _split_planes("out", meta.result_types)
        arrays = (list(_plane_data(inputs, meta.arg_types)) +
                  list(_plane_data(expected, meta.result_types)))

        out = Path(
            output_dir if output_dir is not None else Path("hls_project") /
            self.name)
        data_dir = out / f"{self.name}_tb_data"
        data_dir.mkdir(parents=True, exist_ok=True)
        for (port, _, _), values in zip(in_ports + out_ports, arrays):
            np.savetxt(data_dir / f"{port}.dat",
                       values.reshape(-1),
                       fmt="%.17g")

        (out / f"{self.name}.cpp").write_text(self.source())
        tb = out / f"{self.name}_tb.cpp"
        tb.write_text(
            _testbench_source(self.name, in_ports, out_ports, rtol, atol))
        part, clock = _part_and_clock(self.config)
        (out / f"{self.name}_csim.tcl").write_text(
            _csim_script(self.name, part, clock))
        (out / f"{self.name}_csynth.tcl").write_text(
            _csynth_script(self.name, part, clock))
        _write_header_stubs(out / "stubs")
        return tb

    def write_synthesis_script(self, output_dir=None) -> Path:
        """Writes the design and a Vitis HLS synthesis script into
        `output_dir` (default `hls_project/<top>`), returning the script
        path.

        Unlike `write_testbench` this works for every interface: synthesis
        needs no golden data, so the AXI designs that cannot be csim'd are
        exactly the ones this exists for. Run with
        `vitis_hls -f <top>_csynth.tcl`; see docs/backends.md for what to
        check in the reports.
        """
        out = Path(
            output_dir if output_dir is not None else Path("hls_project") /
            self.name)
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{self.name}.cpp").write_text(self.source())
        part, clock = _part_and_clock(self.config)
        script = out / f"{self.name}_csynth.tcl"
        script.write_text(_csynth_script(self.name, part, clock))
        return script

    def __call__(self, *args, **kwargs):
        raise RuntimeError(
            "HLS designs are emitted as C++ for Vitis HLS synthesis and "
            f"cannot be executed directly; see {self.cpp_path}")

    def __repr__(self) -> str:
        return f"HLSDesign({self.name} @ {self.cpp_path})"


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


def _c_decl(port) -> str:
    name, shape, dtype = port
    dims = "".join(f"[{d}]" for d in shape)
    return f"{_C_TYPES[dtype]} {name}{dims}"


def _testbench_source(top, in_ports, out_ports, rtol, atol) -> str:
    """C-simulation testbench: loads the .dat files, runs the design,
    compares each output plane against the golden data."""
    proto = ",\n    ".join(_c_decl(p) for p in in_ports + out_ports)
    decls = "\n".join(f"  static {_c_decl(p)};" for p in in_ports + out_ports)
    loads = "\n".join(
        f'  if (!load(dir, "{p[0]}.dat", &{p[0]}{"[0]" * len(p[1])}, '
        f"{int(np.prod(p[1]))})) return 2;" for p in in_ports)
    call_args = ", ".join(p[0] for p in in_ports + out_ports)
    checks = "\n".join(f'  fail |= check(dir, "{p[0]}.dat", "{p[0]}", '
                       f'&{p[0]}{"[0]" * len(p[1])}, {int(np.prod(p[1]))});'
                       for p in out_ports)
    return f"""\
// Auto-generated C-simulation testbench for `{top}`.
//
// Vitis HLS:  vitis_hls -f {top}_csim.tcl
// Plain C++ (stub headers stand in for the Vitis ones):
//   c++ -O2 -I stubs {top}.cpp {top}_tb.cpp -o csim -pthread && ./csim
//
// The design runs on a dedicated 1 GiB stack: in csim the on-chip
// buffers are stack arrays, far beyond the default thread stack.

#include <cmath>
#include <cstdio>
#include <fstream>
#include <pthread.h>
#include <string>

void {top}(
    {proto});

namespace {{

// Vitis HLS may copy testbench data flat into the csim working
// directory; fall back from `dir`/`name` to the bare name.
template <typename T>
bool load(const std::string &dir, const std::string &name, T *dst,
          long count) {{
  std::string path = dir + "/" + name;
  std::ifstream f(path);
  if (!f) {{
    path = name;
    f.open(path);
  }}
  if (!f) {{
    std::fprintf(stderr, "cannot open %s/%s or ./%s\\n", dir.c_str(),
                 name.c_str(), name.c_str());
    return false;
  }}
  for (long i = 0; i < count; ++i)
    if (!(f >> dst[i])) {{
      std::fprintf(stderr, "short read in %s\\n", path.c_str());
      return false;
    }}
  return true;
}}

template <typename T>
int check(const std::string &dir, const std::string &name, const char *port,
          const T *got, long count) {{
  static double gold[{max(int(np.prod(p[1])) for p in out_ports)}];
  if (!load(dir, name, gold, count))
    return 2;
  double max_err = 0.0;
  long bad = 0;
  for (long i = 0; i < count; ++i) {{
    double err = std::abs(double(got[i]) - gold[i]);
    max_err = std::fmax(max_err, err);
    if (err > {atol} + {rtol} * std::abs(gold[i]))
      ++bad;
  }}
  std::printf("%-12s max |err| = %.3e  (%ld/%ld out of tolerance)\\n",
              port, max_err, bad, count);
  return bad != 0;
}}

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
  pthread_attr_init(&attr);
  pthread_attr_setstacksize(&attr, 1ull << 30);
  pthread_t worker;
  if (pthread_create(&worker, &attr, trampoline, (void *)&dir) != 0) {{
    std::fprintf(stderr, "cannot start the csim worker thread\\n");
    return 3;
  }}
  void *result = nullptr;
  pthread_join(worker, &result);
  int fail = *static_cast<int *>(result);
  delete static_cast<int *>(result);
  return fail;
}}
"""


_AP_INT_STUB = """\
#pragma once
// Minimal stand-in for the Vitis `ap_int`/`ap_uint` types: wide enough
// for the index arithmetic the designs emit, so any C++ compiler can
// run the C simulation without the Vitis headers.
template <int W> struct ap_int {
  long long v;
  ap_int() : v(0) {}
  ap_int(long long x) : v(x) {}
  operator long long() const { return v; }
};
template <int W> struct ap_uint {
  unsigned long long v;
  ap_uint() : v(0) {}
  ap_uint(unsigned long long x) : v(x) {}
  operator unsigned long long() const { return v; }
};
"""

#: `hls::stream` stand-in. A design with dataflow channels includes this
#: header, so an empty stub would make the package uncompilable outside
#: Vitis -- which is the whole point of shipping stubs. Sequential csim
#: never blocks, so an unbounded queue is behaviourally exact here.
_HLS_STREAM_STUB = """\
#pragma once
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


def _write_header_stubs(stub_dir: Path) -> None:
    """Vitis header stand-ins so the package csims with any compiler."""
    stub_dir.mkdir(parents=True, exist_ok=True)
    (stub_dir / "ap_int.h").write_text(_AP_INT_STUB)
    (stub_dir / "hls_stream.h").write_text(_HLS_STREAM_STUB)
    for header in ("ap_axi_sdata.h", "ap_fixed.h", "hls_math.h",
                   "hls_vector.h"):
        (stub_dir / header).write_text("#pragma once\n#include <cmath>\n")


def _part_and_clock(config) -> tuple:
    """(part, clock period) the generated Vitis scripts name."""
    if config is None:
        return "xcvu13p-fhgb2104-2-e", 10.0
    return config.part, float(config.clock_ns)


def _csim_script(top, part, clock_ns) -> str:
    """Vitis HLS csim script (written against Vitis HLS 2022.2)."""
    return f"""\
# Vitis HLS C simulation for `{top}`: vitis_hls -f {top}_csim.tcl
open_project -reset {top}_csim_proj
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


def _csynth_script(top, part, clock_ns) -> str:
    """Vitis HLS synthesis script (written against Vitis HLS 2022.2).

    Synthesis is the validation csim cannot give: achieved II, timing at
    the target clock, and measured BRAM/URAM/DSP against the config
    budgets. The checklist for reading the reports is in
    docs/backends.md ("Validating a design in Vitis HLS").
    """
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
csynth_design
exit
"""


class Backend(BaseBackend):
    name = "hls"

    @classmethod
    def is_available(cls) -> bool:
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
        config = HLSConfig.resolve(metadata.options)
        check_precision(config, metadata.arg_types, metadata.result_types)
        metadata.extra["hls_config"] = config
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
        facts = autotune.measure_kernel(module_text, metadata)
        metadata.extra["hls_facts"] = facts
        config.adopt(autotune.derive(config, facts))

        def build() -> str:
            cache.write_text("kernel.sar.mlir", module_text)
            # `top_func` renames the kernel's symbol here rather than at
            # the pipeline's `top-func` option alone: that option selects
            # a function by name, so a name the module does not carry
            # finds nothing.
            renamed = _rename_symbol(module_text, metadata.name,
                                     _top_func(config, metadata))
            budget = max(1, int(config.on_chip_budget))
            banded = "true" if config.interp_banded_gather else "false"
            pipeline = ("--sar-to-affine-pipeline="
                        f"reuse-buffer-min-elements="
                        f"{int(config.reuse_buffer_min_elements)} "
                        f"recompute-min-elements="
                        f"{int(config.recompute_min_elements)} "
                        f"transpose-block-bytes={budget // 8} "
                        f"interp-enable-banded-gather={banded} "
                        f"fft-stage-group={int(config.fft_stage_group)}")
            return run_tool("sar-lower", [find_tool("sar-opt"), pipeline, "-"],
                            input_text=renamed)

        return cached_stage(cache, "kernel.affine.mlir", build)

    @staticmethod
    def _stage_hls_cpp(lowered: str, metadata: KernelMetadata, cache) -> str:
        config = metadata.extra["hls_config"]
        # Placement is the one decision that needs the lowered buffers, so
        # it is derived here rather than with the rest -- and, like them,
        # before the cache short-circuits the stage.
        config.adopt(
            autotune.derive(config, metadata.extra["hls_facts"], metadata,
                            lowered))

        def build() -> str:
            top_func = _top_func(config, metadata)
            budget = max(1, int(config.on_chip_budget))

            # `interface` decides how off-chip buffers reach the top
            # function: as AXI master ports (SoC integration), AXI-Stream
            # (streaming front-end), or as plain arrays (what the generated
            # csim testbench drives). Which buffers go off chip is the
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
            # Per-tier budgets steer placement between BRAM, URAM and
            # distributed RAM; 0 means the tier is unbounded and the
            # aggregate `on-chip-bytes` rule applies on its own.
            # `lutram-max-bytes` and `uram-min-bytes` are derived from the
            # config (one bus beat and one physical URAM block).
            lutram_max = int(config.lutram_max_bytes)
            uram_min = int(config.uram_min_bytes)
            hls_arg = (f"-hls-pipeline=top-func={top_func} "
                       f"loop-tile-size={int(config.loop_tile_size)} "
                       f"on-chip-bytes={budget} {axi} " +
                       (f"{stream_flag} " if stream_flag else "") +
                       f"bram-bytes={int(config.bram_bytes)} "
                       f"uram-bytes={int(config.uram_bytes)} "
                       f"lutram-bytes={int(config.lutram_bytes)} "
                       f"lutram-max-bytes={lutram_max} "
                       f"uram-min-bytes={uram_min} "
                       "external-buffer-threshold="
                       f"{int(config.external_buffer_threshold)}")

            scheduled = run_tool("sar-hls",
                                 [find_tool("sar-opt"), hls_arg, "-"],
                                 input_text=lowered)
            cache.write_text("kernel.hls.mlir", scheduled)

            # The bus width is a board property, so the shaping of the AXI
            # masters -- beat width, burst length, how many are in flight
            # -- is derived from it rather than fixed. What the emitter is
            # given is the buffering a port may occupy: one direction's
            # worth of full-length bursts in each direction.
            buffer_bits = (2 * config.axi_max_burst_length *
                           config.axi_max_outstanding * config.axi_bus_bits)
            return run_tool("sar-emit-hls", [
                find_tool("sar-translate"), "-hls-emit-hlscpp",
                "-emit-vitis-directives",
                f"-axi-bus-bits={config.axi_bus_bits}",
                f"-axi-buffer-bits={buffer_bits}", "-"
            ],
                            input_text=scheduled)

        cached_stage(cache, "kernel.hls.cpp", build)
        return str(cache.path("kernel.hls.cpp"))

    # ------------------------------------------------------------------ #
    # Launcher
    # ------------------------------------------------------------------ #

    def make_launcher(self, artifact: str, metadata: KernelMetadata):
        config = metadata.extra["hls_config"]
        # The design is named after the function it emits, which is what
        # the testbench has to call.
        return HLSDesign(artifact,
                         _top_func(config, metadata),
                         metadata=metadata,
                         config=config)
