#!/usr/bin/env python3
"""Bounded, cached Vitis HLS design-space experiments.

Each variant is emitted into an isolated project. Successful syntheses are
content-addressed by source, Tcl, target and Vitis build; failed, timed-out or
incomplete runs are recorded but never become reusable cache entries.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_REPO / "python"), str(_REPO / "examples")]

import sar  # noqa: E402

from algorithms import ALL, GEOMETRIES, load  # noqa: E402
from common.params import ALOS_PARAMS  # noqa: E402
from hls_reports import parse_csynth_bundle, timing_shortfall  # noqa: E402
from hls_reports import validate_constraints  # noqa: E402
from provenance import environment  # noqa: E402

#: Where the hand-written reference keeps its own copy of the acquisition
#: constants, and how those macros map onto `RadarParams` fields. The
#: directory is deliberately self-contained -- it is not a SAR-DSL
#: dependency -- so the constants are duplicated rather than shared.
_HANDWRITTEN_CONFIG = _REPO / "examples/wka/handwritten_hls/config.h"
_HANDWRITTEN_TARGET = (_REPO / "examples/wka/handwritten_hls/hls/targets" /
                       "vu13p.tcl")
_HANDWRITTEN_FIELDS = {
    "WKA_C0": "c",
    "WKA_FC": "fc",
    "WKA_FS": "fs",
    "WKA_PRF": "prf",
    "WKA_VR": "vr",
    "WKA_R0": "r0",
    "WKA_KR": "kr",
}
_HANDWRITTEN_PRODUCTION = {
    "WKA_N": 16384,
    "WKA_AXI_BUS_BITS": 512,
    "WKA_AXI_PLANE_BITS": 256,
    "WKA_AXI_MAX_READ_BURST_LENGTH": 64,
    "WKA_AXI_MAX_WRITE_BURST_LENGTH": 64,
    "WKA_AXI_NUM_READ_OUTSTANDING": 16,
    "WKA_AXI_NUM_WRITE_OUTSTANDING": 16,
    "WKA_PLANE_LANES": 8,
    "WKA_STOLT_OUT_LANES": 4,
    "WKA_STOLT_CACHE_COPIES": 4,
}


def _handwritten_macros() -> dict:
    """Numeric `#define`s of the hand-written configuration header."""
    macros = {}
    for name, value in re.findall(r"^#define\s+(WKA_\w+)\s+(.+?)\s*$",
                                  _HANDWRITTEN_CONFIG.read_text(), re.M):
        value = value.strip()
        while value.startswith("(") and value.endswith(")"):
            value = value[1:-1].strip()
        if value.endswith(("f", "F")):
            value = value[:-1]
        try:
            macros[name] = float(value)
        except ValueError:
            continue
    return macros


def _handwritten_target() -> dict:
    """Scalar `set WKA_*` values from the hand-written target Tcl."""
    values = {}
    for name, value in re.findall(r'^set\s+(WKA_\w+)\s+"?([^"\s]+)"?\s*$',
                                  _HANDWRITTEN_TARGET.read_text(), re.M):
        try:
            values[name] = float(value)
        except ValueError:
            values[name] = value
    return values


def check_handwritten_geometry() -> None:
    """Fails when the hand-written reference no longer describes ALOS.

    The production table puts the generated omega-K design beside the
    hand-written one, and that comparison only means anything while both
    are focusing the same acquisition. Nothing links the two copies of the
    constants, so the agreement is checked here, where the comparison is
    produced, rather than being assumed.
    """
    macros = _handwritten_macros()
    missing = sorted(set(_HANDWRITTEN_FIELDS) - set(macros))
    if missing:
        raise SystemExit(
            f"{_HANDWRITTEN_CONFIG} defines none of {', '.join(missing)}; "
            "the hand-written geometry can no longer be checked against "
            "ALOS_PARAMS")

    shift_samples = macros.get("WKA_STOLT_TIME_SHIFT_SAMPLES")
    expected = {
        field: getattr(ALOS_PARAMS, field)
        for _, field in _HANDWRITTEN_FIELDS.items()
    }
    actual = {
        field: macros[macro]
        for macro, field in _HANDWRITTEN_FIELDS.items()
    }
    if shift_samples is not None:
        expected["t_shift"] = ALOS_PARAMS.t_shift
        actual["t_shift"] = shift_samples / macros["WKA_FS"]

    drifted = [
        f"{field}: hand-written {actual[field]!r} vs ALOS_PARAMS "
        f"{expected[field]!r}" for field in expected
        if abs(actual[field] - expected[field]) > 1e-12 *
        max(1.0, abs(expected[field]))
    ]
    if drifted:
        raise SystemExit(
            "the hand-written omega-K reference and ALOS_PARAMS describe "
            "different acquisitions, so the production comparison would be "
            "measuring two different problems:\n  " + "\n  ".join(drifted))


def check_handwritten_production_contract(size: int, config) -> None:
    """Checks the external and Stolt contracts used for WKA comparison."""
    macros = _handwritten_macros()
    expected = dict(_HANDWRITTEN_PRODUCTION)
    expected["WKA_N"] = int(size)
    expected.update({
        "WKA_AXI_BUS_BITS":
        int(config.axi_bus_bits),
        "WKA_AXI_PLANE_BITS":
        int(config.external_vector_max_lanes) * 32,
        "WKA_PLANE_LANES":
        int(config.external_vector_max_lanes),
        "WKA_AXI_MAX_READ_BURST_LENGTH":
        int(config.axi_max_burst_length),
        "WKA_AXI_MAX_WRITE_BURST_LENGTH":
        int(config.axi_max_burst_length),
        "WKA_AXI_NUM_READ_OUTSTANDING":
        int(config.axi_max_outstanding),
        "WKA_AXI_NUM_WRITE_OUTSTANDING":
        int(config.axi_max_outstanding),
        "WKA_STOLT_OUT_LANES":
        int(config.external_vector_compute_lanes),
        "WKA_STOLT_CACHE_COPIES":
        int(config.interp_cache_copies),
    })
    drifted = [
        f"{name}: hand-written {macros.get(name)!r} vs generated {value!r}"
        for name, value in expected.items() if macros.get(name) != value
    ]
    if drifted:
        raise SystemExit(
            "the hand-written and generated omega-K production contracts "
            "differ, so their synthesis results are not comparable:\n  " +
            "\n  ".join(drifted))

    target = _handwritten_target()
    expected_target = {
        "WKA_PART":
        config.part,
        "WKA_CLOCK_PERIOD_NS":
        float(config.clock_ns),
        "WKA_CLOCK_UNCERTAINTY_NS":
        float(config.clock_ns) * float(config.clock_uncertainty_percent) /
        100.0,
        "WKA_BRAM18K_BUDGET":
        int(config.bram_bytes) // (18 * 1024 // 8),
        "WKA_URAM_BUDGET":
        int(config.uram_bytes) // (288 * 1024 // 8),
        "WKA_DSP_BUDGET":
        int(config.dsp),
        "WKA_FF_BUDGET":
        int(config.ff),
        "WKA_LUT_BUDGET":
        int(config.lut),
    }
    target_drift = [
        f"{name}: hand-written {target.get(name)!r} vs generated {value!r}"
        for name, value in expected_target.items() if target.get(name) != value
    ]
    if target_drift:
        raise SystemExit(
            "the hand-written and generated omega-K target constraints "
            "differ, so their synthesis results are not comparable:\n  " +
            "\n  ".join(target_drift))


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return parsed


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _atomic_json(path: Path, value) -> None:
    # The suffix carries the thread as well as the process: two workers
    # writing one project's result would otherwise pick the same temporary
    # name, and the first rename would pull it out from under the second.
    tag = f"{os.getpid()}.{threading.get_ident()}"
    temporary = path.with_suffix(path.suffix + f".{tag}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _vitis_version(executable: str) -> str:
    result = subprocess.run([executable, "-version"],
                            capture_output=True,
                            text=True,
                            timeout=30)
    text = (result.stdout + result.stderr).strip()
    if result.returncode:
        raise RuntimeError(f"{executable} -version failed:\n{text}")
    return text


def _git_provenance() -> dict:

    def run(*args):
        result = subprocess.run(["git", *args],
                                capture_output=True,
                                text=True,
                                timeout=10)
        return result.stdout.strip() if result.returncode == 0 else None

    return {
        "commit": run("rev-parse", "HEAD"),
        "dirty": bool(run("status", "--porcelain")),
    }


def _directory_bytes(root: Path) -> int:
    total = 0
    for directory, _, names in os.walk(root):
        for name in names:
            try:
                total += (Path(directory) / name).stat().st_size
            except OSError:
                pass
    return total


def _process_tree_rss(root_pid: int) -> int:
    parents = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / "stat").read_text().split()
            parents[int(entry.name)] = int(fields[3])
        except (OSError, ValueError, IndexError):
            continue
    descendants = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, parent in parents.items():
            if parent in descendants and pid not in descendants:
                descendants.add(pid)
                changed = True

    rss = 0
    for pid in descendants:
        try:
            for line in Path(f"/proc/{pid}/status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    rss += int(line.split()[1]) * 1024
                    break
        except (OSError, ValueError):
            pass
    return rss


@dataclass(frozen=True)
class Job:
    algorithm: str
    size: int
    geometry: str
    options: dict
    project: Path
    script: Path
    top: str
    config: object
    key: str


def _terminate_group(process: subprocess.Popen, grace_s: float = 60.0) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=grace_s)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def _run_job(job: Job, executable: str, timeout_s: float, rss_limit: int,
             disk_limit: int, stop_event: threading.Event) -> dict:
    done = job.project / ".done.json"
    if done.is_file():
        cached = json.loads(done.read_text())
        cached["cached"] = True
        return cached

    log = job.project / "vitis_hls.log"
    started = time.monotonic()
    peak_rss = 0
    reason = None
    with log.open("w") as output:
        process = subprocess.Popen([executable, "-f", job.script.name],
                                   cwd=job.project,
                                   stdout=output,
                                   stderr=subprocess.STDOUT,
                                   start_new_session=True)
        while process.poll() is None:
            elapsed = time.monotonic() - started
            peak_rss = max(peak_rss, _process_tree_rss(process.pid))
            if stop_event.is_set():
                reason = "interrupted"
            elif timeout_s > 0 and elapsed > timeout_s:
                reason = "timeout"
            elif rss_limit > 0 and peak_rss > rss_limit:
                reason = "rss"
            elif disk_limit > 0 and _directory_bytes(job.project) > disk_limit:
                reason = "disk"
            if reason:
                _terminate_group(process)
                break
            time.sleep(2.0)
        returncode = process.wait()

    elapsed = time.monotonic() - started
    reports = sorted(
        job.project.glob(
            f"{job.top}_csynth_proj/sol1/syn/report/{job.top}_csynth.xml"))
    result = {
        "algorithm": job.algorithm,
        "size": job.size,
        "geometry": job.geometry,
        "options": job.options,
        "key": job.key,
        "project": str(job.project),
        "returncode": returncode,
        "elapsed_s": elapsed,
        "peak_rss_bytes": peak_rss,
        "failure": reason,
        "cached": False,
    }
    if reports:
        bundle = parse_csynth_bundle(reports[0])
        result["report"] = bundle
        violations = validate_constraints(bundle["top"], job.config)
        result["violations"] = violations
        # Timing is a goal, not a budget: a miss is recorded beside the
        # result rather than failing the job.
        result["timing_shortfall"] = timing_shortfall(bundle["top"],
                                                      job.config)
        if violations and result["failure"] is None:
            result["failure"] = "constraints"
    else:
        result["violations"] = ["missing top-level synthesis XML"]
        if result["failure"] is None:
            result["failure"] = "incomplete"

    _atomic_json(job.project / "result.json", result)
    if returncode == 0 and result["failure"] is None:
        _atomic_json(done, result)
    return result


def _default_variants() -> list[dict]:
    variants = [{}]
    for key, values in (
        ("external_vector_max_lanes", (4, 8)),
        ("fft_parallel_rows", (4, 8, 16)),
        ("fft_stage_group", (2, 3, 4)),
        ("array_partition_max_factor", (4, 8, 16)),
    ):
        variants.extend({key: value} for value in values)
    return variants


def _load_variants(path: str | None) -> list[dict]:
    if path is None:
        return _default_variants()
    value = json.loads(Path(path).read_text())
    if not isinstance(value, list) or any(not isinstance(item, dict)
                                          for item in value):
        raise ValueError("variants JSON must be a list of option objects")
    return value


def _prepare_job(root: Path, executable_version: str, algorithm: str,
                 size: int, dtype, geometry: str, options: dict) -> Job:
    chain = load(algorithm, size, dtype=dtype, geometry=geometry)
    design = chain.compile_kernel(backend="hls", interface="axi", **options)
    staging = Path(tempfile.mkdtemp(prefix="prepare-", dir=root))
    script = design.write_synthesis_script(staging)
    source = (staging / f"{design.name}.cpp").read_bytes()
    header = (staging / f"{design.name}.h").read_bytes()
    tcl = script.read_bytes()
    key_data = json.dumps(
        {
            "source": _sha256(source),
            "header": _sha256(header),
            "tcl": _sha256(tcl),
            "vitis": executable_version,
            "geometry": geometry,
            "part": design.config.part,
            "clock_ns": float(design.config.clock_ns),
        },
        sort_keys=True).encode()
    key = _sha256(key_data)[:24]
    project = root / key
    if project.exists():
        shutil.rmtree(staging)
    else:
        os.replace(staging, project)
    return Job(algorithm, size, geometry, options, project,
               project / script.name, design.name, design.config, key)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algs", nargs="+", choices=ALL, default=["wka"])
    parser.add_argument("--size", type=_positive_int, default=16384)
    parser.add_argument("--dtype", choices=("c64", "c128"), default="c64")
    parser.add_argument("--geometry",
                        choices=tuple(GEOMETRIES),
                        default="alos",
                        help="stripmap collection geometry; the default is "
                        "the acquisition the hand-written reference "
                        "implements, so the two are comparable")
    parser.add_argument("--variants",
                        help="JSON list of HLS option objects; defaults to a "
                        "single-factor production sweep")
    parser.add_argument("--baseline-only",
                        action="store_true",
                        help="synthesize only the compiler-derived baseline, "
                        "without the default single-factor variants")
    parser.add_argument("--output",
                        default="/tmp/sar-dsl-hls-sweep",
                        help="content-addressed project/cache directory")
    parser.add_argument("--workers", type=_positive_int, default=2)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--max-rss-gib", type=float, default=256.0)
    parser.add_argument("--max-disk-gib", type=float, default=50.0)
    parser.add_argument("--vitis-hls", default="vitis_hls")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", help="write aggregate summary here")
    args = parser.parse_args()

    # The hand-written reference is ALOS-only, so it is only a valid
    # comparison point for designs built at that geometry.
    if args.geometry == "alos":
        check_handwritten_geometry()

    executable = shutil.which(args.vitis_hls)
    if executable is None:
        raise SystemExit(f"cannot find {args.vitis_hls!r}")
    version = _vitis_version(executable)
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.baseline_only and args.variants:
        parser.error("--baseline-only and --variants are mutually exclusive")
    variants = [{}] if args.baseline_only else _load_variants(args.variants)
    dtype = getattr(sar, args.dtype)

    jobs = []
    prepared = set()
    prepare_failures = []
    for algorithm in args.algs:
        for variant in variants:
            try:
                job = _prepare_job(root, version, algorithm, args.size, dtype,
                                   args.geometry, variant)
                if (algorithm == "wka" and args.geometry == "alos"
                        and args.dtype == "c64" and args.size == 16384):
                    check_handwritten_production_contract(
                        args.size, job.config)
                # Variants are single-factor, so one that lands on the value
                # the compiler would have derived produces the design another
                # variant already produced. Synthesizing it twice would only
                # race two workers over one project directory.
                if job.key in prepared:
                    continue
                prepared.add(job.key)
                jobs.append(job)
            except (sar.CompilationError, sar.HLSConfigError) as exc:
                prepare_failures.append({
                    "algorithm": algorithm,
                    "size": args.size,
                    "geometry": args.geometry,
                    "options": variant,
                    "key": "-",
                    "project": None,
                    "returncode": None,
                    "elapsed_s": 0.0,
                    "peak_rss_bytes": 0,
                    "failure": "prepare",
                    "error": str(exc),
                    "cached": False,
                    "violations": [str(exc)],
                })
    if args.dry_run:
        for job in jobs:
            print(job.project)
        for failure in prepare_failures:
            print(
                f"{failure['algorithm']} prepare failed: "
                f"{failure['error']}",
                file=sys.stderr)
        if prepare_failures:
            raise SystemExit(1)
        return

    seats = int(os.environ.get("SAR_DSL_VITIS_LICENSE_SEATS", args.workers))
    workers = min(args.workers, seats, len(jobs))
    rss_limit = int(max(0.0, args.max_rss_gib) * 2**30)
    disk_limit = int(max(0.0, args.max_disk_gib) * 2**30)
    results = list(prepare_failures)
    if jobs:
        stop_event = threading.Event()
        with concurrent.futures.ThreadPoolExecutor(
                max_workers=max(1, workers)) as pool:
            futures = [
                pool.submit(_run_job, job, executable, args.timeout, rss_limit,
                            disk_limit, stop_event) for job in jobs
            ]
            try:
                results.extend(future.result() for future in futures)
            except KeyboardInterrupt:
                stop_event.set()
                for future in futures:
                    future.cancel()
                raise

    summary = {
        "environment": environment(),
        "git": _git_provenance(),
        "geometry": args.geometry,
        "vitis_version": version,
        "workers": workers,
        "results": results,
    }
    output = Path(args.json) if args.json else root / "summary.json"
    _atomic_json(output, summary)
    for result in results:
        status = "PASS" if result["failure"] is None else result["failure"]
        print(f"{result['algorithm']:<4} {result.get('key', '-')} {status}")
    if any(result["failure"] is not None for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
