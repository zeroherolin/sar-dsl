"""Small diagnostics CLI: ``python -m sar doctor``."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import shutil
import subprocess

from . import __version__, list_backends
from .compiler.cache import cache_stats, clear_cache
from .compiler.toolchain import find_runtime_library, find_tool
from .runtime import thread_config


def _version(command) -> str | None:
    try:
        result = subprocess.run(command,
                                capture_output=True,
                                text=True,
                                timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return None
    output = result.stdout or result.stderr
    return output.splitlines()[0].strip() if output else None


def doctor() -> dict:
    tools = {}
    for name in ("sar-opt", "sar-translate", "mlir-translate", "clang"):
        try:
            path = find_tool(name, required=False)
            error = None
        except Exception as exc:
            path = None
            error = str(exc)
        tools[name] = {
            "path": path,
            "version": _version([path, "--version"]) if path else None,
            "error": error,
        }
    try:
        runtime = find_runtime_library()
        ctypes.CDLL(runtime)
        runtime_error = None
        threads = thread_config()
    except Exception as exc:  # diagnostics must report, not abort
        runtime = None
        runtime_error = str(exc)
        threads = None
    backends = {}
    try:
        discovered = list_backends()
        discovery_error = None
    except Exception as exc:
        discovered = {}
        discovery_error = str(exc)
    for name, backend in discovered.items():
        try:
            available = bool(backend.is_available())
            error = None
        except Exception as exc:  # external plugin diagnostics
            available = False
            error = str(exc)
        backends[name] = {"available": available, "error": error}
    return {
        "sar_dsl_version": __version__,
        "tools": tools,
        "runtime": {
            "path": runtime,
            "error": runtime_error,
            "threads": threads,
        },
        "vitis_hls": {
            "path":
            shutil.which("vitis_hls"),
            "version": (_version(["vitis_hls", "-version"])
                        if shutil.which("vitis_hls") else None),
        },
        "backends": backends,
        "backend_discovery_error": discovery_error,
        "cache": cache_stats(),
        "environment": {
            key: os.environ.get(key)
            for key in ("SAR_DSL_BUILD_DIR", "SAR_DSL_TOOL_PATH",
                        "SAR_DSL_RUNTIME_LIB", "SAR_DSL_OMP_LIB",
                        "SAR_DSL_HLS_CONFIG", "OMP_NUM_THREADS",
                        "SAR_RT_NUM_THREADS")
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m sar")
    subparsers = parser.add_subparsers(dest="command", required=True)
    doctor_parser = subparsers.add_parser("doctor", help="inspect toolchains")
    doctor_parser.add_argument("--json", action="store_true")
    cache_parser = subparsers.add_parser("cache",
                                         help="inspect artifact cache")
    cache_parser.add_argument("--clear", action="store_true")
    args = parser.parse_args()
    if args.command == "doctor":
        data = doctor()
        if args.json:
            print(json.dumps(data, indent=2, sort_keys=True))
        else:
            print(f"SAR-DSL {data['sar_dsl_version']}")
            for name, info in data["tools"].items():
                print(f"{name:16} {info['path'] or 'not found'}")
            print(f"runtime          {data['runtime']['path'] or 'not found'}")
            for name, info in sorted(data["backends"].items()):
                state = "available" if info["available"] else "unavailable"
                print(f"backend {name:8} {state}")
            cache = data["cache"]
            print(f"cache            {cache['entries']} entries, "
                  f"{cache['bytes']} bytes at {cache['root']}")
    else:
        data = clear_cache() if args.clear else cache_stats()
        print(json.dumps(data, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
