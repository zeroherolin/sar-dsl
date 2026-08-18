"""Locates the tools a backend shells out to (sar-opt, sar-translate,
mlir-translate, clang).

Search order for a tool named `foo-bar`:
1. environment variable ``SAR_DSL_TOOL_FOO_BAR`` (full path);
2. directories listed by the generated ``sar._build_config`` module
   (created by the CMake build);
3. ``PATH``.
"""

from __future__ import annotations

import functools
import importlib.util
import math
import os
import shutil
import subprocess
from pathlib import Path
from types import ModuleType
from typing import List, Optional, Sequence

from ..errors import CompilationError, ToolchainError


def _load_build_config() -> Optional[ModuleType]:
    candidates = []
    explicit = os.environ.get("SAR_DSL_BUILD_CONFIG")
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            raise ToolchainError(
                f"SAR_DSL_BUILD_CONFIG={explicit} does not exist")
        candidates.append(path)
    else:
        build_dir = os.environ.get("SAR_DSL_BUILD_DIR")
        if build_dir:
            path = Path(build_dir) / "python/sar/_build_config.py"
            if not path.is_file():
                raise ToolchainError(f"SAR_DSL_BUILD_DIR={build_dir} has no "
                                     "python/sar/_build_config.py")
            candidates.append(path)
        repo_root = Path(__file__).resolve().parents[3]
        candidates.append(repo_root / "build/python/sar/_build_config.py")
    for path in candidates:
        if not path.is_file():
            continue
        spec = importlib.util.spec_from_file_location(
            "sar._active_build_config", path)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    try:
        from .. import _build_config
        return _build_config
    except ImportError:
        pass
    return None


_config = _load_build_config()


def _config_dirs() -> List[str]:
    dirs = []
    if _config is not None:
        dirs.append(getattr(_config, "SAR_DSL_TOOL_DIR", ""))
        dirs.append(getattr(_config, "LLVM_TOOL_DIR", ""))
    extra = os.environ.get("SAR_DSL_TOOL_PATH", "")
    dirs.extend(p for p in extra.split(os.pathsep) if p)
    return [d for d in dirs if d]


@functools.lru_cache(maxsize=None)
def find_tool(name: str, required: bool = True) -> Optional[str]:
    env_key = "SAR_DSL_TOOL_" + name.upper().replace("-", "_")
    override = os.environ.get(env_key)
    if override:
        if not os.path.isfile(override):
            raise ToolchainError(f"{env_key}={override} does not exist")
        if not os.access(override, os.X_OK):
            raise ToolchainError(f"{env_key}={override} is not executable")
        return override

    for directory in _config_dirs():
        candidate = os.path.join(directory, name)
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate

    found = shutil.which(name)
    if found:
        return found

    if required:
        raise ToolchainError(
            f"required tool '{name}' not found; build the project "
            f"(see README) or set {env_key}")
    return None


@functools.lru_cache(maxsize=None)
def find_runtime_library() -> str:
    override = os.environ.get("SAR_DSL_RUNTIME_LIB")
    if override:
        if not os.path.isfile(override):
            raise ToolchainError(
                f"SAR_DSL_RUNTIME_LIB={override} does not exist")
        return override
    if _config is not None:
        candidate = getattr(_config, "SAR_DSL_RUNTIME_LIB", "")
        if candidate and os.path.isfile(candidate):
            return candidate
    tool = find_tool("sar-opt", required=False)
    if tool:
        candidate = os.path.join(os.path.dirname(os.path.dirname(tool)), "lib",
                                 "libsar_runtime.so")
        if os.path.isfile(candidate):
            return candidate
    raise ToolchainError(
        "libsar_runtime.so not found; build the project or set "
        "SAR_DSL_RUNTIME_LIB")


def run_tool(stage: str,
             command: Sequence[str],
             input_text: Optional[str] = None,
             timeout: Optional[float] = None) -> str:
    """Runs a tool, returning stdout; raises CompilationError on failure."""
    if timeout is None:
        raw_timeout = os.environ.get("SAR_DSL_TOOL_TIMEOUT_SECONDS", "1800")
        try:
            timeout = float(raw_timeout)
        except ValueError as err:
            raise ToolchainError(
                "SAR_DSL_TOOL_TIMEOUT_SECONDS must be a number") from err
        if not math.isfinite(timeout):
            raise ToolchainError("SAR_DSL_TOOL_TIMEOUT_SECONDS must be finite")
        if timeout <= 0:
            timeout = None
    try:
        proc = subprocess.run(list(command),
                              input=input_text,
                              capture_output=True,
                              text=True,
                              timeout=timeout)
    except subprocess.TimeoutExpired as err:
        output = err.stderr or err.stdout or ""
        if isinstance(output, bytes):
            output = output.decode(errors="replace")
        limit = f"{timeout:g}" if timeout is not None else "configured"
        raise CompilationError(
            stage, command,
            f"timed out after {limit} seconds\n{output}") from err
    except OSError as err:
        # The tool vanished or became unrunnable between discovery and
        # use; surface it as the toolchain problem it is.
        raise ToolchainError(f"stage '{stage}' could not run "
                             f"{command[0]}: {err}") from err
    if proc.returncode != 0:
        raise CompilationError(stage, command, proc.stderr or proc.stdout)
    return proc.stdout
