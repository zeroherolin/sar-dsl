"""Backend discovery and registry.

Backends are Python packages exposing a ``Backend`` class (subclass of
`sar.backends.base.BaseBackend`). The built-in ones -- ``cpu`` and ``hls``
-- are subpackages here; out-of-tree backends are discovered from
``$SAR_DSL_BACKEND_PATH`` (os.pathsep-separated directories, each being a
backend package directory)."""

from __future__ import annotations

import importlib
import importlib.util
import hashlib
import os
import re
import sys
import threading
from pathlib import Path
from typing import Dict, Type

from ..errors import SARError, ToolchainError
from .base import (
    BaseBackend,
    KernelMetadata,
    Stage,  # noqa: F401
    cached_stage)

__all__ = [
    "BaseBackend", "KernelMetadata", "Stage", "cached_stage", "get_backend",
    "list_backends", "backend_fingerprint"
]

_registry: Dict[str, Type[BaseBackend]] = {}
_discovered = False
_discovery_lock = threading.RLock()


def register_backend(cls: Type[BaseBackend]) -> None:
    """Adds a backend class to the registry under its ``name``; a name
    already taken by a different class is rejected."""
    with _discovery_lock:
        if not issubclass(cls, BaseBackend):
            raise SARError(f"{cls!r} is not a BaseBackend subclass")
        previous = _registry.get(cls.name)
        if previous is not None and previous is not cls:
            raise SARError(
                f"backend name {cls.name!r} is already registered by "
                f"{previous.__module__}.{previous.__name__}")
        _registry[cls.name] = cls


def _register_from(module) -> None:
    backend_cls = getattr(module, "Backend", None)
    if backend_cls is not None:
        register_backend(backend_cls)


def _load_backend_module(name: str, directory: Path) -> None:
    """Loads an out-of-tree backend package from `directory`."""
    init = directory / "__init__.py"
    compiler = directory / "compiler.py"
    source = init if init.exists() else compiler
    if not source.exists():
        return
    safe_name = re.sub(r"\W+", "_", name).strip("_") or "external"
    path_id = hashlib.sha256(str(directory.resolve()).encode()).hexdigest()[:8]
    module_name = f"sar_backend_{safe_name}_{path_id}"
    if module_name in sys.modules:
        module = sys.modules[module_name]
    else:
        package_paths = [str(directory)] if source == init else None
        spec = importlib.util.spec_from_file_location(
            module_name, source, submodule_search_locations=package_paths)
        if spec is None or spec.loader is None:
            return
        module = importlib.util.module_from_spec(spec)
        # The module has to be visible before exec_module for its own
        # relative imports to resolve; drop it again if it fails to load,
        # so a broken backend cannot masquerade as an imported one.
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            del sys.modules[module_name]
            raise
    _register_from(module)


def _discover() -> None:
    global _discovered
    with _discovery_lock:
        if _discovered:
            return
        previous = dict(_registry)
        try:
            package_dir = Path(__file__).parent
            for child in sorted(package_dir.iterdir()):
                if child.is_dir() and (child / "compiler.py").exists():
                    _register_from(
                        importlib.import_module(
                            f"sar.backends.{child.name}.compiler"))

            for entry in os.environ.get("SAR_DSL_BACKEND_PATH",
                                        "").split(os.pathsep):
                if entry:
                    path = Path(entry)
                    _load_backend_module(
                        path.parent.name
                        if path.name == "backend" else path.name, path)
        except Exception:
            _registry.clear()
            _registry.update(previous)
            raise
        _discovered = True


def backend_fingerprint(cls: Type[BaseBackend]) -> str:
    """Content identity of a backend package, including external plugins."""
    module = sys.modules.get(cls.__module__)
    source = (Path(module.__file__).resolve()
              if module is not None and module.__file__ else None)
    if source is None:
        return f"{cls.__module__}.{cls.__name__}"
    root = source.parent
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if not path.is_file() or (path.suffix != ".py"
                                  and path.suffix not in (".yaml", ".yml")):
            continue
        digest.update(str(path.relative_to(root)).encode())
        try:
            digest.update(path.read_bytes())
        except OSError:
            digest.update(b"unreadable")
    return digest.hexdigest()[:16]


def list_backends() -> Dict[str, Type[BaseBackend]]:
    """All discovered backend classes by name (available or not)."""
    _discover()
    return dict(_registry)


def get_backend(name: str) -> Type[BaseBackend]:
    """The backend class for `name`; raises if unknown or unavailable."""
    _discover()
    if name not in _registry:
        raise ToolchainError(f"unknown backend '{name}'; discovered backends: "
                             f"{sorted(_registry)}")
    cls = _registry[name]
    if not cls.is_available():
        raise ToolchainError(
            f"backend '{name}' is registered but its toolchain is not "
            "available on this machine")
    return cls
