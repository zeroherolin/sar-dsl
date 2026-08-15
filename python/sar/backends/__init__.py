"""Backend discovery and registry.

Backends are Python packages exposing a ``Backend`` class (subclass of
`sar.backends.base.BaseBackend`). They are discovered from, in order:

1. subpackages of ``sar.backends`` (populated by ``setup.py`` /
   ``scripts/setup-dev.sh`` from the ``third_party`` layout);
2. ``$SAR_DSL_BACKEND_PATH`` (os.pathsep-separated directories, each being a
   backend package directory);
3. ``third_party/*/backend`` relative to a source checkout (development
   convenience).
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from pathlib import Path
from typing import Dict, Type

from ..errors import SARError, ToolchainError
from .base import BaseBackend, KernelMetadata

__all__ = ["BaseBackend", "KernelMetadata", "get_backend", "list_backends"]

_registry: Dict[str, Type[BaseBackend]] = {}
_discovered = False


def register_backend(cls: Type[BaseBackend]) -> None:
    if not issubclass(cls, BaseBackend):
        raise SARError(f"{cls!r} is not a BaseBackend subclass")
    _registry[cls.name] = cls


def _load_backend_module(name: str, directory: Path) -> None:
    init = directory / "__init__.py"
    compiler = directory / "compiler.py"
    source = compiler if compiler.exists() else init
    if not source.exists():
        return
    module_name = f"sar_backend_{name}"
    if module_name in sys.modules:
        module = sys.modules[module_name]
    else:
        spec = importlib.util.spec_from_file_location(module_name, source)
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
    backend_cls = getattr(module, "Backend", None)
    if backend_cls is not None:
        register_backend(backend_cls)


def _discover() -> None:
    global _discovered
    if _discovered:
        return
    _discovered = True

    # 1. Installed subpackages of sar.backends.
    package_dir = Path(__file__).parent
    for child in sorted(package_dir.iterdir()):
        if child.is_dir() and (child / "compiler.py").exists():
            _load_backend_module(child.name, child)

    # 2. Explicit search path.
    for entry in os.environ.get("SAR_DSL_BACKEND_PATH", "").split(os.pathsep):
        if entry:
            path = Path(entry)
            _load_backend_module(
                path.parent.name if path.name == "backend" else path.name,
                path)

    # 3. Source-tree third_party/<name>/backend.
    repo_root = package_dir.parent.parent.parent
    third_party = repo_root / "third_party"
    if third_party.is_dir():
        for vendor in sorted(third_party.iterdir()):
            backend_dir = vendor / "backend"
            if backend_dir.is_dir():
                _load_backend_module(vendor.name, backend_dir)


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
