"""Backend abstraction.

A backend contributes an ordered set of *stages*: each stage is a
function ``(artifact, metadata, cache) -> artifact`` that refines the
previous artifact, starting from the serialized `sar` dialect module.
The final artifact is wrapped into a launcher by `make_launcher`.

New hardware targets drop a package with a ``Backend`` subclass into
``sar/backends/<name>`` or any directory on ``$SAR_DSL_BACKEND_PATH`` --
no changes to the core are needed (see docs/backends.md).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from ..ir import TensorType

__all__ = ["BaseBackend", "KernelMetadata", "Stage", "cached_stage"]

#: One compilation stage: ``(artifact, metadata, cache) -> artifact``.
Stage = Callable[[object, "KernelMetadata", object], object]


def cached_stage(cache, filename: str, build: Callable[[], str]) -> str:
    """Runs one stage through the artifact cache: returns the cached text
    when present, otherwise calls `build`, stores its result under
    `filename` and returns it."""
    cached = cache.read_if_cached(filename)
    if cached is not None:
        return cached
    out = build()
    cache.write_text(filename, out)
    return out


@dataclass
class KernelMetadata:
    """Information shared across compilation stages."""

    name: str
    arg_types: List[TensorType]
    result_types: List[TensorType]
    options: Dict[str, object] = field(default_factory=dict)
    extra: Dict[str, object] = field(default_factory=dict)
    #: The kernel's Python parameter names (None when the signature does
    #: not expose them); the traced IR carries them as `sar.arg_names`.
    param_names: Optional[List[str]] = None


class BaseBackend(ABC):
    """Interface every SAR-DSL backend implements."""

    #: unique backend name used in ``kernel.compile(backend=...)``
    name: str = "<abstract>"

    @classmethod
    @abstractmethod
    def is_available(cls) -> bool:
        """Whether the backend's toolchain is usable on this machine."""

    @abstractmethod
    def add_stages(self, stages: "Dict[str, Stage]",
                   metadata: KernelMetadata) -> None:
        """Populates `stages` (ordered dict) with compilation stages."""

    @abstractmethod
    def make_launcher(self, artifact, metadata: KernelMetadata):
        """Wraps the final artifact into a user-facing object."""
