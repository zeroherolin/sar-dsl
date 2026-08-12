"""Backend abstraction.

A backend contributes an ordered set of *stages* (FlagTree/Triton style):
each stage is a function ``(artifact, metadata, cache) -> artifact`` that
refines the previous artifact, starting from the serialized `sar` dialect
module. The final artifact is wrapped into a launcher by `make_launcher`.

New hardware targets drop a package with a ``Backend`` subclass into
``third_party/<name>/backend`` (development) or ``sar/backends/<name>``
(installed) -- no changes to the core are needed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Dict, List

from ..ir import TensorType

__all__ = ["BaseBackend", "KernelMetadata", "Stage"]

Stage = Callable  # (artifact, metadata: KernelMetadata, cache) -> artifact


@dataclass
class KernelMetadata:
    """Information shared across compilation stages."""

    name: str
    arg_types: List[TensorType]
    result_types: List[TensorType]
    options: Dict[str, object] = field(default_factory=dict)
    extra: Dict[str, object] = field(default_factory=dict)


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
