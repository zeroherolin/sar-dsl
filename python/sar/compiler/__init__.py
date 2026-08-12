"""Compilation driver: runs backend stages over the traced kernel."""

from __future__ import annotations

from collections import OrderedDict
from typing import Optional

from ..backends import KernelMetadata, get_backend
from .cache import KernelCache

__all__ = ["compile"]


def compile(kernel, backend: str = "cpu", options: Optional[dict] = None):
    """Compiles a traced kernel with the given backend.

    Returns the backend-specific launcher: a callable executing the kernel
    for execution backends, or an artifact handle for emission backends.
    """
    options = dict(options or {})
    backend_cls = get_backend(backend)
    backend_obj = backend_cls()

    module_text = kernel.to_mlir()
    metadata = KernelMetadata(
        name=kernel.name,
        arg_types=list(kernel.arg_types),
        result_types=list(kernel.declared_result_types),
        options=options,
    )

    stages = OrderedDict()
    backend_obj.add_stages(stages, metadata)

    cache = KernelCache(module_text, backend, options)
    artifact = module_text
    for stage_name, stage_fn in stages.items():
        artifact = stage_fn(artifact, metadata, cache)

    return backend_obj.make_launcher(artifact, metadata)
