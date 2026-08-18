"""Compilation driver: runs backend stages over the traced kernel."""

from __future__ import annotations

from typing import Optional

from ..backends import KernelMetadata, backend_fingerprint, get_backend
from .cache import KernelCache

__all__ = ["compile"]


def compile(kernel, backend: str = "cpu", options: Optional[dict] = None):
    """Compiles a traced kernel with the given backend.

    Returns the backend-specific launcher: a callable executing the kernel
    for execution backends, or an artifact handle for emission backends.
    """
    backend_cls = get_backend(backend)
    backend_obj = backend_cls()

    module_text = kernel.to_mlir()
    metadata = KernelMetadata(
        name=kernel.name,
        arg_types=list(kernel.arg_types),
        result_types=list(kernel.declared_result_types),
        options=dict(options or {}),
        param_names=getattr(kernel, "param_names", None),
    )

    stages = {}
    backend_obj.add_stages(stages, metadata)

    # Keyed on `metadata.options`, not the caller's `options`: `add_stages`
    # may rewrite them into their fully resolved form (the HLS backend
    # folds in its config files), and it is the resolved set that
    # identifies the artifacts.
    cache = KernelCache(module_text, backend, metadata.options,
                        backend_fingerprint(backend_cls))
    artifact = module_text
    for stage_fn in stages.values():
        artifact = stage_fn(artifact, metadata, cache)

    return backend_obj.make_launcher(artifact, metadata)
