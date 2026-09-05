"""Compilation driver: runs backend stages over the traced kernel."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional

from ..backends import KernelMetadata, backend_fingerprint, get_backend
from .cache import KernelCache

__all__ = ["compile", "dump_pipeline"]


def _prepare(kernel, backend: str, options):
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
    cache = KernelCache(module_text, backend, metadata.options,
                        backend_fingerprint(backend_cls))
    return backend_obj, module_text, metadata, stages, cache


def compile(kernel, backend: str = "cpu", options: Optional[dict] = None):
    """Compiles a traced kernel with the given backend.

    Returns the backend-specific launcher: a callable executing the kernel
    for execution backends, or an artifact handle for emission backends.
    """
    backend_obj, module_text, metadata, stages, cache = _prepare(
        kernel, backend, options)
    with cache:
        artifact = module_text
        for stage_fn in stages.values():
            artifact = stage_fn(artifact, metadata, cache)
        return backend_obj.make_launcher(artifact, metadata)


def dump_pipeline(kernel,
                  output_dir,
                  backend: str = "cpu",
                  options: Optional[dict] = None) -> Path:
    """Compiles and copies every cached intermediate into ``output_dir``."""
    backend_obj, module_text, metadata, stages, cache = _prepare(
        kernel, backend, options)
    out = Path(output_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and not out.is_dir():
        raise FileExistsError(f"pipeline output {out} is not a directory")
    if (out.exists() and any(out.iterdir())
            and not (out / "pipeline_manifest.json").is_file()):
        raise FileExistsError(
            f"pipeline output {out} is not empty and is not a previous "
            "pipeline export")
    staging = Path(tempfile.mkdtemp(prefix=f".{out.name}.", dir=out.parent))
    try:
        with cache:
            artifact = module_text
            for stage_fn in stages.values():
                artifact = stage_fn(artifact, metadata, cache)
            backend_obj.make_launcher(artifact, metadata)
            for source in cache.dir.iterdir():
                # Dotfiles are cache bookkeeping (.lock, the .accessed
                # LRU marker, .*.tmp scratch files), not artifacts.
                if source.name.startswith(".") or not source.is_file():
                    continue
                shutil.copy2(source, staging / source.name)
            files = sorted(path.name for path in staging.iterdir())
            (staging / "pipeline_manifest.json").write_text(
                json.dumps(
                    {
                        "kernel": metadata.name,
                        "backend": backend,
                        "cache_key": cache.key,
                        "stages": list(stages),
                        "options": metadata.options,
                        "files": files,
                    },
                    indent=2,
                    sort_keys=True) + "\n")
        if out.is_dir():
            shutil.rmtree(out)
        os.replace(staging, out)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return out
