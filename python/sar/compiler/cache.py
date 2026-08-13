"""On-disk artifact cache, keyed by kernel content and backend options.

Layout: ``$SAR_DSL_CACHE_DIR/<key>/<stage-artifact>`` where the key is a
SHA-256 over the MLIR module, the backend name and its options. Set
``SAR_DSL_DISABLE_CACHE=1`` to force recompilation.
"""

from __future__ import annotations

import hashlib
import os
import uuid
from pathlib import Path

__all__ = ["KernelCache"]

_VERSION = "2"  # bump to invalidate all caches


def _default_cache_root() -> Path:
    root = os.environ.get("SAR_DSL_CACHE_DIR")
    if root:
        return Path(root)
    return Path.home() / ".cache" / "sar-dsl"


class KernelCache:
    def __init__(self, module_text: str, backend: str, options: dict):
        digest = hashlib.sha256()
        digest.update(_VERSION.encode())
        digest.update(module_text.encode())
        digest.update(backend.encode())
        digest.update(repr(sorted(options.items())).encode())
        self.key = digest.hexdigest()[:24]
        self.dir = _default_cache_root() / self.key
        self.enabled = os.environ.get("SAR_DSL_DISABLE_CACHE", "0") != "1"
        self.dir.mkdir(parents=True, exist_ok=True)

    def path(self, filename: str) -> Path:
        return self.dir / filename

    def has(self, filename: str) -> bool:
        return self.enabled and self.path(filename).exists()

    def read_text(self, filename: str) -> str:
        return self.path(filename).read_text()

    def write_text(self, filename: str, content: str) -> Path:
        """Atomic write (temp file + rename): concurrent compilations of the
        same kernel may duplicate work but never observe partial files."""
        p = self.path(filename)
        tmp = p.with_name(f".{p.name}.{uuid.uuid4().hex[:8]}.tmp")
        tmp.write_text(content)
        os.replace(tmp, p)
        return p

    def scratch_path(self, filename: str) -> Path:
        """A unique temporary path inside the cache directory; pass to tools
        that write files directly, then `publish` the result."""
        return self.path(f".{filename}.{uuid.uuid4().hex[:8]}.tmp")

    def publish(self, scratch: Path, filename: str) -> Path:
        p = self.path(filename)
        os.replace(scratch, p)
        return p
