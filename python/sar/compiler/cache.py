"""On-disk artifact cache, keyed by kernel content and backend options.

Layout: ``$SAR_DSL_CACHE_DIR/<key>/<stage-artifact>``. The key is a SHA-256
over a toolchain fingerprint, the MLIR module, the backend name and its
options, so rebuilding ``sar-opt`` invalidates stale artifacts without a
manual version bump. Set ``SAR_DSL_DISABLE_CACHE=1`` to force
recompilation.

The cache is bounded: once the total size exceeds
``SAR_DSL_CACHE_MAX_SIZE`` bytes (default 2 GiB), least-recently-used
entries are evicted. Eviction runs at most once per process.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import uuid
from pathlib import Path

__all__ = ["KernelCache"]

#: Eviction threshold in bytes; 0 disables pruning entirely.
_MAX_CACHE_SIZE = int(
    os.environ.get("SAR_DSL_CACHE_MAX_SIZE", str(2 * 1024**3)))

#: Marker file recording the last use of an entry (LRU ordering).
_ACCESS_MARKER = ".accessed"

_pruned_this_process = False


def _default_cache_root() -> Path:
    root = os.environ.get("SAR_DSL_CACHE_DIR")
    if root:
        return Path(root)
    return Path.home() / ".cache" / "sar-dsl"


#: Tools whose output the cache holds. Every one of them has to be in the
#: fingerprint: a kernel passes through all of them, so rebuilding any one
#: can change the artifacts while the source and options stay identical.
_FINGERPRINTED_TOOLS = ("sar-opt", "scalehls-opt", "scalehls-translate")


def _toolchain_fingerprint() -> str:
    """Identity of the compilers that produced the cached artifacts.

    Each tool's size and mtime stand in for its content: rebuilding one
    changes them, which changes every cache key. This replaces the
    hand-maintained version counter that had to be bumped after ABI changes.
    """
    from .toolchain import find_tool

    digest = hashlib.sha256()
    for name in _FINGERPRINTED_TOOLS:
        try:
            tool = find_tool(name, required=False)
            info = os.stat(tool) if tool else None
        except Exception:  # pragma: no cover - fingerprint is best-effort
            info = None
        digest.update(f"{name}:".encode())
        digest.update(b"absent" if info is
                      None else f"{info.st_mtime_ns}:{info.st_size}".encode())
    return digest.hexdigest()[:16]


def _entry_size(path: Path) -> int:
    total = 0
    try:
        for item in path.rglob("*"):
            if item.is_file():
                total += item.stat().st_size
    except OSError:  # pragma: no cover - racing eviction
        pass
    return total


def _prune_lru(root: Path, max_size: int) -> None:
    """Evicts least-recently-used entries until the cache fits `max_size`."""
    global _pruned_this_process
    if _pruned_this_process or max_size <= 0 or not root.is_dir():
        return
    _pruned_this_process = True

    entries = []
    total = 0
    try:
        children = list(root.iterdir())
    except OSError:  # pragma: no cover
        return
    for entry in children:
        if not entry.is_dir():
            continue
        marker = entry / _ACCESS_MARKER
        try:
            used = (marker if marker.exists() else entry).stat().st_mtime
        except OSError:  # pragma: no cover - racing eviction
            continue
        size = _entry_size(entry)
        entries.append((used, size, entry))
        total += size

    if total <= max_size:
        return

    entries.sort(key=lambda item: item[0])  # oldest first
    for _, size, entry in entries:
        if total <= max_size:
            break
        try:
            shutil.rmtree(entry)
            total -= size
        except OSError:  # pragma: no cover - racing eviction
            pass


class KernelCache:

    def __init__(self, module_text: str, backend: str, options: dict):
        digest = hashlib.sha256()
        digest.update(_toolchain_fingerprint().encode())
        digest.update(module_text.encode())
        digest.update(backend.encode())
        digest.update(repr(sorted(options.items())).encode())
        self.key = digest.hexdigest()[:24]
        root = _default_cache_root()
        self.dir = root / self.key
        self.enabled = os.environ.get("SAR_DSL_DISABLE_CACHE", "0") != "1"
        # Artifacts are written even when lookups are disabled, so the
        # directory always has to exist.
        self.dir.mkdir(parents=True, exist_ok=True)
        if self.enabled:
            _prune_lru(root, _MAX_CACHE_SIZE)
            self._touch()

    def _touch(self) -> None:
        """Records this entry as most recently used."""
        try:
            (self.dir / _ACCESS_MARKER).touch()
        except OSError:  # pragma: no cover - read-only cache dir
            pass

    def path(self, filename: str) -> Path:
        return self.dir / filename

    def has(self, filename: str) -> bool:
        if not (self.enabled and self.path(filename).exists()):
            return False
        self._touch()
        return True

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
