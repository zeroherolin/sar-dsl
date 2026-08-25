"""On-disk artifact cache, keyed by kernel content and backend options.

Layout: ``$SAR_DSL_CACHE_DIR/<key>/<stage-artifact>``. The key is a SHA-256
over a toolchain fingerprint, the MLIR module, the backend name and its
options, so rebuilding ``sar-opt`` invalidates stale artifacts without a
manual version bump. Set ``SAR_DSL_DISABLE_CACHE=1`` to force
recompilation.

The cache is bounded: once the total size exceeds
``SAR_DSL_CACHE_MAX_SIZE`` bytes (default 2 GiB), least-recently-used
entries are evicted. Eviction runs once per cache root per process.
"""

from __future__ import annotations

import hashlib
import functools
import os
import platform
import shutil
import time
import uuid
from pathlib import Path
from typing import Optional

try:
    import fcntl
except ImportError:  # pragma: no cover - cache locking is Linux-specific
    fcntl = None

__all__ = ["KernelCache"]

#: Marker file recording the last use of an entry (LRU ordering).
_ACCESS_MARKER = ".accessed"

#: Entries used within this window are never evicted: a fresh marker
#: usually means another process is compiling into the entry right now.
_PRUNE_GRACE_SECONDS = 600

#: Last prune time per root; a full directory scan is throttled.
_pruned_roots = {}
_PRUNE_INTERVAL_SECONDS = 30


def _max_cache_size() -> int:
    """Eviction threshold in bytes (0 disables pruning); read per cache
    construction rather than at import, so the environment can change it
    in a running process."""
    raw = os.environ.get("SAR_DSL_CACHE_MAX_SIZE", str(2 * 1024**3))
    try:
        value = int(raw)
    except ValueError as err:
        raise ValueError(
            "SAR_DSL_CACHE_MAX_SIZE must be a non-negative integer") from err
    if value < 0:
        raise ValueError(
            "SAR_DSL_CACHE_MAX_SIZE must be a non-negative integer")
    return value


def _default_cache_root() -> Path:
    root = os.environ.get("SAR_DSL_CACHE_DIR")
    if root:
        return Path(root)
    return Path.home() / ".cache" / "sar-dsl"


#: Tools whose output the cache holds. Every one of them has to be in the
#: fingerprint: a kernel passes through all of them, so rebuilding any one
#: can change the artifacts while the source and options stay identical.
#: `clang` and `mlir-translate` produce the CPU backend's final object, and
#: libsar_runtime is linked into it, so all three belong here too.
_FINGERPRINTED_TOOLS = ("sar-opt", "sar-translate", "mlir-translate", "clang")


@functools.lru_cache(maxsize=32)
def _file_identity(path: str, mtime_ns: int, size: int) -> bytes:
    digest = hashlib.sha256()
    digest.update(str(Path(path).resolve()).encode())
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.digest()


def _driver_fingerprint() -> str:
    """Content identity of the Python code that orchestrates compilation."""
    package = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(package.rglob("*")):
        if not path.is_file() or path.name == "_build_config.py":
            continue
        if path.suffix != ".py" and path.name != "hls_config.yaml":
            continue
        digest.update(str(path.relative_to(package)).encode())
        try:
            digest.update(path.read_bytes())
        except OSError:
            digest.update(b"unreadable")
    return digest.hexdigest()[:16]


def _cpu_identity() -> str:
    """Identity of the host CPU, as far as codegen is concerned.

    `-march=native` bakes the build machine's ISA into the CPU backend's
    artifacts, so a cache shared across machines must miss on a different
    microarchitecture rather than serve a binary that traps. The model
    name plus a digest of the feature flags is what actually varies the
    output; `platform.processor()` is the fallback where /proc/cpuinfo
    does not exist.
    """
    model = flags = ""
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if not model and line.startswith("model name"):
                    model = line.partition(":")[2].strip()
                elif not flags and line.startswith("flags"):
                    flags = line.partition(":")[2].strip()
                if model and flags:
                    break
    except OSError:  # pragma: no cover - non-Linux host
        pass
    if model or flags:
        return f"{model}:{hashlib.sha256(flags.encode()).hexdigest()[:12]}"
    return platform.processor()


def _toolchain_fingerprint() -> str:
    """Identity of the compilers that produced the cached artifacts.

    The path and content are authoritative. Size and mtime only memoize the
    content hash within this process.
    """
    from .toolchain import find_runtime_library, find_tool

    def stamp(path) -> bytes:
        try:
            info = os.stat(path) if path else None
        except OSError:  # pragma: no cover - fingerprint is best-effort
            info = None
        if info is None:
            return b"absent"
        try:
            return _file_identity(str(path), info.st_mtime_ns, info.st_size)
        except OSError:  # pragma: no cover - racing tool replacement
            return b"unreadable"

    digest = hashlib.sha256()
    for name in _FINGERPRINTED_TOOLS:
        try:
            tool = find_tool(name, required=False)
        except Exception:  # pragma: no cover - fingerprint is best-effort
            tool = None
        digest.update(f"{name}:".encode())
        digest.update(stamp(tool))
    try:
        runtime = find_runtime_library()
    except Exception:  # pragma: no cover - fingerprint is best-effort
        runtime = None
    digest.update(b"libsar_runtime:")
    digest.update(stamp(runtime))
    omp = os.environ.get("SAR_DSL_OMP_LIB")
    if not omp:
        try:
            clang = find_tool("clang", required=False)
        except Exception:
            clang = None
        if clang:
            candidate = (Path(clang).resolve().parent.parent / "lib" /
                         "libomp.so")
            omp = str(candidate) if candidate.is_file() else None
    digest.update(b"openmp:")
    digest.update(stamp(omp))
    # Host codegen identity: `-march=native` bakes the build machine's ISA
    # into the artifact, so a shared cache must not hand it to a different
    # CPU. platform.machine() alone is too coarse; the CPU model and flag
    # set are what actually vary the output.
    digest.update(b"host:")
    digest.update(f"{platform.machine()}:{platform.system()}:"
                  f"{_cpu_identity()}".encode())
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
    if max_size <= 0 or not root.is_dir():
        return
    now = time.monotonic()
    last = _pruned_roots.get(root)
    if last is not None and now - last < _PRUNE_INTERVAL_SECONDS:
        return
    _pruned_roots[root] = now

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

    fresh = time.time() - _PRUNE_GRACE_SECONDS
    entries.sort(key=lambda item: item[0])  # oldest first
    for used, size, entry in entries:
        if total <= max_size:
            break
        if used > fresh:
            break
        lock = None
        try:
            if fcntl is not None:
                lock = (entry / ".lock").open("a+")
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            shutil.rmtree(entry)
            total -= size
        except (BlockingIOError, OSError):  # pragma: no cover - racing use
            pass
        finally:
            if lock is not None:
                lock.close()


class KernelCache:

    def __init__(self,
                 module_text: str,
                 backend: str,
                 options: dict,
                 backend_fingerprint: str = ""):
        digest = hashlib.sha256()
        digest.update(_toolchain_fingerprint().encode())
        digest.update(_driver_fingerprint().encode())
        digest.update(backend_fingerprint.encode())
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
        self._lock = (self.dir / ".lock").open("a+")
        if fcntl is not None:
            fcntl.flock(self._lock, fcntl.LOCK_SH)
        _prune_lru(root, _max_cache_size())
        self._touch()
        self._root = root

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

    def read_if_cached(self, filename: str) -> Optional[str]:
        """The cached text, or None on a miss. A hit whose read then fails
        is also a miss: a concurrent prune in another process can drop the
        entry between the existence check and the read."""
        if not self.has(filename):
            return None
        try:
            return self.read_text(filename)
        except OSError:
            return None

    def write_text(self, filename: str, content: str) -> Path:
        """Atomic write (temp file + rename): concurrent compilations of the
        same kernel may duplicate work but never observe partial files."""
        p = self.path(filename)
        tmp = p.with_name(f".{p.name}.{uuid.uuid4().hex[:8]}.tmp")
        try:
            tmp.write_text(content)
            os.replace(tmp, p)
        finally:
            # A failed write or rename must not litter the entry with .tmp
            # files; after a successful rename there is nothing to unlink.
            tmp.unlink(missing_ok=True)
        _prune_lru(self._root, _max_cache_size())
        return p

    def scratch_path(self, filename: str) -> Path:
        """A unique temporary path inside the cache directory; pass to tools
        that write files directly, then `publish` the result."""
        return self.path(f".{filename}.{uuid.uuid4().hex[:8]}.tmp")

    def publish(self, scratch: Path, filename: str) -> Path:
        p = self.path(filename)
        os.replace(scratch, p)
        _prune_lru(self._root, _max_cache_size())
        return p
