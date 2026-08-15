"""Compiler infrastructure tests: cache behavior and backend discovery."""

import os

import numpy as np

import sar
from sar.compiler import cache as cache_module
from sar.compiler.cache import KernelCache

from conftest import requires_cpu


def test_cache_key_stable_and_distinct():
    a = KernelCache("module-a", "cpu", {})
    a2 = KernelCache("module-a", "cpu", {})
    b = KernelCache("module-b", "cpu", {})
    opt = KernelCache("module-a", "cpu", {"opt_level": 3})
    assert a.key == a2.key
    assert a.key != b.key
    assert a.key != opt.key


def test_cache_key_tracks_toolchain(tmp_path, monkeypatch):
    """Rebuilding sar-opt must invalidate cached artifacts: the key mixes in
    a fingerprint of the binary instead of a hand-maintained version."""
    monkeypatch.setenv("SAR_DSL_CACHE_DIR", str(tmp_path))
    fake = tmp_path / "sar-opt"
    fake.write_text("#!/bin/sh\n")
    fake.chmod(0o755)

    monkeypatch.setattr(cache_module, "_toolchain_fingerprint", lambda: "aaa")
    before = KernelCache("m", "cpu", {}).key
    monkeypatch.setattr(cache_module, "_toolchain_fingerprint", lambda: "bbb")
    after = KernelCache("m", "cpu", {}).key
    assert before != after


def test_cache_writes_when_lookups_disabled(tmp_path, monkeypatch):
    """SAR_DSL_DISABLE_CACHE suppresses hits but still populates the cache,
    so the directory must exist regardless."""
    monkeypatch.setenv("SAR_DSL_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("SAR_DSL_DISABLE_CACHE", "1")
    cache = KernelCache("m", "cpu", {})
    cache.write_text("artifact.txt", "hello")
    assert cache.has("artifact.txt") is False
    assert cache.read_text("artifact.txt") == "hello"


def test_cache_evicts_least_recently_used(tmp_path, monkeypatch):
    """Entries are dropped oldest-first once the cache exceeds its budget."""
    monkeypatch.setenv("SAR_DSL_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(cache_module, "_MAX_CACHE_SIZE", 4096)

    keys = []
    for i in range(4):
        monkeypatch.setattr(cache_module, "_pruned_this_process", False)
        entry = KernelCache(f"module-{i}", "cpu", {})
        entry.write_text("blob.bin", "x" * 2048)
        # Space the access markers so the LRU order is unambiguous.
        marker = entry.dir / cache_module._ACCESS_MARKER
        os.utime(marker, (1_000_000 + i, 1_000_000 + i))
        keys.append(entry.key)

    monkeypatch.setattr(cache_module, "_pruned_this_process", False)
    KernelCache("module-trigger", "cpu", {})

    surviving = {p.name for p in tmp_path.iterdir() if p.is_dir()}
    assert keys[0] not in surviving  # oldest evicted
    assert keys[-1] in surviving  # newest retained


def test_cache_atomic_write(tmp_path, monkeypatch):
    monkeypatch.setenv("SAR_DSL_CACHE_DIR", str(tmp_path))
    cache = KernelCache("m", "cpu", {})
    cache.write_text("artifact.txt", "hello")
    assert cache.read_text("artifact.txt") == "hello"
    # No temp litter left behind.
    leftovers = [p for p in cache.dir.iterdir() if p.name.endswith(".tmp")]
    assert not leftovers


@requires_cpu
def test_recompilation_hits_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("SAR_DSL_CACHE_DIR", str(tmp_path))
    N = 8

    @sar.func
    def k(a: sar.f32[N]) -> sar.f32[N]:
        return a * 4.0

    first = sar.compile(k, backend="cpu")
    so = tmp_path / KernelCache(k.to_mlir(), "cpu", {}).key / "kernel.so"
    assert so.exists()
    mtime = so.stat().st_mtime_ns

    second = sar.compile(k, backend="cpu")  # fresh driver run, same key
    assert so.stat().st_mtime_ns == mtime  # artifact reused

    x = np.ones(N, dtype=np.float32)
    np.testing.assert_allclose(first(x), second(x))


def test_backend_registry_contents():
    backends = sar.list_backends()
    assert "cpu" in backends and "scalehls" in backends


def test_unknown_backend_message():
    N = 4

    @sar.func
    def k(a: sar.f32[N]) -> sar.f32[N]:
        return a + 1.0

    try:
        k.compile(backend="nonexistent")
    except sar.ToolchainError as e:
        assert "nonexistent" in str(e) and "cpu" in str(e)
    else:
        raise AssertionError("expected ToolchainError")
