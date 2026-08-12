"""Compiler infrastructure tests: cache behavior and backend discovery."""

import numpy as np

import sar
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

    @sar.jit
    def k(a: sar.f32[N]) -> sar.f32[N]:
        return a * 4.0

    first = sar.compile(k, backend="cpu")
    so = tmp_path / KernelCache(k.to_mlir(), "cpu", {}).key / "kernel.so"
    assert so.exists()
    mtime = so.stat().st_mtime_ns

    second = sar.compile(k, backend="cpu")  # fresh driver run, same key
    assert so.stat().st_mtime_ns == mtime   # artifact reused

    x = np.ones(N, dtype=np.float32)
    np.testing.assert_allclose(first(x), second(x))


def test_backend_registry_contents():
    backends = sar.list_backends()
    assert "cpu" in backends and "scalehls" in backends


def test_unknown_backend_message():
    N = 4

    @sar.jit
    def k(a: sar.f32[N]) -> sar.f32[N]:
        return a + 1.0

    try:
        k.compile(backend="nonexistent")
    except sar.ToolchainError as e:
        assert "nonexistent" in str(e) and "cpu" in str(e)
    else:
        raise AssertionError("expected ToolchainError")
