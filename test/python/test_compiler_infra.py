"""Compiler infrastructure tests: cache behavior and backend discovery."""

import os
import sys

import numpy as np
import pytest

import sar
from sar.compiler import cache as cache_module
from sar.compiler.cache import KernelCache

from conftest import requires_cpu, requires_hls


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
    monkeypatch.setenv("SAR_DSL_CACHE_MAX_SIZE", "4096")

    keys = []
    for i in range(4):
        cache_module._pruned_roots.clear()
        entry = KernelCache(f"module-{i}", "cpu", {})
        entry.write_text("blob.bin", "x" * 2048)
        # Space the access markers so the LRU order is unambiguous.
        marker = entry.dir / cache_module._ACCESS_MARKER
        os.utime(marker, (1_000_000 + i, 1_000_000 + i))
        keys.append(entry.key)

    cache_module._pruned_roots.clear()
    KernelCache("module-trigger", "cpu", {})

    surviving = {p.name for p in tmp_path.iterdir() if p.is_dir()}
    assert keys[0] not in surviving  # oldest evicted
    assert keys[-1] in surviving  # newest retained


def test_cache_prune_spares_recently_used_entries(tmp_path, monkeypatch):
    """An entry used within the grace window is never evicted, even when
    the cache is over budget: a fresh access marker usually means another
    process is compiling into the entry right now."""
    monkeypatch.setenv("SAR_DSL_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("SAR_DSL_CACHE_MAX_SIZE", "4096")
    cache_module._pruned_roots.clear()
    fresh = KernelCache("fresh-entry", "cpu", {})
    fresh.write_text("blob.bin", "x" * 8192)  # over budget on its own

    cache_module._pruned_roots.clear()
    KernelCache("trigger", "cpu", {})
    assert fresh.dir.exists()


def test_cache_atomic_write(tmp_path, monkeypatch):
    monkeypatch.setenv("SAR_DSL_CACHE_DIR", str(tmp_path))
    cache = KernelCache("m", "cpu", {})
    cache.write_text("artifact.txt", "hello")
    assert cache.read_text("artifact.txt") == "hello"
    # No temp litter left behind.
    leftovers = [p for p in cache.dir.iterdir() if p.name.endswith(".tmp")]
    assert not leftovers


def test_cache_write_failure_leaves_no_tmp_litter(tmp_path, monkeypatch):
    """A write that fails midway must clean up its temp file, or the
    entry fills with orphaned .tmp files over time."""
    monkeypatch.setenv("SAR_DSL_CACHE_DIR", str(tmp_path))
    cache = KernelCache("m", "cpu", {})
    original_replace = os.replace

    def failing_replace(src, dst):
        raise OSError("simulated rename failure")

    monkeypatch.setattr(cache_module.os, "replace", failing_replace)
    with pytest.raises(OSError, match="simulated"):
        cache.write_text("artifact.txt", "hello")
    monkeypatch.setattr(cache_module.os, "replace", original_replace)
    leftovers = [p for p in cache.dir.iterdir() if p.name.endswith(".tmp")]
    assert not leftovers


def test_cache_read_if_cached_treats_races_as_misses(tmp_path, monkeypatch):
    """A hit whose read then fails (a concurrent prune dropped the entry
    between the existence check and the read) must be a miss, not an
    OSError escaping into the compilation driver."""
    monkeypatch.setenv("SAR_DSL_CACHE_DIR", str(tmp_path))
    cache = KernelCache("m", "cpu", {})
    assert cache.read_if_cached("a.txt") is None
    cache.write_text("a.txt", "hello")
    assert cache.read_if_cached("a.txt") == "hello"

    def vanished(filename):
        raise FileNotFoundError(filename)

    monkeypatch.setattr(cache, "read_text", vanished)
    assert cache.read_if_cached("a.txt") is None


def test_cache_fingerprint_tracks_cpu_identity(monkeypatch):
    """`-march=native` bakes the host ISA into CPU artifacts, so a cache
    on a shared filesystem must miss on a different microarchitecture."""
    monkeypatch.setattr(cache_module, "_cpu_identity", lambda: "cpu-a")
    on_a = cache_module._toolchain_fingerprint()
    monkeypatch.setattr(cache_module, "_cpu_identity", lambda: "cpu-b")
    on_b = cache_module._toolchain_fingerprint()
    assert on_a != on_b
    assert cache_module._cpu_identity()  # non-empty on this platform


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
    assert "cpu" in backends and "hls" in backends


def test_builtin_backends_are_the_real_modules():
    """Built-in backends must be imported under their package names, not
    re-executed from their file paths: a second execution would duplicate
    every class object, so isinstance checks against sar.backends.hls
    (e.g. HLSDesign) would fail on registry-built objects."""
    import sar.backends.cpu
    import sar.backends.hls

    backends = sar.list_backends()
    assert backends["cpu"] is sar.backends.cpu.Backend
    assert backends["hls"] is sar.backends.hls.Backend
    assert "sar_backend_cpu" not in sys.modules
    assert "sar_backend_hls" not in sys.modules


@requires_cpu
def test_compile_memo_reuses_and_tolerates_unhashable_options():
    N = 8

    @sar.func
    def k(a: sar.f32[N]) -> sar.f32[N]:
        return a * 3.0

    assert k.compile("cpu") is k.compile("cpu")  # memo hit
    # An unhashable option value skips the in-process memo (the on-disk
    # cache still applies) instead of raising a bare TypeError.
    launcher = k.compile("cpu", options={"unhashable": [1, 2]})
    x = np.ones(N, dtype=np.float32)
    np.testing.assert_allclose(launcher(x), 3 * x)


@requires_hls
def test_compile_config_file_changes_are_picked_up(tmp_path):
    """`options={"config": path}` points at a file whose contents may
    change between compiles; the in-process memo must not serve a design
    built against the old contents."""
    n = 8

    @sar.func
    def memo_cfg(x: sar.c64[n, n]) -> sar.c64[n, n]:
        return x * 2.0

    cfg = tmp_path / "hls.yaml"
    cfg.write_text("top_func: first_top\n")
    first = memo_cfg.compile("hls", options={"config": str(cfg)})
    cfg.write_text("top_func: second_top\n")
    second = memo_cfg.compile("hls", options={"config": str(cfg)})
    assert first.name == "first_top"
    assert second.name == "second_top"


@requires_cpu
def test_op_defaults_share_one_specialization():
    """A call relying on a default and one spelling it explicitly are the
    same specialization; without normalization the second would trace and
    compile an identical variant."""
    traces = []

    @sar.op
    def scaled(x, factor=2.0):
        traces.append(1)
        return x * factor

    x = np.ones((4, ), dtype=np.float32)
    np.testing.assert_allclose(scaled(x), 2 * x)
    np.testing.assert_allclose(scaled(x, factor=2.0), 2 * x)
    assert len(traces) == 1


@requires_cpu
def test_openmp_rpath_flags_survive_commas(tmp_path, monkeypatch):
    """-Wl,-rpath,<dir> splits on commas inside <dir>; the -Xlinker
    spelling hands the path to the linker in one piece."""
    from sar.backends.cpu import compiler as cpu_compiler

    weird = tmp_path / "lib,with,commas"
    weird.mkdir()
    (weird / "libomp.so").write_text("")
    monkeypatch.setenv("SAR_DSL_OMP_LIB", str(weird / "libomp.so"))
    flags = cpu_compiler._openmp_link_flags()
    assert "-Xlinker" in flags
    assert f"-rpath={weird}" in flags
    assert not any(f.startswith("-Wl,-rpath") for f in flags)


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
