"""Compiler infrastructure tests: cache behavior and backend discovery."""

import os
import subprocess
import sys

import numpy as np
import pytest

import sar
from sar.compiler import cache as cache_module
from sar.compiler.cache import KernelCache

from conftest import REPO_ROOT, requires_cpu, requires_hls


def test_cache_key_stable_and_distinct():
    a = KernelCache("module-a", "cpu", {})
    a2 = KernelCache("module-a", "cpu", {})
    b = KernelCache("module-b", "cpu", {})
    opt = KernelCache("module-a", "cpu", {"opt_level": 3})
    assert a.key == a2.key
    assert a.key != b.key
    assert a.key != opt.key
    assert len(a.key) == 64
    for cache in (a, a2, b, opt):
        cache.close()


def test_cache_key_tracks_toolchain(tmp_path, monkeypatch):
    """Rebuilding sar-opt must invalidate cached artifacts: the key mixes in
    a fingerprint of the binary instead of a hand-maintained version."""
    monkeypatch.setenv("SAR_DSL_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(cache_module, "_toolchain_fingerprint", lambda: "aaa")
    before = KernelCache("m", "cpu", {}).key
    monkeypatch.setattr(cache_module, "_toolchain_fingerprint", lambda: "bbb")
    after = KernelCache("m", "cpu", {}).key
    assert before != after


def test_compilation_error_exposes_structured_diagnostic():
    error = sar.CompilationError(
        "lower", ["sar-opt", "-"],
        'kernel.py:12:3: error: bad tensor shape\n  note: details\n')
    assert error.location == {"file": "kernel.py", "line": 12, "column": 3}
    assert error.diagnostic == "bad tensor shape"
    assert error.as_dict()["stage"] == "lower"


def test_cache_key_tracks_python_compiler_strategy(monkeypatch):
    monkeypatch.setattr(cache_module, "_driver_fingerprint", lambda: "aaa")
    before = KernelCache("m", "hls", {}).key
    monkeypatch.setattr(cache_module, "_driver_fingerprint", lambda: "bbb")
    after = KernelCache("m", "hls", {}).key
    assert before != after


def test_backend_fingerprint_tracks_external_source(tmp_path, monkeypatch):
    import types

    from sar.backends import BaseBackend, backend_fingerprint

    source = tmp_path / "compiler.py"
    source.write_text("VERSION = 1\n")
    module_name = "sar_test_external_fingerprint"
    monkeypatch.setitem(sys.modules, module_name,
                        types.SimpleNamespace(__file__=str(source)))
    plugin = type("Plugin", (BaseBackend, ), {"__module__": module_name})
    before = backend_fingerprint(plugin)
    source.write_text("VERSION = 2\n")
    after = backend_fingerprint(plugin)
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


def test_disabled_cache_still_honours_lru_limit(tmp_path, monkeypatch):
    monkeypatch.setenv("SAR_DSL_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("SAR_DSL_CACHE_MAX_SIZE", "4096")
    monkeypatch.setenv("SAR_DSL_DISABLE_CACHE", "1")
    entries = []
    for i in range(3):
        cache_module._pruned_roots.clear()
        entry = KernelCache(f"disabled-{i}", "cpu", {})
        entry.write_text("blob.bin", "x" * 3072)
        os.utime(entry.dir / cache_module._ACCESS_MARKER,
                 (1_000_000 + i, 1_000_000 + i))
        entry._lock.close()
        entries.append(entry)
    cache_module._pruned_roots.clear()
    KernelCache("disabled-trigger", "cpu", {})
    assert sum(path.is_dir() for path in tmp_path.iterdir()) < len(entries) + 1


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


@pytest.mark.parametrize("value", ["invalid", "-1"])
def test_cache_size_must_be_non_negative(value, monkeypatch):
    monkeypatch.setenv("SAR_DSL_CACHE_MAX_SIZE", value)
    with pytest.raises(ValueError, match="non-negative integer"):
        cache_module._max_cache_size()


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


def test_cache_context_releases_lock_descriptor(tmp_path, monkeypatch):
    monkeypatch.setenv("SAR_DSL_CACHE_DIR", str(tmp_path))
    with KernelCache("m", "cpu", {}) as cache:
        lock = cache._lock
        assert not lock.closed
    assert lock.closed


def test_cache_stats_and_clear_skip_locked_entries(tmp_path, monkeypatch):
    monkeypatch.setenv("SAR_DSL_CACHE_DIR", str(tmp_path))
    active = KernelCache("active", "cpu", {})
    active.write_text("artifact", "x" * 16)
    inactive = KernelCache("inactive", "cpu", {})
    inactive.write_text("artifact", "y" * 32)
    inactive.close()

    stats = cache_module.cache_stats()
    assert stats["entries"] == 2
    assert stats["bytes"] >= 48
    cleared = cache_module.clear_cache()
    assert cleared == {
        "removed": 1,
        "retained": 1,
        "root": str(tmp_path),
    }
    assert active.dir.exists()
    assert not inactive.dir.exists()
    active.close()
    assert cache_module.clear_cache()["removed"] == 1


def test_clear_cache_spares_entry_mid_initialization(tmp_path, monkeypatch):
    """A fresh directory with no access marker is a KernelCache.__init__
    in flight (mkdir done, flock not yet held); clear_cache must not
    rmtree it out from under the constructor."""
    monkeypatch.setenv("SAR_DSL_CACHE_DIR", str(tmp_path))
    mid_init = tmp_path / ("0" * 64)
    mid_init.mkdir()
    cleared = cache_module.clear_cache()
    assert cleared == {"removed": 0, "retained": 1, "root": str(tmp_path)}
    assert mid_init.is_dir()


def test_cache_init_survives_concurrent_clear(tmp_path, monkeypatch):
    """clear_cache in another process can remove the entry between our
    open and flock; __init__ must detect the unlinked .lock inode and
    recreate the entry instead of writing into a removed directory."""
    monkeypatch.setenv("SAR_DSL_CACHE_DIR", str(tmp_path))
    real_flock = cache_module.fcntl.flock
    raced = []

    def racing_flock(fd, op):
        real_flock(fd, op)
        if not raced and op == cache_module.fcntl.LOCK_SH:
            raced.append(True)
            for entry in tmp_path.iterdir():
                if entry.is_dir():
                    __import__("shutil").rmtree(entry)

    monkeypatch.setattr(cache_module.fcntl, "flock", racing_flock)
    cache = KernelCache("m", "cpu", {})
    assert raced
    assert cache.dir.is_dir()
    cache.write_text("artifact.txt", "ok")
    assert cache.read_text("artifact.txt") == "ok"
    cache.close()


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


def test_cache_fingerprint_tracks_openmp_runtime(tmp_path, monkeypatch):
    omp = tmp_path / "libomp.so"
    omp.write_bytes(b"one")
    monkeypatch.setenv("SAR_DSL_OMP_LIB", str(omp))
    before = cache_module._toolchain_fingerprint()
    omp.write_bytes(b"different-runtime")
    after = cache_module._toolchain_fingerprint()
    assert before != after


@requires_cpu
def test_recompilation_hits_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("SAR_DSL_CACHE_DIR", str(tmp_path))
    N = 8

    @sar.func
    def k(a: sar.f32[N]) -> sar.f32[N]:
        return a * 4.0

    first = sar.compile(k, backend="cpu")
    defaults = {"opt_level": 3, "native_codegen": True}
    from sar.backends import backend_fingerprint
    fingerprint = backend_fingerprint(sar.get_backend("cpu"))
    so = tmp_path / KernelCache(k.to_mlir(), "cpu", defaults,
                                fingerprint).key / "kernel.so"
    assert so.exists()
    mtime = so.stat().st_mtime_ns

    second = sar.compile(k, backend="cpu")  # fresh driver run, same key
    assert so.stat().st_mtime_ns == mtime  # artifact reused

    x = np.ones(N, dtype=np.float32)
    np.testing.assert_allclose(first(x), second(x))


@requires_hls
def test_pipeline_dump_exports_ir_facts_and_manifest(tmp_path):

    @sar.func
    def scale(x: sar.f32[8]) -> sar.f32[8]:
        return x * 2.0

    out = scale.dump_pipeline(tmp_path, backend="hls")
    names = {path.name for path in out.iterdir()}
    assert {
        "kernel.sar.mlir", "kernel.affine.mlir", "kernel.hls.mlir",
        "kernel.sar.facts.json", "kernel.affine.facts.json",
        "pipeline_manifest.json"
    } <= names
    manifest = __import__("json").loads(
        (out / "pipeline_manifest.json").read_text())
    assert manifest["kernel"] == "scale"
    assert manifest["backend"] == "hls"
    assert manifest["stages"] == ["lower", "hls"]


@requires_hls
def test_pipeline_dump_replaces_a_previous_export(tmp_path):

    @sar.func
    def scale(x: sar.f32[8]) -> sar.f32[8]:
        return x * 2.0

    out = tmp_path / "pipeline"
    scale.dump_pipeline(out, backend="hls")
    (out / "stale.txt").write_text("old export")
    scale.dump_pipeline(out, backend="cpu")
    assert not (out / "stale.txt").exists()
    manifest = __import__("json").loads(
        (out / "pipeline_manifest.json").read_text())
    assert manifest["backend"] == "cpu"


def test_pipeline_dump_excludes_cache_bookkeeping(tmp_path, monkeypatch):
    """Every cache entry holds dotfile bookkeeping (.lock, the .accessed
    LRU marker, possibly crash-leftover .*.tmp scratch files); an export
    must contain artifacts only."""
    from sar import compiler
    from sar.backends import KernelMetadata

    monkeypatch.setenv("SAR_DSL_CACHE_DIR", str(tmp_path / "cache"))
    cache = KernelCache("m", "cpu", {})
    cache.scratch_path("kernel.so").write_text("crashed tool leftover")

    def stage(artifact, metadata, stage_cache):
        stage_cache.write_text("kernel.stage.txt", artifact)
        return artifact

    backend_obj = __import__("types").SimpleNamespace(
        make_launcher=lambda artifact, metadata: artifact)
    metadata = KernelMetadata(name="k", arg_types=[], result_types=[])
    monkeypatch.setattr(
        compiler, "_prepare", lambda kernel, backend, options:
        (backend_obj, "module", metadata, {
            "stage": stage
        }, cache))

    out = compiler.dump_pipeline(object(), tmp_path / "dump")
    names = {path.name for path in out.iterdir()}
    assert names == {"kernel.stage.txt", "pipeline_manifest.json"}
    manifest = __import__("json").loads(
        (out / "pipeline_manifest.json").read_text())
    assert manifest["files"] == ["kernel.stage.txt"]
    assert manifest["stages"] == ["stage"]


def test_backend_registry_contents():
    backends = sar.list_backends()
    assert "cpu" in backends and "hls" in backends


def test_doctor_reports_tools_backends_and_cache(tmp_path, monkeypatch):
    from sar import __main__ as cli

    monkeypatch.setenv("SAR_DSL_CACHE_DIR", str(tmp_path))
    data = cli.doctor()
    assert data["sar_dsl_version"] == sar.__version__
    assert {"sar-opt", "sar-translate", "mlir-translate", "clang"} <= \
        set(data["tools"])
    assert {"cpu", "hls"} <= set(data["backends"])
    assert data["cache"]["root"] == str(tmp_path)


def test_doctor_reports_invalid_tool_override(tmp_path, monkeypatch):
    from sar import __main__ as cli

    monkeypatch.setenv("SAR_DSL_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("SAR_DSL_TOOL_CLANG", str(tmp_path / "missing"))
    data = cli.doctor()
    assert data["tools"]["clang"]["path"] is None
    assert "does not exist" in data["tools"]["clang"]["error"]


def test_doctor_json_cli_is_machine_readable(tmp_path):
    env = dict(os.environ,
               PYTHONPATH=str(REPO_ROOT / "python"),
               SAR_DSL_CACHE_DIR=str(tmp_path))
    result = subprocess.run([sys.executable, "-m", "sar", "doctor", "--json"],
                            capture_output=True,
                            text=True,
                            env=env,
                            check=True)
    data = __import__("json").loads(result.stdout)
    assert "backends" in data and "cache" in data


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


def test_external_backend_package_supports_relative_imports(tmp_path):
    package = tmp_path / "my_backend"
    package.mkdir()
    (package / "__init__.py").write_text("from .compiler import Backend\n")
    (package / "helper.py").write_text("NAME = 'external_test'\n")
    (package / "compiler.py").write_text("""
from sar.backends.base import BaseBackend
from .helper import NAME

class Backend(BaseBackend):
    name = NAME

    @classmethod
    def is_available(cls):
        return True

    def add_stages(self, stages, metadata):
        pass

    def make_launcher(self, artifact, metadata):
        return artifact
""")
    code = ("import sar; "
            "backend = sar.get_backend('external_test'); "
            "print(backend.__module__)")
    env = dict(os.environ,
               PYTHONPATH=str(REPO_ROOT / "python"),
               SAR_DSL_BACKEND_PATH=str(package))
    result = subprocess.run([sys.executable, "-c", code],
                            capture_output=True,
                            text=True,
                            env=env)
    assert result.returncode == 0, result.stderr
    assert "sar_backend_my_backend_" in result.stdout


def test_external_backend_discovery_is_thread_safe(tmp_path):
    package = tmp_path / "threaded_backend"
    package.mkdir()
    (package / "__init__.py").write_text("import time\n"
                                         "time.sleep(0.1)\n"
                                         "from .compiler import Backend\n")
    (package / "compiler.py").write_text("""
from sar.backends.base import BaseBackend

class Backend(BaseBackend):
    name = "threaded_test"

    @classmethod
    def is_available(cls):
        return True

    def add_stages(self, stages, metadata):
        pass

    def make_launcher(self, artifact, metadata):
        return artifact
""")
    code = """
import threading
import sar

barrier = threading.Barrier(2)
results = []
def discover():
    barrier.wait()
    results.append("threaded_test" in sar.list_backends())
threads = [threading.Thread(target=discover) for _ in range(2)]
for thread in threads:
    thread.start()
for thread in threads:
    thread.join()
assert results == [True, True], results
"""
    env = dict(os.environ,
               PYTHONPATH=str(REPO_ROOT / "python"),
               SAR_DSL_BACKEND_PATH=str(package))
    result = subprocess.run([sys.executable, "-c", code],
                            capture_output=True,
                            text=True,
                            env=env)
    assert result.returncode == 0, result.stderr


@requires_cpu
def test_compile_memo_reuses_and_validates_options():
    N = 8

    @sar.func
    def k(a: sar.f32[N]) -> sar.f32[N]:
        return a * 3.0

    assert k.compile("cpu") is k.compile("cpu")  # memo hit
    # Distinct options are distinct launchers, not a stale memo hit.
    other = k.compile("cpu", options={"opt_level": 1})
    assert other is not k.compile("cpu")
    # A typo'd option is rejected instead of silently ignored.
    with pytest.raises(sar.errors.ToolchainError, match="unknown cpu"):
        k.compile("cpu", options={"opt_levle": 1})
    with pytest.raises(sar.errors.ToolchainError, match="opt_level"):
        k.compile("cpu", options={"opt_level": 4})
    with pytest.raises(sar.errors.ToolchainError, match="native_codegen"):
        k.compile("cpu", options={"native_codegen": "yes"})


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


def test_generic_specializations_are_bounded_and_clearable(monkeypatch):
    monkeypatch.setenv("SAR_DSL_SPECIALIZATIONS_MAX", "2")

    @sar.func
    def scale(x):
        return x * 2.0

    first = scale.specialize(sar.f32[2])
    scale.specialize(sar.f32[3])
    scale.specialize(sar.f32[4])
    assert scale.specialization_stats() == {"variants": 2}
    assert scale.specialize(sar.f32[2]) is not first  # oldest was evicted
    scale.clear_specializations()
    assert scale.specialization_stats() == {"variants": 0}


def test_compiled_launcher_memo_is_bounded_and_clearable(monkeypatch):
    monkeypatch.setenv("SAR_DSL_COMPILED_MAX", "2")

    @sar.func
    def scale(x: sar.f32[2]) -> sar.f32[2]:
        return x * 2.0

    from sar import compiler
    launches = []

    def fake_compile(*args, **kwargs):
        launches.append(object())
        return launches[-1]

    monkeypatch.setattr(compiler, "compile", fake_compile)
    scale.compile("cpu", {"opt_level": 0})
    scale.compile("cpu", {"opt_level": 1})
    scale.compile("cpu", {"opt_level": 2})
    assert scale.compilation_stats() == {"compiled_launchers": 2}
    scale.clear_compiled()
    assert scale.compilation_stats() == {"compiled_launchers": 0}


def test_operator_constant_array_key_is_a_digest():
    import sar.language as language

    value = np.arange(1024, dtype=np.float64)
    frozen = language._freeze_for_key(value)
    assert frozen[:3] == ("ndarray", value.shape, str(value.dtype))
    assert len(frozen[3]) == 64
    assert value.tobytes() not in frozen


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


def test_operator_exposes_direct_specialization_api():

    @sar.op
    def twice(x):
        return x * 2.0

    variant = twice.specialize(sar.f32[4])
    assert "tensor<4xf32>" in variant.to_mlir()


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


def test_tool_timeout_is_reported(monkeypatch):
    from sar.compiler.toolchain import run_tool

    monkeypatch.setenv("SAR_DSL_TOOL_TIMEOUT_SECONDS", "0.01")
    with pytest.raises(sar.CompilationError, match="timed out"):
        run_tool("slow-test",
                 [sys.executable, "-c", "import time; time.sleep(1)"])


@pytest.mark.parametrize("value", ["not-a-number", "nan", "inf"])
def test_tool_timeout_must_be_finite(value, monkeypatch):
    from sar.compiler.toolchain import run_tool

    monkeypatch.setenv("SAR_DSL_TOOL_TIMEOUT_SECONDS", value)
    with pytest.raises(sar.ToolchainError,
                       match="must be (?:a number|finite)"):
        run_tool("timeout-config", [sys.executable, "-c", "pass"])


def test_explicit_build_config_precedes_stale_package_config(
        tmp_path, monkeypatch):
    import types

    from sar.compiler import toolchain

    config = tmp_path / "_build_config.py"
    config.write_text("SAR_DSL_TOOL_DIR = 'fresh-tools'\n")
    monkeypatch.setenv("SAR_DSL_BUILD_CONFIG", str(config))
    monkeypatch.setenv("SAR_DSL_BUILD_DIR", str(tmp_path / "missing-build"))
    monkeypatch.setattr(sar,
                        "_build_config",
                        types.SimpleNamespace(SAR_DSL_TOOL_DIR="stale-tools"),
                        raising=False)
    loaded = toolchain._load_build_config()
    assert loaded.SAR_DSL_TOOL_DIR == "fresh-tools"


@pytest.mark.parametrize(
    "variable,value,expected",
    [
        ("SAR_DSL_BUILD_CONFIG", "missing.py", "does not exist"),
        ("SAR_DSL_BUILD_DIR", "missing-build", "has no"),
    ],
)
def test_explicit_build_locations_fail_when_missing(variable, value, expected,
                                                    tmp_path, monkeypatch):
    from sar.compiler import toolchain

    monkeypatch.delenv("SAR_DSL_BUILD_CONFIG", raising=False)
    monkeypatch.delenv("SAR_DSL_BUILD_DIR", raising=False)
    monkeypatch.setenv(variable, str(tmp_path / value))
    with pytest.raises(sar.ToolchainError, match=expected):
        toolchain._load_build_config()


def test_tool_override_must_be_executable(tmp_path, monkeypatch):
    from sar.compiler import toolchain

    probe = tmp_path / "probe"
    probe.write_text("#!/bin/sh\n")
    monkeypatch.setenv("SAR_DSL_TOOL_NOT_EXECUTABLE_PROBE", str(probe))
    toolchain.find_tool.cache_clear()
    with pytest.raises(sar.ToolchainError, match="not executable"):
        toolchain.find_tool("not-executable-probe")
    toolchain.find_tool.cache_clear()


def test_tool_override_changes_are_visible_in_process(tmp_path, monkeypatch):
    from sar.compiler import toolchain

    name = "dynamic-tool-probe"
    variable = "SAR_DSL_TOOL_DYNAMIC_TOOL_PROBE"
    monkeypatch.delenv(variable, raising=False)
    assert toolchain.find_tool(name, required=False) is None

    probe = tmp_path / name
    probe.write_text("#!/bin/sh\nexit 0\n")
    probe.chmod(0o755)
    monkeypatch.setenv(variable, str(probe))
    assert toolchain.find_tool(name) == str(probe)
