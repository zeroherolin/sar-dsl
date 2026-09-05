"""Runtime library loading tests: handle memoization and the error
channel fallback. Native execution is covered by test_runtime_direct.py;
these tests fake the dlopen so they run without a built runtime."""

import types

import pytest

import sar.runtime as runtime


class _FakeLibrary:
    """Stands in for ctypes.CDLL: every symbol lookup succeeds and the
    prototype assignments land on plain namespaces."""

    def __init__(self, path):
        self.path = path

    def __getattr__(self, name):
        fn = types.SimpleNamespace()
        setattr(self, name, fn)
        return fn


def test_runtime_library_handle_is_memoized_per_path(tmp_path, monkeypatch):
    """The dlopen handle is reused for a given resolved path, while path
    resolution itself stays uncached: pointing SAR_DSL_RUNTIME_LIB at a
    different file must load the new library."""
    loaded = []

    def fake_cdll(path):
        loaded.append(path)
        return _FakeLibrary(path)

    monkeypatch.setattr(runtime.ctypes, "CDLL", fake_cdll)
    lib_a = tmp_path / "a.so"
    lib_a.write_bytes(b"")
    lib_b = tmp_path / "b.so"
    lib_b.write_bytes(b"")

    monkeypatch.setenv("SAR_DSL_RUNTIME_LIB", str(lib_a))
    first = runtime._runtime_library()
    assert runtime._runtime_library() is first
    monkeypatch.setenv("SAR_DSL_RUNTIME_LIB", str(lib_b))
    second = runtime._runtime_library()
    assert second is not first
    assert loaded == [str(lib_a), str(lib_b)]


def test_missing_error_channel_warns(monkeypatch):
    """A kernel library that hides sar_rt_error_get (static or
    hidden-visibility deployment) disables runtime error checking; that
    downgrade must be announced, not silent."""

    class NoErrorApiLibrary:

        def __init__(self, path):
            self._mlir_ciface_k = types.SimpleNamespace()

        def __getattr__(self, name):
            raise AttributeError(name)

    monkeypatch.setattr(runtime.ctypes, "CDLL", NoErrorApiLibrary)
    with pytest.warns(RuntimeWarning, match="fail silently"):
        kernel = runtime.CompiledKernel("fake.so", "k", [], [])
    assert kernel._error_clear is None
    assert kernel._error_get is None
