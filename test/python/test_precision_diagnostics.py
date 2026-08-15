"""Trace-time precision diagnostics (sar.PrecisionWarning).

Promotion follows numpy, so host data of a wider dtype pulls the whole
pipeline up with it. That is correct but easy to do by accident -- numpy
defaults to float64 -- and on an HLS target it doubles every operator and
buffer downstream, so the trace reports it.
"""

import warnings

import numpy as np

import sar

N = 16


def _messages(fn):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fn.to_mlir()
    return [
        str(w.message) for w in caught
        if issubclass(w.category, sar.PrecisionWarning)
    ]


def test_double_host_array_warns():

    @sar.func
    def k(x: sar.f32[N, N]) -> sar.f64[N, N]:
        return x * np.ones((N, N))

    messages = _messages(k)
    assert len(messages) == 1, messages
    assert "f64" in messages[0] and "f32 pipeline" in messages[0]


def test_single_host_array_is_silent():

    @sar.func
    def k(x: sar.f32[N, N]) -> sar.f32[N, N]:
        return x * np.ones((N, N), dtype=np.float32)

    assert _messages(k) == []


def test_double_pipeline_is_silent():
    """Nothing widens when the kernel is already double."""

    @sar.func
    def k(x: sar.f64[N, N]) -> sar.f64[N, N]:
        return x * np.ones((N, N))

    assert _messages(k) == []


def test_complex_pipeline_reports_the_host_side():
    """A real host array widening a complex pipeline still names the host."""

    @sar.func
    def k(z: sar.c64[N, N]) -> sar.c128[N, N]:
        return z * np.ones((N, N))

    messages = _messages(k)
    assert len(messages) == 1, messages
    assert "c128" in messages[0]


def test_kernel_arguments_do_not_warn():
    """Only host data is reported: an argument's dtype is the caller's
    contract, not something the trace can suggest changing."""

    @sar.func
    def k(x: sar.f32[N, N], y: sar.f64[N, N]) -> sar.f64[N, N]:
        return x * y

    assert _messages(k) == []


def test_widening_is_reported_through_intermediate_ops():
    """The host array and the tensor it widens need not meet directly: a
    value computed only from host data carries its width along."""

    @sar.func
    def k(x: sar.f32[N, N]) -> sar.f64[N, N]:
        table = sar.broadcast(np.linspace(0.0, 1.0, N), (N, N), dim=0)
        return x * sar.sqrt(table + 1.0)

    messages = _messages(k)
    assert len(messages) == 1, messages
    assert "f64" in messages[0]
