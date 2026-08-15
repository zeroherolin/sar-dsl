"""Trace-time spectral-domain diagnostics (sar.DomainWarning).

The tracker follows per-axis time/frequency state through the ops it can
reason about and stays silent on anything unknown, so host-provided
inputs never produce false positives.
"""

import warnings

import pytest

import sar

N = 16


def _messages(fn):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fn.to_mlir()
    return [
        str(w.message) for w in caught
        if issubclass(w.category, sar.DomainWarning)
    ]


def test_double_fft_warns():

    @sar.func
    def k(z: sar.c64[N, N]) -> sar.c64[N, N]:
        return sar.fft(sar.fft(z, dim=1), dim=1)

    assert any("already in the frequency domain" in m for m in _messages(k))


def test_ifft_of_centered_spectrum_warns():

    @sar.func
    def k(z: sar.c64[N, N]) -> sar.c64[N, N]:
        return sar.ifft(sar.fftshift(sar.fft(z, dim=0), dim=0), dim=0)

    assert any("still centered" in m for m in _messages(k))


def test_ifft_of_time_domain_warns():

    @sar.func
    def k(z: sar.c64[N, N]) -> sar.c64[N, N]:
        return sar.ifft(sar.ifft(sar.fft(z, dim=1), dim=1), dim=1)

    assert any("time domain" in m for m in _messages(k))


def test_mixing_domains_warns():

    @sar.func
    def k(z: sar.c64[N, N]) -> sar.c64[N, N]:
        return sar.fft(z, dim=1) * sar.ifft(sar.fft(z, dim=1), dim=1)

    assert any("mix time-domain and frequency-domain" in m
               for m in _messages(k))


def test_proper_chain_is_silent():

    @sar.func
    def k(z: sar.c64[N, N], w: sar.f64[N]) -> sar.c64[N, N]:
        s = sar.fftshift(sar.fft(z, dim=1), dim=1)
        s = s * sar.cast(sar.broadcast(w, (N, N), dim=1), sar.c64)
        return sar.ifft(sar.ifftshift(s, dim=1), dim=1)

    assert _messages(k) == []


def test_imaging_algorithms_trace_clean():
    from common.params import synthetic_params
    from csa.algorithm import build_kernel as build_csa
    from rda.algorithm import build_kernel as build_rda
    from wka.algorithm import build_kernel as build_wka

    params = synthetic_params(N)
    with warnings.catch_warnings():
        warnings.simplefilter("error", sar.DomainWarning)
        for build in (build_wka, build_rda, build_csa):
            build(N, params).to_mlir()


def test_transpose_swaps_axis_state():

    @sar.func
    def k(z: sar.c64[N, N]) -> sar.c64[N, N]:
        # fft on dim 1, corner turn, fft on dim 1 again: different physical
        # axis, so no warning.
        return sar.fft(sar.transpose(sar.fft(z, dim=1)), dim=1)

    assert _messages(k) == []


def test_warnings_can_be_escalated():

    @sar.func
    def k(z: sar.c64[N, N]) -> sar.c64[N, N]:
        return sar.fft(sar.fft(z, dim=1), dim=1)

    with warnings.catch_warnings():
        warnings.simplefilter("error", sar.DomainWarning)
        with pytest.raises(sar.DomainWarning):
            k.to_mlir()
