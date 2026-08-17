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


def test_centered_and_uncentered_spectra_warn():
    """Multiplying a shifted spectrum by an unshifted one: same domain, but
    the phase reference differs by a half-band rotation."""

    @sar.func
    def k(z: sar.c64[N, N], y: sar.c64[N, N]) -> sar.c64[N, N]:
        return sar.fftshift(sar.fft(z, dim=1), dim=1) * sar.fft(y, dim=1)

    assert any("centered spectrum" in m for m in _messages(k))


def test_matched_centering_is_silent():

    @sar.func
    def k(z: sar.c64[N, N], y: sar.c64[N, N]) -> sar.c64[N, N]:
        a = sar.fftshift(sar.fft(z, dim=1), dim=1)
        b = sar.fftshift(sar.fft(y, dim=1), dim=1)
        return a * b

    assert _messages(k) == []


def test_domain_state_survives_circshift():
    """A roll keeps every axis in its domain, so the tracker must not lose
    the state across one -- otherwise a double FFT slips through."""

    @sar.func
    def k(z: sar.c64[N, N]) -> sar.c64[N, N]:
        rolled = sar.circshift(sar.fft(z, dim=1), 3, dim=1)
        return sar.fft(rolled, dim=1)

    assert any("already in the frequency domain" in m for m in _messages(k))


def test_warning_blames_the_kernel_line():
    """The report has to point at user code, not at the DSL internals, or
    module-scoped warning filters cannot match it."""

    @sar.func
    def k(z: sar.c64[N, N]) -> sar.c64[N, N]:
        return sar.fft(sar.fft(z, dim=1), dim=1)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        k.to_mlir()
    hits = [w for w in caught if issubclass(w.category, sar.DomainWarning)]
    assert hits and all(w.filename == __file__ for w in hits)


def test_warnings_repeat_on_every_trace():
    """`to_mlir` reuses the first trace (so its warnings fire once), but an
    explicit re-trace is not memoized: escalating the filter after a first
    call still has to fire."""

    @sar.func
    def k(z: sar.c64[N, N]) -> sar.c64[N, N]:
        return sar.fft(sar.fft(z, dim=1), dim=1)

    assert _messages(k)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        k.trace()
    assert [w for w in caught if issubclass(w.category, sar.DomainWarning)]
