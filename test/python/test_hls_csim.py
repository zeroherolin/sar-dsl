"""Generated validation packages exercised in C-sim through Vitis HLS."""

import shutil
import subprocess

import numpy as np
import pytest

import sar

from conftest import requires_hls, requires_vitis, run_hls_csim

pytestmark = requires_hls

#: Every protocol the backend emits a top signature for. A package that
#: cannot be built and run is a package a user cannot check their design
#: with, whichever of the three they chose.
INTERFACES = ("ap_memory", "axi", "stream")


def test_portable_cpp_fallback_is_explicit(tmp_path, monkeypatch):
    """Without Vitis, the named portable script is the only fallback path."""
    n = 16

    @sar.func
    def portable_scale(x: sar.f32[n]) -> sar.f32[n]:
        return x * 2.0

    values = np.arange(n, dtype=np.float32)
    design = portable_scale.compile(backend="hls")
    design.write_testbench([values], [values * 2.0], tmp_path)

    real_which = shutil.which
    monkeypatch.setattr(
        shutil, "which", lambda executable: None
        if executable == "vitis_hls" else real_which(executable))
    mode, _ = run_hls_csim(tmp_path, "portable_scale")
    assert mode == "portable_cpp_sim"
    assert (tmp_path / "portable_scale_portable_cpp_sim.sh").is_file()
    assert (tmp_path / "portable_scale_hls_csim.tcl").is_file()
    assert not (tmp_path / "portable_scale_csim.tcl").exists()
    assert (tmp_path / "portable_cpp_sim").is_file()


@pytest.mark.parametrize("interface", INTERFACES)
def test_elementwise_package_hls_csim_on_every_interface(interface, tmp_path):
    """The protocol decides the top signature and the testbench that drives
    it: plain arrays, AXI masters, or `hls::stream` ports. All three have to
    produce a package that builds and passes."""
    n = 64

    @sar.func
    def scale(x: sar.c64[n]) -> sar.c64[n]:
        return x * 2.0

    values = (np.arange(n) + 1j * np.arange(n)).astype(np.complex64)
    design = scale.compile(backend="hls", options={"interface": interface})
    design.write_testbench([values], [values * 2.0], tmp_path)
    run_hls_csim(tmp_path, "scale")


@pytest.mark.parametrize("interface", ("ap_memory", "axi"))
def test_transform_package_hls_csim_on_addressed_interfaces(
        interface, tmp_path):
    """A transform reads a plane along one axis, so it needs addresses --
    `stream` rejects such a chain by design. The two addressed protocols
    must carry it, since that is every real imaging kernel."""
    n = 64

    @sar.func
    def roundtrip(x: sar.c64[n]) -> sar.c64[n]:
        return sar.ifft(sar.fft(x, dim=0), dim=0)

    rng = np.random.default_rng(7)
    values = (rng.normal(size=n) + 1j * rng.normal(size=n)).astype(
        np.complex64)
    design = roundtrip.compile(backend="hls", options={"interface": interface})
    design.write_testbench([values], [values.astype(np.complex128)],
                           tmp_path,
                           rtol=1e-3,
                           atol=1e-3)
    run_hls_csim(tmp_path, "roundtrip")


def test_integer_and_float_ports_hls_csim_together(tmp_path):
    """Mixed integer and floating ports in one package: the integer planes
    are compared exactly and the floating ones within tolerance, so the
    testbench has to carry both comparators at once."""
    n = 32

    @sar.func
    def mixed(counts: sar.i64[n],
              gain: sar.f64[n]) -> (sar.i64[n], sar.f64[n]):
        return counts + counts, gain * 2.0

    counts = (np.arange(n, dtype=np.int64) + 1) * 1234567890123
    gain = np.linspace(0.5, 3.0, n)
    design = mixed.compile(backend="hls")
    design.write_testbench([counts, gain], [counts + counts, gain * 2.0],
                           tmp_path)
    run_hls_csim(tmp_path, "mixed")


@requires_vitis
def test_cosim_passes_at_the_reference_raster(tmp_path):
    """RTL co-simulation of a transform design, at one raster.

    This is the check C simulation cannot give: the generated RTL, driven by
    the same golden data, against the same testbench. It is pinned to 256
    because co-simulation time grows with the raster and a production grid
    would not finish -- the script ships with every package, so a user who
    wants a larger grid runs the same command themselves.
    """
    n = 256

    @sar.func
    def cosim_fft(x: sar.c64[n]) -> sar.c64[n]:
        return sar.ifft(sar.fft(x, dim=0), dim=0)

    rng = np.random.default_rng(3)
    values = (rng.normal(size=n) + 1j * rng.normal(size=n)).astype(
        np.complex64)
    design = cosim_fft.compile(backend="hls", options={"interface": "axi"})
    design.write_testbench([values], [values.astype(np.complex128)],
                           tmp_path,
                           rtol=1e-3,
                           atol=1e-3)

    run = subprocess.run(["vitis_hls", "-f", "cosim_fft_cosim.tcl"],
                         cwd=tmp_path,
                         capture_output=True,
                         text=True,
                         timeout=3000)
    assert run.returncode == 0, run.stdout[-3000:]
    assert "co-simulation finished: PASS" in run.stdout, run.stdout[-3000:]
