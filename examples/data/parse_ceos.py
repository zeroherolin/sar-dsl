#!/usr/bin/env python3
"""Extracts WKA-relevant radar parameters from an ALOS-1 CEOS L1.0 product.

Reads the Leader (LED-*) dataset summary record and the first Image (IMG-*)
line prefix to recover:

    c    propagation speed (m/s)         fc   carrier frequency (Hz)
    Fs   range sampling rate (Hz)        PRF  pulse repetition freq. (Hz)
    R0   reference slant range (m)       Kr   range chirp rate (Hz/s)
    Rank slant range gate line number

Byte offsets follow the ALOS PALSAR CEOS format description and were
verified against sample products. The effective radar velocity Vr is not
recoverable from these records; configure it explicitly in the imaging
parameters (e.g. Vr = 7155.0 for the San Francisco scene).
"""

import glob
import os
import struct

SPEED_OF_LIGHT = 299792458.0
#: System delay for ALOS-1 PALSAR FBS mode (Fs ~ 32 MHz).
SYSTEM_DELAY_S = 3.07e-5


def _read_dssr(led_filepath):
    """Returns the 4096-byte dataset summary record body, or None."""
    with open(led_filepath, "rb") as f:
        f.seek(720)  # file descriptor record
        while True:
            header = f.read(12)
            if len(header) < 12:
                return None
            record_len = struct.unpack(">I", header[8:12])[0]
            body = f.read(record_len - 12)
            if record_len == 4096:
                return body


def _ascii_float(body, start_byte, length):
    """Reads an ASCII float at a 1-based record offset (header included)."""
    idx = start_byte - 13
    try:
        text = body[idx:idx + length].decode("ascii").strip()
        return float(text) if text else None
    except (UnicodeDecodeError, ValueError):
        return None


def parse_ceos_leader(led_filepath):
    """Parses the product and returns a parameter dict (None if missing)."""
    print(f"Parsing CEOS leader file: {led_filepath}\n")

    c = SPEED_OF_LIGHT
    fc = fs = prf = None

    dssr = _read_dssr(led_filepath)
    if dssr:
        wavelength = _ascii_float(dssr, 501, 16)  # meters
        sample_rate_mhz = _ascii_float(dssr, 711, 16)
        prf_milli_hz = _ascii_float(dssr, 935, 16)
        if wavelength:
            fc = c / wavelength
        if sample_rate_mhz:
            fs = sample_rate_mhz * 1e6
        if prf_milli_hz:
            prf = prf_milli_hz / 1000.0

    # R0 comes from the image line prefix: rank and sampling window delay.
    r0 = None
    rank = 0
    img_pattern = os.path.basename(led_filepath).replace("LED-", "IMG-??-")
    img_files = glob.glob(
        os.path.join(os.path.dirname(led_filepath), img_pattern))
    if img_files and prf:
        with open(img_files[0], "rb") as f:
            f.seek(720)
            prefix = f.read(412)
        rank = struct.unpack(">H", prefix[116:118])[0]
        delay_ns = struct.unpack(">I", prefix[120:124])[0]
        total_time = rank / prf + delay_ns * 1e-9 - SYSTEM_DELAY_S
        r0 = total_time * c / 2.0

    # Chirp rate from the mode bandwidth (28 MHz for FBS, 14 MHz for FBD).
    kr = None
    if fs:
        pulse_width = 27.0e-6
        bandwidth = 28.0e6 if abs(fs - 32e6) < 1e5 else 14.0e6
        kr = -(bandwidth / pulse_width)

    params = {
        "c": c,
        "fc": fc,
        "Fs": fs,
        "PRF": prf,
        "R0": r0,
        "Kr": kr,
        "Rank": rank
    }

    print("====== Extracted Radar Parameters ======")
    print(f"c    = {c} m/s")
    print(f"fc   = {fc:.2f} Hz" if fc else "fc   = not found")
    print(f"Fs   = {fs:.2f} Hz" if fs else "Fs   = not found")
    print(f"PRF  = {prf:.3f} Hz" if prf else "PRF  = not found")
    print(f"Rank = {rank}")
    print(f"R0   = {r0:.3f} m" if r0 else "R0   = not found")
    print(f"Kr   = {kr:.3e} Hz/s" if kr else "Kr   = not found")
    print("\n[note] Vr is not recoverable from these records; configure it "
          "explicitly in the imaging parameters.")
    return params


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    parse_ceos_leader(
        os.path.join(here, "ALPSRP275140740-L1.0",
                     "LED-ALPSRP275140740-H1.0__A"))
