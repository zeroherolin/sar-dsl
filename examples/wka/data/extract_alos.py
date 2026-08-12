#!/usr/bin/env python3
"""Extracts raw echoes from an ALOS-1 PALSAR CEOS L1.0 image file.

Reads a block of azimuth lines, removes the DC bias of the 8-bit I/Q
samples, zero-pads the range dimension to a power of two and writes a
complex64 binary consumable by `run_alos.py`.
"""

import os
import struct

import numpy as np


def extract_alos_raw(input_file, output_file, target_na=16384,
                     target_nr=16384, start_line=2000):
    print(f"Processing ALOS-1 PALSAR file: {input_file}")

    # Line length from the first CEOS record header (big-endian, bytes 8-11
    # after the 720-byte file descriptor).
    with open(input_file, "rb") as f:
        f.seek(720)
        record_len = struct.unpack(">I", f.read(12)[8:12])[0]

    # Each line: 412-byte prefix + Nr interleaved 8-bit (I, Q) pairs.
    nr_actual = (record_len - 412) // 2
    print(f"Detected range samples (Nr): {nr_actual}")

    line_dtype = np.dtype([
        ("header", "V412"),
        ("iq", "u1", (nr_actual, 2)),
    ])

    file_size = os.path.getsize(input_file)
    na_total = (file_size - 720) // record_len
    print(f"Detected total azimuth lines (Na): {na_total}")
    if na_total < target_na + start_line:
        raise ValueError(
            f"insufficient azimuth lines: need {target_na + start_line}, "
            f"found {na_total}")

    print(f"Extracting {target_na} lines starting at line {start_line} ...")
    with open(input_file, "rb") as f:
        f.seek(720 + start_line * record_len)
        raw = np.frombuffer(f.read(target_na * record_len),
                            dtype=line_dtype)

    # Unsigned 0-255 samples carry a DC bias; subtract the channel means to
    # obtain a zero-mean baseband signal.
    iq = raw["iq"].astype(np.float32)
    i_mean, q_mean = iq[:, :, 0].mean(), iq[:, :, 1].mean()
    data = (iq[:, :, 0] - i_mean) + 1j * (iq[:, :, 1] - q_mean)
    print(f"DC bias removed (I: {i_mean:.2f}, Q: {q_mean:.2f})")

    print(f"Zero-padding range dimension: {nr_actual} -> {target_nr} ...")
    padded = np.zeros((target_na, target_nr), dtype=np.complex64)
    padded[:, :nr_actual] = data

    print(f"Writing {output_file}")
    padded.tofile(output_file)
    size_gb = os.path.getsize(output_file) / 1024 ** 3
    print(f"Done. Output size: {size_gb:.2f} GiB")


if __name__ == "__main__":
    # Input sits next to this script; output goes to examples/wka/ where
    # run_alos.py expects it.
    here = os.path.dirname(os.path.abspath(__file__))
    extract_alos_raw(
        input_file=os.path.join(
            here, "ALPSRP275140740-L1.0", "IMG-HH-ALPSRP275140740-H1.0__A"),
        output_file=os.path.join(here, os.pardir,
                                 "alos_raw_16384x16384.bin"),
        target_na=16384,
        target_nr=16384,
        start_line=14000,
    )
