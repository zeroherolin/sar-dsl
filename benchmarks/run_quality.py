#!/usr/bin/env python3
"""Point-target image-quality report for the stripmap imaging chains.

Focuses one scene-center scatterer with each compiled chain and reports
IRW / PSLR / ISLR plus the peak position error (see `metrics.py`).

Usage:
    python benchmarks/run_quality.py [--n 512]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from algorithms import LABELS, focus_point_target  # noqa: E402
from metrics import measure_image  # noqa: E402


def _report(name: str, metrics: dict) -> None:
    rng, azm = metrics["range"], metrics["azimuth"]
    print(f"{name:>14}: peak_err={metrics['peak_error']}  "
          f"range[IRW={rng['irw']:5.2f} PSLR={rng['pslr']:7.2f} dB "
          f"ISLR={rng['islr']:7.2f} dB]  "
          f"azimuth[IRW={azm['irw']:5.2f} PSLR={azm['pslr']:7.2f} dB "
          f"ISLR={azm['islr']:7.2f} dB]")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=512)
    args = parser.parse_args()

    center = (args.n // 2, args.n // 2)
    for name, image in focus_point_target(args.n):
        _report(LABELS[name], measure_image(image, expected_peak=center))


if __name__ == "__main__":
    main()
