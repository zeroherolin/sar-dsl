"""Machine-readable environment metadata shared by benchmark runners."""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

_REPO = Path(__file__).resolve().parents[1]


def _git(*args) -> Optional[str]:
    try:
        result = subprocess.run(["git", "-C", str(_REPO), *args],
                                capture_output=True,
                                text=True,
                                timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def environment() -> dict:
    """Captures enough context to compare or reproduce a benchmark run."""
    status = _git("status", "--porcelain")
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": None if status is None else bool(status),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "sar_rt_num_threads": os.environ.get("SAR_RT_NUM_THREADS"),
    }
