"""Machine-readable environment metadata shared by benchmark runners."""

from __future__ import annotations

import hashlib
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

_REPO = Path(__file__).resolve().parents[1]


def _git(*args, raw: bool = False):
    """Runs one git command; bytes with ``raw`` (for --binary/-z output)."""
    try:
        result = subprocess.run(["git", "-C", str(_REPO), *args],
                                capture_output=True,
                                text=not raw,
                                timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout if raw else result.stdout.strip()


def _dirty_tree_sha256() -> Optional[str]:
    """Digest of tracked diffs and untracked source files, or None."""
    diff = _git("diff", "--binary", "HEAD", "--", raw=True)
    listing = _git("ls-files",
                   "--others",
                   "--exclude-standard",
                   "-z",
                   raw=True)
    if diff is None or listing is None:
        return None
    untracked = listing.split(b"\0")
    digest = hashlib.sha256()
    digest.update(diff)
    for encoded in sorted(path for path in untracked if path):
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
        try:
            content = (_REPO / os.fsdecode(encoded)).read_bytes()
        except OSError:
            content = b"<unreadable>"
        digest.update(len(content).to_bytes(8, "little"))
        digest.update(content)
    return digest.hexdigest() if diff or any(untracked) else None


def environment() -> dict:
    """Captures enough context to compare or reproduce a benchmark run."""
    status = _git("status", "--porcelain")
    dirty = None if status is None else bool(status)
    return {
        "timestamp_utc":
        datetime.now(timezone.utc).isoformat(),
        "git_commit":
        _git("rev-parse", "HEAD"),
        "git_dirty":
        dirty,
        "git_diff_sha256":
        _dirty_tree_sha256() if dirty else None,
        "python":
        sys.version.split()[0],
        "numpy":
        np.__version__,
        "platform":
        platform.platform(),
        "machine":
        platform.machine(),
        "processor":
        platform.processor(),
        "cpu_count":
        os.cpu_count(),
        "cpu_affinity_count": (len(os.sched_getaffinity(0)) if hasattr(
            os, "sched_getaffinity") else None),
        "omp_num_threads":
        os.environ.get("OMP_NUM_THREADS"),
        "sar_rt_num_threads":
        os.environ.get("SAR_RT_NUM_THREADS"),
    }


def result_environment(allow_dirty: bool = False) -> dict:
    """Provenance for a persisted result, refusing unauditable source state."""
    data = environment()
    if data["git_dirty"] is None and not allow_dirty:
        raise RuntimeError(
            "cannot determine the git source state for benchmark output; "
            "pass --allow-dirty to record results without it")
    if data["git_dirty"] and not allow_dirty:
        raise RuntimeError(
            "refusing to write benchmark results from a dirty worktree; "
            "commit/stash the changes or pass --allow-dirty to record their "
            "git_diff_sha256")
    if data["git_dirty"] and not data["git_diff_sha256"]:
        raise RuntimeError("dirty benchmark source has no auditable diff hash")
    return data


def check_result_preconditions(allow_dirty: bool = False) -> None:
    """Raises before any measurement when result output would be refused.

    Runners call this right after argument parsing when JSON output is
    requested; the epilogue-time `result_environment` call stays
    authoritative, since the source state can change mid-run.
    """
    result_environment(allow_dirty)
