import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))
sys.path.insert(0, str(REPO_ROOT / "examples" / "wka"))

import sar  # noqa: E402


def _backend_available(name: str) -> bool:
    backends = sar.list_backends()
    return name in backends and backends[name].is_available()


requires_cpu = pytest.mark.skipif(
    not _backend_available("cpu"),
    reason="CPU backend toolchain not available (build the project first)")

requires_scalehls = pytest.mark.skipif(
    not _backend_available("scalehls"),
    reason="ScaleHLS toolchain not available")
