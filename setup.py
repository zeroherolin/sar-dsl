"""Packaging entry point.

Backend plugin packages living in ``third_party/<vendor>/backend`` are
copied into ``sar/backends/<vendor>`` at build time so the installed
package is self-contained.
"""

import shutil
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py

ROOT = Path(__file__).resolve().parent


class BuildPyWithBackends(build_py):

    def run(self):
        backends_dir = ROOT / "python" / "sar" / "backends"
        for vendor_dir in sorted((ROOT / "third_party").glob("*/backend")):
            vendor = vendor_dir.parent.name
            target = backends_dir / vendor
            if target.is_symlink() or target.exists():
                continue
            shutil.copytree(vendor_dir, target)
        super().run()


setup(cmdclass={"build_py": BuildPyWithBackends})
