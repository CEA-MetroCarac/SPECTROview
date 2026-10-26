# setup.py
"""Build hook for SPECTROview.

Automatically embeds the build timestamp as BUILD_NUMBER (format: YYYY-MM-DD HH-MM-SS)
into spectroview/_build_info.py during wheel (.whl) and source distribution (.tar.gz) packaging,
leaving the source development directory intact (with default BUILD_NUMBER = "Dev").
"""

import datetime
from pathlib import Path
from setuptools import setup
from setuptools.command.build_py import build_py
from setuptools.command.sdist import sdist


def get_build_timestamp() -> str:
    """Return build timestamp in format YYYY-MM-DD HH-MM-SS."""
    return datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")


def write_build_info(target_dir: Path, timestamp: str) -> None:
    """Write _build_info.py into the specified target spectroview package folder."""
    pkg_dir = target_dir / "spectroview"
    pkg_dir.mkdir(parents=True, exist_ok=True)
    build_info_path = pkg_dir / "_build_info.py"
    content = (
        f'# Automatically generated build info - DO NOT EDIT MANUALLY\n'
        f'BUILD_NUMBER = "{timestamp}"\n'
    )
    build_info_path.write_text(content, encoding="utf-8")


class CustomBuildPy(build_py):
    """Custom build_py command that injects the build timestamp into the built wheel."""

    def run(self):
        super().run()
        timestamp = get_build_timestamp()
        write_build_info(Path(self.build_lib), timestamp)


class CustomSdist(sdist):
    """Custom sdist command that injects the build timestamp into release distributions."""

    def make_release_tree(self, base_dir, files):
        super().make_release_tree(base_dir, files)
        timestamp = get_build_timestamp()
        write_build_info(Path(base_dir), timestamp)


setup(
    cmdclass={
        "build_py": CustomBuildPy,
        "sdist": CustomSdist,
    }
)
