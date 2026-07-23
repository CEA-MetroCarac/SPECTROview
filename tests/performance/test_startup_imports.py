"""Guard the startup import graph.

Importing ``spectroview.main`` dominates application startup time. The LLM
provider SDKs (~4.5 s) and scipy (~1.3 s) are deliberately imported at their
point of use instead of at module level; this test fails if one of them creeps
back onto the startup path.

Each check runs in a subprocess so it observes a pristine interpreter -- an
in-process assertion would be meaningless once the test session has already
imported scipy for other tests.
"""
import subprocess
import sys

import pytest

# Packages that must NOT be in sys.modules after importing spectroview.main.
DEFERRED_PACKAGES = ["anthropic", "openai", "ollama", "mcp", "scipy"]

_PROBE = (
    "import sys; import spectroview.main; "
    "print(','.join(sorted(m for m in sys.modules if '.' not in m)))"
)


@pytest.fixture(scope="module")
def startup_modules() -> set:
    """Top-level module names present after a bare ``import spectroview.main``."""
    out = subprocess.run(
        [sys.executable, "-c", _PROBE],
        capture_output=True, text=True, timeout=300,
    )
    assert out.returncode == 0, f"probe failed:\n{out.stderr}"
    return set(out.stdout.strip().splitlines()[-1].split(","))


@pytest.mark.parametrize("package", DEFERRED_PACKAGES)
def test_package_stays_off_startup_path(startup_modules, package):
    assert package not in startup_modules, (
        f"`{package}` is imported at SPECTROview startup again. It costs seconds "
        f"to import and is not needed to paint the first window -- move the import "
        f"into the function that uses it (see docs/developer/index.md)."
    )
