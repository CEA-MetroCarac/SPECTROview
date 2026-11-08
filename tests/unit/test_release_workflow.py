"""Structural and contract validation tests for .github/workflows/release.yml."""

from __future__ import annotations

from pathlib import Path
import yaml


def _load_workflow() -> dict:
    wf_path = Path(__file__).resolve().parents[2] / ".github" / "workflows" / "release.yml"
    data = yaml.safe_load(wf_path.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def test_release_workflow_preserves_pypi_trusted_publishing_and_adds_all_platforms():
    wf = _load_workflow()
    # PyYAML parses unquoted `on:` as boolean True
    triggers = wf.get("on") if "on" in wf else wf[True]
    assert "v[0-9]*" in triggers["push"]["tags"]
    assert "workflow_dispatch" in triggers

    jobs = wf["jobs"]
    expected_jobs = {
        "verify",
        "build-python",
        "build-windows",
        "build-macos",
        "build-linux",
        "release",
        "publish-pypi",
    }
    assert expected_jobs.issubset(jobs.keys())

    # Python 3.14 & hardened pip/certifi minimums across jobs
    raw_text = (Path(__file__).resolve().parents[2] / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    assert 'python-version: "3.14"' in raw_text
    assert "certifi>=2026.10.23" in raw_text
    assert "pip>=26.2.2" in raw_text
    assert "SHA256SUMS.txt" in raw_text
    # Ensure wheel verification runs outside repo root so CWD does not resolve ./spectroview as DEV_SOURCE
    assert "cd /tmp &&" in raw_text
    # Ensure step env secrets are evaluated cleanly inside shell rather than unsupported step-env `if:`
    assert "env.APPLE_CERTIFICATE_P12" not in raw_text

    # Preserve PyPI Trusted Publisher exactly
    pypi_job = jobs["publish-pypi"]
    assert pypi_job["environment"]["name"] == "pypi"
    assert pypi_job["permissions"]["id-token"] == "write"
    uses_list = [step.get("uses", "") for step in pypi_job["steps"]]
    assert "pypa/gh-action-pypi-publish@release/v1" in uses_list

    # Preserve macOS matrix covering arm64 & x64
    mac_matrix = jobs["build-macos"]["strategy"]["matrix"]["include"]
    arches = {item["arch"] for item in mac_matrix}
    assert arches == {"arm64", "x64"}
