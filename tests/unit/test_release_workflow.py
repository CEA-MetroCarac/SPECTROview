"""Structural and contract validation tests for .github/workflows/release.yml."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import re
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

    # macOS matrix targets Apple Silicon arm64 only (macos-15)
    mac_matrix = jobs["build-macos"]["strategy"]["matrix"]["include"]
    arches = {item["arch"] for item in mac_matrix}
    runners = {item["runner"] for item in mac_matrix}
    assert arches == {"arm64"}
    assert runners == {"macos-15"}
    assert "macos-15-intel" not in raw_text
    assert "SPECTROview_${VER}_macOS_x64.dmg" not in raw_text
    assert "SPECTROview_${VER}_macOS_arm64.dmg" in raw_text


def test_release_workflow_spectroview_imports_resolve():
    repo_root = Path(__file__).resolve().parents[2]
    wf_path = repo_root / ".github" / "workflows" / "release.yml"
    raw_text = wf_path.read_text(encoding="utf-8")

    assert "spectroview.model.system.m_tls" in raw_text
    assert "spectroview.model.system.m_installation" in raw_text
    assert "spectroview.model.m_tls" not in raw_text
    assert "spectroview.model.m_installation" not in raw_text

    matches = re.findall(
        r"(?:from|import)\s+(spectroview(?:\.[a-zA-Z0-9_]+)+)", raw_text
    )
    assert len(matches) >= 2
    unresolved = [
        mod for mod in matches if importlib.util.find_spec(mod) is None
    ]
    assert unresolved == [], (
        f"Unresolved spectroview modules in release.yml: {unresolved}"
    )

    packaging_scripts = sorted((repo_root / "packaging").rglob("*.py"))
    assert len(packaging_scripts) >= 1
    pkg_unresolved: list[str] = []
    for script_path in packaging_scripts:
        tree = ast.parse(
            script_path.read_text(encoding="utf-8"),
            filename=str(script_path),
        )
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "spectroview" or alias.name.startswith(
                        "spectroview."
                    ):
                        if importlib.util.find_spec(alias.name) is None:
                            pkg_unresolved.append(
                                f"{script_path.name}:{alias.name}"
                            )
            elif isinstance(node, ast.ImportFrom) and node.module:
                if node.module == "spectroview" or node.module.startswith(
                    "spectroview."
                ):
                    if importlib.util.find_spec(node.module) is None:
                        pkg_unresolved.append(
                            f"{script_path.name}:{node.module}"
                        )
    assert pkg_unresolved == [], (
        f"Unresolved spectroview modules in packaging/: {pkg_unresolved}"
    )


def test_main_entrypoint_and_spec_imports_are_pyinstaller_traceable():
    repo_root = Path(__file__).resolve().parents[2]
    pkg_root = repo_root / "spectroview"

    lazy_facade_pkgs = {
        ".".join(init_path.parent.relative_to(repo_root).parts)
        for init_path in pkg_root.rglob("__init__.py")
        if "def __getattr__" in init_path.read_text(encoding="utf-8")
    }

    main_tree = ast.parse(
        (pkg_root / "main.py").read_text(encoding="utf-8"),
        filename="spectroview/main.py",
    )
    main_from_modules = {
        node.module
        for node in ast.walk(main_tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    lazy_imports = sorted(main_from_modules & lazy_facade_pkgs)
    assert lazy_imports == [], (
        "spectroview/main.py must not import through PEP 562 lazy facades "
        f"(breaks PyInstaller ModuleGraph): {lazy_imports}"
    )

    for expected_mod in (
        "spectroview.view.workspaces.v_workspace_spectra",
        "spectroview.view.workspaces.v_workspace_maps",
        "spectroview.view.workspaces.graphs.v_workspace_graphs",
    ):
        assert expected_mod in main_from_modules

    spec_text = (repo_root / "main.spec").read_text(encoding="utf-8")
    spec_tree = ast.parse(spec_text, filename="main.spec")
    spec_modules: set[str] = set()
    for node in ast.walk(spec_tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("spectroview."):
                    spec_modules.add(alias.name)
        elif (
            isinstance(node, ast.ImportFrom)
            and node.module
            and node.module.startswith("spectroview.")
        ):
            spec_modules.add(node.module)
        elif isinstance(node, ast.Assign):
            is_hidden = any(
                isinstance(t, ast.Name) and t.id == "hidden_imports"
                for t in node.targets
            )
            if is_hidden and isinstance(node.value, ast.List):
                for elt in node.value.elts:
                    if (
                        isinstance(elt, ast.Constant)
                        and isinstance(elt.value, str)
                        and elt.value.startswith("spectroview.")
                    ):
                        spec_modules.add(elt.value)

    assert "spectroview.view.workspaces.v_workspace_spectra" in spec_modules
    unresolved = sorted(
        mod for mod in spec_modules if importlib.util.find_spec(mod) is None
    )
    assert unresolved == [], (
        f"Unresolved spectroview modules in main.spec: {unresolved}"
    )
