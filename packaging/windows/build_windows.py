"""Build script for Windows standalone distributions (PyInstaller + Velopack + Portable ZIP).

Usage:
    python packaging/windows/build_windows.py [--output-dir release_artifacts] [--skip-velopack] [--smoke-test]
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from spectroview.model.m_tls import assert_safe_build_tls  # noqa: E402


def read_version() -> str:
    init_py = PROJECT_ROOT / "spectroview" / "__init__.py"
    text = init_py.read_text(encoding="utf-8")
    match = re.search(r'^VERSION\s*=\s*["\']([^"\']+)["\']', text, re.MULTILINE)
    if not match:
        raise RuntimeError(f"Could not extract VERSION from {init_py}")
    return match.group(1)


def write_win_version_info(version: str, target_path: Path) -> None:
    """Write a PyInstaller Windows PE version-resource file for SPECTROview.exe."""
    parts = [int(x) for x in re.findall(r"\d+", version)[:4]]
    while len(parts) < 4:
        parts.append(0)
    ver_tuple = f"({parts[0]}, {parts[1]}, {parts[2]}, {parts[3]})"

    content = f"""# UTF-8
VSVersionInfo(
  ffi=FixedFileInfo(
    filevers={ver_tuple},
    prodvers={ver_tuple},
    mask=0x3f,
    flags=0x0,
    OS=0x40004,
    fileType=0x1,
    subtype=0x0,
    date=(0, 0)
  ),
  kids=[
    StringFileInfo(
      [
      StringTable(
        '040904B0',
        [StringStruct('CompanyName', 'CEA-MetroCarac'),
        StringStruct('FileDescription', 'SPECTROview — Spectroscopic Data Processing & Visualization'),
        StringStruct('FileVersion', '{version}'),
        StringStruct('InternalName', 'SPECTROview'),
        StringStruct('LegalCopyright', 'GPL-3.0-or-later'),
        StringStruct('OriginalFilename', 'SPECTROview.exe'),
        StringStruct('ProductName', 'SPECTROview'),
        StringStruct('ProductVersion', '{version}')])
      ]),
    VarFileInfo([VarStruct('Translation', [1033, 1200])])
  ]
)
"""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(content, encoding="utf-8")


def verify_onedir_bundle(dist_dir: Path) -> None:
    """Assert critical runtime binaries, resources, and AI/TLS assets exist cleanly."""
    required_paths = [
        dist_dir / "SPECTROview.exe",
        dist_dir / "_internal" / "spectroview" / "resources" / "build_info.json",
        dist_dir / "_internal" / "spectroview" / "ai_agent" / "config" / "model.yaml",
        dist_dir / "_internal" / "spectroview" / "ai_agent" / "prompts" / "system.md",
        dist_dir / "_internal" / "certifi" / "cacert.pem",
    ]
    missing = [str(p) for p in required_paths if not p.exists()]
    if missing:
        raise RuntimeError(
            "PyInstaller Windows bundle is missing required assets:\n- "
            + "\n- ".join(missing)
        )


def run_smoke_test(dist_dir: Path) -> None:
    """Verify the frozen Windows executable launches offscreen cleanly."""
    exe = dist_dir / "SPECTROview.exe"
    probe_env = os.environ.copy()
    probe_env["QT_QPA_PLATFORM"] = "offscreen"
    # Launch briefly to confirm Qt platform plugin, Python runtime, and C extensions load
    proc = subprocess.Popen(
        [str(exe)],
        cwd=str(dist_dir),
        env=probe_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        proc.wait(timeout=8)
    except subprocess.TimeoutExpired:
        # Still alive after 8s GUI startup window -> healthy launch! Terminate cleanly.
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        return
    if proc.returncode not in (0, None):
        stderr = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
        raise RuntimeError(f"Smoke launch exited early with code {proc.returncode}:\n{stderr}")


def create_portable_zip(dist_dir: Path, out_zip: Path) -> Path:
    """Package the PyInstaller --onedir folder cleanly into a Portable ZIP."""
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for file_path in sorted(dist_dir.rglob("*")):
            if file_path.is_file():
                arcname = Path("SPECTROview") / file_path.relative_to(dist_dir)
                zf.write(file_path, arcname=arcname.as_posix())
    return out_zip


def package_velopack(
    dist_dir: Path,
    version: str,
    release_out_dir: Path,
    *,
    repo_url: str = "https://github.com/CEA-MetroCarac/SPECTROview",
    download_previous: bool = True,
) -> list[Path]:
    """Run ``vpk pack`` cleanly and export Setup.exe, releases.win.json, and .nupkg files."""
    vpk_bin = shutil.which("vpk")
    if not vpk_bin:
        raise FileNotFoundError(
            "Velopack CLI ('vpk') was not found on PATH. Install .NET 8+ SDK and run "
            "'dotnet tool install -g vpk'."
        )

    vpk_work = PROJECT_ROOT / "build" / "vpk_releases"
    if vpk_work.exists():
        shutil.rmtree(vpk_work)
    vpk_work.mkdir(parents=True, exist_ok=True)

    if download_previous:
        dl_cmd = [vpk_bin, "download", "github", "--repoUrl", repo_url, "--outputDir", str(vpk_work)]
        print(f"[build_windows] Fetching prior release delta history (best-effort): {' '.join(dl_cmd)}")
        subprocess.run(dl_cmd, check=False)

    icon_path = PROJECT_ROOT / "spectroview" / "resources" / "icons" / "logo_spectroview.ico"
    pack_cmd = [
        vpk_bin,
        "pack",
        "--packId",
        "SPECTROview",
        "--packTitle",
        "SPECTROview",
        "--packAuthors",
        "Van Hoan Le / CEA-MetroCarac",
        "--packVersion",
        version,
        "--packDir",
        str(dist_dir),
        "--mainExe",
        "SPECTROview.exe",
        "--icon",
        str(icon_path),
        "--shortcuts",
        "Desktop,StartMenuRoot",
        "--outputDir",
        str(vpk_work),
    ]
    print(f"[build_windows] Running: {' '.join(pack_cmd)}")
    subprocess.run(pack_cmd, check=True)

    exported: list[Path] = []
    setup_target = release_out_dir / f"SPECTROview_{version}_Windows_x64_Setup.exe"
    setup_candidates = sorted(vpk_work.glob("*Setup.exe"))
    if not setup_candidates:
        raise RuntimeError(f"vpk pack completed but no *Setup.exe found in {vpk_work}")
    shutil.copy2(setup_candidates[0], setup_target)
    exported.append(setup_target)

    # Export Velopack feed index & nupkg delta/full assets verbatim so UpdateManager
    # and future `vpk download github` delta chaining locate exact filenames.
    for pattern in ("releases*.json", "*.nupkg"):
        for src in sorted(vpk_work.glob(pattern)):
            dst = release_out_dir / src.name
            shutil.copy2(src, dst)
            exported.append(dst)

    return exported


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Windows standalone SPECTROview packages.")
    parser.add_argument("--output-dir", default="release_artifacts", help="Directory for output artifacts.")
    parser.add_argument("--skip-velopack", action="store_true", help="Build PyInstaller + Portable ZIP only.")
    parser.add_argument("--no-download-previous", action="store_true", help="Skip fetching prior GitHub Velopack history.")
    parser.add_argument("--smoke-test", action="store_true", help="Run offscreen startup probe against dist EXE.")
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)
    version = read_version()
    print(f"[build_windows] Building SPECTROview {version} for Windows x64...")

    tls_report = assert_safe_build_tls(require_pip=True)
    print(
        f"[build_windows] TLS security verified: certifi={tls_report['certifi_version']}, "
        f"pip={tls_report['pip_version']}, truststore={tls_report['truststore_available']}"
    )

    write_win_version_info(version, PROJECT_ROOT / "build" / "win_version_info.txt")

    pyinstaller_cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "main.spec",
    ]
    print(f"[build_windows] Running: {' '.join(pyinstaller_cmd)}")
    subprocess.run(pyinstaller_cmd, check=True)

    dist_dir = PROJECT_ROOT / "dist" / "SPECTROview"
    verify_onedir_bundle(dist_dir)

    if args.smoke_test:
        print("[build_windows] Running offscreen executable smoke test...")
        run_smoke_test(dist_dir)
        print("[build_windows] Smoke test passed.")

    out_dir = (PROJECT_ROOT / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    portable_zip = out_dir / f"SPECTROview_{version}_Windows_x64_Portable.zip"
    print(f"[build_windows] Packaging Portable ZIP -> {portable_zip}")
    create_portable_zip(dist_dir, portable_zip)

    artifacts = [portable_zip]
    if not args.skip_velopack:
        artifacts.extend(
            package_velopack(
                dist_dir,
                version,
                out_dir,
                download_previous=not args.no_download_previous,
            )
        )

    print("[build_windows] Built artifacts:")
    for item in artifacts:
        size_mb = item.stat().st_size / (1024 * 1024)
        print(f"  - {item.name} ({size_mb:.2f} MiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

