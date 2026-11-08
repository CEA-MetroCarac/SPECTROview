"""Build script for Linux standalone distributions (AppImage and Debian .deb).

Generates:
- ``SPECTROview_<VERSION>_Linux_x86_64.AppImage``
- ``SPECTROview_<VERSION>_Linux_amd64.deb``

Usage:
    python3 packaging/linux/build_linux.py [--format all|appimage|deb] [--output-dir release_artifacts] [--smoke-test]
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import stat
import subprocess
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from spectroview.model.m_tls import assert_safe_build_tls

LINUX_DEB_DEPENDS: tuple[str, ...] = (
    "libc6 (>= 2.34)",
    "libgl1",
    "libegl1",
    "libdbus-1-3",
    "libfontconfig1",
    "libfreetype6",
    "libglib2.0-0",
    "libx11-6",
    "libxext6",
    "libxrender1",
    "libxkbcommon0",
    "libxkbcommon-x11-0",
    "libxcb-cursor0 | libxcb-util1",
    "libxcb-icccm4",
    "libxcb-image0",
    "libxcb-keysyms1",
    "libxcb-randr0",
    "libxcb-render-util0",
    "libxcb-shape0",
    "libxcb-xfixes0",
    "libxcb-xinerama0",
    "zlib1g",
)


def read_version() -> str:
    init_py = PROJECT_ROOT / "spectroview" / "__init__.py"
    text = init_py.read_text(encoding="utf-8")
    match = re.search(r'^VERSION\s*=\s*["\']([^"\']+)["\']', text, re.MULTILINE)
    if not match:
        raise RuntimeError(f"Could not extract VERSION from {init_py}")
    return match.group(1)


def appimage_filename(version: str) -> str:
    return f"SPECTROview_{version}_Linux_x86_64.AppImage"


def deb_filename(version: str) -> str:
    return f"SPECTROview_{version}_Linux_amd64.deb"


def debian_package_version(version: str) -> str:
    """Convert PEP 440 / SemVer tags into valid Debian Policy versions cleanly."""
    clean = str(version).strip().lstrip("v")
    # Convert pre-releases like 26.47.0-rc.1 -> 26.47.0~rc.1 so stable upgrades cleanly
    return re.sub(r"[-.](alpha|beta|rc|dev)", r"~\1", clean, count=1, flags=re.IGNORECASE)


def verify_linux_bundle(dist_dir: Path) -> None:
    """Assert critical frozen binaries, Qt platform plugin hints, TLS CAs, and AI assets."""
    required = [
        dist_dir / "SPECTROview",
        dist_dir / "_internal" / "spectroview" / "resources" / "build_info.json",
        dist_dir / "_internal" / "spectroview" / "ai_agent" / "config" / "model.yaml",
        dist_dir / "_internal" / "spectroview" / "ai_agent" / "prompts" / "system.md",
        dist_dir / "_internal" / "certifi" / "cacert.pem",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise RuntimeError(
            "PyInstaller Linux bundle is missing required assets:\n- "
            + "\n- ".join(missing)
        )


def run_smoke_test(dist_dir: Path) -> None:
    """Launch the frozen Linux executable under offscreen QPA briefly."""
    exe = (dist_dir / "SPECTROview").resolve()
    probe_env = os.environ.copy()
    probe_env["QT_QPA_PLATFORM"] = "offscreen"
    proc = subprocess.Popen(
        [str(exe), "--smoke-test"],
        cwd=str(dist_dir),
        env=probe_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        stdout, stderr = proc.communicate(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, stderr = proc.communicate()
        err_msg = stderr.decode("utf-8", errors="replace") if stderr else ""
        raise RuntimeError(f"Linux smoke launch timed out after 15s:\n{err_msg}")
    if proc.returncode != 0:
        err_msg = stderr.decode("utf-8", errors="replace") if stderr else ""
        raise RuntimeError(f"Linux smoke launch exited with code {proc.returncode}:\n{err_msg}")


def _set_exec_bit(path: Path) -> None:
    try:
        mode = path.stat().st_mode
        os.chmod(path, mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH | stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    except OSError:
        pass


def render_deb_control(version: str, installed_size_kb: int) -> str:
    """Render Debian ``DEBIAN/control`` file content."""
    deb_ver = debian_package_version(version)
    depends = ", ".join(LINUX_DEB_DEPENDS)
    return (
        "Package: spectroview\n"
        f"Version: {deb_ver}\n"
        "Section: science\n"
        "Priority: optional\n"
        "Architecture: amd64\n"
        f"Installed-Size: {max(1, int(installed_size_kb))}\n"
        "Maintainer: Van Hoan Le <van-hoan.le@cea.fr>\n"
        "Homepage: https://github.com/CEA-MetroCarac/SPECTROview\n"
        f"Depends: {depends}\n"
        "Description: Spectroscopic Data Processing, Fitting, and Visualization\n"
        " SPECTROview is an open-source desktop application for 1D spectroscopic data,\n"
        " hyperspectral 2D/wafer map analysis, batched curve fitting (VBF), multivariate\n"
        " analysis (PCA/NMF), publication plotting, and optional AI-assisted exploration.\n"
    )


def _dir_size_kb(root: Path) -> int:
    total = 0
    for item in root.rglob("*"):
        if item.is_file() and not item.is_symlink():
            total += item.stat().st_size
    return max(1, (total + 1023) // 1024)


def stage_appdir(
    dist_dir: Path,
    appdir: Path,
    *,
    desktop_file: Optional[Path] = None,
    icon_file: Optional[Path] = None,
) -> Path:
    """Assemble an AppImage ``AppDir`` directory around *dist_dir*."""
    desktop_src = desktop_file or (PROJECT_ROOT / "packaging" / "linux" / "spectroview.desktop")
    icon_src = icon_file or (PROJECT_ROOT / "spectroview" / "resources" / "icons" / "logo_spectroview.png")

    if appdir.exists():
        shutil.rmtree(appdir)

    usr_bin = appdir / "usr" / "bin"
    usr_apps = appdir / "usr" / "share" / "applications"
    usr_icons = appdir / "usr" / "share" / "icons" / "hicolor" / "256x256" / "apps"
    usr_bin.mkdir(parents=True, exist_ok=True)
    usr_apps.mkdir(parents=True, exist_ok=True)
    usr_icons.mkdir(parents=True, exist_ok=True)

    # Copy full --onedir contents into AppDir/usr/bin/ so _internal resolves cleanly beside SPECTROview
    shutil.copytree(dist_dir, usr_bin, symlinks=True, dirs_exist_ok=True)
    _set_exec_bit(usr_bin / "SPECTROview")

    desktop_text = desktop_src.read_text(encoding="utf-8").replace("Exec=spectroview %F", "Exec=SPECTROview %F")
    (appdir / "spectroview.desktop").write_text(desktop_text, encoding="utf-8")
    (usr_apps / "spectroview.desktop").write_text(desktop_text, encoding="utf-8")

    shutil.copy2(icon_src, appdir / "spectroview.png")
    shutil.copy2(icon_src, appdir / ".DirIcon")
    shutil.copy2(icon_src, usr_icons / "spectroview.png")

    apprun = appdir / "AppRun"
    apprun.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'HERE="$(cd "$(dirname "$(readlink -f "${0}")")" && pwd)"\n'
        'export PATH="${HERE}/usr/bin:${PATH}"\n'
        'exec "${HERE}/usr/bin/SPECTROview" "$@"\n',
        encoding="utf-8",
    )
    _set_exec_bit(apprun)
    return appdir


def stage_deb_root(
    dist_dir: Path,
    deb_root: Path,
    *,
    version: str,
    desktop_file: Optional[Path] = None,
    icon_file: Optional[Path] = None,
) -> Path:
    """Assemble a Debian ``dpkg-deb`` root tree under ``/opt/spectroview`` cleanly."""
    desktop_src = desktop_file or (PROJECT_ROOT / "packaging" / "linux" / "spectroview.desktop")
    icon_src = icon_file or (PROJECT_ROOT / "spectroview" / "resources" / "icons" / "logo_spectroview.png")

    if deb_root.exists():
        shutil.rmtree(deb_root)

    opt_dir = deb_root / "opt" / "spectroview"
    usr_bin = deb_root / "usr" / "bin"
    usr_apps = deb_root / "usr" / "share" / "applications"
    usr_icons = deb_root / "usr" / "share" / "icons" / "hicolor" / "256x256" / "apps"
    debian_dir = deb_root / "DEBIAN"

    opt_dir.mkdir(parents=True, exist_ok=True)
    usr_bin.mkdir(parents=True, exist_ok=True)
    usr_apps.mkdir(parents=True, exist_ok=True)
    usr_icons.mkdir(parents=True, exist_ok=True)
    debian_dir.mkdir(parents=True, exist_ok=True)

    shutil.copytree(dist_dir, opt_dir, symlinks=True, dirs_exist_ok=True)
    _set_exec_bit(opt_dir / "SPECTROview")

    # Explicit marker inspected by InstallationManager on Linux
    (opt_dir / ".spectroview-deb").write_text(f"version={version}\n", encoding="utf-8")

    bin_link = usr_bin / "spectroview"
    try:
        os.symlink("/opt/spectroview/SPECTROview", bin_link)
    except OSError:
        if sys.platform != "win32":
            raise
        bin_link.write_text("#!/bin/sh\nexec /opt/spectroview/SPECTROview \"$@\"\n", encoding="utf-8")
        _set_exec_bit(bin_link)

    desktop_text = desktop_src.read_text(encoding="utf-8").replace(
        "Exec=spectroview %F", "Exec=/opt/spectroview/SPECTROview %F"
    )
    (usr_apps / "spectroview.desktop").write_text(desktop_text, encoding="utf-8")
    shutil.copy2(icon_src, usr_icons / "spectroview.png")

    installed_kb = _dir_size_kb(deb_root)
    (debian_dir / "control").write_text(
        render_deb_control(version, installed_kb),
        encoding="utf-8",
    )

    postinst = debian_dir / "postinst"
    postinst.write_text(
        "#!/bin/sh\n"
        "set -e\n"
        "if command -v update-desktop-database >/dev/null 2>&1; then\n"
        "    update-desktop-database -q /usr/share/applications || true\n"
        "fi\n"
        "if command -v gtk-update-icon-cache >/dev/null 2>&1 && [ -d /usr/share/icons/hicolor ]; then\n"
        "    gtk-update-icon-cache -q -t /usr/share/icons/hicolor || true\n"
        "fi\n",
        encoding="utf-8",
    )
    _set_exec_bit(postinst)

    postrm = debian_dir / "postrm"
    postrm.write_text(
        "#!/bin/sh\n"
        "set -e\n"
        "# Intentionally preserve user configuration (~/.config/CEA-Leti/) and Working Folders.\n"
        "if command -v update-desktop-database >/dev/null 2>&1; then\n"
        "    update-desktop-database -q /usr/share/applications || true\n"
        "fi\n",
        encoding="utf-8",
    )
    _set_exec_bit(postrm)
    try:
        os.chmod(debian_dir, 0o755)
    except OSError:
        pass

    return deb_root


def build_appimage_binary(appdir: Path, out_file: Path, *, appimagetool_path: Optional[str] = None) -> Path:
    tool = (
        appimagetool_path
        or os.environ.get("APPIMAGETOOL")
        or shutil.which("appimagetool")
        or shutil.which("appimagetool-x86_64.AppImage")
    )
    if not tool:
        raise FileNotFoundError(
            "appimagetool was not found. Install appimagetool-x86_64.AppImage or set APPIMAGETOOL."
        )

    out_file.parent.mkdir(parents=True, exist_ok=True)
    if out_file.exists():
        out_file.unlink()

    env = os.environ.copy()
    env.setdefault("ARCH", "x86_64")
    # Containers/CI without FUSE2 require APPIMAGE_EXTRACT_AND_RUN=1 when appimagetool is itself an AppImage.
    env.setdefault("APPIMAGE_EXTRACT_AND_RUN", "1")
    cmd = [tool, "--no-appstream", str(appdir), str(out_file)]
    print(f"[build_linux] Running: {' '.join(cmd)}")
    subprocess.run(cmd, env=env, check=True)
    _set_exec_bit(out_file)
    return out_file


def build_deb_package(deb_root: Path, out_file: Path) -> Path:
    dpkg_deb = shutil.which("dpkg-deb")
    if not dpkg_deb:
        raise FileNotFoundError("'dpkg-deb' was not found on PATH.")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    if out_file.exists():
        out_file.unlink()
    cmd = [dpkg_deb, "--root-owner-group", "--build", str(deb_root), str(out_file)]
    print(f"[build_linux] Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    return out_file


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Linux AppImage and .deb packages for SPECTROview.")
    parser.add_argument("--format", choices=("all", "appimage", "deb"), default="all")
    parser.add_argument("--output-dir", default="release_artifacts")
    parser.add_argument("--appimagetool", default="", help="Optional explicit path to appimagetool.")
    parser.add_argument("--smoke-test", action="store_true", help="Launch bundled Linux binary offscreen briefly.")
    parser.add_argument("--skip-build", action="store_true", help="Reuse existing dist/SPECTROview bundle.")
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)
    version = read_version()
    print(f"[build_linux] Building SPECTROview {version} for Linux ({args.format})...")

    tls_report = assert_safe_build_tls(require_pip=True)
    print(
        f"[build_linux] TLS security verified: certifi={tls_report['certifi_version']}, "
        f"pip={tls_report['pip_version']}, truststore={tls_report['truststore_available']}"
    )

    if not args.skip_build:
        build_env = os.environ.copy()
        build_env["SPECTROVIEW_PACKAGE_FORMAT"] = f"linux_{args.format}"
        cmd = [sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", "main.spec"]
        print(f"[build_linux] Running: {' '.join(cmd)}")
        subprocess.run(cmd, env=build_env, check=True)

    dist_dir = PROJECT_ROOT / "dist" / "SPECTROview"
    verify_linux_bundle(dist_dir)

    if args.smoke_test:
        print("[build_linux] Running offscreen Linux executable smoke test...")
        run_smoke_test(dist_dir)
        print("[build_linux] Smoke test passed.")

    out_dir = (PROJECT_ROOT / args.output_dir).resolve()
    built: list[Path] = []

    if args.format in ("all", "appimage"):
        appdir = PROJECT_ROOT / "build" / "AppDir"
        stage_appdir(dist_dir, appdir)
        built.append(
            build_appimage_binary(
                appdir,
                out_dir / appimage_filename(version),
                appimagetool_path=args.appimagetool or None,
            )
        )

    if args.format in ("all", "deb"):
        deb_root = PROJECT_ROOT / "build" / "deb_root"
        stage_deb_root(dist_dir, deb_root, version=version)
        built.append(build_deb_package(deb_root, out_dir / deb_filename(version)))

    print("[build_linux] Built artifacts:")
    for item in built:
        size_mb = item.stat().st_size / (1024 * 1024)
        print(f"  - {item.name} ({size_mb:.2f} MiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
