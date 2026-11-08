"""Build script for macOS Apple Silicon (``arm64``) standalone .app bundle and .dmg disk image.

Targets Apple Silicon (``arm64``) builds using the unified ``main.spec``
configuration (Intel ``x64`` / ``x86_64`` macOS builds are no longer
supported). By default generates an unsigned / ad-hoc disk image without
requiring Apple Developer credentials; optional Developer ID signing and
notarization activate automatically when signing secrets are present in the
environment.

Usage:
    python3 packaging/macos/build_dmg.py [--arch arm64|auto] [--output-dir release_artifacts] [--smoke-test]
"""

from __future__ import annotations

import argparse
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from spectroview.model.m_tls import assert_safe_build_tls


def read_version() -> str:
    init_py = PROJECT_ROOT / "spectroview" / "__init__.py"
    text = init_py.read_text(encoding="utf-8")
    match = re.search(r'^VERSION\s*=\s*["\']([^"\']+)["\']', text, re.MULTILINE)
    if not match:
        raise RuntimeError(f"Could not extract VERSION from {init_py}")
    return match.group(1)


def normalize_arch(raw_arch: Optional[str] = None) -> str:
    """Normalise machine architecture strings for Apple Silicon (``arm64``).

    macOS builds target Apple Silicon (``arm64``) exclusively; Intel (``x64`` /
    ``x86_64``) builds are no longer supported.
    """
    val = (raw_arch or platform.machine() or "arm64").strip().lower()
    if val in ("arm64", "aarch64", "apple_silicon"):
        return "arm64"
    if val in ("x86_64", "amd64", "x64", "intel"):
        raise ValueError(
            f"Unsupported macOS architecture: {raw_arch or platform.machine()!r}. "
            "Intel macOS builds are no longer supported; only Apple Silicon (arm64) is supported."
        )
    raise ValueError(f"Unsupported macOS architecture: {raw_arch!r}")


def dmg_filename(version: str, arch: str) -> str:
    """Return canonical macOS disk image filename for *version* and *arch*."""
    return f"SPECTROview_{version}_macOS_{normalize_arch(arch)}.dmg"


def verify_app_bundle(app_path: Path) -> None:
    """Assert critical macOS .app structure, TLS CA bundle, and AI assets exist."""
    contents = app_path / "Contents"
    info_plist = contents / "Info.plist"
    main_bin = contents / "MacOS" / "SPECTROview"

    required = [info_plist, main_bin]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise RuntimeError(
            "PyInstaller macOS .app bundle is missing required files:\n- "
            + "\n- ".join(missing)
        )

    # PyInstaller 6 places bundled data inside Contents/Frameworks and symlinks into Contents/Resources.
    resource_roots = [contents / "Resources", contents / "Frameworks"]
    for rel in (
        "spectroview/resources/build_info.json",
        "spectroview/ai_agent/config/model.yaml",
        "spectroview/ai_agent/prompts/system.md",
        "certifi/cacert.pem",
    ):
        if not any((root / rel).exists() for root in resource_roots):
            raise RuntimeError(
                f"PyInstaller macOS .app bundle missing bundled asset '{rel}' "
                f"under {[str(r) for r in resource_roots]}"
            )


def run_smoke_test(app_path: Path) -> None:
    """Launch the bundled macOS binary offscreen briefly to confirm runtime load."""
    exe = (app_path / "Contents" / "MacOS" / "SPECTROview").resolve()
    probe_env = os.environ.copy()
    probe_env["QT_QPA_PLATFORM"] = "offscreen"
    proc = subprocess.Popen(
        [str(exe), "--smoke-test"],
        cwd=str(exe.parent),
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
        raise RuntimeError(f"macOS .app smoke launch timed out after 15s:\n{err_msg}")
    if proc.returncode != 0:
        err_msg = stderr.decode("utf-8", errors="replace") if stderr else ""
        raise RuntimeError(f"macOS .app smoke launch exited with code {proc.returncode}:\n{err_msg}")


def build_codesign_commands(
    app_path: Path,
    *,
    signing_identity: Optional[str] = None,
    entitlements_path: Optional[Path] = None,
) -> list[list[str]]:
    """Build codesign command list based on whether Developer ID identity is provided.

    Without ``signing_identity``, falls back to ad-hoc signing (``-s -``) on
    macOS so Apple Silicon Gatekeeper can run after the user explicitly allows it,
    or keeps signing optional without requiring Apple Developer credentials.
    """
    ent = str(entitlements_path or (PROJECT_ROOT / "packaging" / "macos" / "entitlements.plist"))
    identity = (signing_identity or "").strip()
    if identity:
        return [
            [
                "codesign",
                "--deep",
                "--force",
                "--options",
                "runtime",
                "--timestamp",
                "--entitlements",
                ent,
                "--sign",
                identity,
                str(app_path),
            ],
            ["codesign", "--verify", "--deep", "--strict", str(app_path)],
        ]
    return [
        [
            "codesign",
            "--deep",
            "--force",
            "--sign",
            "-",
            str(app_path),
        ]
    ]


def stage_dmg_contents(
    app_path: Path,
    staging_dir: Path,
    *,
    version: str,
    arch: str,
    signed: bool = False,
) -> Path:
    """Populate *staging_dir* with SPECTROview.app, Applications symlink, and launch note."""
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)

    target_app = staging_dir / "SPECTROview.app"
    shutil.copytree(app_path, target_app, symlinks=True)

    apps_link = staging_dir / "Applications"
    try:
        os.symlink("/Applications", apps_link)
    except OSError:
        # Windows hosts without symlink privilege inside unit tests may skip OS symlink;
        # macOS build hosts always support POSIX symlinks.
        if sys.platform != "win32":
            raise

    if not signed:
        note = staging_dir / "First Launch (Unsigned Build).txt"
        note.write_text(
            f"SPECTROview {version} (macOS {arch})\n"
            "=====================================\n\n"
            "1. Drag SPECTROview.app onto the Applications folder icon.\n"
            "2. Because this release is not yet signed with an Apple Developer ID,\n"
            "   macOS Gatekeeper may warn on first launch.\n"
            "3. To open SPECTROview:\n"
            "   - Control-click (or right-click) SPECTROview.app in /Applications -> Open,\n"
            "     OR open System Settings -> Privacy & Security -> Click 'Open Anyway'.\n\n"
            "Your settings and Working Folder are stored outside SPECTROview.app and\n"
            "are preserved whenever you drag a newer version into Applications.\n",
            encoding="utf-8",
        )
    return staging_dir


def create_dmg(
    staging_dir: Path,
    dmg_path: Path,
    *,
    version: str,
    signing_identity: Optional[str] = None,
) -> Path:
    """Create compressed macOS disk image via ``hdiutil`` (and optionally codesign)."""
    dmg_path.parent.mkdir(parents=True, exist_ok=True)
    if dmg_path.exists():
        dmg_path.unlink()

    cmd = [
        "hdiutil",
        "create",
        "-volname",
        f"SPECTROview {version}",
        "-srcfolder",
        str(staging_dir),
        "-ov",
        "-format",
        "UDZO",
        str(dmg_path),
    ]
    print(f"[build_dmg] Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    identity = (signing_identity or "").strip()
    if identity:
        subprocess.run(
            ["codesign", "--force", "--timestamp", "--sign", identity, str(dmg_path)],
            check=True,
        )
    return dmg_path


def maybe_notarize_dmg(dmg_path: Path) -> bool:
    """Notarize and staple *dmg_path* when Apple notarization env vars exist."""
    apple_id = os.environ.get("APPLE_ID", "").strip()
    team_id = os.environ.get("APPLE_TEAM_ID", "").strip()
    app_pw = os.environ.get("APPLE_APP_PASSWORD", "").strip()
    if not (apple_id and team_id and app_pw):
        print("[build_dmg] Notarization credentials not set; skipping notarytool cleanly.")
        return False

    subprocess.run(
        [
            "xcrun",
            "notarytool",
            "submit",
            str(dmg_path),
            "--apple-id",
            apple_id,
            "--team-id",
            team_id,
            "--password",
            app_pw,
            "--wait",
        ],
        check=True,
    )
    subprocess.run(["xcrun", "stapler", "staple", str(dmg_path)], check=True)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Build macOS SPECTROview .app and .dmg.")
    parser.add_argument(
        "--arch",
        default="auto",
        choices=["arm64", "auto"],
        help="Target macOS architecture: Apple Silicon (arm64, default) or auto.",
    )
    parser.add_argument("--output-dir", default="release_artifacts", help="Directory for output .dmg.")
    parser.add_argument("--smoke-test", action="store_true", help="Launch bundled .app offscreen briefly.")
    parser.add_argument("--skip-dmg", action="store_true", help="Build and verify .app only.")
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)
    version = read_version()
    arch = normalize_arch(None if args.arch == "auto" else args.arch)
    print(f"[build_dmg] Building SPECTROview {version} for macOS ({arch})...")

    tls_report = assert_safe_build_tls(require_pip=True)
    print(
        f"[build_dmg] TLS security verified: certifi={tls_report['certifi_version']}, "
        f"pip={tls_report['pip_version']}, truststore={tls_report['truststore_available']}"
    )

    build_env = os.environ.copy()
    build_env["SPECTROVIEW_PACKAGE_FORMAT"] = f"darwin_{arch}"
    pyinstaller_cmd = [sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", "main.spec"]
    print(f"[build_dmg] Running: {' '.join(pyinstaller_cmd)}")
    subprocess.run(pyinstaller_cmd, env=build_env, check=True)

    app_path = PROJECT_ROOT / "dist" / "SPECTROview.app"
    verify_app_bundle(app_path)

    signing_identity = os.environ.get("APPLE_SIGNING_IDENTITY", "").strip() or None
    for cs_cmd in build_codesign_commands(app_path, signing_identity=signing_identity):
        print(f"[build_dmg] Running: {' '.join(cs_cmd)}")
        subprocess.run(cs_cmd, check=False if not signing_identity else True)

    if args.smoke_test:
        print("[build_dmg] Running offscreen .app smoke test...")
        run_smoke_test(app_path)
        print("[build_dmg] Smoke test passed.")

    if args.skip_dmg:
        return 0

    staging_dir = PROJECT_ROOT / "build" / "dmg_staging"
    stage_dmg_contents(
        app_path,
        staging_dir,
        version=version,
        arch=arch,
        signed=bool(signing_identity),
    )

    out_dir = (PROJECT_ROOT / args.output_dir).resolve()
    dmg_out = out_dir / dmg_filename(version, arch)
    create_dmg(staging_dir, dmg_out, version=version, signing_identity=signing_identity)
    if signing_identity:
        maybe_notarize_dmg(dmg_out)

    size_mb = dmg_out.stat().st_size / (1024 * 1024)
    print(f"[build_dmg] Built macOS DMG: {dmg_out.name} ({size_mb:.2f} MiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
