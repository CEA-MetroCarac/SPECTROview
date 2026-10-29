# -*- mode: python ; coding: utf-8 -*-

import sys
import os
import re

def get_version():
    """Extract version from spectroview/__init__.py"""
    init_path = os.path.join("spectroview", "__init__.py")
    if not os.path.exists(init_path):
        return "0.0.0"
    with open(init_path, "r", encoding="utf-8") as f:
        content = f.read()
    match = re.search(r'^VERSION\s*=\s*["\']([^"\']+)["\']', content, re.MULTILINE)
    return match.group(1) if match else "0.0.0"

VERSION = get_version()
APP_NAME = f"SPECTROview_{VERSION}"

block_cipher = None

is_windows = sys.platform.startswith("win")
is_macos = sys.platform == "darwin"

# ------------------------------------------------------------
# Build Environment Snapshot & Distribution Metadata
# ------------------------------------------------------------
spec_dir = os.path.dirname(os.path.abspath(SPEC)) if 'SPEC' in locals() else os.getcwd()
if spec_dir not in sys.path:
    sys.path.insert(0, spec_dir)

try:
    from spectroview.settings.build_info import generate_build_snapshot
    snapshot = generate_build_snapshot()
    pkg_count = len(snapshot.get("packages", {}))
    print(f"[main.spec] Captured build environment snapshot: {pkg_count} packages recorded.")
except Exception as exc:
    print(f"[main.spec] Warning: Could not generate build snapshot: {exc}")

extra_datas = []
try:
    from PyInstaller.utils.hooks import copy_metadata
    for pkg in [
        'PySide6', 'scipy', 'matplotlib', 'pandas', 'pybaselines',
        'openpyxl', 'pyarrow', 'pyqt-toast-notification', 'openai',
        'anthropic', 'mcp', 'pyyaml', 'renishawWiRE', 'pywin32'
    ]:
        try:
            extra_datas += copy_metadata(pkg)
        except Exception:
            pass
except Exception as exc:
    print(f"[main.spec] Warning: copy_metadata unavailable: {exc}")

# ------------------------------------------------------------
# Icon handling (platform specific)
# ------------------------------------------------------------
if is_windows:
    icon_file = "spectroview/resources/icons/logo_spectroview.ico"
elif is_macos:
    icon_file = "spectroview/resources/icons/logo_spectroview.icns"
else:
    icon_file = None


# ------------------------------------------------------------
# Analysis
# ------------------------------------------------------------
a = Analysis(
    ['spectroview/main.py'],
    pathex=['spectroview'],
    binaries=[],
    datas=[
        ('spectroview/resources', 'spectroview/resources'),
    ] + extra_datas,
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'PyQt5',
        'PyQt5.QtCore',
        'PyQt5.QtGui',
        'PyQt5.QtWidgets',
        'PyQt6',
        'PyQt6.QtCore',
        'PyQt6.QtGui',
        'PyQt6.QtWidgets',
        'PySide6.QtWebEngine',
        'PySide6.QtWebEngineCore',
        'PySide6.QtWebEngineWidgets',
        'PySide6.QtQml',
        'PySide6.QtQuick',
        'PySide6.Qt3DCore',
        'PySide6.Qt3DRender',
        'tkinter',
        'IPython',
        'notebook',
        'jedi',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)


# ------------------------------------------------------------
# PYZ
# ------------------------------------------------------------
pyz = PYZ(
    a.pure,
    a.zipped_data,
    cipher=block_cipher,
)


if is_macos:
    exe = EXE(
        pyz,
        a.scripts,
        [],
        exclude_binaries=True,
        name=APP_NAME,
        debug=False,
        strip=False,
        upx=False,
        console=False,
        icon=icon_file,
    )

    coll = COLLECT(
        exe,
        a.binaries,
        a.zipfiles,
        a.datas,
        strip=False,
        upx=False,
        name=APP_NAME,
    )

    app = BUNDLE(
        coll,
        name=f'{APP_NAME}.app',
        icon=icon_file,
        bundle_identifier='com.cea.spectroview',
    )

elif is_windows:
    exe = EXE(
        pyz,
        a.scripts,
        [],
        exclude_binaries=True,
        name=APP_NAME,
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=True,
        console=False,
        icon=icon_file,
    )

    coll = COLLECT(
        exe,
        a.binaries,
        a.zipfiles,
        a.datas,
        strip=False,
        upx=True,
        upx_exclude=[],
        name=APP_NAME,
    )
