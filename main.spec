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
        ('doc', 'doc'),
    ],
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
        console=True,
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
        a.binaries,
        a.zipfiles,
        a.datas,
        [],
        name=APP_NAME,
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=True,
        upx_exclude=[],
        runtime_tmpdir=None,
        console=True,
        icon=icon_file,
    )

