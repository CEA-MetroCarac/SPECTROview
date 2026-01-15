# -*- mode: python ; coding: utf-8 -*-

block_cipher = None


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


# ------------------------------------------------------------
# EXE
# ------------------------------------------------------------
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='SPECTROview',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # windowed app
    icon='spectroview/resources/icons/logo_spectroview.ico',
    version='version.txt',
)
