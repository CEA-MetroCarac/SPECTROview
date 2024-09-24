# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['C:\\Users\\VL251876\\Documents\\Python\\SPECTROview\\app\\main.py'],
    pathex=['C:\\Users\\VL251876\\Documents\\Python\\SPECTROview\\app'],
    binaries=[],
    datas=[('C:\\Users\\VL251876\\Documents\\Python\\SPECTROview\\app\\ui\\iconpack\\icon3.ico', '.'), ('C:\\Users\\VL251876\\Documents\\Python\\SPECTROview', 'SPECTROview/'), ('C:\\Users\\VL251876\\Documents\\Python\\SPECTROview\\app', 'app/'), ('C:\\Users\\VL251876\\Documents\\Python\\SPECTROview\\app\\doc', 'doc/'), ('C:\\Users\\VL251876\\Documents\\Python\\SPECTROview\\app\\resources', 'resources/'), ('C:\\Users\\VL251876\\Documents\\Python\\SPECTROview\\app\\ui', 'ui/'), ('C:\\Users\\VL251876\\Documents\\Python\\SPECTROview\\app\\ui\\iconpack', 'iconpack/')],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='main',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['C:\\Users\\VL251876\\Documents\\Python\\SPECTROview\\app\\ui\\iconpack\\icon3.ico'],
)
