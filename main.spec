# -*- mode: python ; coding: utf-8 -*-

import glob
import os
import re
import sys

def get_version():
    """Extract version from spectroview/__init__.py."""
    init_path = os.path.join("spectroview", "__init__.py")
    if not os.path.exists(init_path):
        return "0.0.0"
    with open(init_path, "r", encoding="utf-8") as f:
        content = f.read()
    match = re.search(r'^VERSION\s*=\s*["\']([^"\']+)["\']', content, re.MULTILINE)
    return match.group(1) if match else "0.0.0"


VERSION = get_version()
APP_NAME = "SPECTROview"
BUNDLE_IDENTIFIER = "fr.cea.spectroview"

block_cipher = None

is_windows = sys.platform.startswith("win")
is_macos = sys.platform == "darwin"
is_linux = sys.platform.startswith("linux")

spec_dir = os.path.dirname(os.path.abspath(SPEC)) if "SPEC" in locals() else os.getcwd()
if spec_dir not in sys.path:
    sys.path.insert(0, spec_dir)

# ------------------------------------------------------------
# Security Gate: Verify CA / TLS Trust Store Before Freezing
# ------------------------------------------------------------
from spectroview.model.m_tls import assert_safe_build_tls

tls_report = assert_safe_build_tls(require_pip=False)
print(
    f"[main.spec] TLS gate passed: certifi={tls_report['certifi_version']}, "
    f"truststore={tls_report['truststore_available']}, pip={tls_report['pip_version']}"
)

# ------------------------------------------------------------
# Build Environment Snapshot -> staged build/build_info.json
# ------------------------------------------------------------
build_dir = os.path.join(spec_dir, "build")
os.makedirs(build_dir, exist_ok=True)
build_info_stage = os.path.join(build_dir, "build_info.json")
package_format = os.environ.get("SPECTROVIEW_PACKAGE_FORMAT") or None

try:
    from spectroview.settings.build_info import generate_build_snapshot

    snapshot = generate_build_snapshot(
        target_path=build_info_stage,
        package_format=package_format,
    )
    pkg_count = len(snapshot.get("packages", {}))
    print(f"[main.spec] Staged build snapshot at {build_info_stage}: {pkg_count} packages recorded.")
except Exception as exc:
    print(f"[main.spec] Warning: Could not generate build snapshot: {exc}")

# ------------------------------------------------------------
# Extra Datas, Metadata, and AI Agent Assets
# ------------------------------------------------------------
extra_datas = []

# Static AI Agent prompts/rules/knowledge/examples/config (never CHATLOG or TEMPLATE)
for rel_pattern in [
    "spectroview/ai_agent/config/*.yaml",
    "spectroview/ai_agent/prompts/*.md",
    "spectroview/ai_agent/rules/*.md",
    "spectroview/ai_agent/knowledge/*.md",
    "spectroview/ai_agent/examples/*.md",
]:
    for src_path in glob.glob(rel_pattern):
        dest_dir = os.path.dirname(src_path).replace("\\", "/")
        extra_datas.append((src_path, dest_dir))

try:
    from PyInstaller.utils.hooks import collect_data_files, copy_metadata

    for pkg in [
        "PySide6",
        "numpy",
        "scipy",
        "matplotlib",
        "pandas",
        "pybaselines",
        "openpyxl",
        "superqt",
        "pyarrow",
        "certifi",
        "truststore",
        "openai",
        "anthropic",
        "ollama",
        "mcp",
        "pyyaml",
        "markdown",
        "pywin32",
        "velopack",
    ]:
        try:
            extra_datas += copy_metadata(pkg)
        except Exception:
            pass

    for data_pkg in ["certifi", "truststore"]:
        try:
            extra_datas += collect_data_files(data_pkg)
        except Exception:
            pass
except Exception as exc:
    print(f"[main.spec] Warning: PyInstaller hook helper issue: {exc}")

# ------------------------------------------------------------
# Icon & Version Info Handling
# ------------------------------------------------------------
if is_windows:
    icon_file = "spectroview/resources/icons/logo_spectroview.ico"
elif is_macos:
    icon_file = "spectroview/resources/icons/logo_spectroview.icns"
else:
    icon_file = "spectroview/resources/icons/logo_spectroview.png"

win_version_file = os.path.join(build_dir, "win_version_info.txt")
if is_windows and not os.path.isfile(win_version_file):
    win_version_file = None

# ------------------------------------------------------------
# Hidden Imports (lazy-loaded modules + platform backends)
# ------------------------------------------------------------
hidden_imports = [
    "truststore",
    "certifi",
    "httpx",
    "yaml",
    "spectroview.ai_agent",
    "spectroview.ai_agent.v_chat_panel",
    "spectroview.ai_agent.vm_chat",
    "spectroview.ai_agent.m_llm_client",
    "spectroview.ai_agent.mcp.server",
    "spectroview.ai_agent.mcp.runtime",
    "spectroview.ai_agent.mcp.hub",
    "mcp",
    "mcp.server.fastmcp",
    "starlette",
    "sse_starlette",
    "uvicorn",
    "uvicorn.logging",
    "uvicorn.loops.auto",
    "uvicorn.protocols.http.auto",
    "uvicorn.lifespan.on",
]
if is_windows:
    hidden_imports.extend(["velopack"])

# ------------------------------------------------------------
# Analysis
# ------------------------------------------------------------
a = Analysis(
    ["spectroview/main.py"],
    pathex=[spec_dir],
    binaries=[],
    datas=[
        ("spectroview/resources", "spectroview/resources"),
    ]
    + extra_datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=["packaging/macos/pyinstaller_runtime_hook.py"] if is_macos else [],
    excludes=[
        "PyQt5",
        "PyQt5.QtCore",
        "PyQt5.QtGui",
        "PyQt5.QtWidgets",
        "PyQt6",
        "PyQt6.QtCore",
        "PyQt6.QtGui",
        "PyQt6.QtWidgets",
        "PySide6.QtWebEngine",
        "PySide6.QtWebEngineCore",
        "PySide6.QtWebEngineWidgets",
        "PySide6.QtQml",
        "PySide6.QtQuick",
        "PySide6.Qt3DCore",
        "PySide6.Qt3DRender",
        "tkinter",
        "IPython",
        "notebook",
        "jedi",
        # Heavy optional ML & speech packages present in local dev environments
        "torch",
        "torchvision",
        "torchaudio",
        "timm",
        "numba",
        "llvmlite",
        "transformers",
        "huggingface_hub",
        "speech_recognition",
        "pocketsphinx",
        "babel",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# Replace any copied build_info.json cleanly with staged build/build_info.json
if os.path.isfile(build_info_stage):
    norm_target = os.path.normpath("spectroview/resources/build_info.json")
    a.datas = [
        entry for entry in a.datas if os.path.normpath(entry[0]) != norm_target
    ]
    a.datas.append(
        ("spectroview/resources/build_info.json", build_info_stage, "DATA")
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
        name=f"{APP_NAME}.app",
        icon=icon_file,
        bundle_identifier=BUNDLE_IDENTIFIER,
        version=VERSION,
        info_plist={
            "CFBundleName": APP_NAME,
            "CFBundleDisplayName": APP_NAME,
            "CFBundleIdentifier": BUNDLE_IDENTIFIER,
            "CFBundleVersion": VERSION,
            "CFBundleShortVersionString": VERSION,
            "NSHighResolutionCapable": True,
            "LSMinimumSystemVersion": "12.0",
            "NSPrincipalClass": "NSApplication",
        },
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
        upx=False,
        console=False,
        icon=icon_file,
        version=win_version_file,
    )

    coll = COLLECT(
        exe,
        a.binaries,
        a.zipfiles,
        a.datas,
        strip=False,
        upx=False,
        upx_exclude=[],
        name=APP_NAME,
    )

else:
    # Linux (--onedir bundle reused for AppImage and .deb packaging)
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
