"""Model system subpackage: installation detection, TLS trust store, updater, and OS patches."""

from spectroview.model.system.m_installation import (
    InstallationInfo,
    InstallationManager,
    InstallType,
)
from spectroview.model.system.m_macos_font_patch import (
    apply_macos_font_cache_fix,
    is_macos_font_cache_fix_applied,
)
from spectroview.model.system.m_tls import (
    InsecureTrustStoreError,
    assert_safe_build_tls,
    build_ssl_context,
    ensure_secure_tls,
    get_safe_certifi_cafile,
    inject_truststore,
)

__all__ = [
    "InstallType",
    "InstallationInfo",
    "InstallationManager",
    "InsecureTrustStoreError",
    "UpdateCheckerWorker",
    "UpdateManager",
    "apply_macos_font_cache_fix",
    "assert_safe_build_tls",
    "build_ssl_context",
    "ensure_secure_tls",
    "get_safe_certifi_cafile",
    "inject_truststore",
    "is_macos_font_cache_fix_applied",
]

_LAZY_SUBMODULES = frozenset({"m_update_checker", "m_update_manager"})


def __getattr__(name: str):
    if name in _LAZY_SUBMODULES:
        import importlib

        return importlib.import_module(f"{__name__}.{name}")
    if name == "UpdateCheckerWorker":
        from spectroview.model.system.m_update_checker import UpdateCheckerWorker

        return UpdateCheckerWorker
    if name == "UpdateManager":
        from spectroview.model.system.m_update_manager import UpdateManager

        return UpdateManager
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
