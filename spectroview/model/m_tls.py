"""Centralized TLS trust-store configuration and CA security validation.

Addresses the October 2026 eMudhra / emSign root CA security incident
(`GHSA-6q48-6c7w-9gw2`) by:

1. Prioritising explicit CA overrides (``SSL_CERT_FILE`` /
   ``REQUESTS_CA_BUNDLE``) for institutional proxies and custom PKIs.
2. Preferring the host operating system's native certificate store via
   ``truststore`` so OS-level CA revocations apply immediately.
3. Refusing outdated ``certifi`` bundles (< ``2026.10.19``) on fallback when
   ``truststore`` is unavailable, rather than silently trusting compromised
   intermediate/root certificates.
4. Providing build-time and pip-upgrade verification helpers so standalone
   installers and in-place wheel upgrades cannot reintroduce stale CAs.

This module belongs strictly to the Model layer: it performs no Qt imports and
defers importing optional packages until called.
"""

from __future__ import annotations

import importlib.metadata
import logging
import os
import re
from typing import Any, Optional

log = logging.getLogger(__name__)

MIN_SAFE_CERTIFI_VERSION: tuple[int, ...] = (2026, 10, 19)
RECOMMENDED_CERTIFI_VERSION: str = "2026.10.23"
MIN_SAFE_PIP_VERSION: tuple[int, ...] = (26, 2, 2)
MIN_TRUSTSTORE_PIP_VERSION: tuple[int, ...] = (24, 2)

_TRUSTSTORE_INJECTED = False


class InsecureTrustStoreError(RuntimeError):
    """Raised when only a compromised or unpatched CA trust store is available."""


def parse_calver_version(version_str: str) -> tuple[int, ...]:
    """Parse a dot-separated CalVer/PEP 440 version prefix into an integer tuple."""
    cleaned = str(version_str or "").strip().lstrip("v")
    parts: list[int] = []
    for segment in cleaned.split("."):
        match = re.match(r"(\d+)", segment)
        if not match:
            break
        parts.append(int(match.group(1)))
    return tuple(parts)


def get_custom_ca_bundle() -> str:
    """Return an existing custom CA bundle path from environment variables, or ``""``."""
    ca_bundle = (
        os.environ.get("SSL_CERT_FILE")
        or os.environ.get("REQUESTS_CA_BUNDLE")
        or ""
    ).strip()
    if ca_bundle and os.path.isfile(ca_bundle):
        return ca_bundle
    return ""


def _enforce_min_tls(ctx: Any) -> Any:
    """Restrict *ctx* to TLS 1.2 or newer when supported by the stdlib/backend."""
    try:
        import ssl

        if hasattr(ssl, "TLSVersion") and hasattr(ctx, "minimum_version"):
            ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    except Exception:
        pass
    return ctx


def is_certifi_safe(version_str: Optional[str] = None) -> tuple[bool, str]:
    """Return ``(is_safe, version)`` checking ``certifi >= 2026.10.19``."""
    ver = version_str
    if ver is None:
        try:
            import certifi

            ver = str(getattr(certifi, "__version__", "") or "")
        except ImportError:
            return False, ""
    parsed = parse_calver_version(ver)
    if not parsed:
        # Unknown stub without version attribute cannot prove freshness cleanly,
        # but don't treat missing version metadata inside mock tests as older CalVer
        # unless a version string was actually supplied.
        return False, ver
    # Pad lengths aren't an issue for YYYY.MM.DD, compare cleanly:
    return parsed >= MIN_SAFE_CERTIFI_VERSION, ver


def is_pip_safe(version_str: Optional[str] = None) -> tuple[bool, str]:
    """Return ``(is_safe, version)`` checking ``pip >= 26.2.2``."""
    ver = version_str
    if ver is None:
        try:
            ver = importlib.metadata.version("pip")
        except Exception:
            try:
                import pip

                ver = str(getattr(pip, "__version__", "") or "")
            except Exception:
                return False, ""
    parsed = parse_calver_version(ver)
    if not parsed:
        return False, ver
    return parsed >= MIN_SAFE_PIP_VERSION, ver


def get_safe_certifi_cafile() -> str:
    """Return ``certifi.where()`` only when ``certifi >= 2026.10.19``, else ``""``."""
    try:
        import certifi
    except ImportError:
        return ""
    ver = str(getattr(certifi, "__version__", "") or "")
    safe, _ = is_certifi_safe(ver)
    if not safe:
        return ""
    ca_path = str(certifi.where() or "")
    return ca_path if ca_path and os.path.isfile(ca_path) else ""


def inject_truststore() -> bool:
    """Inject the native OS trust store into ``ssl.SSLContext`` cleanly (idempotent)."""
    global _TRUSTSTORE_INJECTED
    if get_custom_ca_bundle():
        # Respect explicit user/enterprise CA bundle overrides cleanly.
        return False
    if _TRUSTSTORE_INJECTED:
        return True
    try:
        import truststore

        truststore.inject_into_ssl()
        _TRUSTSTORE_INJECTED = True
        return True
    except Exception as exc:
        log.debug("truststore injection unavailable: %s", exc)
        return False


def build_ssl_context(strict_certifi: bool = True) -> Optional[Any]:
    """Build an SSL context prioritising custom CA -> OS truststore -> patched certifi.

    Raises
    ------
    InsecureTrustStoreError
        When ``strict_certifi`` is ``True``, neither a custom CA bundle nor
        ``truststore`` is usable, and the fallback ``certifi`` package is older
        than ``2026.10.19`` (vulnerable to compromised eMudhra / emSign roots).
    """
    import ssl

    ca_bundle = get_custom_ca_bundle()
    if ca_bundle:
        try:
            return _enforce_min_tls(ssl.create_default_context(cafile=ca_bundle))
        except Exception:
            pass

    try:
        import truststore

        ctx = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        return _enforce_min_tls(ctx)
    except Exception:
        pass

    try:
        import certifi
    except ImportError:
        return None

    ver = str(getattr(certifi, "__version__", "") or "")
    safe, _ = is_certifi_safe(ver)
    if strict_certifi and ver and not safe:
        raise InsecureTrustStoreError(
            f"Refusing HTTPS connection: installed certifi ({ver}) predates "
            f"2026.10.19 (GHSA-6q48-6c7w-9gw2 eMudhra/emSign CA revocation) and "
            "the native OS truststore backend is unavailable. Upgrade with: "
            f"pip install --upgrade \"pip>={'.'.join(map(str, MIN_SAFE_PIP_VERSION))}\" "
            f"\"certifi>={RECOMMENDED_CERTIFI_VERSION}\" truststore"
        )

    try:
        return _enforce_min_tls(ssl.create_default_context(cafile=certifi.where()))
    except Exception:
        return None


def ensure_secure_tls(strict_certifi: bool = True) -> str:
    """Validate TLS verification for stdlib / httpx / SDK clients and return backend name.

    Returns one of ``"custom_ca_bundle"``, ``"os_truststore"``, or ``"certifi"``.
    Raises :class:`InsecureTrustStoreError` if only a known-vulnerable CA store exists.
    """
    if get_custom_ca_bundle():
        return "custom_ca_bundle"

    if inject_truststore():
        return "os_truststore"

    safe, ver = is_certifi_safe()
    if safe:
        return "certifi"

    if strict_certifi:
        ver_label = ver or "not installed"
        raise InsecureTrustStoreError(
            f"HTTPS verification blocked: native OS truststore is unavailable and "
            f"certifi ({ver_label}) does not meet >= 2026.10.19 (GHSA-6q48-6c7w-9gw2). "
            f"Please upgrade certifi>={RECOMMENDED_CERTIFI_VERSION} and pip>=26.2.2."
        )
    return "unverified_fallback"


def get_tls_security_report() -> dict[str, Any]:
    """Return a structured diagnostic snapshot of active CA/TLS protections."""
    custom_ca = get_custom_ca_bundle()

    truststore_available = False
    try:
        import truststore  # noqa: F401

        truststore_available = True
    except Exception:
        truststore_available = False

    certifi_safe, certifi_ver = is_certifi_safe()
    pip_safe, pip_ver = is_pip_safe()

    if custom_ca:
        backend = "custom_ca_bundle"
    elif truststore_available:
        backend = "os_truststore"
    elif certifi_safe:
        backend = "certifi"
    else:
        backend = "insecure_or_missing"

    warnings: list[str] = []
    if not certifi_safe:
        warnings.append(
            f"certifi ({certifi_ver or 'missing'}) < {RECOMMENDED_CERTIFI_VERSION} "
            "(GHSA-6q48-6c7w-9gw2 eMudhra/emSign advisory)"
        )
    if pip_ver and not pip_safe:
        warnings.append(
            f"pip ({pip_ver}) < {'.'.join(map(str, MIN_SAFE_PIP_VERSION))} "
            "(vendored certifi may predate eMudhra/emSign removal)"
        )

    return {
        "active_backend": backend,
        "custom_ca_bundle": custom_ca,
        "truststore_available": truststore_available,
        "truststore_injected": _TRUSTSTORE_INJECTED,
        "certifi_version": certifi_ver or "Not installed",
        "certifi_safe": certifi_safe,
        "pip_version": pip_ver or "Not installed",
        "pip_safe": pip_safe,
        "warnings": warnings,
    }


def assert_safe_build_tls(require_pip: bool = True) -> dict[str, Any]:
    """Fail build/CI packaging if ``certifi`` or ``pip`` carries compromised CAs."""
    report = get_tls_security_report()
    errors: list[str] = []
    if not report["certifi_safe"]:
        errors.append(
            f"certifi {report['certifi_version']} does not satisfy >= "
            f"{'.'.join(map(str, MIN_SAFE_CERTIFI_VERSION))} (GHSA-6q48-6c7w-9gw2)."
        )
    if require_pip and report["pip_version"] != "Not installed" and not report["pip_safe"]:
        errors.append(
            f"pip {report['pip_version']} does not satisfy >= "
            f"{'.'.join(map(str, MIN_SAFE_PIP_VERSION))} (GHSA-6q48-6c7w-9gw2)."
        )
    if errors:
        raise InsecureTrustStoreError("Build TLS gate failed:\n- " + "\n- ".join(errors))
    return report

