"""Unit tests for centralized TLS configuration and eMudhra/emSign CA safeguards."""

from __future__ import annotations

import builtins
import ssl
import types

import pytest

from spectroview.model import m_tls


class TestCalVerAndChecks:
    def test_parse_calver_version(self):
        assert m_tls.parse_calver_version("2026.10.23") == (2026, 10, 23)
        assert m_tls.parse_calver_version("26.2.2") == (26, 2, 2)
        assert m_tls.parse_calver_version("v2026.10.19.post1") == (2026, 10, 19)

    def test_certifi_safety_threshold(self):
        assert m_tls.is_certifi_safe("2026.10.18") == (False, "2026.10.18")
        assert m_tls.is_certifi_safe("2026.10.19") == (True, "2026.10.19")
        assert m_tls.is_certifi_safe("2026.10.23") == (True, "2026.10.23")

    def test_pip_safety_threshold(self):
        assert m_tls.is_pip_safe("26.2.1") == (False, "26.2.1")
        assert m_tls.is_pip_safe("26.2.2") == (True, "26.2.2")
        assert m_tls.is_pip_safe("26.3") == (True, "26.3")


class TestBuildSslContext:
    def test_rejects_outdated_certifi_when_truststore_unavailable(self, monkeypatch):
        real_import = builtins.__import__
        fake_certifi = types.SimpleNamespace(__version__="2026.08.08", where=lambda: "/tmp/cacert.pem")

        def fake_import(name, *args, **kwargs):
            if name == "truststore":
                raise ImportError("no truststore")
            if name == "certifi":
                return fake_certifi
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        monkeypatch.delenv("SSL_CERT_FILE", raising=False)
        monkeypatch.delenv("REQUESTS_CA_BUNDLE", raising=False)

        with pytest.raises(m_tls.InsecureTrustStoreError, match="GHSA-6q48-6c7w-9gw2"):
            m_tls.build_ssl_context(strict_certifi=True)

    def test_allows_patched_certifi_when_truststore_unavailable(self, monkeypatch, tmp_path):
        import certifi as real_certifi
        from pathlib import Path

        real_import = builtins.__import__
        ca_file = tmp_path / "cacert.pem"
        ca_file.write_bytes(Path(real_certifi.where()).read_bytes())
        fake_certifi = types.SimpleNamespace(__version__="2026.10.23", where=lambda: str(ca_file))

        def fake_import(name, *args, **kwargs):
            if name == "truststore":
                raise ImportError("no truststore")
            if name == "certifi":
                return fake_certifi
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        monkeypatch.delenv("SSL_CERT_FILE", raising=False)
        monkeypatch.delenv("REQUESTS_CA_BUNDLE", raising=False)

        ctx = m_tls.build_ssl_context(strict_certifi=True)
        assert isinstance(ctx, ssl.SSLContext)

    def test_prioritises_custom_ca_bundle_env(self, monkeypatch, tmp_path):
        import certifi as real_certifi
        from pathlib import Path

        ca_file = tmp_path / "corp.pem"
        ca_file.write_bytes(Path(real_certifi.where()).read_bytes())
        monkeypatch.setenv("REQUESTS_CA_BUNDLE", str(ca_file))
        monkeypatch.delenv("SSL_CERT_FILE", raising=False)

        calls = []
        orig = ssl.create_default_context

        def spy_create(*args, **kwargs):
            calls.append(kwargs.get("cafile"))
            return orig(*args, **kwargs)

        monkeypatch.setattr(ssl, "create_default_context", spy_create)
        ctx = m_tls.build_ssl_context()
        assert ctx is not None
        assert str(ca_file) in calls

    def test_ensure_secure_tls_blocks_vulnerable_fallback(self, monkeypatch):
        monkeypatch.delenv("SSL_CERT_FILE", raising=False)
        monkeypatch.delenv("REQUESTS_CA_BUNDLE", raising=False)
        monkeypatch.setattr(m_tls, "inject_truststore", lambda: False)
        monkeypatch.setattr(m_tls, "is_certifi_safe", lambda: (False, "2025.11.12"))

        with pytest.raises(m_tls.InsecureTrustStoreError, match="2026.10.19"):
            m_tls.ensure_secure_tls(strict_certifi=True)

    def test_assert_safe_build_tls_checks_versions(self, monkeypatch):
        monkeypatch.setattr(
            m_tls,
            "get_tls_security_report",
            lambda: {
                "active_backend": "os_truststore",
                "custom_ca_bundle": "",
                "truststore_available": True,
                "truststore_injected": True,
                "certifi_version": "2026.01.04",
                "certifi_safe": False,
                "pip_version": "26.2.1",
                "pip_safe": False,
                "warnings": ["outdated"],
            },
        )
        with pytest.raises(m_tls.InsecureTrustStoreError, match="Build TLS gate failed"):
            m_tls.assert_safe_build_tls(require_pip=True)

        monkeypatch.setattr(
            m_tls,
            "get_tls_security_report",
            lambda: {
                "active_backend": "os_truststore",
                "custom_ca_bundle": "",
                "truststore_available": True,
                "truststore_injected": True,
                "certifi_version": "2026.10.23",
                "certifi_safe": True,
                "pip_version": "26.3",
                "pip_safe": True,
                "warnings": [],
            },
        )
        report = m_tls.assert_safe_build_tls(require_pip=True)
        assert report["certifi_safe"] is True
        assert report["pip_safe"] is True

