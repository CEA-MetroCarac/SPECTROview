"""
Tests for spectroview/ai_agent/m_llm_client.py — corporate-CA / TLS support
and the error-message formatter.

These cover the plumbing that makes an internal HTTPS endpoint (whose CA isn't
in certifi) usable, and that surfaces the real cause instead of the SDK's
opaque "Connection error." No network or optional SDK is required — the helpers
are pure Python.
"""
import spectroview.ai_agent.m_llm_client as m


class TestFormatException:
    def test_plain_exception_returns_its_message(self):
        out = m._format_exception(ValueError("bad value"))
        assert out == "ValueError: bad value"

    def test_walks_cause_chain_to_the_root(self):
        # Mirrors the real openai failure: APIConnectionError("Connection error.")
        # caused by an SSL certificate-verification failure.
        try:
            try:
                raise OSError(
                    "[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed"
                )
            except OSError as inner:
                raise ConnectionError("Connection error.") from inner
        except ConnectionError as exc:
            out = m._format_exception(exc)

        assert "Connection error." in out
        # The actionable root cause must be surfaced, not hidden.
        assert "CERTIFICATE_VERIFY_FAILED" in out
        assert "Caused by:" in out

    def test_cycle_is_handled_without_hanging(self):
        # A self-referential context must not loop forever.
        exc = RuntimeError("loop")
        exc.__context__ = exc
        assert m._format_exception(exc) == "RuntimeError: loop"


class TestMakeHttpClient:
    def test_returns_none_without_ca_bundle_env(self, monkeypatch):
        # No SSL_CERT_FILE / REQUESTS_CA_BUNDLE configured → default verification.
        monkeypatch.setattr(m, "_CA_BUNDLE_ENV", "")
        assert m._make_http_client() is None

    def test_returns_none_for_missing_bundle_file(self, monkeypatch):
        monkeypatch.setattr(m, "_CA_BUNDLE_ENV", "/no/such/ca-bundle.pem")
        assert m._make_http_client() is None
