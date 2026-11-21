"""Tests for localized online documentation URL resolution."""

import pytest

from spectroview import DOCS_BASE_URL, docs_url


@pytest.mark.parametrize(
    ("lang", "path", "expected"),
    [
        ("en", "", DOCS_BASE_URL),
        ("en", "user_manual/", f"{DOCS_BASE_URL}user_manual/"),
        ("fr", "", f"{DOCS_BASE_URL}fr/"),
        ("fr", "user_manual/", f"{DOCS_BASE_URL}fr/user_manual/"),
        ("de", "", f"{DOCS_BASE_URL}de/"),
        ("de", "user_manual/", f"{DOCS_BASE_URL}de/user_manual/"),
        ("ru", "", f"{DOCS_BASE_URL}ru/"),
        ("ru", "user_manual/", f"{DOCS_BASE_URL}ru/user_manual/"),
        ("zh", "", f"{DOCS_BASE_URL}zh/"),
        ("zh", "user_manual/", f"{DOCS_BASE_URL}zh/user_manual/"),
        ("es", "user_manual/", f"{DOCS_BASE_URL}user_manual/"),
        ("unknown", "", DOCS_BASE_URL),
    ],
)
def test_docs_url(lang: str, path: str, expected: str) -> None:
    assert docs_url(path, lang) == expected
