"""Unit tests for release asset selection and updater helper generation."""

import json
from pathlib import Path

import spectroview.model.m_update_checker as update_checker
from spectroview.model.m_update_checker import (
    UpdateCheckerWorker,
    _find_wheel_asset,
    _parse_version,
    _update_script_content,
)


def test_parse_version_accepts_tags_and_non_numeric_components():
    assert _parse_version("v26.32.2") == (26, 32, 2)
    assert _parse_version("26.32.beta") == (26, 32, 0)


def test_find_wheel_asset_uses_the_release_wheel_and_digest():
    url, digest = _find_wheel_asset(
        {
            "assets": [
                {"name": "source.zip", "browser_download_url": "https://example.test/source.zip"},
                {
                    "name": "spectroview-26.32.2-py3-none-any.whl",
                    "browser_download_url": "https://example.test/spectroview.whl",
                    "digest": "sha256:" + "a" * 64,
                },
            ]
        }
    )

    assert url == "https://example.test/spectroview.whl"
    assert digest == "a" * 64


def test_find_wheel_asset_returns_empty_values_when_release_has_no_wheel():
    assert _find_wheel_asset({"assets": []}) == ("", "")


def test_update_checker_emits_the_release_wheel_url(monkeypatch, qapp):
    release = {
        "tag_name": "v26.32.2",
        "html_url": "https://example.test/releases/v26.32.2",
        "body": "Release notes",
        "assets": [
            {
                "name": "spectroview-26.32.2-py3-none-any.whl",
                "browser_download_url": "https://example.test/spectroview.whl",
                "digest": "sha256:" + "b" * 64,
            }
        ],
    }

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def read(self):
            return json.dumps(release).encode("utf-8")

    monkeypatch.setattr(update_checker, "urlopen", lambda *args, **kwargs: Response())
    monkeypatch.setattr(update_checker, "_ssl_context", lambda: None)
    worker = UpdateCheckerWorker("26.32.1")
    emitted = []
    worker.update_available.connect(lambda *args: emitted.append(args))

    worker.run()

    assert emitted == [
        (
            "v26.32.2",
            "Release notes",
            "https://example.test/releases/v26.32.2",
            "https://example.test/spectroview.whl",
            "b" * 64,
        )
    ]


def test_update_helper_installs_the_wheel_and_relaunches_from_temp_directory():
    content = _update_script_content(
        Path("C:/Temp/spectroview.whl"),
        Path("C:/Python/python.exe"),
        Path("C:/Python/pythonw.exe"),
    )

    assert '"-m", "pip", "install", "--upgrade"' in content
    assert '"-m", "spectroview.main"' in content
    assert "cwd=tempfile.gettempdir()" in content
    assert 'restart_environment.pop("PYTHONPATH", None)' in content
