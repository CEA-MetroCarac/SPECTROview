"""Behaviour tests for the automatic-update banner action."""

from spectroview.view.components.v_update_banner import VUpdateBanner


def _banner(on_update, wheel_url="https://example.test/spectroview-26.32.2-py3-none-any.whl"):
    return VUpdateBanner(
        tag="v26.32.2",
        html_url="https://example.test/releases/v26.32.2",
        on_skip=lambda tag: None,
        on_dismiss=lambda: None,
        on_update=on_update,
        wheel_url=wheel_url,
        wheel_sha256="a" * 64,
    )


def test_update_button_forwards_the_exact_release_asset(qapp):
    requested = []
    banner = _banner(lambda *args: requested.append(args))

    banner.btn_update.click()

    assert requested == [
        (
            "v26.32.2",
            "https://example.test/spectroview-26.32.2-py3-none-any.whl",
            "a" * 64,
        )
    ]


def test_update_button_is_disabled_when_the_release_has_no_wheel(qapp):
    banner = _banner(lambda *args: None, wheel_url="")

    assert banner.btn_update.isEnabled() is False


def test_update_banner_supports_custom_action_label_and_version_transition(qapp):
    from PySide6.QtWidgets import QLabel

    banner = VUpdateBanner(
        tag="v26.47.0",
        html_url="https://example.test/releases/v26.47.0",
        on_skip=lambda tag: None,
        on_dismiss=lambda: None,
        on_update=lambda *args: None,
        wheel_url="https://example.test/SPECTROview_26.47.0_macOS_arm64.dmg",
        action_label="Download Update",
        action_tooltip="Download verified DMG.",
        current_version="26.46.1",
    )

    labels = [lbl.text() for lbl in banner.findChildren(QLabel)]
    assert any("26.46.1 → 26.47.0" in t for t in labels)
    assert banner.btn_update.text() == "Download Update"
    assert banner.btn_update.isEnabled() is True
