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
