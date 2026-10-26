"""Unit tests for VAboutDialog and BUILD_NUMBER."""
from spectroview import VERSION, BUILD_NUMBER
from spectroview.view.components.v_about import VAboutDialog
from PySide6.QtWidgets import QTextBrowser


class TestAboutDialog:
    def test_build_number_exported(self):
        """BUILD_NUMBER must be a non-empty string."""
        assert isinstance(BUILD_NUMBER, str)
        assert len(BUILD_NUMBER) > 0

    def test_v_about_dialog_contains_version_and_build(self, qapp):
        """VAboutDialog must show VERSION and BUILD_NUMBER."""
        dialog = VAboutDialog()
        browsers = dialog.findChildren(QTextBrowser)
        assert len(browsers) == 1
        html = browsers[0].toHtml()

        assert f"Version {VERSION}" in html
        assert f"Build: {BUILD_NUMBER}" in html
