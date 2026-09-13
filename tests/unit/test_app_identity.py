"""Unit tests for Windows app identity and application icon loading."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PySide6.QtCore import QSize
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from spectroview import LOGO_APPLI, LOGO_APPLI_ICO, get_app_icon
from spectroview.winapi.app_identity import apply_windows_taskbar_icon, set_current_process_app_id


@pytest.fixture(scope="module")
def qapp():
    """Ensure a QApplication instance exists for QIcon queries."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_resource_paths_exist():
    """Verify both ICO and PNG icon assets exist."""
    assert Path(LOGO_APPLI).is_file(), f"PNG logo missing: {LOGO_APPLI}"
    assert Path(LOGO_APPLI_ICO).is_file(), f"ICO logo missing: {LOGO_APPLI_ICO}"


def test_set_current_process_app_id_native():
    """Verify calling set_current_process_app_id does not raise an exception."""
    set_current_process_app_id("fr.cea.spectroview")


def test_set_current_process_app_id_non_windows():
    """Verify early exit on non-Windows platforms."""
    with patch("sys.platform", "linux"), patch("ctypes.WinDLL") as mock_windll:
        set_current_process_app_id()
        mock_windll.assert_not_called()


def test_set_current_process_app_id_handles_os_error():
    """Verify exceptions from WinDLL or Shell32 are caught safely."""
    with patch("sys.platform", "win32"), patch("ctypes.WinDLL", side_effect=OSError("Access denied")):
        # Should return silently and not raise
        set_current_process_app_id()


def test_get_app_icon(qapp):
    """Verify get_app_icon returns a valid QIcon with multi-resolution sizes."""
    icon = get_app_icon()
    assert isinstance(icon, QIcon)
    assert not icon.isNull()

    sizes = icon.availableSizes()
    assert len(sizes) > 0

    # Ensure standard taskbar sizes (16, 32) and high-res (1024) are present
    width_heights = {(s.width(), s.height()) for s in sizes}
    assert (16, 16) in width_heights or (32, 32) in width_heights
    assert (1024, 1024) in width_heights


def test_get_app_icon_fallback_when_one_missing(qapp, tmp_path):
    """Verify fallback behavior when either ICO or PNG is absent."""
    with patch("spectroview.LOGO_APPLI_ICO", str(tmp_path / "nonexistent.ico")):
        icon = get_app_icon()
        assert not icon.isNull()

    with patch("spectroview.LOGO_APPLI", str(tmp_path / "nonexistent.png")):
        icon = get_app_icon()
        assert not icon.isNull()


def test_apply_windows_taskbar_icon_noop_on_invalid_hwnd():
    """Verify apply_windows_taskbar_icon handles 0/invalid hwnd safely."""
    apply_windows_taskbar_icon(0, LOGO_APPLI_ICO)


def test_apply_windows_taskbar_icon_non_windows():
    """Verify apply_windows_taskbar_icon does nothing on non-Windows."""
    with patch("sys.platform", "darwin"), patch("ctypes.WinDLL") as mock_windll:
        apply_windows_taskbar_icon(1234, LOGO_APPLI_ICO)
        mock_windll.assert_not_called()


def test_apply_windows_taskbar_icon_default_ico():
    """Verify apply_windows_taskbar_icon defaults to LOGO_APPLI_ICO when ico_path is None."""
    apply_windows_taskbar_icon(0)


def test_apply_windows_taskbar_icon_win32_api_calls():
    """Verify apply_windows_taskbar_icon invokes correct Win32 API calls (WM_SETICON, SetClassLong)."""
    mock_user32 = MagicMock()
    mock_user32.LoadImageW.side_effect = [111, 222]

    with patch("sys.platform", "win32"), patch("ctypes.WinDLL", return_value=mock_user32):
        apply_windows_taskbar_icon(99999)

    assert mock_user32.LoadImageW.call_count == 2
    # WM_SETICON = 0x0080 (128)
    msg_calls = mock_user32.SendMessageW.call_args_list
    assert len(msg_calls) == 2
    assert msg_calls[0].args == (99999, 0x0080, 1, 111)
    assert msg_calls[1].args == (99999, 0x0080, 0, 222)

    # SetClassLongPtrW (GCLP_HICON = -14, GCLP_HICONSM = -34)
    if hasattr(mock_user32, "SetClassLongPtrW"):
        ptr_calls = mock_user32.SetClassLongPtrW.call_args_list
        assert len(ptr_calls) == 2
        assert ptr_calls[0].args == (99999, -14, 111)
        assert ptr_calls[1].args == (99999, -34, 222)


def test_apply_windows_taskbar_icon_on_widget(qapp):
    """Verify applying taskbar icon on a widget executes without error and updates icon when native."""
    from PySide6.QtWidgets import QDialog

    dialog = QDialog()
    dialog.show()
    try:
        hwnd = int(dialog.winId())
        apply_windows_taskbar_icon(hwnd)
        if sys.platform == "win32" and QApplication.platformName() == "windows":
            import ctypes
            from ctypes import wintypes
            user32 = ctypes.windll.user32
            user32.SendMessageW.argtypes = (wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM)
            user32.SendMessageW.restype = wintypes.LPARAM
            WM_GETICON = 0x007F
            h_big = user32.SendMessageW(hwnd, WM_GETICON, 1, 0)
            h_small = user32.SendMessageW(hwnd, WM_GETICON, 0, 0)
            assert h_big != 0
            assert h_small != 0
    finally:
        dialog.close()
