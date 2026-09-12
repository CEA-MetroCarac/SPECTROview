"""Windows taskbar identity for a SPECTROview process launched by a pip entry point."""

from __future__ import annotations

import ctypes
import sys


def set_current_process_app_id(app_id: str = "fr.cea.spectroview") -> None:
    """Give taskbar windows a stable SPECTROview identity instead of the Python launcher identity.

    The call must run before the first top-level Qt window is created. It is deliberately a
    best-effort no-op off Windows and on older/restricted shells.
    """
    if sys.platform != "win32":
        return
    try:
        shell32 = ctypes.WinDLL("shell32", use_last_error=True)
        function = shell32.SetCurrentProcessExplicitAppUserModelID
        function.argtypes = (ctypes.c_wchar_p,)
        function.restype = ctypes.c_long
        function(app_id)
    except (AttributeError, OSError):
        return


def apply_windows_taskbar_icon(hwnd: int, ico_path: str) -> None:
    """Explicitly send WM_SETICON with native HICON handles to ensure Windows taskbar displays the icon."""
    if sys.platform != "win32" or not hwnd:
        return
    try:
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        IMAGE_ICON = 1
        LR_LOADFROMFILE = 0x00000010
        LR_DEFAULTSIZE = 0x00000040

        # Load native large icon (for taskbar and Alt-Tab) and small icon (for titlebar)
        h_big = user32.LoadImageW(None, str(ico_path), IMAGE_ICON, 0, 0, LR_LOADFROMFILE | LR_DEFAULTSIZE)
        h_small = user32.LoadImageW(None, str(ico_path), IMAGE_ICON, 16, 16, LR_LOADFROMFILE)

        WM_SETICON = 0x007F
        ICON_SMALL = 0
        ICON_BIG = 1

        if h_big:
            user32.SendMessageW(hwnd, WM_SETICON, ICON_BIG, h_big)
        if h_small:
            user32.SendMessageW(hwnd, WM_SETICON, ICON_SMALL, h_small)
    except Exception:
        pass
