"""Windows taskbar identity for a SPECTROview process launched by a pip entry point."""

from __future__ import annotations

import ctypes
import sys
from pathlib import Path


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


def apply_windows_taskbar_icon(hwnd: int, ico_path: str | Path | None = None) -> None:
    """Explicitly send WM_SETICON and update window class icon with native HICON handles.

    This ensures Windows taskbar and Alt-Tab display the native SPECTROview icon rather than
    the generic Python launcher logo when running in development or via python -m spectroview.
    """
    if sys.platform != "win32" or not hwnd:
        return
    try:
        from ctypes import wintypes

        if ico_path is None:
            from spectroview import LOGO_APPLI_ICO

            ico_path = LOGO_APPLI_ICO
        if not ico_path or not Path(ico_path).exists():
            return

        user32 = ctypes.WinDLL("user32", use_last_error=True)
        user32.LoadImageW.argtypes = (
            wintypes.HINSTANCE,
            wintypes.LPCWSTR,
            wintypes.UINT,
            ctypes.c_int,
            ctypes.c_int,
            wintypes.UINT,
        )
        user32.LoadImageW.restype = wintypes.HANDLE

        user32.SendMessageW.argtypes = (
            wintypes.HWND,
            wintypes.UINT,
            wintypes.WPARAM,
            wintypes.LPARAM,
        )
        user32.SendMessageW.restype = wintypes.LPARAM

        IMAGE_ICON = 1
        LR_LOADFROMFILE = 0x00000010
        LR_DEFAULTSIZE = 0x00000040

        ico_str = str(ico_path)
        # Load large icon (system default for taskbar/Alt-Tab) and small icon (16x16 for titlebar)
        h_big = user32.LoadImageW(None, ico_str, IMAGE_ICON, 0, 0, LR_LOADFROMFILE | LR_DEFAULTSIZE)
        h_small = user32.LoadImageW(None, ico_str, IMAGE_ICON, 16, 16, LR_LOADFROMFILE)

        WM_SETICON = 0x0080
        ICON_SMALL = 0
        ICON_BIG = 1
        GCLP_HICON = -14
        GCLP_HICONSM = -34

        if h_big:
            user32.SendMessageW(hwnd, WM_SETICON, ICON_BIG, h_big)
            try:
                if ctypes.sizeof(ctypes.c_void_p) == 8:
                    user32.SetClassLongPtrW.argtypes = (wintypes.HWND, ctypes.c_int, ctypes.c_void_p)
                    user32.SetClassLongPtrW.restype = ctypes.c_void_p
                    user32.SetClassLongPtrW(hwnd, GCLP_HICON, h_big)
                else:
                    user32.SetClassLongW.argtypes = (wintypes.HWND, ctypes.c_int, ctypes.c_long)
                    user32.SetClassLongW.restype = ctypes.c_long
                    user32.SetClassLongW(hwnd, GCLP_HICON, h_big)
            except Exception:
                pass

        if h_small:
            user32.SendMessageW(hwnd, WM_SETICON, ICON_SMALL, h_small)
            try:
                if ctypes.sizeof(ctypes.c_void_p) == 8:
                    user32.SetClassLongPtrW.argtypes = (wintypes.HWND, ctypes.c_int, ctypes.c_void_p)
                    user32.SetClassLongPtrW.restype = ctypes.c_void_p
                    user32.SetClassLongPtrW(hwnd, GCLP_HICONSM, h_small)
                else:
                    user32.SetClassLongW.argtypes = (wintypes.HWND, ctypes.c_int, ctypes.c_long)
                    user32.SetClassLongW.restype = ctypes.c_long
                    user32.SetClassLongW(hwnd, GCLP_HICONSM, h_small)
            except Exception:
                pass
    except Exception:
        pass
