"""Entry point for ``python -m spectroview``."""

from __future__ import annotations

# Set Windows taskbar identity BEFORE importing spectroview.main,
# which triggers PySide6 imports at module level.
from spectroview.winapi import set_current_process_app_id
set_current_process_app_id()

from spectroview.main import launcher

if __name__ == "__main__":
    launcher()
