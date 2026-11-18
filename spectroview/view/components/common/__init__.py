"""Shared application shell, notification, calculator, and utility widgets."""

from spectroview.view.components.common.customized_widgets import (
    CustomizedPalette,
    ExpressionLineEdit,
    NoDoubleClickZoomToolbar,
)
from spectroview.view.components.common.markdown_utils import ensure_markdown_spacing
from spectroview.view.components.common.v_about import VAboutDialog
from spectroview.view.components.common.v_menubar import THEME_OPTIONS, VMenuBar
from spectroview.view.components.common.v_notification import (
    NotificationManager,
    NotificationType,
    ToastPreset,
    VNotificationArea,
    VNotificationBadge,
    show_notification,
)
from spectroview.view.components.common.v_quick_calculators import (
    MQuickCalc,
    PenetrationDepthCalculator,
    ResizableImageLabel,
    SpotSizeCalculator,
    UnitConverterCalculator,
)
from spectroview.view.components.common.v_update_banner import VUpdateBanner

__all__ = [
    "CustomizedPalette",
    "ExpressionLineEdit",
    "MQuickCalc",
    "NoDoubleClickZoomToolbar",
    "NotificationManager",
    "NotificationType",
    "PenetrationDepthCalculator",
    "ResizableImageLabel",
    "SpotSizeCalculator",
    "THEME_OPTIONS",
    "ToastPreset",
    "UnitConverterCalculator",
    "VAboutDialog",
    "VMenuBar",
    "VNotificationArea",
    "VNotificationBadge",
    "VUpdateBanner",
    "ensure_markdown_spacing",
    "show_notification",
]
