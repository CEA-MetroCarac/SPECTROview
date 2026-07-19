"""Shared spinbox widgets for the Customize Graph dialog's "optional
override" fields (axis limits, per-series style overrides): a sentinel
value marks "unset", displayed as a grayed placeholder word instead of a
blank box or the raw sentinel number.

Plain QSpinBox/QDoubleSpinBox with setSpecialValueText() alone still has a
usability bug: clicking the up/down arrow from the sentinel steps *from*
that huge number (e.g. -999999 -> -999998), which briefly shows a nonsense
value instead of landing on something sensible. PlaceholderSpinBox/
PlaceholderDoubleSpinBox fix that by overriding stepBy() so the first
arrow-click from "unset" jumps straight to a configurable start value.
"""
from PySide6.QtWidgets import QDoubleSpinBox, QSpinBox


class _PlaceholderMixin:
    def _init_placeholder(self, unset_value, start_value, placeholder_text="default"):
        self._unset_value = unset_value
        self._start_value = start_value
        self.setSpecialValueText(placeholder_text)
        self.valueChanged.connect(self._update_placeholder_style)
        self._update_placeholder_style()

    def set_start_value(self, value):
        """Update the value the first arrow-click from "unset" jumps to
        (e.g. refreshed per-graph to the current rendered axis limit)."""
        self._start_value = value

    def set_placeholder_value(self, value):
        """Show `value` -- the real current/effective value (still grayed
        via _update_placeholder_style) -- as the sentinel's placeholder
        text, instead of a generic "default" word that doesn't tell the
        user what will actually be used. Formatted with this spinbox's own
        configured decimals() (QDoubleSpinBox only; QSpinBox is always a
        plain integer)."""
        if hasattr(self, 'decimals'):
            self.setSpecialValueText(f"{value:.{self.decimals()}f}")
        else:
            self.setSpecialValueText(str(int(round(value))))

    def _update_placeholder_style(self, *_):
        gray = self.value() == self._unset_value
        le = self.lineEdit()
        
        # Use QPalette instead of setStyleSheet to avoid breaking
        # native widget margins/height on macOS, which causes text clipping.
        if not hasattr(self, '_default_palette'):
            self._default_palette = le.palette()
            
        if gray:
            from PySide6.QtGui import QPalette
            from PySide6.QtCore import Qt
            palette = le.palette()
            palette.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.gray)
            le.setPalette(palette)
        else:
            le.setPalette(self._default_palette)

    def stepEnabled(self):
        enabled = super().stepEnabled()
        if self.value() == self._unset_value:
            from PySide6.QtWidgets import QAbstractSpinBox
            enabled |= QAbstractSpinBox.StepEnabledFlag.StepDownEnabled
        return enabled

    def stepBy(self, steps):
        if self.value() == self._unset_value and steps != 0:
            self.setValue(self._start_value)
        else:
            super().stepBy(steps)


class PlaceholderDoubleSpinBox(_PlaceholderMixin, QDoubleSpinBox):
    """Optional-override QDoubleSpinBox: shows a grayed placeholder word at
    its sentinel (range-minimum) value instead of a blank box."""

    def __init__(self, unset_value, start_value, placeholder_text="default", parent=None):
        super().__init__(parent)
        self._init_placeholder(unset_value, start_value, placeholder_text)


class PlaceholderSpinBox(_PlaceholderMixin, QSpinBox):
    """Integer counterpart of PlaceholderDoubleSpinBox."""

    def __init__(self, unset_value, start_value, placeholder_text="default", parent=None):
        super().__init__(parent)
        self._init_placeholder(unset_value, start_value, placeholder_text)
