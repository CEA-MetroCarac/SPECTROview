"""Floating metadata tooltip popup and hover event manager for list widgets."""

from __future__ import annotations

from PySide6.QtCore import QEvent, QObject, QPoint, Qt, QTimer, Slot
from PySide6.QtGui import QCursor, QPalette
from PySide6.QtWidgets import QApplication, QFrame, QLabel, QVBoxLayout

from spectroview.viewmodel.common.spectral_utils import format_metadata_tooltip


class MetadataTooltipPopup(QFrame):
    """Floating tooltip frame with theme-adaptive styling that stays visible as long as needed."""

    def __init__(self, parent=None):
        super().__init__(
            parent,
            Qt.ToolTip | Qt.FramelessWindowHint | Qt.WindowDoesNotAcceptFocus,
        )
        self.setObjectName("MetadataTooltipPopup")
        self.setAttribute(Qt.WA_ShowWithoutActivating, True)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(0)

        self.label = QLabel(self)
        self.label.setTextFormat(Qt.RichText)
        self.label.setWordWrap(False)
        layout.addWidget(self.label)

    def show_tip(self, html: str, global_pos: QPoint):
        """Update content and position near cursor, constrained to screen bounds."""
        app = QApplication.instance()
        if app and not app.styleSheet():
            pal = app.palette()
            bg = pal.color(QPalette.ToolTipBase).name()
            fg = pal.color(QPalette.ToolTipText).name()
            border = pal.color(QPalette.Mid).name()
            self.setStyleSheet(f"""
                QFrame#MetadataTooltipPopup {{
                    background-color: {bg};
                    color: {fg};
                    border: 1px solid {border};
                    border-radius: 6px;
                }}
                QLabel {{
                    background: transparent;
                    color: {fg};
                    font-size: 11px;
                    line-height: 1.4;
                }}
            """)
        elif self.styleSheet():
            self.setStyleSheet("")

        self.label.setText(html)
        self.adjustSize()

        pos = global_pos + QPoint(16, 16)
        screen = self.screen()
        if screen:
            sg = screen.availableGeometry()
            if pos.x() + self.width() > sg.right():
                pos.setX(max(sg.left(), global_pos.x() - self.width() - 8))
            if pos.y() + self.height() > sg.bottom():
                pos.setY(max(sg.top(), global_pos.y() - self.height() - 8))

        self.move(pos)
        self.show()
        self.raise_()


class MetadataTooltipManager(QObject):
    """Manages metadata tooltips for QListWidget items.

    Requirements:
    - Display tooltip when user hovers an item (with initial show_delay_ms delay, native-like).
    - Can optionally require Ctrl key if require_ctrl=True.
    - Once showing, seamlessly and immediately updates without delay when moving between items.
    - Stays displayed continuously as long as the mouse cursor is over items.
    - Does NOT automatically disappear after 2 seconds (independent of OS tooltip timeout).
    - Disappears when mouse leaves items or widget.
    - Disappears immediately on click, drag, scroll, or window hide so it never interferes.
    """

    def __init__(
        self,
        list_widget,
        metadata_provider=None,
        require_ctrl: bool = False,
        show_delay_ms: int = 1000,
    ):
        super().__init__(list_widget)
        self.list_widget = list_widget
        self.metadata_provider = metadata_provider
        self.require_ctrl = require_ctrl
        self.show_delay_ms = show_delay_ms
        self._hovered_item = None
        self._pending_item = None
        self._pending_global_pos = None
        self._is_showing = False
        self._app_filtered = False

        self._popup = MetadataTooltipPopup()

        self._delay_timer = QTimer(self)
        self._delay_timer.setSingleShot(True)
        self._delay_timer.timeout.connect(self._on_delay_timeout)

        self.list_widget.setMouseTracking(True)
        viewport = self.list_widget.viewport()
        self._viewport = viewport
        if viewport:
            viewport.setMouseTracking(True)
            viewport.installEventFilter(self)
        self.list_widget.installEventFilter(self)

        app = QApplication.instance()
        if app:
            app.installEventFilter(self)
            self._app_filtered = True

        self.list_widget.destroyed.connect(self._on_widget_destroyed)

    def set_metadata_provider(self, provider):
        """Update the metadata provider callable."""
        self.metadata_provider = provider

    def set_delay_ms(self, delay: int):
        """Set initial hover delay in milliseconds before showing tooltip."""
        self.show_delay_ms = max(0, delay)

    def _cancel_delay(self):
        """Cancel pending hover delay timer and reset pending tracking."""
        try:
            if (
                hasattr(self, "_delay_timer")
                and self._delay_timer is not None
                and self._delay_timer.isActive()
            ):
                self._delay_timer.stop()
        except RuntimeError:
            pass
        self._pending_item = None
        self._pending_global_pos = None

    def _on_delay_timeout(self):
        """Triggered when initial hover delay expires."""
        item = self._pending_item
        gpos = self._pending_global_pos
        self._pending_item = None
        self._pending_global_pos = None

        if not item or item.flags() == Qt.NoItemFlags or not item.text():
            return

        if self.require_ctrl and not self._is_ctrl_held():
            return

        try:
            lw = self.list_widget
            if not lw or not lw.isVisible():
                return
        except (RuntimeError, AttributeError):
            return

        if gpos is None:
            gpos = QCursor.pos()

        self._show_tooltip_for_item(item, gpos)

    def _remove_app_filter(self):
        """Cleanly remove application event filter."""
        if getattr(self, "_app_filtered", False):
            app = QApplication.instance()
            if app:
                try:
                    app.removeEventFilter(self)
                except Exception:
                    pass
            self._app_filtered = False

    def _disconnect_list_widget(self):
        """Disconnect destroyed signal to avoid dead-slot invocations during GC."""
        try:
            if hasattr(self, "list_widget") and self.list_widget is not None:
                self.list_widget.destroyed.disconnect(self._on_widget_destroyed)
        except Exception:
            pass

    @Slot(object)
    def _on_widget_destroyed(self, obj=None):
        """Cleanly remove application event filter and popup when widget is destroyed."""
        self._disconnect_list_widget()
        self._cancel_delay()
        self._remove_app_filter()
        popup = getattr(self, "_popup", None)
        if popup:
            try:
                popup.hide()
                popup.deleteLater()
            except Exception:
                pass
            self._popup = None
        self.list_widget = None

    def __del__(self):
        self._disconnect_list_widget()
        self._cancel_delay()
        self._remove_app_filter()
        popup = getattr(self, "_popup", None)
        if popup:
            try:
                popup.hide()
                popup.deleteLater()
            except Exception:
                pass
            self._popup = None

    @staticmethod
    def _is_ctrl_held() -> bool:
        """Check if Ctrl (or Cmd on macOS) modifier is held down."""
        mods = QApplication.keyboardModifiers()
        return bool(mods & (Qt.ControlModifier | Qt.MetaModifier))

    def _hide_tooltip(self):
        """Dismiss active tooltip and reset state."""
        self._cancel_delay()
        if self._popup:
            self._popup.hide()
        self._is_showing = False
        self._hovered_item = None

    def _fetch_metadata(self, item):
        """Retrieve metadata for the given item."""
        if self.metadata_provider:
            try:
                return self.metadata_provider(item)
            except TypeError:
                return self.metadata_provider(item.text())
        return item.data(Qt.UserRole + 2) or {}

    def _show_tooltip_for_item(self, item, global_pos):
        """Render and display tooltip for a valid list item."""
        if not item or item.flags() == Qt.NoItemFlags or not item.text():
            self._hide_tooltip()
            return

        if not self._popup:
            return

        metadata = self._fetch_metadata(item)
        tooltip_html = format_metadata_tooltip(metadata)

        self._popup.show_tip(tooltip_html, global_pos)
        self._hovered_item = item
        self._is_showing = True

    def eventFilter(self, watched, event):
        """Handle mouse movement, leave, and interaction events."""
        lw = getattr(self, "list_widget", None)
        vp = getattr(self, "_viewport", None)
        if watched != lw and watched != vp:
            if (
                not getattr(self, "_is_showing", False)
                and getattr(self, "_pending_item", None) is None
                and not getattr(self, "require_ctrl", False)
            ):
                return False

        etype = event.type()
        if etype == QEvent.Destroy and watched == lw:
            self._on_widget_destroyed()
            return False

        if etype not in (
            QEvent.MouseMove,
            QEvent.ToolTip,
            QEvent.Leave,
            QEvent.MouseButtonPress,
            QEvent.DragEnter,
            QEvent.Wheel,
            QEvent.KeyRelease,
            QEvent.KeyPress,
            QEvent.Hide,
            QEvent.WindowDeactivate,
        ):
            return False

        try:
            lw = self.list_widget
            if lw is None:
                self._remove_app_filter()
                return False
            viewport = lw.viewport()
            if viewport is None:
                self._remove_app_filter()
                return False
        except (RuntimeError, AttributeError):
            self._remove_app_filter()
            return False

        if etype == QEvent.MouseMove and watched == viewport:
            if self.require_ctrl and not self._is_ctrl_held():
                self._hide_tooltip()
                return False

            pos = event.position().toPoint() if hasattr(event, "position") else event.pos()
            item = lw.itemAt(pos)
            gpos = (
                event.globalPosition().toPoint()
                if hasattr(event, "globalPosition")
                else event.globalPos()
            )

            if not item or item.flags() == Qt.NoItemFlags or not item.text():
                self._hide_tooltip()
                return False

            if self._is_showing:
                if item != self._hovered_item:
                    self._cancel_delay()
                    self._show_tooltip_for_item(item, gpos)
                elif self._popup:
                    pos_pop = gpos + QPoint(16, 16)
                    screen = self._popup.screen()
                    if screen:
                        sg = screen.availableGeometry()
                        if pos_pop.x() + self._popup.width() > sg.right():
                            pos_pop.setX(max(sg.left(), gpos.x() - self._popup.width() - 8))
                        if pos_pop.y() + self._popup.height() > sg.bottom():
                            pos_pop.setY(max(sg.top(), gpos.y() - self._popup.height() - 8))
                    self._popup.move(pos_pop)
            else:
                if self.show_delay_ms <= 0:
                    self._show_tooltip_for_item(item, gpos)
                else:
                    if item != self._pending_item:
                        self._pending_item = item
                        self._pending_global_pos = gpos
                        self._delay_timer.start(self.show_delay_ms)
                    else:
                        self._pending_global_pos = gpos
                        if not self._delay_timer.isActive():
                            self._delay_timer.start(self.show_delay_ms)
            return False

        if etype == QEvent.ToolTip and watched in (viewport, lw):
            return True

        if etype == QEvent.Leave and watched == viewport:
            self._hide_tooltip()
            return False

        if etype in (QEvent.MouseButtonPress, QEvent.DragEnter, QEvent.Wheel):
            self._hide_tooltip()
            return False

        if etype == QEvent.KeyRelease:
            if self.require_ctrl and event.key() in (Qt.Key_Control, Qt.Key_Meta):
                self._hide_tooltip()
            return False

        if etype == QEvent.KeyPress:
            if event.key() in (Qt.Key_Control, Qt.Key_Meta):
                try:
                    if viewport and viewport.isVisible():
                        gpos = QCursor.pos()
                        vpos = viewport.mapFromGlobal(gpos)
                        if viewport.rect().contains(vpos):
                            item = lw.itemAt(vpos)
                            if item and item.flags() != Qt.NoItemFlags and item.text():
                                self._cancel_delay()
                                self._show_tooltip_for_item(item, gpos)
                except (RuntimeError, AttributeError):
                    pass
            return False

        if etype == QEvent.Hide and watched in (lw, viewport, lw.window()):
            self._hide_tooltip()
            return False

        if etype == QEvent.WindowDeactivate and watched == lw.window():
            self._hide_tooltip()
            return False

        return False
