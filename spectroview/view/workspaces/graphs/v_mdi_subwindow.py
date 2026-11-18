"""Custom MDI subwindow widget for the Graphs workspace."""

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QIcon, QPalette, QPixmap
from PySide6.QtWidgets import QMdiSubWindow


class MdiSubWindow(QMdiSubWindow):
    """Custom MDI subwindow."""

    closed = Signal(int)

    def __init__(self, graph_id, mdi_area, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.graph_id = graph_id
        self.mdi_area = mdi_area  # Reference to the parent QMdiArea
        # Drop the default Qt window icon from the title bar. A null icon is
        # not enough -- Fusion then falls back to the generic app logo -- so
        # set a fully transparent pixmap, which paints nothing.
        _blank = QPixmap(16, 16)
        _blank.fill(Qt.transparent)
        self.setWindowIcon(QIcon(_blank))
        # Rubber-band resize/move: Qt drags a lightweight outline and resizes
        # the (expensive-to-redraw) matplotlib canvas once on mouse release,
        # instead of re-rendering the figure on every intermediate size.
        self.setOption(QMdiSubWindow.SubWindowOption.RubberBandResize, True)
        self.setOption(QMdiSubWindow.SubWindowOption.RubberBandMove, True)
        self._fixing_palette = False
        self._fix_title_shadow()

    def _fix_title_shadow(self):
        """Neutralise the Fusion title-bar text shadow.

        Fusion computes the shadow colour as ``palette.text().color().lighter(120)``.
        By setting ``Active/Text = highlight.darker(120)``, the shadow becomes
        identical to the title-bar background (which is the Highlight colour)
        and is therefore invisible. The inner widget's palette is reset to prevent inheritance.
        """
        if self._fixing_palette:
            return
        self._fixing_palette = True
        try:
            pal = self.palette()
            highlight = pal.color(
                QPalette.ColorGroup.Active, QPalette.ColorRole.Highlight
            )
            target_text = highlight.darker(120)

            if pal.color(QPalette.ColorGroup.Active, QPalette.ColorRole.Text) != target_text:
                pal.setColor(
                    QPalette.ColorGroup.Active,
                    QPalette.ColorRole.Text,
                    target_text,
                )
                self.setPalette(pal)

                # Prevent inheritance into the graph/content
                child = self.widget()
                if child:
                    from PySide6.QtWidgets import QApplication
                    child.setPalette(QApplication.palette())
        finally:
            self._fixing_palette = False

    def changeEvent(self, event):
        """Re-apply the shadow fix whenever the palette changes (theme switch)."""
        super().changeEvent(event)
        from PySide6.QtCore import QEvent
        if event.type() == QEvent.Type.PaletteChange:
            self._fix_title_shadow()

    def setWidget(self, widget):
        """Override to prevent the title shadow palette fix from inheriting."""
        super().setWidget(widget)
        if widget:
            from PySide6.QtWidgets import QApplication
            widget.setPalette(QApplication.palette())

    def closeEvent(self, event):
        """Override close event."""
        self.mdi_area.clearFocus()
        self.mdi_area.setActiveSubWindow(None)
        self.closed.emit(self.graph_id)
        super().closeEvent(event)

    def focusInEvent(self, event):
        """Override focus event."""
        if not self.mdi_area.activeSubWindow():
            self.mdi_area.setActiveSubWindow(None)
        super().focusInEvent(event)
