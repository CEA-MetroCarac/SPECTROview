# spectroview/view/components/customized_widgets.py

from PySide6.QtGui import QIcon, QImage, QPixmap
from PySide6.QtCore import Qt, QSize
from PySide6.QtWidgets import QComboBox, QLineEdit

import numpy as np
import matplotlib.cm as cm
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT

from spectroview import PALETTE


class NoDoubleClickZoomToolbar(NavigationToolbar2QT):
    """NavigationToolbar that ignores double-clicks in zoom mode.

    By default, matplotlib's zoom tool resets the view on a double-click.
    This subclass suppresses that behaviour so that double-clicks can be
    used exclusively for legend customisation without side-effects.
    """

    def press_zoom(self, event):
        if getattr(event, "dblclick", False):
            return  # swallow double-clicks – do not zoom
        super().press_zoom(event)


class CustomizedPalette(QComboBox):
    """Custom QComboBox to show color palette previews along with their names."""
    def __init__(self, palette_list=None, parent=None, icon_size=(99, 12)):
        super().__init__(parent)
        self.icon_width, self.icon_height = icon_size
        self.setIconSize(QSize(*icon_size))
        self.setMinimumWidth(100)

        self.palette_list = palette_list or PALETTE
        self._populate_with_previews()

    def _populate_with_previews(self):
        self.clear()
        for cmap_name in self.palette_list:
            icon = QIcon(self._create_colormap_preview(cmap_name))
            self.addItem(icon, cmap_name)

    def _create_colormap_preview(self, cmap_name):
        """Generate a horizontal gradient preview image for the colormap."""
        width, height = self.icon_width, self.icon_height
        gradient = np.linspace(0, 1, 20).reshape(1, -1)

        fig = Figure(figsize=(width / 100, height / 100), dpi=100)
        canvas = FigureCanvas(fig)
        ax = fig.add_axes([0, 0, 1, 1], frameon=False)
        ax.imshow(gradient, aspect='auto', cmap=cm.get_cmap(cmap_name))
        ax.set_axis_off()
        canvas.draw()

        image = np.array(canvas.buffer_rgba())
        qimage = QImage(image.data, image.shape[1], image.shape[0],
                        QImage.Format_RGBA8888)
        return QPixmap.fromImage(qimage)


class ExpressionLineEdit(QLineEdit):
    """Custom QLineEdit with smart autocomplete for mathematical expressions.
    
    Supports autocomplete for column names anywhere in the expression,
    not just at the beginning.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._completer = None
    
    def setCompleter(self, completer):
        """Set the completer for this line edit."""
        if self._completer:
            self._completer.activated.disconnect()
        
        self._completer = completer
        
        if not self._completer:
            return
        
        self._completer.setWidget(self)
        self._completer.activated.connect(self._insert_completion)
    
    def completer(self):
        """Return the current completer."""
        return self._completer
    
    def _insert_completion(self, completion):
        """Insert the selected completion at cursor position."""
        if not self._completer:
            return
        
        # Get the text being completed
        text = self.text()
        cursor_pos = self.cursorPosition()
        
        # Find word boundaries around cursor
        start = cursor_pos
        while start > 0 and self._is_word_char(text[start - 1]):
            start -= 1
        
        # Replace the partial word with the completion
        new_text = text[:start] + completion + text[cursor_pos:]
        self.setText(new_text)
        self.setCursorPosition(start + len(completion))
    
    def _is_word_char(self, char):
        """Check if character is part of a word (column name)."""
        # Column names can contain letters, numbers, underscores, and backticks
        return char.isalnum() or char in ('_', '`')
    
    def _get_text_under_cursor(self):
        """Get the partial word being typed at cursor position."""
        text = self.text()
        cursor_pos = self.cursorPosition()
        
        # Find start of current word
        start = cursor_pos
        while start > 0 and self._is_word_char(text[start - 1]):
            start -= 1
        
        # Extract the partial word
        return text[start:cursor_pos]
    
    def keyPressEvent(self, event):
        """Handle key press events to show completer popup."""
        if self._completer and self._completer.popup().isVisible():
            # Let completer handle these keys when popup is visible
            if event.key() in (Qt.Key_Enter, Qt.Key_Return, Qt.Key_Escape, 
                              Qt.Key_Tab, Qt.Key_Backtab):
                event.ignore()
                return
        
        # Process the key normally
        super().keyPressEvent(event)
        
        # Update completer with current word
        if self._completer:
            completion_prefix = self._get_text_under_cursor()
            
            # Only show completer if we have at least 1 character
            if len(completion_prefix) >= 1:
                if completion_prefix != self._completer.completionPrefix():
                    self._completer.setCompletionPrefix(completion_prefix)
                    popup = self._completer.popup()
                    popup.setCurrentIndex(self._completer.completionModel().index(0, 0))
                
                # Position popup at cursor
                cursor_rect = self.cursorRect()
                cursor_rect.setWidth(
                    self._completer.popup().sizeHintForColumn(0) +
                    self._completer.popup().verticalScrollBar().sizeHint().width()
                )
                self._completer.complete(cursor_rect)
            else:
                self._completer.popup().hide()
