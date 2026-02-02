# spectroview/view/components/v_expression_lineedit.py
"""Custom QLineEdit with smart autocomplete for mathematical expressions."""

from PySide6.QtWidgets import QLineEdit
from PySide6.QtCore import Qt


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
