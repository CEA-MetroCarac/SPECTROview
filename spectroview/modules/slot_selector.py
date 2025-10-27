# slot_selector.py
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QCheckBox, QScrollArea
from PySide6.QtCore import Signal

class SlotSelector(QWidget):
    selectionChanged = Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.checkbox_rows = []  # Will hold two horizontal layouts
        self.checkboxes = {}  # Map slot number -> QCheckBox
        self._build_ui()
        self.setVisible(False)

    def _build_ui(self):
        # Select All checkbox
        self.cb_select_all = QCheckBox("Select All")
        self.cb_select_all.stateChanged.connect(self._toggle_select_all)
        self.layout.addWidget(self.cb_select_all)

        # Two horizontal layouts for checkboxes
        row1 = QHBoxLayout()
        row2 = QHBoxLayout()
        self.layout.addLayout(row1)
        self.layout.addLayout(row2)
        self.checkbox_rows = [row1, row2]

        # Create checkboxes (1–25) but do not add to layout yet
        for i in range(1, 26):
            cb = QCheckBox(str(i))  # Only number as title
            cb.stateChanged.connect(self._emit_selection)
            self.checkboxes[i] = cb

    def update_slots(self, available_slots):
        """Show only available slot checkboxes and arrange in two rows."""
        self.setVisible(True)
        self.cb_select_all.setChecked(False)

        # Clear layouts first
        for row in self.checkbox_rows:
            while row.count():
                item = row.takeAt(0)
                widget = item.widget()
                if widget:
                    widget.setParent(None)

        # Add only available slots
        for idx, slot in enumerate(sorted(available_slots)):
            cb = self.checkboxes[slot]
            cb.setEnabled(True)
            cb.setChecked(True)
            # Arrange first half in row1, second half in row2
            target_row = self.checkbox_rows[0] if idx < len(available_slots)/2 else self.checkbox_rows[1]
            target_row.addWidget(cb)

        # Disable and uncheck all unavailable slots
        for i, cb in self.checkboxes.items():
            if i not in available_slots:
                cb.setEnabled(False)
                cb.setChecked(False)

        self._emit_selection()

    def _toggle_select_all(self, state):
        """Select or deselect all available slots."""
        checked = bool(state)
        for cb in self.checkboxes.values():
            if cb.isEnabled():
                cb.setChecked(checked)
        self._emit_selection()

    def _emit_selection(self):
        """Emit list of currently selected slot numbers."""
        selected = [i for i, cb in self.checkboxes.items() if cb.isChecked()]
        self.selectionChanged.emit(selected)
