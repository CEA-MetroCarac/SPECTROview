# view/components/v_dataframe_table.py

import numpy as np
import pandas as pd
from PySide6.QtWidgets import (
    QVBoxLayout, QTableView, QWidget, QApplication, QHeaderView
)
from PySide6.QtGui import QAction, QColor
from PySide6.QtCore import Qt, QAbstractTableModel, QModelIndex


class DataFrameTableModel(QAbstractTableModel):
    """Virtualized Qt table model backed directly by a pandas DataFrame.
    
    Avoids creating Qt item objects per cell, enabling instant display of
    tens of thousands of rows with minimal memory overhead.
    """

    DEFAULT_PALETTE = ['#bda16d', '#a27ba0', '#cb5b12', '#23993b', '#008281', '#147ce4']

    def __init__(self, df: pd.DataFrame = None, fill_colors: bool = True, palette: list[str] = None):
        super().__init__()
        self._palette = palette or self.DEFAULT_PALETTE
        self._fill_colors = fill_colors
        self._df = pd.DataFrame()
        self._raw_cols: list[np.ndarray] = []
        self._col_colors: list[QColor] = []
        self.set_data(df)

    def set_data(self, df: pd.DataFrame | None):
        """Update model data and notify views with a model reset."""
        self.beginResetModel()
        self._df = df if (df is not None and not df.empty) else pd.DataFrame()
        if not self._df.empty:
            self._raw_cols = [self._df[c].to_numpy() for c in self._df.columns]
            self._col_colors = self._compute_column_colors(self._df.columns)
        else:
            self._raw_cols = []
            self._col_colors = []
        self.endResetModel()

    def _compute_column_colors(self, columns) -> list[QColor]:
        prefix_colors = {}
        col_colors = []
        for col_name in columns:
            prefix = col_name.split("_")[0] if "_" in col_name else col_name
            if prefix not in prefix_colors:
                color_index = len(prefix_colors) % len(self._palette)
                prefix_colors[prefix] = QColor(self._palette[color_index])
            col_colors.append(prefix_colors[prefix])
        return col_colors

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(self._df)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(self._df.columns)

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole):
        if not index.isValid():
            return None

        row = index.row()
        col = index.column()

        if role == Qt.DisplayRole:
            if col < len(self._raw_cols):
                val = self._raw_cols[col][row]
                if pd.isna(val):
                    return ""
                return str(val)
            return ""

        elif role == Qt.BackgroundRole and self._fill_colors:
            if col < len(self._col_colors):
                return self._col_colors[col]

        elif role == Qt.TextAlignmentRole:
            return Qt.AlignLeft | Qt.AlignVCenter

        return None

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.DisplayRole):
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal and section < len(self._df.columns):
                return str(self._df.columns[section])
            elif orientation == Qt.Vertical:
                return str(section + 1)
        return None

    def sort(self, column: int, order: Qt.SortOrder = Qt.AscendingOrder):
        """Sort underlying DataFrame in-place and refresh view."""
        if self._df.empty or column < 0 or column >= len(self._df.columns):
            return
        self.layoutAboutToBeChanged.emit()
        col_name = self._df.columns[column]
        ascending = (order == Qt.AscendingOrder)
        self._df = self._df.sort_values(by=col_name, ascending=ascending, kind='quicksort')
        self._raw_cols = [self._df[c].to_numpy() for c in self._df.columns]
        self.layoutChanged.emit()


class VDataframeTable(QWidget):
    """View component to display a pandas DataFrame in a virtualized QTableView with color-coded columns.
    
    Features:
    - Displays DataFrame using a virtual QAbstractTableModel for O(1) performance
    - Supports fast sorting on click
    - Color-codes columns by parameter prefix
    - Fast copy to clipboard (Ctrl+C or context menu)
    - Fast smart auto-resizing of column widths
    """

    def __init__(self, layout=None):
        """Initialize the DataFrame table widget.
        
        Args:
            layout: Optional external QVBoxLayout to add this widget to
        """
        super().__init__()
        self.external_layout = layout
        self.model: DataFrameTableModel | None = None
        self._init_ui()

    def _init_ui(self):
        """Initialize the user interface."""
        if self.external_layout:
            while self.external_layout.count():
                item = self.external_layout.takeAt(0)
                widget = item.widget()
                if widget:
                    widget.deleteLater()

        # Set the internal layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        # Create QTableView
        self.table_view = QTableView()
        self.table_view.setSortingEnabled(True)
        self.table_view.setAlternatingRowColors(False)
        self.table_view.setSelectionBehavior(QTableView.SelectItems)
        self.table_view.setSelectionMode(QTableView.ExtendedSelection)
        self.table_view.horizontalHeader().setStretchLastSection(False)
        self.table_view.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        layout.addWidget(self.table_view)

        # Alias for backward compatibility
        self.table_widget = self.table_view

        # Enable copy action via context menu
        self.table_view.setContextMenuPolicy(Qt.ActionsContextMenu)
        copy_action = QAction("Copy", self)
        copy_action.triggered.connect(self._copy_data)
        self.table_view.addAction(copy_action)

        # Add this widget to the external layout if provided
        if self.external_layout:
            self.external_layout.addWidget(self)

    def show(self, df: pd.DataFrame, fill_colors: bool = True):
        """Populate the table with data from a pandas DataFrame.
        
        Args:
            df: pandas DataFrame to display
            fill_colors: If True, apply color-coding to columns by prefix
        """
        if df is not None and not df.empty:
            self.model = DataFrameTableModel(df, fill_colors=fill_colors)
            self.table_view.setModel(self.model)
            self._resize_columns_smart(df)
        else:
            self.clear()

    def _resize_columns_smart(self, df: pd.DataFrame):
        """Quickly compute sensible column widths based on header and sample data."""
        fm = self.table_view.fontMetrics()
        sample_df = df.head(30)  # check at most first 30 rows
        for col_idx, col_name in enumerate(df.columns):
            w = fm.horizontalAdvance(str(col_name)) + 28
            for val in sample_df[col_name]:
                if pd.notna(val):
                    val_str = str(val)
                    if len(val_str) < 30:
                        w = max(w, fm.horizontalAdvance(val_str) + 16)
            self.table_view.setColumnWidth(col_idx, max(60, min(w, 240)))

    def _get_column_colors(self, columns):
        """Generate color for each column based on parameter prefix (legacy helper)."""
        palette = DataFrameTableModel.DEFAULT_PALETTE
        prefix_colors = {}
        column_colors = []
        for col_name in columns:
            prefix = col_name.split("_")[0] if "_" in col_name else col_name
            if prefix not in prefix_colors:
                color_index = len(prefix_colors) % len(palette)
                prefix_colors[prefix] = QColor(palette[color_index])
            column_colors.append(prefix_colors[prefix])
        return column_colors

    def clear(self):
        """Clear all data from the table."""
        if self.model is not None:
            self.model.set_data(None)
        else:
            self.table_view.setModel(None)

    def _copy_data(self):
        """Copy selected data from the table to clipboard (tab-separated)."""
        selection_model = self.table_view.selectionModel()
        if not selection_model or not selection_model.hasSelection():
            return

        selected_indexes = selection_model.selectedIndexes()
        if not selected_indexes or self.model is None or self.model._df.empty:
            return

        rows = sorted(set(index.row() for index in selected_indexes))
        cols = sorted(set(index.column() for index in selected_indexes))

        sub_df = self.model._df.iloc[rows, cols]
        tsv_data = sub_df.to_csv(sep='\t', index=False, header=False)
        clipboard = QApplication.clipboard()
        clipboard.setText(tsv_data)

    def keyPressEvent(self, event):
        """Handle key press events to enable copying with Ctrl+C."""
        if event.modifiers() & Qt.ControlModifier and event.key() == Qt.Key_C:
            self._copy_data()
        else:
            super().keyPressEvent(event)

