"""Top-level "Customize Graph" dialog: hosts the Axis / Legend-Color /
Annotations / More-options tabs and coordinates their shared Apply/Cancel.

The tab widgets themselves used to live in this file too (~1700 lines across
8 largely independent classes); they've been split into their own modules
for maintainability. Everything is re-imported and re-exported here so
existing imports of `spectroview.view.components.customize_graph_dialog`
(e.g. `EditLineDialog`/`EditTextDialog` from view/components/v_graph.py, or
the individual tab widgets from tests) keep working unchanged.
"""
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QTabWidget, QWidget

from spectroview import ICON_DIR
from spectroview.view.components.customize_legend import CustomizeLegend
from spectroview.view.components.customize_annotations import CustomizeAnnotations
from spectroview.view.components.customize_axis import CustomizeAxis
from spectroview.view.components.customize_more_options import CustomizeMoreOptions
from spectroview.view.components.customize_annotation_dialogs import (
    EditLineDialog, EditTextDialog, ColorDelegate,
)

__all__ = [
    "CustomizeGraphDialog", "CustomizeLegend", "CustomizeAnnotations",
    "CustomizeAxis", "CustomizeMoreOptions", "EditLineDialog",
    "EditTextDialog", "ColorDelegate",
]


class CustomizeGraphDialog(QDialog):
    """Dialog for customizing graph"""

    def __init__(self, graph_widget, graph_id, parent=None):
        super().__init__(parent)
        self.graph_widget = graph_widget
        self.graph_id = graph_id

        self.setWindowTitle(f"Customize Graph {graph_id}")
        self.setModal(False)
        # Qt.Tool keeps the dialog above its parent on both macOS and Windows,
        # hides when switching to another app, and preserves the close button.
        self.setWindowFlags(
            Qt.Window | Qt.Tool | Qt.CustomizeWindowHint
            | Qt.WindowTitleHint | Qt.WindowCloseButtonHint
        )
        self.resize(450, 550)

        self._setup_ui()


    def _setup_ui(self):
        """Setup dialog UI with tabs."""
        # Main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(8)

        # Create tab widget
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        # Create tabs
        tab_annotations = self._create_annotations_tab()
        tab_legend = self._create_legend_tab()
        tab_general = self._create_general_tab()
        tab_axis = self._create_axis_tab()

        # Add tabs to widget
        self.tabs.addTab(tab_axis, "Axis")
        self.tabs.addTab(tab_legend, "Legend / Color")
        self.tabs.addTab(tab_annotations, "Annotations")
        self.tabs.addTab(tab_general, "More options")

        # Universal Apply / Cancel buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        btn_cancel = QPushButton("Cancel")
        btn_cancel.setIcon(QIcon(f"{ICON_DIR}/close.png"))
        btn_cancel.clicked.connect(self.cancel_all)
        btn_layout.addWidget(btn_cancel)

        btn_apply = QPushButton("Apply")
        btn_apply.setIcon(QIcon(f"{ICON_DIR}/done.png"))
        btn_apply.clicked.connect(self.apply_all)
        btn_layout.addWidget(btn_apply)

        layout.addLayout(btn_layout)

    def apply_all(self):
        """Apply changes from all tabs and close dialog."""
        self.legend_widget.apply_changes()
        self.axis_widget._apply_axis_settings()
        self.more_options_widget._apply()

        # After more options are applied (which may replot and recreate legend properties),
        # reload the legend tab to ensure it reflects the newly generated colors and order.
        self.legend_widget.load_legend_properties()

    def cancel_all(self):
        """Cancel changes and close dialog."""
        self.legend_widget.cancel_changes()


    def _create_legend_tab(self):
        """Create legend customization tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(8)

        self.legend_widget = CustomizeLegend(self.graph_widget, parent=tab)
        layout.addWidget(self.legend_widget)
        layout.addStretch()

        return tab

    def _create_annotations_tab(self):
        """Create annotations customization tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(8)

        self.annotations_widget = CustomizeAnnotations(self.graph_widget, parent=tab)
        layout.addWidget(self.annotations_widget)
        return tab

    def _create_general_tab(self):
        """Create general settings tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(8)

        self.more_options_widget = CustomizeMoreOptions(self.graph_widget, parent=tab)
        layout.addWidget(self.more_options_widget)
        return tab

    def _create_axis_tab(self):
        """Create axis customization tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(8)

        self.axis_widget = CustomizeAxis(self.graph_widget, parent=tab)
        layout.addWidget(self.axis_widget)
        return tab

    def open_legend_tab(self):
        """Open the dialog and switch to the Legend tab."""
        self.legend_widget.load_legend_properties()
        self.tabs.setCurrentIndex(1) # Switch to Legend tab (index 1)
        self.show()
        self.raise_()
        self.activateWindow()

    def open_axis_tab(self):
        """Open the dialog and switch to the AXIS tab."""
        self.tabs.setCurrentIndex(0) # Switch to Axis tab (index 0)
        self.show()
        self.raise_()
        self.activateWindow()

    def switch_graph(self, graph_widget, graph_id):
        """Switch the dialog to a different graph widget.

        Re-binds all child widgets (legend, annotations, axis, more_options) to the new
        graph and reloads their content.
        """
        if self.graph_id == graph_id:
            return  # Already showing this graph

        self.graph_widget = graph_widget
        self.graph_id = graph_id
        self.setWindowTitle(f"Customize Graph {graph_id}")

        # Switch each child widget to the new graph
        self.legend_widget.switch_graph(graph_widget)
        self.annotations_widget.switch_graph(graph_widget)
        self.axis_widget.switch_graph(graph_widget)
        self.more_options_widget.switch_graph(graph_widget)
