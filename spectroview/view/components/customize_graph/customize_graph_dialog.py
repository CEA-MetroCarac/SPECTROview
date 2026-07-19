"""Top-level "Customize Graph" dialog: hosts the Axis / Legend-Colors /
Annotations / More-options tabs and coordinates their shared Apply/Cancel.

The tab widgets themselves used to live in this file too (~1700 lines across
8 largely independent classes); they've been split into their own modules,
all living together under the `customize_graph/` package for maintainability.
Everything is re-imported and re-exported here so existing imports of
`spectroview.view.components.customize_graph.customize_graph_dialog`
(e.g. `EditLineDialog`/`EditTextDialog` from view/components/v_graph.py, or
the individual tab widgets from tests) keep working unchanged.
"""
import copy

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QTabWidget, QWidget

from spectroview import ICON_DIR
from spectroview.model.graph_style import can_restyle_without_replot
from spectroview.view.components.graph_commit import snapshot, diff
from spectroview.view.components.customize_graph.customize_legend import CustomizeLegend
from spectroview.view.components.customize_graph.customize_annotations import CustomizeAnnotations
from spectroview.view.components.customize_graph.customize_axis import CustomizeAxis
from spectroview.view.components.customize_graph.customize_more_options import CustomizeMoreOptions
from spectroview.view.components.customize_graph.customize_annotation_dialogs import (
    EditLineDialog, EditTextDialog, EditArrowDialog, EditSpanDialog,
    EditBoxDialog, EditCalloutDialog, ColorDelegate,
)

__all__ = [
    "CustomizeGraphDialog", "CustomizeLegend", "CustomizeAnnotations",
    "CustomizeAxis", "CustomizeMoreOptions",
    "EditLineDialog", "EditTextDialog", "EditArrowDialog", "EditSpanDialog",
    "EditBoxDialog", "EditCalloutDialog", "ColorDelegate",
]


class CustomizeGraphDialog(QDialog):
    """Dialog for customizing graph"""

    def __init__(self, graph_widget, graph_id, parent=None):
        super().__init__(parent)
        self.graph_widget = graph_widget
        self.graph_id = graph_id

        self.setWindowTitle(f"Customize Graph {graph_id}")
        self.setModal(False)
        self.setWindowFlags(
            Qt.Window | Qt.Tool | Qt.CustomizeWindowHint
            | Qt.WindowTitleHint | Qt.WindowCloseButtonHint
        )
        # Prevent the tool window from hiding when switching to another app on macOS.
        self.setAttribute(Qt.WA_MacAlwaysShowToolWindow)
        self.resize(560, 760)

        self._setup_ui()

        self._original_snapshot = snapshot(self.graph_widget)

        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(400)
        self._preview_timer.timeout.connect(self._preview_apply)
        self._wire_live_preview()


    def _setup_ui(self):
        """Setup dialog UI with tabs."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(8)

        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        tab_axis = self._create_axis_tab()
        tab_legend = self._create_legend_tab()
        tab_annotations = self._create_annotations_tab()
        tab_general = self._create_general_tab()

        self.tabs.addTab(tab_axis, "Axis")
        self.tabs.addTab(tab_legend, "Legend/Colors")
        self.tabs.addTab(tab_annotations, "Annotations")
        self.tabs.addTab(tab_general, "More options")

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

    def _vm(self):
        """The owning VMWorkspaceGraphs, when this dialog was opened from
        the real app (parent=VWorkspaceGraphs) -- None in the handful of
        tests that construct this dialog standalone, where undo-batching is
        simply skipped rather than raising on a missing parent."""
        return getattr(self.parent(), 'vm', None)

    _LIVE_PREVIEW_SIGNALS = ('valueChanged', 'toggled', 'currentIndexChanged', 'textChanged')

    def _wire_live_preview(self):
        seen = set()
        for w in self.findChildren(QWidget):
            if id(w) in seen:
                continue
            seen.add(id(w))
            for signal_name in self._LIVE_PREVIEW_SIGNALS:
                signal = getattr(w, signal_name, None)
                if signal is not None and hasattr(signal, 'connect'):
                    try:
                        signal.connect(self._schedule_preview)
                    except (TypeError, RuntimeError):
                        pass

    def _schedule_preview(self, *_args):
        self._preview_timer.start()

    def _preview_apply(self):
        """Debounced live preview, fired ~400ms after the user stops
        interacting with any control.

        A preview must stay purely visual -- mutate the widget and repaint,
        but commit nothing -- so that cancel_all() reverting the widget to
        _original_snapshot."""

        gw = self.graph_widget
        before = snapshot(gw)
        gw.blockSignals(True)
        try:
            self.legend_widget.apply_changes(replot=False)
            self.axis_widget._apply_axis_settings(silent=True, replot=False)
            self.more_options_widget._apply(replot=False)
        finally:
            gw.blockSignals(False)

        changed = set(diff(gw, before).keys())
        if not changed:
            return  # nothing actually changed this tick

        if can_restyle_without_replot(changed) and gw.restyle():
            return

        if gw.df is not None:
            gw.plot(gw.df)
        else:
            gw.canvas.draw_idle()

        if gw.plot_style == 'trendline':
            self.more_options_widget._refresh_equation_table()

    def apply_all(self):
        """Apply changes from all tabs and close dialog."""
        self._preview_timer.stop()
        vm = self._vm()
        if vm:
            vm.begin_undo_batch()
        try:
            self.legend_widget.apply_changes()
            self.axis_widget._apply_axis_settings()
            self.more_options_widget._apply()
        finally:
            if vm:
                vm.end_undo_batch()

        self.legend_widget.load_legend_properties()

        self._original_snapshot = snapshot(self.graph_widget)
        self._preview_timer.stop()

    def cancel_all(self):
        """Discard every live-previewed change made since the dialog was
        opened (or since the last Apply) and restore that baseline."""
        self._preview_timer.stop()
        gw = self.graph_widget
        for field, value in self._original_snapshot.items():
            setattr(gw, field, copy.deepcopy(value))
        if gw.df is not None:
            gw.plot(gw.df)
        else:
            gw.canvas.draw()

        self.axis_widget.load_axis_settings()
        self.more_options_widget.load_settings()
        self.legend_widget.load_legend_properties()
        self.annotations_widget.load_annotations()
        self._preview_timer.stop()


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
        """Create axis customization tab (scale/limits/breaks/inset/secondary axes)."""
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
        self.tabs.setCurrentIndex(self.tabs.indexOf(self.legend_widget.parent()))
        self.show()
        self.raise_()
        self.activateWindow()

    def open_axis_tab(self):
        """Open the dialog and switch to the AXIS tab."""
        self.tabs.setCurrentIndex(self.tabs.indexOf(self.axis_widget.parent()))
        self.show()
        self.raise_()
        self.activateWindow()

    def switch_graph(self, graph_widget, graph_id):
        """Switch the dialog to a different graph widget.

        Re-binds all child widgets (legend, annotations, axis, more_options)
        to the new graph and reloads their content.
        """
        if self.graph_id == graph_id:
            return

        self._preview_timer.stop()

        self.graph_widget = graph_widget
        self.graph_id = graph_id
        self.setWindowTitle(f"Customize Graph {graph_id}")

        self.legend_widget.switch_graph(graph_widget)
        self.annotations_widget.switch_graph(graph_widget)
        self.axis_widget.switch_graph(graph_widget)
        self.more_options_widget.switch_graph(graph_widget)

        self._original_snapshot = snapshot(self.graph_widget)
        self._preview_timer.stop()
