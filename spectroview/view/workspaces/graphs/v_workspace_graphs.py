"""View for Graphs Workspace - main UI coordinator for graph plotting and visualization."""

from typing import Optional

from PySide6.QtWidgets import QWidget

from spectroview.i18n import combo_data_or_text, on_language_changed, tr
from spectroview.model.graphs.m_plot_recipe_store import MPlotRecipeStore
from spectroview.model.graphs.m_style_template_store import MStyleTemplateStore
from spectroview.model.m_settings import MSettings
from spectroview.view.components.common.v_notification import show_notification
from spectroview.view.workspaces.graphs.v_graphs_actions import GraphsActionsMixin
from spectroview.view.workspaces.graphs.v_graphs_dataframes import GraphsDataFramesMixin
from spectroview.view.workspaces.graphs.v_graphs_mdi import GraphsMdiMixin
from spectroview.view.workspaces.graphs.v_graphs_ui import GraphsUIMixin
from spectroview.view.workspaces.graphs.v_mdi_subwindow import MdiSubWindow
from spectroview.viewmodel.graphs.vm_workspace_graphs import VMWorkspaceGraphs


class VWorkspaceGraphs(
    GraphsUIMixin,
    GraphsDataFramesMixin,
    GraphsMdiMixin,
    GraphsActionsMixin,
    QWidget,
):
    """View for Graphs Workspace."""

    _SIDE_PANEL_SHARED_FIELDS = frozenset({
        'plot_title', 'plot_subtitle', 'xlabel', 'ylabel', 'zlabel',
        'color_palette', 'y2label', 'y3label', 'x2label',
        'y2logscale', 'y3logscale', 'x2logscale',
    })
    _SECONDARY_SHARED_FIELDS = (
        'y2label', 'y3label', 'x2label',
        'y2logscale', 'y3logscale', 'x2logscale',
    )

    def __init__(self, parent=None):
        super().__init__(parent)
        self.m_settings = MSettings()
        self.vm = VMWorkspaceGraphs(self.m_settings)

        # Starting stores; handlers refresh from current Working Folder setting
        self.recipe_store = MPlotRecipeStore(self.m_settings.get_plot_recipe_folder())
        self.style_template_store = MStyleTemplateStore(self.m_settings.get_plot_style_folder())

        # Session-level "copied style" clipboard for Copy Style/Paste Style
        self._copied_style: Optional[dict] = None

        # Graph storage: {graph_id: (Graph widget, QDialog, QMdiSubWindow)}
        self.graph_widgets = {}

        # Singleton customize dialog
        self._customize_dialog = None

        self.init_ui()
        self.setup_connections()
        on_language_changed(self, self._retranslate_workspace)

    def _retranslate_workspace(self):
        """Refresh dynamic placeholder and 'None' sentinel combobox items on language switch."""
        if getattr(self, "_has_df_placeholder", False):
            for i in range(self.df_listbox.count()):
                item = self.df_listbox.item(i)
                if item and item.text():
                    item.setText(tr("Drag and drop file(s) anywhere to open"))
                    break
        for cbb in (self.cbb_z, self.cbb_y2, self.cbb_y3, self.cbb_x2):
            if cbb.count() > 0 and cbb.itemData(0) == "None":
                was_blocked = cbb.blockSignals(True)
                try:
                    cbb.setItemText(0, tr("None"))
                finally:
                    cbb.blockSignals(was_blocked)

    def setup_connections(self):
        """Connect signals and slots."""
        vm = self.vm

        # DataFrame management
        self.df_listbox.itemSelectionChanged.connect(self._on_df_selected)
        self.btn_view_df.clicked.connect(self._on_view_df)
        self.btn_remove_df.clicked.connect(self._on_remove_df)
        self.btn_save_df.clicked.connect(self._on_save_df)
        self.btn_refresh_df.clicked.connect(self._on_refresh_df)

        # Filter connections
        self.v_data_filter.apply_requested.connect(self._on_apply_filters)

        # Slot selector connections
        self.select_all_checkbox.stateChanged.connect(self._on_select_all_slots)
        self.btn_add_multi_wafer.clicked.connect(self._on_plot_multi_wafer)

        # Plot buttons
        self.btn_add_plot.clicked.connect(self._on_add_plot)
        self.btn_update_plot.clicked.connect(self._on_update_plot)

        # Plot recipe buttons
        self.btn_apply_recipe.clicked.connect(self._on_apply_recipe_clicked)
        self.btn_save_as_recipe.clicked.connect(self._on_save_as_recipe_clicked)
        self.btn_compose_figure.clicked.connect(self._on_compose_figure_clicked)
        self.btn_undo.clicked.connect(self._on_undo_clicked)
        self.btn_redo.clicked.connect(self._on_redo_clicked)

        # Plot style connection
        self.cbb_plot_style.currentIndexChanged.connect(
            lambda _idx: self._on_plot_style_changed(
                combo_data_or_text(self.cbb_plot_style)
            )
        )

        # Bottom toolbar connections
        self.cbb_graph_list.currentIndexChanged.connect(self._on_graph_selected_toolbar)
        self.btn_minimize_all.clicked.connect(self._on_minimize_all)
        self.btn_delete_all.clicked.connect(self._on_delete_all)

        # MDI area connections
        self.mdi_area.subWindowActivated.connect(self._on_subwindow_activated)

        # ViewModel → View signal connections
        vm.dataframes_changed.connect(self._update_df_list)
        vm.dataframe_columns_changed.connect(self._update_column_combos)
        vm.dataframe_columns_changed.connect(self._update_slot_selector)
        vm.graphs_changed.connect(self._update_graph_list)
        vm.undo_state_changed.connect(self._on_undo_state_changed)
        vm.notify.connect(self._show_notification)

        self._setup_keyboard_shortcuts()

    def _show_notification(self, message: str, notification_type=None, duration: int = 3500, title: str | None = None):
        """Show native floating notification in the application toolbar."""
        show_notification(
            message=message,
            notification_type=notification_type,
            duration=duration,
            title=title,
            parent=self,
        )

    def _show_toast_notification(self, message: str):
        """Backward-compatible alias for _show_notification."""
        self._show_notification(message)


__all__ = ["MdiSubWindow", "VWorkspaceGraphs"]
