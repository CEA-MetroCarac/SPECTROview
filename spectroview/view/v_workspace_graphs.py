# view/v_workspace_graphs.py
"""View for Graphs Workspace - main UI coordinator for graph plotting and visualization."""

import os
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QListWidget,
    QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox, QLineEdit, QSplitter,
    QMdiArea, QMdiSubWindow, QTabWidget, QGroupBox, QMessageBox, QFrame, QScrollArea
)
from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtGui import QIcon

from spectroview import ICON_DIR, PLOT_STYLES
from spectroview.model.m_settings import MSettings
from spectroview.view.components.v_data_filter import VDataFilter
from spectroview.viewmodel.vm_workspace_graphs import VMWorkspaceGraphs
from spectroview.viewmodel.utils import CustomizedPalette


class VWorkspaceGraphs(QWidget):
    """View for Graphs Workspace - manages DataFrame plotting and visualization."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.m_settings = MSettings()
        self.vm = VMWorkspaceGraphs(self.m_settings)
        
        self.init_ui()
        self.setup_connections()
    
    def init_ui(self):
        """Initialize the user interface."""
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(4, 4, 4, 4)
        
        # Main splitter: MDI Area (left) | Control Panel (right)
        main_splitter = QSplitter(Qt.Horizontal)
        
        # Setup left and right panels
        left_panel = self._setup_left_panel()
        right_panel = self._setup_right_panel()
        right_panel.setMaximumWidth(400)
        
        # Add panels to splitter
        main_splitter.addWidget(left_panel)
        main_splitter.addWidget(right_panel)
        main_splitter.setSizes([650, 350])
        
        main_layout.addWidget(main_splitter)
    
    def _setup_left_panel(self):
        """Setup the left panel with MDI area and bottom toolbar."""
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(2)
        
        # MDI Area
        self.mdi_area = QMdiArea()
        self.mdi_area.setMinimumWidth(600)
        left_layout.addWidget(self.mdi_area)
        
        # Bottom Toolbar
        bottom_toolbar = self._create_bottom_toolbar()
        left_layout.addWidget(bottom_toolbar)
        
        return left_panel
    
    def _create_bottom_toolbar(self):
        """Create the bottom toolbar with plot controls."""
        bottom_toolbar = QFrame()
        bottom_toolbar.setFrameShape(QFrame.StyledPanel)
        bottom_toolbar.setMaximumHeight(40)
        toolbar_layout = QHBoxLayout(bottom_toolbar)
        toolbar_layout.setContentsMargins(4, 2, 4, 2)
        toolbar_layout.setSpacing(8)
        
        # Graph list combobox
        toolbar_layout.addWidget(QLabel("Graph list:"))
        self.cbb_graph_list_toolbar = QComboBox()
        self.cbb_graph_list_toolbar.setMinimumWidth(150)
        toolbar_layout.addWidget(self.cbb_graph_list_toolbar)
        
        # Minimize all button
        self.btn_minimize_all = QPushButton("Minimize All")
        self.btn_minimize_all.setMaximumWidth(100)
        toolbar_layout.addWidget(self.btn_minimize_all)
        
        # Plot size label
        self.lbl_plot_size = QLabel("(600x500)")
        self.lbl_plot_size.setMinimumWidth(70)
        toolbar_layout.addWidget(self.lbl_plot_size)
        
        # DPI spinbox
        toolbar_layout.addWidget(QLabel("DPI:"))
        self.spin_dpi_toolbar = QSpinBox()
        self.spin_dpi_toolbar.setRange(50, 300)
        self.spin_dpi_toolbar.setValue(110)
        self.spin_dpi_toolbar.setMaximumWidth(60)
        toolbar_layout.addWidget(self.spin_dpi_toolbar)
        
        # X label rotation
        toolbar_layout.addWidget(QLabel("X label rotation:"))
        self.spin_xlabel_rotation = QSpinBox()
        self.spin_xlabel_rotation.setRange(0, 360)
        self.spin_xlabel_rotation.setValue(0)
        self.spin_xlabel_rotation.setMaximumWidth(60)
        toolbar_layout.addWidget(self.spin_xlabel_rotation)
        
        # Legend outside checkbox
        self.cb_legend_outside_toolbar = QCheckBox("Legend outside")
        toolbar_layout.addWidget(self.cb_legend_outside_toolbar)
        
        # Legend location combobox
        self.cbb_legend_loc_toolbar = QComboBox()
        self.cbb_legend_loc_toolbar.addItems(['lower left', 'upper right', 'upper left', 'lower right',
                                               'right', 'center left', 'center right', 'lower center',
                                               'upper center', 'center'])
        self.cbb_legend_loc_toolbar.setMaximumWidth(100)
        toolbar_layout.addWidget(self.cbb_legend_loc_toolbar)
        
        # Grid checkbox
        self.cb_grid_toolbar = QCheckBox("Grid")
        toolbar_layout.addWidget(self.cb_grid_toolbar)
        
        # Copy button
        self.btn_copy_figure = QPushButton()
        self.btn_copy_figure.setIcon(QIcon(os.path.join(ICON_DIR, "copy.png")))
        self.btn_copy_figure.setToolTip("Copy selected figure to clipboard")
        self.btn_copy_figure.setMaximumWidth(35)
        toolbar_layout.addWidget(self.btn_copy_figure)
        
        toolbar_layout.addStretch()
        
        return bottom_toolbar
    
    def _setup_right_panel(self):
        """Setup the right control panel."""
        right_panel = QFrame()
        right_panel.setFrameShape(QFrame.StyledPanel)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(4, 4, 4, 4)
        right_layout.setSpacing(4)
        
        # Add sections
        self._setup_dataframe_section(right_layout)
        self._setup_filter_section(right_layout)
        self._setup_plot_tabs(right_layout)
        self._setup_action_buttons(right_layout)
        
        return right_panel
    
    def _setup_dataframe_section(self, parent_layout):
        """Setup the DataFrame section with listbox and buttons."""
        lbl_dfs = QLabel("Loaded dataframe(s):")
        parent_layout.addWidget(lbl_dfs)
        
        # DataFrame listbox and buttons side by side
        df_section_layout = QHBoxLayout()
        
        # DataFrame listbox
        self.df_listbox = QListWidget()
        self.df_listbox.setMaximumHeight(80)
        df_section_layout.addWidget(self.df_listbox)
        
        # DataFrame buttons (vertical layout on the right)
        df_buttons_layout = QVBoxLayout()
        
        self.btn_view_df = QPushButton()
        self.btn_view_df.setIcon(QIcon(os.path.join(ICON_DIR, "view.png")))
        self.btn_view_df.setIconSize(QSize(20, 20))
        self.btn_view_df.setToolTip("View DataFrame")
        self.btn_view_df.setMaximumWidth(35)
        
        self.btn_remove_df = QPushButton()
        self.btn_remove_df.setIcon(QIcon(os.path.join(ICON_DIR, "trash3.png")))
        self.btn_remove_df.setIconSize(QSize(20, 20))
        self.btn_remove_df.setToolTip("Remove DataFrame")
        self.btn_remove_df.setMaximumWidth(35)
        
        self.btn_save_df = QPushButton()
        self.btn_save_df.setIcon(QIcon(os.path.join(ICON_DIR, "save.png")))
        self.btn_save_df.setIconSize(QSize(20, 20))
        self.btn_save_df.setToolTip("Save DataFrame to Excel")
        self.btn_save_df.setMaximumWidth(35)
        
        df_buttons_layout.addWidget(self.btn_view_df)
        df_buttons_layout.addWidget(self.btn_remove_df)
        df_buttons_layout.addWidget(self.btn_save_df)
        df_buttons_layout.addStretch()
        
        df_section_layout.addLayout(df_buttons_layout)
        parent_layout.addLayout(df_section_layout)
    
    def _setup_filter_section(self, parent_layout):
        """Setup the data filter section."""
        self.v_data_filter = VDataFilter()
        self.v_data_filter.setMaximumHeight(150)
        parent_layout.addWidget(self.v_data_filter)
    
    def _setup_plot_tabs(self, parent_layout):
        """Setup the plot configuration tabs."""
        self.plot_tabs = QTabWidget()
        
        # Create tabs
        plot_tab = self._create_plot_tab()
        more_options_tab = self._create_more_options_tab()
        
        self.plot_tabs.addTab(plot_tab, "Plot")
        self.plot_tabs.addTab(more_options_tab, "More options")
        
        parent_layout.addWidget(self.plot_tabs, stretch=1)
    
    def _create_plot_tab(self):
        """Create the Plot tab with all plot configuration controls."""
        tab_plot = QWidget()
        tab_plot_layout = QVBoxLayout(tab_plot)
        tab_plot_layout.setContentsMargins(5, 5, 5, 5)
        
        # Plot style and color palette
        plot_style_layout = QHBoxLayout()
        lbl_ps = QLabel("Plot style:")
        lbl_ps.setFixedWidth(60)
        plot_style_layout.addWidget(lbl_ps)
        self.cbb_plot_style = QComboBox()
        self.cbb_plot_style.setIconSize(QSize(30, 30))
        for style in PLOT_STYLES:
            icon_path = os.path.join(ICON_DIR, f"{style}.png")
            if os.path.exists(icon_path):
                self.cbb_plot_style.addItem(QIcon(icon_path), style)
            else:
                self.cbb_plot_style.addItem(style)
        plot_style_layout.addWidget(self.cbb_plot_style)
        
        # Color palette using CustomizedPalette
        self.cbb_colormap = CustomizedPalette()
        self.cbb_colormap.setFixedWidth(120)
        plot_style_layout.addWidget(self.cbb_colormap)
        tab_plot_layout.addLayout(plot_style_layout)
        
        # X, Y, Z axes
        self._create_axes_controls(tab_plot_layout)
        
        # Plot title and axis labels
        self._create_title_and_labels(tab_plot_layout)
        
        # Axis limits
        self._create_axis_limits(tab_plot_layout)
        
        tab_plot_layout.addStretch()
        
        # Wrap in QScrollArea
        scroll_area_plot = QScrollArea()
        scroll_area_plot.setWidgetResizable(True)
        scroll_area_plot.setWidget(tab_plot)
        
        return scroll_area_plot
    
    def _create_axes_controls(self, parent_layout):
        """Create X, Y, Z axis selection controls."""
        axes_layout = QVBoxLayout()
        
        # X axis
        x_layout = QHBoxLayout()
        lbl_x= QLabel("X:")
        lbl_x.setFixedWidth(15)
        x_layout.addWidget(lbl_x)
        self.cbb_x = QComboBox()
        self.cbb_x.setFixedWidth(200)
        x_layout.addWidget(self.cbb_x)
        self.cb_xlog = QCheckBox("Log scale")
        x_layout.addWidget(self.cb_xlog)
        x_layout.addStretch()
        axes_layout.addLayout(x_layout)
        
        # Y axis
        y_layout = QHBoxLayout()
        lbl_y=QLabel("Y:")
        lbl_y.setFixedWidth(15)
        y_layout.addWidget(lbl_y)
        self.cbb_y = QComboBox()
        self.cbb_y.setFixedWidth(200)
        y_layout.addWidget(self.cbb_y)
        self.cb_ylog = QCheckBox("Log scale")
        y_layout.addWidget(self.cb_ylog)
        y_layout.addStretch()
        axes_layout.addLayout(y_layout)
        
        # Z axis
        z_layout = QHBoxLayout()
        lbl_z=QLabel("Z:")
        lbl_z.setFixedWidth(15)
        z_layout.addWidget(lbl_z)
        self.cbb_z = QComboBox()
        self.cbb_z.setFixedWidth(200)
        z_layout.addWidget(self.cbb_z)
        
        # Wafer size combobox
        lbl_wafer = QLabel("Wafer size:")
        z_layout.addWidget(lbl_wafer)
        self.cbb_wafer_size = QComboBox()
        self.cbb_wafer_size.addItems(['300', '200', '150', '100'])
        self.cbb_wafer_size.setCurrentText('300')
        self.cbb_wafer_size.setMaximumWidth(80)
        z_layout.addWidget(self.cbb_wafer_size)
        
        z_layout.addStretch()
        axes_layout.addLayout(z_layout)
        
        parent_layout.addLayout(axes_layout)
    
    def _create_title_and_labels(self, parent_layout):
        """Create plot title and axis label inputs."""
        # Create group box
        title_labels_group = QGroupBox("Title and labels:")
        group_layout = QVBoxLayout(title_labels_group)
        group_layout.setContentsMargins(5, 2, 5, 2)  # Reduce margins
        group_layout.setSpacing(2)  # Reduce spacing
        
        # Plot title
        title_layout = QHBoxLayout()
        lbl_title = QLabel("Plot title:")
        lbl_title.setFixedWidth(60)
        title_layout.addWidget(lbl_title)
        self.edit_plot_title = QLineEdit()
        self.edit_plot_title.setPlaceholderText("Type to modify the plot title")
        title_layout.addWidget(self.edit_plot_title)
        group_layout.addLayout(title_layout)
        
        # X label
        xlabel_layout = QHBoxLayout()
        lbl_xlabel = QLabel("X label:")
        lbl_xlabel.setFixedWidth(60)
        xlabel_layout.addWidget(lbl_xlabel)
        self.edit_xlabel = QLineEdit()
        self.edit_xlabel.setPlaceholderText("X axis label")
        xlabel_layout.addWidget(self.edit_xlabel)
        group_layout.addLayout(xlabel_layout)
        
        # Y label
        ylabel_layout = QHBoxLayout()
        lbl_ylabel = QLabel("Y label:")
        lbl_ylabel.setFixedWidth(60)
        ylabel_layout.addWidget(lbl_ylabel)
        self.edit_ylabel = QLineEdit()
        self.edit_ylabel.setPlaceholderText("Y axis label")
        ylabel_layout.addWidget(self.edit_ylabel)
        group_layout.addLayout(ylabel_layout)
        
        # Z label
        zlabel_layout = QHBoxLayout()
        lbl_zlabel = QLabel("Z label:")
        lbl_zlabel.setFixedWidth(60)
        zlabel_layout.addWidget(lbl_zlabel)
        self.edit_zlabel = QLineEdit()
        self.edit_zlabel.setPlaceholderText("Z axis label")
        zlabel_layout.addWidget(self.edit_zlabel)
        group_layout.addLayout(zlabel_layout)
        
        parent_layout.addWidget(title_labels_group)
    
    def _create_axis_limits(self, parent_layout):
        """Create axis limits controls."""
        limits_group = QGroupBox("Axis limits:")
        limits_layout = QVBoxLayout(limits_group)
        limits_layout.setContentsMargins(5, 2, 5, 2)  # Reduce margins
        limits_layout.setSpacing(2)  # Reduce spacing
        
        # X limits
        x_limits_layout = QHBoxLayout()
        x_limits_layout.addWidget(QLabel("X limits:"))
        self.spin_xmin = QDoubleSpinBox()
        self.spin_xmin.setRange(-999999, 999999)
        self.spin_xmax = QDoubleSpinBox()
        self.spin_xmax.setRange(-999999, 999999)
        x_limits_layout.addWidget(QLabel("min"))
        x_limits_layout.addWidget(self.spin_xmin)
        x_limits_layout.addWidget(QLabel("max"))
        x_limits_layout.addWidget(self.spin_xmax)
        limits_layout.addLayout(x_limits_layout)
        
        # Y limits
        y_limits_layout = QHBoxLayout()
        y_limits_layout.addWidget(QLabel("Y limits:"))
        self.spin_ymin = QDoubleSpinBox()
        self.spin_ymin.setRange(-999999, 999999)
        self.spin_ymax = QDoubleSpinBox()
        self.spin_ymax.setRange(-999999, 999999)
        y_limits_layout.addWidget(QLabel("min"))
        y_limits_layout.addWidget(self.spin_ymin)
        y_limits_layout.addWidget(QLabel("max"))
        y_limits_layout.addWidget(self.spin_ymax)
        limits_layout.addLayout(y_limits_layout)
        
        # Limit buttons
        limits_btn_layout = QHBoxLayout()
        self.btn_set_limits = QPushButton("Set current XY limits")
        self.btn_clear_limits = QPushButton("Clear XY limits")
        limits_btn_layout.addWidget(self.btn_set_limits)
        limits_btn_layout.addWidget(self.btn_clear_limits)
        limits_layout.addLayout(limits_btn_layout)
        
        # Z limits / color range
        z_limits_layout = QHBoxLayout()
        z_limits_layout.addWidget(QLabel("Z limits:"))
        self.spin_zmin = QDoubleSpinBox()
        self.spin_zmin.setRange(-999999, 999999)
        self.spin_zmax = QDoubleSpinBox()
        self.spin_zmax.setRange(-999999, 999999)
        z_limits_layout.addWidget(QLabel("min"))
        z_limits_layout.addWidget(self.spin_zmin)
        z_limits_layout.addWidget(QLabel("max"))
        z_limits_layout.addWidget(self.spin_zmax)
        limits_layout.addLayout(z_limits_layout)
        
        parent_layout.addWidget(limits_group)
    
    def _create_more_options_tab(self):
        """Create the More Options tab."""
        tab_more = QWidget()
        tab_more_layout = QVBoxLayout(tab_more)
        tab_more_layout.setContentsMargins(5, 5, 5, 5)
        
        # Plot-specific options
        self.cb_error_bar = QCheckBox("error bar for 'bar_plot'")
        tab_more_layout.addWidget(self.cb_error_bar)
        
        self.cb_wafer_stats = QCheckBox("stats on 'wafer_plot'")
        self.cb_wafer_stats.setChecked(True)
        tab_more_layout.addWidget(self.cb_wafer_stats)
        
        self.cb_join_point_plot = QCheckBox("join for 'point_plot'")
        tab_more_layout.addWidget(self.cb_join_point_plot)
        
        # Trendline equation with order spinbox
        trendline_layout = QHBoxLayout()
        self.cb_trendline_eq = QCheckBox("add trendline equation (oder")
        trendline_layout.addWidget(self.cb_trendline_eq)
        
        self.spin_trendline_order = QSpinBox()
        self.spin_trendline_order.setRange(1, 10)
        self.spin_trendline_order.setValue(1)
        self.spin_trendline_order.setMaximumWidth(50)
        trendline_layout.addWidget(self.spin_trendline_order)
        
        trendline_layout.addWidget(QLabel(")"))
        trendline_layout.addStretch()
        tab_more_layout.addLayout(trendline_layout)
        
        tab_more_layout.addStretch()
        
        return tab_more
    
    def _setup_action_buttons(self, parent_layout):
        """Setup the action buttons (Add plot, Update plot)."""
        action_buttons_layout = QHBoxLayout()
        
        self.btn_add_plot = QPushButton("Add plot")
        self.btn_add_plot.setIcon(QIcon(os.path.join(ICON_DIR, "add.png")))
        self.btn_add_plot.setMinimumHeight(35)
        
        self.btn_update_plot = QPushButton("Update plot")
        self.btn_update_plot.setIcon(QIcon(os.path.join(ICON_DIR, "refresh.png")))
        self.btn_update_plot.setMinimumHeight(35)
        
        action_buttons_layout.addWidget(self.btn_add_plot)
        action_buttons_layout.addWidget(self.btn_update_plot)
        
        parent_layout.addLayout(action_buttons_layout)
    
    def setup_connections(self):
        """Connect ViewModel signals and slots to View components."""
        vm = self.vm
        
        # DataFrame management
        self.df_listbox.itemSelectionChanged.connect(self._on_df_selected)
        self.btn_remove_df.clicked.connect(self._on_remove_df)
        self.btn_save_df.clicked.connect(self._on_save_df)
        
        # Filter connections
        self.v_data_filter.apply_requested.connect(self._on_apply_filters)
        
        # Plot buttons
        self.btn_add_plot.clicked.connect(self._on_add_plot)
        self.btn_update_plot.clicked.connect(self._on_update_plot)
        
        # Bottom toolbar connections
        self.cbb_graph_list_toolbar.currentIndexChanged.connect(self._on_graph_selected_toolbar)
        self.btn_minimize_all.clicked.connect(self._on_minimize_all)
        self.spin_dpi_toolbar.valueChanged.connect(self._on_dpi_changed_toolbar)
        self.spin_xlabel_rotation.valueChanged.connect(self._on_xlabel_rotation_changed)
        self.cb_legend_outside_toolbar.stateChanged.connect(self._on_legend_outside_changed_toolbar)
        self.cbb_legend_loc_toolbar.currentTextChanged.connect(self._on_legend_loc_changed_toolbar)
        self.cb_grid_toolbar.stateChanged.connect(self._on_grid_changed_toolbar)
        self.btn_copy_figure.clicked.connect(self._on_copy_figure)
        
        # ViewModel → View
        vm.dataframes_changed.connect(self._update_df_list)
        vm.dataframe_columns_changed.connect(self._update_column_combos)
        vm.graphs_changed.connect(self._update_graph_list)
        vm.notify.connect(lambda msg: QMessageBox.information(self, "Graphs", msg))
    
    def _on_df_selected(self):
        """Handle DataFrame selection."""
        current_item = self.df_listbox.currentItem()
        if current_item:
            df_name = current_item.text()
            self.vm.select_dataframe(df_name)
    
    def _on_remove_df(self):
        """Remove selected DataFrame."""
        current_item = self.df_listbox.currentItem()
        if current_item:
            df_name = current_item.text()
            self.vm.remove_dataframe(df_name)
    
    def _on_save_df(self):
        """Save selected DataFrame to Excel."""
        current_item = self.df_listbox.currentItem()
        if current_item:
            df_name = current_item.text()
            self.vm.save_dataframe_to_excel(df_name)
    
    def _on_apply_filters(self):
        """Apply filters to selected DataFrame."""
        if not self.vm.selected_df_name:
            return
        
        filters = self.v_data_filter.get_filters()
        self.vm.apply_filters(self.vm.selected_df_name, filters)
    
    def _on_add_plot(self):
        """Add a new plot."""
        # This will be implemented when adding matplotlib integration
        self.vm.notify.emit("Plot functionality will be implemented in Phase 2")
    
    def _on_update_plot(self):
        """Update selected plot."""
        # This will be implemented when adding matplotlib integration
        self.vm.notify.emit("Plot functionality will be implemented in Phase 2")
    
    def _update_df_list(self, df_names: list):
        """Update DataFrame listbox."""
        self.df_listbox.clear()
        self.df_listbox.addItems(df_names)
    
    def _update_column_combos(self, columns: list):
        """Update column comboboxes."""
        self.cbb_x.clear()
        self.cbb_y.clear()
        self.cbb_z.clear()
        
        self.cbb_x.addItems(columns)
        self.cbb_y.addItems(columns)
        self.cbb_z.addItem("None")
        self.cbb_z.addItems(columns)
    
    def _update_graph_list(self, graph_ids: list):
        """Update graph list combobox in toolbar."""
        self.cbb_graph_list_toolbar.clear()
        for gid in graph_ids:
            graph = self.vm.get_graph(gid)
            if graph:
                display_name = graph.get_display_name()
                self.cbb_graph_list_toolbar.addItem(display_name, gid)
    
    def _on_graph_selected_toolbar(self):
        """Handle graph selection from toolbar combobox."""
        index = self.cbb_graph_list_toolbar.currentIndex()
        if index >= 0:
            graph_id = self.cbb_graph_list_toolbar.currentData()
            if graph_id:
                self.vm.select_graph(graph_id)
    
    def _on_minimize_all(self):
        """Minimize all MDI subwindows."""
        for window in self.mdi_area.subWindowList():
            window.showMinimized()
    
    def _on_dpi_changed_toolbar(self, value: int):
        """Handle DPI change from toolbar."""
        # TODO: Update current plot DPI in Phase 2
        pass
    
    def _on_xlabel_rotation_changed(self, value: int):
        """Handle X label rotation change."""
        # TODO: Update current plot X label rotation in Phase 2
        pass
    
    def _on_legend_outside_changed_toolbar(self, state: int):
        """Handle legend outside toggle from toolbar."""
        # Sync with More options tab
        self.cb_legend_outside.setChecked(state == Qt.Checked)
        # TODO: Update current plot legend in Phase 2
    
    def _on_legend_loc_changed_toolbar(self, location: str):
        """Handle legend location change from toolbar."""
        # Sync with More options tab
        index = self.cbb_legend_loc.findText(location)
        if index >= 0:
            self.cbb_legend_loc.setCurrentIndex(index)
        # TODO: Update current plot legend location in Phase 2
    
    def _on_grid_changed_toolbar(self, state: int):
        """Handle grid toggle from toolbar."""
        # Sync with More options tab
        self.cb_grid.setChecked(state == Qt.Checked)
        # TODO: Update current plot grid in Phase 2
    
    def _on_copy_figure(self):
        """Copy selected figure to clipboard."""
        # TODO: Implement clipboard copy in Phase 2
        QMessageBox.information(self, "Copy Figure", "Copy to clipboard will be implemented in Phase 2")
    
    def save_workspace(self):
        """Save workspace (called from main menu)."""
        self.vm.save_workspace()
    
    def load_workspace(self, file_path: str):
        """Load workspace (called from main menu)."""
        self.vm.load_workspace(file_path)
    
    def clear_workspace(self):
        """Clear workspace (called from main menu)."""
        self.vm.clear_workspace()

