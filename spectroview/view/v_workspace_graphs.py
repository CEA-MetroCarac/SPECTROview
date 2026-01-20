# view/v_workspace_graphs.py
"""View for Graphs Workspace - main UI coordinator for graph plotting and visualization."""

import os
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QListWidget,
    QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox, QLineEdit, QSplitter,
    QMdiArea, QMdiSubWindow, QTabWidget, QGroupBox, QMessageBox, QFrame, QScrollArea,
    QDialog, QGridLayout
)
from PySide6.QtCore import Qt, Signal, QSize, QTimer
from PySide6.QtGui import QIcon

from spectroview import ICON_DIR, PLOT_STYLES, LEGEND_LOCATION
from spectroview.model.m_settings import MSettings
from spectroview.view.components.v_data_filter import VDataFilter
from spectroview.view.components.v_dataframe_table import VDataframeTable
from spectroview.view.components.v_graph import VGraph
from spectroview.viewmodel.vm_workspace_graphs import VMWorkspaceGraphs
from spectroview.viewmodel.utils import CustomizedPalette, show_toast_notification, copy_fig_to_clb

class VWorkspaceGraphs(QWidget):
    """View for Graphs Workspace."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.m_settings = MSettings()
        self.vm = VMWorkspaceGraphs(self.m_settings)
        
        # Graph storage: {graph_id: (Graph widget, QDialog, QMdiSubWindow)}
        self.graph_widgets = {}
        
        self.init_ui()
        self.setup_connections()
    
    def init_ui(self):
        """Initialize UI."""
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
        main_splitter.setSizes([650, 370])
        
        main_layout.addWidget(main_splitter)
    
    def _setup_left_panel(self):
        """Setup left panel."""
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
        """Create bottom toolbar."""
        bottom_toolbar = QFrame()
        bottom_toolbar.setFrameShape(QFrame.StyledPanel)
        bottom_toolbar.setMaximumHeight(40)
        toolbar_layout = QHBoxLayout(bottom_toolbar)
        toolbar_layout.setContentsMargins(4, 2, 4, 2)
        toolbar_layout.setSpacing(8)
        

        # Delete all button
        self.btn_delete_all = QPushButton()
        self.btn_delete_all.setIcon(QIcon(os.path.join(ICON_DIR, "trash3.png")))    
        self.btn_delete_all.setToolTip("Delete all graphs from workspace")
        
        self.btn_delete_all.setMaximumWidth(100)
        toolbar_layout.addWidget(self.btn_delete_all)

        # Graph list combobox
        self.cbb_graph_list = QComboBox()
        self.cbb_graph_list.setMinimumWidth(150)
        toolbar_layout.addWidget(self.cbb_graph_list)
        
        # Minimize all button
        self.btn_minimize_all = QPushButton("Minimize All")
        self.btn_minimize_all.setMaximumWidth(100)
        toolbar_layout.addWidget(self.btn_minimize_all)
        
        # Plot size label
        self.lbl_plot_size = QLabel("(480x400)")
        self.lbl_plot_size.setMinimumWidth(70)
        toolbar_layout.addWidget(self.lbl_plot_size)
        
        # DPI spinbox
        toolbar_layout.addWidget(QLabel("DPI:"))
        self.spin_dpi_toolbar = QSpinBox()
        self.spin_dpi_toolbar.setRange(50, 300)
        self.spin_dpi_toolbar.setValue(100)
        self.spin_dpi_toolbar.setSingleStep(10)
        self.spin_dpi_toolbar.setMaximumWidth(60)
        toolbar_layout.addWidget(self.spin_dpi_toolbar)
        
        # X label rotation
        toolbar_layout.addWidget(QLabel("X label rotation:"))
        self.spin_xlabel_rotation = QSpinBox()
        self.spin_xlabel_rotation.setRange(0, 90)
        self.spin_xlabel_rotation.setValue(0)
        self.spin_xlabel_rotation.setSingleStep(10)
        self.spin_xlabel_rotation.setMaximumWidth(60)
        toolbar_layout.addWidget(self.spin_xlabel_rotation)
        
        # Legend outside checkbox
        self.cb_legend_outside_toolbar = QCheckBox("Legend outside")
        toolbar_layout.addWidget(self.cb_legend_outside_toolbar)
        
        # Legend location combobox
        self.cbb_legend_loc_toolbar = QComboBox()
        self.cbb_legend_loc_toolbar.addItems(LEGEND_LOCATION)
        self.cbb_legend_loc_toolbar.setMaximumWidth(100)
        toolbar_layout.addWidget(self.cbb_legend_loc_toolbar)
        
        # Grid checkbox
        self.cb_grid_toolbar = QCheckBox("Grid")
        toolbar_layout.addWidget(self.cb_grid_toolbar)
        
        toolbar_layout.addStretch()
        
        return bottom_toolbar
    
    def _setup_right_panel(self):
        """Setup right panel."""
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
        """Setup DataFrame section."""
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
        """Setup filter section."""
        self.v_data_filter = VDataFilter()
        self.v_data_filter.setMaximumHeight(150)
        parent_layout.addWidget(self.v_data_filter)
    
    def _setup_slot_selector_section(self, parent_layout):
        """Setup slot selector section."""
        # Container widget for slot selector
        self.slot_selector_widget = QWidget()
        self.slot_selector_widget.setMaximumHeight(120)
        slot_layout = QVBoxLayout(self.slot_selector_widget)
        slot_layout.setContentsMargins(2, 2, 2, 2)
        slot_layout.setSpacing(2)
        
        # Label and Select All checkbox in same row
        header_layout = QHBoxLayout()
        lbl_slot = QLabel("Select wafer slot(s):")
        header_layout.addWidget(lbl_slot)
        
        self.select_all_checkbox = QCheckBox("Select All")
        self.select_all_checkbox.setChecked(True)
        self.select_all_checkbox.setVisible(False)
        header_layout.addWidget(self.select_all_checkbox)
        header_layout.addStretch()
        
        slot_layout.addLayout(header_layout)
        
        # Grid layout for checkboxes
        self.slot_grid_layout = QGridLayout()
        self.slot_grid_layout.setSpacing(4)
        slot_layout.addLayout(self.slot_grid_layout)
        
        # Storage for checkboxes
        self.slot_checkboxes = []
        
        # Initially hide the whole widget
        self.slot_selector_widget.setVisible(False)
        parent_layout.addWidget(self.slot_selector_widget)
    
    def _setup_plot_tabs(self, parent_layout):
        """Setup plot tabs."""
        self.plot_tabs = QTabWidget()
        
        # Create tabs
        plot_tab = self._create_plot_tab()
        more_options_tab = self._create_more_options_tab()
        
        self.plot_tabs.addTab(plot_tab, "Plot")
        self.plot_tabs.addTab(more_options_tab, "More options")
        
        parent_layout.addWidget(self.plot_tabs, stretch=1)
    
    def _create_plot_tab(self):
        """Create plot tab."""
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
        
        # Slot selector section (for wafer plots)
        self._setup_slot_selector_section(tab_plot_layout)
        
        tab_plot_layout.addStretch()
        
        # Wrap in QScrollArea
        scroll_area_plot = QScrollArea()
        scroll_area_plot.setWidgetResizable(True)
        scroll_area_plot.setWidget(tab_plot)
        
        return scroll_area_plot
    
    def _create_axes_controls(self, parent_layout):
        """Create axes controls."""
        axes_layout = QVBoxLayout()
        
        # X, Y, Z axes with log scale checkboxes
        for axis_name, label_text in [('x', 'X:'), ('y', 'Y:'), ('z', 'Z:')]:
            h_layout = QHBoxLayout()
            lbl = QLabel(label_text)
            lbl.setFixedWidth(15)
            h_layout.addWidget(lbl)
            
            cbb = QComboBox()
            cbb.setFixedWidth(200)
            setattr(self, f'cbb_{axis_name}', cbb)
            h_layout.addWidget(cbb)
            
            if axis_name != 'z':
                cb_log = QCheckBox("Log scale")
                setattr(self, f'cb_{axis_name}log', cb_log)
                h_layout.addWidget(cb_log)
            else:
                # Wafer size for Z axis
                lbl_wafer = QLabel("Wafer size:")
                h_layout.addWidget(lbl_wafer)
                self.cbb_wafer_size = QComboBox()
                self.cbb_wafer_size.addItems(['300', '200', '150', '100'])
                self.cbb_wafer_size.setCurrentText('300')
                self.cbb_wafer_size.setMaximumWidth(80)
                h_layout.addWidget(self.cbb_wafer_size)
            
            h_layout.addStretch()
            axes_layout.addLayout(h_layout)
        
        parent_layout.addLayout(axes_layout)
    
    def _create_title_and_labels(self, parent_layout):
        """Create title and labels controls."""
        title_labels_group = QGroupBox("Title and labels:")
        group_layout = QVBoxLayout(title_labels_group)
        group_layout.setContentsMargins(5, 2, 5, 2)
        group_layout.setSpacing(2)
        
        # Create all label inputs
        labels = [
            ('plot_title', 'Plot title:', 'Type to modify the plot title'),
            ('xlabel', 'X label:', 'X axis label'),
            ('ylabel', 'Y label:', 'Y axis label'),
            ('zlabel', 'Z label:', 'Z axis label')
        ]
        
        for attr_name, label_text, placeholder in labels:
            h_layout = QHBoxLayout()
            lbl = QLabel(label_text)
            lbl.setFixedWidth(60)
            h_layout.addWidget(lbl)
            
            line_edit = QLineEdit()
            line_edit.setPlaceholderText(placeholder)
            setattr(self, f'edit_{attr_name}', line_edit)
            h_layout.addWidget(line_edit)
            group_layout.addLayout(h_layout)
        
        parent_layout.addWidget(title_labels_group)
    
    def _create_axis_limits(self, parent_layout):
        """Create axis limits controls."""
        limits_group = QGroupBox("Axis limits:")
        limits_layout = QVBoxLayout(limits_group)
        limits_layout.setContentsMargins(5, 2, 5, 2)
        limits_layout.setSpacing(2)
        
        # X, Y limits
        for axis in ['X', 'Y']:
            h_layout = QHBoxLayout()
            h_layout.addWidget(QLabel(f"{axis} limits:"))
            
            for limit_type in ['min', 'max']:
                spin = QDoubleSpinBox()
                spin.setRange(-999999, 999999)
                setattr(self, f'spin_{axis.lower()}{limit_type}', spin)
                h_layout.addWidget(QLabel(limit_type))
                h_layout.addWidget(spin)
            limits_layout.addLayout(h_layout)
        
        # Limit buttons
        limits_btn_layout = QHBoxLayout()
        self.btn_set_limits = QPushButton("Set current XY limits")
        self.btn_clear_limits = QPushButton("Clear XY limits")
        limits_btn_layout.addWidget(self.btn_set_limits)
        limits_btn_layout.addWidget(self.btn_clear_limits)
        limits_layout.addLayout(limits_btn_layout)
        
        # Z limits
        z_limits_layout = QHBoxLayout()
        z_limits_layout.addWidget(QLabel("Z limits:"))
        for limit_type in ['min', 'max']:
            spin = QDoubleSpinBox()
            spin.setRange(-999999, 999999)
            setattr(self, f'spin_z{limit_type}', spin)
            z_limits_layout.addWidget(QLabel(limit_type))
            z_limits_layout.addWidget(spin)
        limits_layout.addLayout(z_limits_layout)
        
        parent_layout.addWidget(limits_group)
    
    def _create_more_options_tab(self):
        """Create more options tab."""
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
        """Setup action buttons."""
        action_buttons_layout = QHBoxLayout()
        
        self.btn_add_plot = QPushButton("Add plot")
        self.btn_add_plot.setIcon(QIcon(os.path.join(ICON_DIR, "add.png")))
        self.btn_add_plot.setIconSize(QSize(20, 20))
        self.btn_add_plot.setToolTip("Add new plot with current configuration")
        self.btn_add_plot.setMinimumHeight(25)
        
        self.btn_update_plot = QPushButton("Update plot")
        self.btn_update_plot.setIcon(QIcon(os.path.join(ICON_DIR, "refresh.png")))
        self.btn_update_plot.setIconSize(QSize(20, 20))
        self.btn_update_plot.setToolTip("Update selected plot with current configuration")
        self.btn_update_plot.setMinimumHeight(25)
        
        # Multi-wafer plot button (shown when Slot column exists)
        self.btn_add_multi_wafer = QPushButton("Add Multi-Wafer")
        self.btn_add_multi_wafer.setIcon(QIcon(os.path.join(ICON_DIR, "add.png")))
        self.btn_add_multi_wafer.setIconSize(QSize(20, 20))
        self.btn_add_multi_wafer.setToolTip("Create wafer plots for selected slots")
        self.btn_add_multi_wafer.setMinimumHeight(25)
        self.btn_add_multi_wafer.setVisible(False)  # Initially hidden
        
        action_buttons_layout.addWidget(self.btn_add_plot)
        action_buttons_layout.addWidget(self.btn_update_plot)
        action_buttons_layout.addWidget(self.btn_add_multi_wafer)
        
        parent_layout.addLayout(action_buttons_layout)
    
    def setup_connections(self):
        """Connect signals and slots."""
        vm = self.vm
        
        # DataFrame management
        self.df_listbox.itemSelectionChanged.connect(self._on_df_selected)
        self.btn_view_df.clicked.connect(self._on_view_df)
        self.btn_remove_df.clicked.connect(self._on_remove_df)
        self.btn_save_df.clicked.connect(self._on_save_df)
        
        # Filter connections
        self.v_data_filter.apply_requested.connect(self._on_apply_filters)
        
        # Slot selector connections
        self.select_all_checkbox.stateChanged.connect(self._on_select_all_slots)
        self.btn_add_multi_wafer.clicked.connect(self._on_plot_multi_wafer)
        
        # Plot buttons
        self.btn_add_plot.clicked.connect(self._on_add_plot)
        self.btn_update_plot.clicked.connect(self._on_update_plot)
        
        # Limits buttons
        self.btn_set_limits.clicked.connect(self._on_set_current_limits)
        self.btn_clear_limits.clicked.connect(self._on_clear_limits)
        
        # Plot style connection
        self.cbb_plot_style.currentTextChanged.connect(self._on_plot_style_changed)
        
        # Bottom toolbar connections
        self.cbb_graph_list.currentIndexChanged.connect(self._on_graph_selected_toolbar)
        self.btn_minimize_all.clicked.connect(self._on_minimize_all)
        self.btn_delete_all.clicked.connect(self._on_delete_all)
        self.spin_dpi_toolbar.valueChanged.connect(self._on_dpi_changed_toolbar)
        self.spin_xlabel_rotation.valueChanged.connect(self._on_xlabel_rotation_changed)
        self.cb_legend_outside_toolbar.stateChanged.connect(self._on_legend_outside_changed_toolbar)
        self.cbb_legend_loc_toolbar.currentTextChanged.connect(self._on_legend_loc_changed_toolbar)
        self.cb_grid_toolbar.stateChanged.connect(self._on_grid_changed_toolbar)
        
        # MDI area connections
        self.mdi_area.subWindowActivated.connect(self._on_subwindow_activated)
        
        # ViewModel → View
        vm.dataframes_changed.connect(self._update_df_list)
        vm.dataframe_columns_changed.connect(self._update_column_combos)
        vm.dataframe_columns_changed.connect(self._update_slot_selector)
        vm.graphs_changed.connect(self._update_graph_list)
        vm.notify.connect(self._show_toast_notification)
    
    def _on_plot_style_changed(self, plot_style: str):
        """Handle plot style change."""
        if plot_style == 'wafer':
            # Find and set X and Y columns if they exist
            x_index = self.cbb_x.findText('X')
            y_index = self.cbb_y.findText('Y')
            
            if x_index >= 0:
                self.cbb_x.setCurrentIndex(x_index)
            if y_index >= 0:
                self.cbb_y.setCurrentIndex(y_index)
    
    def _on_df_selected(self):
        """Handle DataFrame selection."""
        current_item = self.df_listbox.currentItem()
        if current_item:
            df_name = current_item.text()
            self.vm.select_dataframe(df_name)
    
    def _on_view_df(self):
        """View DataFrame in table."""
        current_item = self.df_listbox.currentItem()
        if not current_item:
            QMessageBox.warning(self, "No DataFrame", "Please select a DataFrame to view.")
            return
        
        df_name = current_item.text()
        df = self.vm.get_dataframe(df_name)
        
        if df is None or df.empty:
            QMessageBox.warning(self, "Empty DataFrame", "Selected DataFrame is empty.")
            return
        
        # Apply current filters if any
        filters = self.v_data_filter.get_filters()
        active_filters = [f for f in filters if f.get("state", False)]
        
        if active_filters:
            # Apply filters to get filtered DataFrame
            df = self.vm.apply_filters(df_name, filters)
            if df is None or df.empty:
                QMessageBox.warning(self, "Empty Result", "No data after applying filters.")
                return
        
        # Create dialog to show DataFrame
        dialog = QDialog(self)
        title = f"DataFrame: {df_name}"
        if active_filters:
            title += f" (with {len(active_filters)} filter(s))"
        dialog.setWindowTitle(title)
        dialog.resize(800, 600)
        
        # Create layout for dialog
        layout = QVBoxLayout()
        
        # Create and show DataFrame table
        table_view = VDataframeTable(layout)
        table_view.show(df, fill_colors=False)  # No color coding for regular DataFrames
        
        dialog.setLayout(layout)
        dialog.exec()
    
    def _on_remove_df(self):
        """Remove DataFrame."""
        current_item = self.df_listbox.currentItem()
        if current_item:
            df_name = current_item.text()
            self.vm.remove_dataframe(df_name)
    
    def _on_save_df(self):
        """Save DataFrame to Excel."""
        current_item = self.df_listbox.currentItem()
        if current_item:
            df_name = current_item.text()
            self.vm.save_dataframe_to_excel(df_name)
    
    def _on_apply_filters(self):
        """Apply filters."""
        if not self.vm.selected_df_name:
            return
        
        filters = self.v_data_filter.get_filters()
        self.vm.apply_filters(self.vm.selected_df_name, filters)
    
    def _show_toast_notification(self, message: str):
        """Show toast notification."""
        show_toast_notification(
            parent=self,
            message=message,
            duration=3000
        )
    
    def _on_add_plot(self):
        """Add new plot."""
        # Capture filters FIRST before any GUI state changes
        current_filters = self.v_data_filter.get_filters()
        
        if not self._validate_plot_request():
            return
        
        plot_config = self._collect_plot_config()
        if not plot_config['x'] or not plot_config['y']:
            QMessageBox.warning(self, "Missing Axes", "Please select X and Y axes.")
            return
        
        self._create_and_display_plot(plot_config, filters=current_filters)
    
    def _validate_plot_request(self) -> bool:
        """Validate DataFrame selection."""
        if not self.vm.selected_df_name:
            QMessageBox.warning(self, "No DataFrame", "Please select a DataFrame first.")
            return False
        
        df = self.vm.get_dataframe(self.vm.selected_df_name)
        if df is None or df.empty:
            QMessageBox.warning(self, "Empty DataFrame", "Selected DataFrame is empty.")
            return False
        return True
    
    def _create_and_display_plot(self, plot_config: dict, select_in_list: bool = True, filters: list = None):
        """Create and display a plot from configuration."""
        # Use provided filters or get current ones
        if filters is None:
            filters = self.v_data_filter.get_filters()
        
        graph_model = self.vm.create_graph(plot_config)
        filtered_df = self.vm.apply_filters(self.vm.selected_df_name, filters)
        
        graph_widget = VGraph(graph_id=graph_model.graph_id)
        self._configure_graph_from_model(graph_widget, graph_model)
        graph_widget.create_plot_widget(graph_model.dpi)
        
        self._render_plot(graph_widget, filtered_df, graph_model)
        
        self.vm.update_graph(graph_model.graph_id, {'legend_properties': graph_widget.legend_properties})
        
        sub_window = self._create_mdi_subwindow(graph_widget, graph_model)
        graph_dialog = self._wrap_graph_in_dialog(graph_widget)
        sub_window.setWidget(graph_dialog)
        
        self.graph_widgets[graph_model.graph_id] = (graph_widget, graph_dialog, sub_window)
        self.mdi_area.addSubWindow(sub_window)
        sub_window.show()
        
        self._update_graph_list(self.vm.get_graph_ids())
        
        if select_in_list:
            for i in range(self.cbb_graph_list.count()):
                if self.cbb_graph_list.itemData(i) == graph_model.graph_id:
                    self.cbb_graph_list.setCurrentIndex(i)
                    break
    
    def _wrap_graph_in_dialog(self, graph_widget: VGraph) -> QDialog:
        """Wrap graph widget in dialog."""
        from PySide6.QtWidgets import QDialog, QVBoxLayout
        graph_dialog = QDialog(self)
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(graph_widget)
        graph_dialog.setLayout(layout)
        return graph_dialog
    
    def create_plot_from_config(self, df_name: str, plot_config: dict) -> bool:
        """Create plot from configuration."""
        if self.vm.get_dataframe(df_name) is None:
            return False
        
        self.vm.select_dataframe(df_name)
        self._create_and_display_plot(plot_config, select_in_list=False)
        return True
    
    def _on_update_plot(self):
        """Update selected plot."""
        # Get currently active MDI subwindow
        active_subwindow = self.mdi_area.activeSubWindow()
        if not active_subwindow:
            QMessageBox.warning(self, "No Plot Selected", "Please select a plot to update.")
            return
        
        # Find the corresponding graph
        graph_widget = None
        graph_model = None
        for gid, (gw, gd, sw) in self.graph_widgets.items():
            if sw == active_subwindow:
                graph_widget = gw
                graph_model = self.vm.get_graph(gid)
                break
        
        if not graph_widget or not graph_model:
            return
        
        # Collect updated plot properties from GUI
        plot_config = self._collect_plot_config()
        
        # Check if Z-axis has changed (reset legend properties if so)
        if plot_config['z'] != graph_model.z:
            graph_widget.legend_properties = []
        
        # Check if filters have changed
        current_filters = self.v_data_filter.get_filters()
        if current_filters != graph_model.filters:
            graph_widget.legend_properties = []
        
        # Update graph model
        self.vm.update_graph(graph_model.graph_id, plot_config)
        
        # Get updated model
        graph_model = self.vm.get_graph(graph_model.graph_id)
        
        # Apply filters
        filtered_df = self.vm.apply_filters(self.vm.selected_df_name, current_filters)
        
        # Reconfigure and re-render
        self._configure_graph_from_model(graph_widget, graph_model)
        graph_widget.create_plot_widget(graph_model.dpi)

        
        self._render_plot(graph_widget, filtered_df, graph_model)
        
        # Save legend properties back to model after rendering
        self.vm.update_graph(graph_model.graph_id, {
            'legend_properties': graph_widget.legend_properties
        })
        
        # Update window title
        title = f"{graph_model.graph_id}-{graph_model.plot_style}: [{graph_model.x}] vs [{graph_model.y[0] if graph_model.y else 'None'}]"
        if graph_model.z:
            title += f" - [{graph_model.z}]"
        active_subwindow.setWindowTitle(title)
    
    def _update_df_list(self, df_names: list):
        """Update DataFrame list."""
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
    
    def _update_slot_selector(self, columns: list):
        """Update slot selector."""
        # Clear existing checkboxes
        for cb in self.slot_checkboxes:
            cb.deleteLater()
        self.slot_checkboxes.clear()
        
        # Clear grid layout
        while self.slot_grid_layout.count():
            item = self.slot_grid_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        # Check if selected DataFrame has Slot column
        if not self.vm.selected_df_name:
            self.slot_selector_widget.setVisible(False)
            self.btn_add_multi_wafer.setVisible(False)
            return
        
        if not self.vm.has_slot_column(self.vm.selected_df_name):
            self.slot_selector_widget.setVisible(False)
            self.btn_add_multi_wafer.setVisible(False)
            return
        
        # Get unique slots
        unique_slots = self.vm.get_unique_slots(self.vm.selected_df_name)
        
        if not unique_slots:
            self.slot_selector_widget.setVisible(False)
            self.btn_add_multi_wafer.setVisible(False)
            return
        
        # Show slot selector and multi-wafer button
        self.slot_selector_widget.setVisible(True)
        self.select_all_checkbox.setVisible(True)
        self.btn_add_multi_wafer.setVisible(True)
        
        # Create checkboxes in grid (9 columns max)
        row, col = 0, 0
        for slot in unique_slots:
            cb = QCheckBox(str(slot))
            cb.setChecked(True)
            cb.stateChanged.connect(self._on_slot_checkbox_changed)
            self.slot_grid_layout.addWidget(cb, row, col)
            self.slot_checkboxes.append(cb)
            
            col += 1
            if col >= 8:
                col = 0
                row += 1
    
    def _on_select_all_slots(self, state):
        """Toggle all slots."""
        checked = bool(state)  # Convert Qt state to boolean (0=False, 2=True)
        for cb in self.slot_checkboxes:
            cb.blockSignals(True)
            cb.setChecked(checked)
            cb.blockSignals(False)
    
    def _on_slot_checkbox_changed(self):
        """Update select all checkbox."""
        all_checked = all(cb.isChecked() for cb in self.slot_checkboxes)
        self.select_all_checkbox.blockSignals(True)
        self.select_all_checkbox.setChecked(all_checked)
        self.select_all_checkbox.blockSignals(False)
    
    def _on_plot_multi_wafer(self):
        """Create multi-wafer plots."""
        # Capture filters FIRST before any GUI state changes
        current_filters = self.v_data_filter.get_filters()
        
        checked_slots = [int(cb.text()) for cb in self.slot_checkboxes if cb.isChecked()]
        
        if not checked_slots:
            QMessageBox.warning(self, "No Slots Selected", "Please select at least one slot.")
            return
        
        if not self._validate_plot_request():
            return
        
        # Force wafer plot style
        wafer_index = self.cbb_plot_style.findText('wafer')
        if wafer_index >= 0:
            self.cbb_plot_style.setCurrentIndex(wafer_index)
        
        plot_config = self._collect_plot_config()
        plot_config['plot_style'] = 'wafer'
        
        created_graphs = self.vm.create_multi_wafer_graphs(
            self.vm.selected_df_name,
            checked_slots,
            plot_config,
            current_filters
        )
        
        for graph_model in created_graphs:
            filtered_df = self.vm.apply_filters(self.vm.selected_df_name, graph_model.filters)
            
            # Create Graph widget
            graph_widget = VGraph(graph_id=graph_model.graph_id)
            self._configure_graph_from_model(graph_widget, graph_model)
            
            # Create plot
            graph_widget.create_plot_widget(graph_model.dpi)

            
            self._render_plot(graph_widget, filtered_df, graph_model)
            
            self.vm.update_graph(graph_model.graph_id, {'legend_properties': graph_widget.legend_properties})
            
            sub_window = self._create_mdi_subwindow(graph_widget, graph_model)
            graph_dialog = self._wrap_graph_in_dialog(graph_widget)
            sub_window.setWidget(graph_dialog)
            
            self.graph_widgets[graph_model.graph_id] = (graph_widget, graph_dialog, sub_window)
            self.mdi_area.addSubWindow(sub_window)
            sub_window.show()
        
        # Update graph list
        self._update_graph_list(self.vm.get_graph_ids())
        
        self.vm.notify.emit(f"Created {len(created_graphs)} wafer plots")
    
    def _update_graph_list(self, graph_ids: list):
        """Update graph list."""
        self.cbb_graph_list.clear()
        for gid in graph_ids:
            graph = self.vm.get_graph(gid)
            if graph:
                display_name = graph.get_display_name()
                self.cbb_graph_list.addItem(display_name, gid)
    
    def _on_graph_selected_toolbar(self):
        """Handle graph selection."""
        index = self.cbb_graph_list.currentIndex()
        if index >= 0:
            graph_id = self.cbb_graph_list.currentData()
            if graph_id and graph_id in self.graph_widgets:
                # Activate the corresponding MDI subwindow
                _, _, sub_window = self.graph_widgets[graph_id]
                self.mdi_area.setActiveSubWindow(sub_window)
                sub_window.showNormal()  # Restore if minimized
    
    def _on_minimize_all(self):
        """Minimize all windows."""
        for window in self.mdi_area.subWindowList():
            window.showMinimized()
    
    def _on_delete_all(self):
        """Delete all graphs."""
        # Confirm with user
        reply = QMessageBox.question(
            self,
            "Delete All Graphs",
            "Are you sure you want to delete all graphs?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # Close all MDI subwindows
            for sub_window in self.mdi_area.subWindowList():
                self.mdi_area.removeSubWindow(sub_window)
                sub_window.close()
            
            # Delete graphs from ViewModel
            for graph_id in list(self.graph_widgets.keys()):
                self.vm.delete_graph(graph_id)
            
            # Clear graph widgets storage
            self.graph_widgets.clear()
            
            # Update graph list combobox
            self._update_graph_list([])
    
    def _on_set_current_limits(self):
        """Get current axis limits."""
        active_subwindow = self.mdi_area.activeSubWindow()
        if not active_subwindow:
            QMessageBox.warning(self, "No Plot Selected", "Please select a plot first.")
            return
        
        # Find the corresponding graph
        for gid, (gw, gd, sw) in self.graph_widgets.items():
            if sw == active_subwindow:
                try:
                    # Get current limits from matplotlib axes
                    xmin, xmax = gw.ax.get_xlim()
                    ymin, ymax = gw.ax.get_ylim()
                    
                    # Update spinboxes only - model will be updated when "Update plot" is clicked
                    self.spin_xmin.setValue(round(xmin, 3))
                    self.spin_xmax.setValue(round(xmax, 3))
                    self.spin_ymin.setValue(round(ymin, 3))
                    self.spin_ymax.setValue(round(ymax, 3))
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Error getting limits: {str(e)}")
                break
    
    def _on_clear_limits(self):
        """Clear axis limits."""
        self.spin_xmin.setValue(-999999)
        self.spin_xmax.setValue(-999999)
        self.spin_ymin.setValue(-999999)
        self.spin_ymax.setValue(-999999)
        self.spin_zmin.setValue(-999999)
        self.spin_zmax.setValue(-999999)
    
    def _on_dpi_changed_toolbar(self, value: int):
        """Handle DPI change from toolbar (will apply on Update plot)."""
        # Don't update immediately - wait for user to click "Update plot"
        pass
    
    def _on_xlabel_rotation_changed(self, value: int):
        """Handle X label rotation change (will apply on Update plot)."""
        # Don't update immediately - wait for user to click "Update plot"
        pass
    
    def _on_legend_outside_changed_toolbar(self, state: int):
        """Handle legend outside toggle from toolbar (will apply on Update plot)."""
        # Don't update immediately - wait for user to click "Update plot"
        pass
    
    def _on_legend_loc_changed_toolbar(self, location: str):
        """Handle legend location change from toolbar (will apply on Update plot)."""
        # Don't update immediately - wait for user to click "Update plot"
        pass
    
    def _on_grid_changed_toolbar(self, state: int):
        """Handle grid toggle from toolbar (will apply on Update plot)."""
        # Don't update immediately - wait for user to click "Update plot"
        pass
    
    def _on_subwindow_activated(self, sub_window):
        """Handle subwindow activation."""
        if not sub_window:
            return
        
        # Find the corresponding graph
        graph_widget = None
        graph_model = None
        for gid, (gw, gd, sw) in self.graph_widgets.items():
            if sw == sub_window:
                graph_widget = gw
                graph_model = self.vm.get_graph(gid)
                break
        
        if not graph_widget or not graph_model:
            return
        
        # Update plot size label
        size = sub_window.size()
        self.lbl_plot_size.setText(f"({size.width()}x{size.height()})")
        
        # Sync GUI controls with graph properties
        self._sync_gui_from_graph(graph_model)
        
        # Update graph list combobox selection (block signals to prevent conflicts)
        self.cbb_graph_list.blockSignals(True)
        for i in range(self.cbb_graph_list.count()):
            if self.cbb_graph_list.itemData(i) == graph_model.graph_id:
                self.cbb_graph_list.setCurrentIndex(i)
                break
        self.cbb_graph_list.blockSignals(False)
    
    def _sync_gui_from_graph(self, model):
        """Sync GUI from graph model."""
        self.cbb_plot_style.blockSignals(True)
        self.cbb_x.blockSignals(True)
        self.cbb_y.blockSignals(True)
        self.cbb_z.blockSignals(True)
        
        try:
            # Combos
            for cbb_name, value in [('plot_style', model.plot_style), ('x', model.x), ('z', model.z or "None")]:
                cbb = getattr(self, f'cbb_{cbb_name}')
                idx = cbb.findText(value)
                if idx >= 0:
                    cbb.setCurrentIndex(idx)
            
            if model.y:
                idx = self.cbb_y.findText(model.y[0])
                if idx >= 0:
                    self.cbb_y.setCurrentIndex(idx)
            
            # Checkboxes
            self.cb_xlog.setChecked(model.xlogscale)
            self.cb_ylog.setChecked(model.ylogscale)
            self.cb_legend_outside_toolbar.setChecked(model.legend_outside)
            self.cb_grid_toolbar.setChecked(model.grid)
            self.cb_error_bar.setChecked(model.show_bar_plot_error_bar)
            self.cb_wafer_stats.setChecked(model.wafer_stats)
            self.cb_join_point_plot.setChecked(model.join_for_point_plot)
            self.cb_trendline_eq.setChecked(model.show_trendline_eq)
            
            # Text inputs
            self.edit_plot_title.setText(model.plot_title or "")
            self.edit_xlabel.setText(model.xlabel or "")
            self.edit_ylabel.setText(model.ylabel or "")
            self.edit_zlabel.setText(model.zlabel or "")
            
            # Limits
            self.spin_xmin.setValue(model.xmin if model.xmin is not None else -999999)
            self.spin_xmax.setValue(model.xmax if model.xmax is not None else -999999)
            self.spin_ymin.setValue(model.ymin if model.ymin is not None else -999999)
            self.spin_ymax.setValue(model.ymax if model.ymax is not None else -999999)
            self.spin_zmin.setValue(model.zmin if model.zmin is not None else -999999)
            self.spin_zmax.setValue(model.zmax if model.zmax is not None else -999999)
            
            # Toolbar controls
            self.spin_dpi_toolbar.setValue(model.dpi)
            self.spin_xlabel_rotation.setValue(model.x_rot)
            self.spin_trendline_order.setValue(model.trendline_order)
            
            # Legend location
            idx = self.cbb_legend_loc_toolbar.findText(model.legend_location)
            if idx >= 0:
                self.cbb_legend_loc_toolbar.setCurrentIndex(idx)
            
            # Filters
            self.v_data_filter.set_filters(model.filters)
        finally:
            self.cbb_plot_style.blockSignals(False)
            self.cbb_x.blockSignals(False)
            self.cbb_y.blockSignals(False)
            self.cbb_z.blockSignals(False)
    
    # ═════════════════════════════════════════════════════════════════════
    # Phase 2: Plotting Helper Methods
    # ═════════════════════════════════════════════════════════════════════
    
    def _collect_plot_config(self) -> dict:
        """Collect plot configuration."""
        plot_style = self.cbb_plot_style.currentText()
        z_value = self.cbb_z.currentText() if self.cbb_z.currentText() != "None" else None
        
        # Only use color palette for wafer/2D maps or when Z axis is selected
        use_palette = plot_style in ['wafer', '2Dmap'] or z_value is not None
        
        return {
            'df_name': self.vm.selected_df_name,
            'plot_style': plot_style,
            'x': self.cbb_x.currentText(),
            'y': [self.cbb_y.currentText()],
            'z': z_value,
            'xlogscale': self.cb_xlog.isChecked(),
            'ylogscale': self.cb_ylog.isChecked(),
            'plot_title': self.edit_plot_title.text() or None,
            'xlabel': self.edit_xlabel.text() or None,
            'ylabel': self.edit_ylabel.text() or None,
            'zlabel': self.edit_zlabel.text() or None,
            'xmin': self.spin_xmin.value() if self.spin_xmin.value() != -999999 else None,
            'xmax': self.spin_xmax.value() if self.spin_xmax.value() != -999999 else None,
            'ymin': self.spin_ymin.value() if self.spin_ymin.value() != -999999 else None,
            'ymax': self.spin_ymax.value() if self.spin_ymax.value() != -999999 else None,
            'zmin': self.spin_zmin.value() if self.spin_zmin.value() != -999999 else None,
            'zmax': self.spin_zmax.value() if self.spin_zmax.value() != -999999 else None,
            'color_palette': self.cbb_colormap.get_selected_palette() if use_palette else 'jet',
            'wafer_size': float(self.cbb_wafer_size.currentText()),
            'wafer_stats': self.cb_wafer_stats.isChecked(),
            'dpi': self.spin_dpi_toolbar.value(),
            'x_rot': self.spin_xlabel_rotation.value(),
            'legend_visible': True,  # From More Options tab when implemented
            'legend_location': self.cbb_legend_loc_toolbar.currentText(),
            'legend_outside': self.cb_legend_outside_toolbar.isChecked(),
            'grid': self.cb_grid_toolbar.isChecked(),
            'show_bar_plot_error_bar': self.cb_error_bar.isChecked(),
            'show_trendline_eq': self.cb_trendline_eq.isChecked(),
            'trendline_order': self.spin_trendline_order.value(),
            'join_for_point_plot': self.cb_join_point_plot.isChecked(),
            'filters': self.v_data_filter.get_filters()
        }
    
    def _configure_graph_from_model(self, graph_widget: VGraph, model):
        """Configure graph widget."""
        # Data source
        graph_widget.df_name = model.df_name
        graph_widget.filters = model.filters
        
        # Plot style and dimensions
        graph_widget.plot_style = model.plot_style
        graph_widget.plot_width = model.plot_width
        graph_widget.plot_height = model.plot_height
        graph_widget.dpi = model.dpi
        
        # Axes
        graph_widget.x = model.x
        graph_widget.y = model.y.copy() if model.y else []
        graph_widget.z = model.z
        
        # Axis limits
        graph_widget.xmin = model.xmin
        graph_widget.xmax = model.xmax
        graph_widget.ymin = model.ymin
        graph_widget.ymax = model.ymax
        graph_widget.zmin = model.zmin
        graph_widget.zmax = model.zmax
        
        # Axis scales
        graph_widget.xlogscale = model.xlogscale
        graph_widget.ylogscale = model.ylogscale
        
        # Labels
        graph_widget.plot_title = model.plot_title
        graph_widget.xlabel = model.xlabel
        graph_widget.ylabel = model.ylabel
        graph_widget.zlabel = model.zlabel
        
        # Visual properties
        graph_widget.x_rot = model.x_rot
        graph_widget.grid = model.grid
        
        # Legend
        graph_widget.legend_visible = model.legend_visible
        graph_widget.legend_location = model.legend_location
        graph_widget.legend_outside = model.legend_outside
        graph_widget.legend_properties = model.legend_properties.copy() if model.legend_properties else []
        
        # Plot-specific
        graph_widget.color_palette = model.color_palette
        graph_widget.wafer_size = model.wafer_size
        graph_widget.wafer_stats = model.wafer_stats
        graph_widget.trendline_order = model.trendline_order
        graph_widget.show_trendline_eq = model.show_trendline_eq
        graph_widget.show_bar_plot_error_bar = model.show_bar_plot_error_bar
        graph_widget.join_for_point_plot = model.join_for_point_plot
    
    def _render_plot(self, graph_widget: VGraph, filtered_df, model):
        """Render plot."""
        try:
            # The Graph class has a single plot() method that handles all plot types
            graph_widget.plot(filtered_df)
        except Exception as e:
            QMessageBox.critical(self, "Plot Error", 
                               f"Error rendering plot: {str(e)}")
            print(f"Plot rendering error: {e}")
    
    def _create_mdi_subwindow(self, graph_widget: VGraph, model) -> QMdiSubWindow:
        """Create MDI subwindow."""
        
        sub_window = MdiSubWindow(
            graph_id=model.graph_id,
            figsize_label=self.lbl_plot_size,
            mdi_area=self.mdi_area
        )
        
        # Set window title
        title = f"{model.graph_id}-{model.plot_style}: [{model.x}] vs [{model.y[0] if model.y else 'None'}]"
        if model.z:
            title += f" - [{model.z}]"
        sub_window.setWindowTitle(title)
        
        # Set size
        sub_window.resize(model.plot_width, model.plot_height)
        
        # Connect close signal to cleanup
        sub_window.closed.connect(lambda gid=model.graph_id: self._on_graph_closed(gid))
        
        return sub_window
    
    def _on_graph_closed(self, graph_id: int):
        """Handle graph closing."""
        # Remove from storage
        if graph_id in self.graph_widgets:
            del self.graph_widgets[graph_id]
        
        # Remove from ViewModel
        self.vm.delete_graph(graph_id)
        
        # Update graph list
        self._update_graph_list(self.vm.get_graph_ids())
    
    # ═════════════════════════════════════════════════════════════════════
    # Workspace Management
    # ═════════════════════════════════════════════════════════════════════
    
    def save_workspace(self):
        """Save workspace."""
        # Update all graph models with current state before saving
        for gid, (gw, gd, sw) in self.graph_widgets.items():
            size = sw.size()
            self.vm.update_graph(gid, {
                'plot_width': size.width(),
                'plot_height': size.height(),
                'legend_properties': gw.legend_properties,
                'legend_visible': gw.legend_visible,
                'legend_location': gw.legend_location,
                'legend_outside': gw.legend_outside
            })
        
        # Save workspace
        self.vm.save_workspace()
    
    def load_workspace(self, file_path: str):
        """Load workspace."""
        self.clear_workspace()
        
        # Load data into ViewModel
        self.vm.load_workspace(file_path)
        
        # Select first DataFrame if available
        if self.df_listbox.count() > 0:
            self.df_listbox.setCurrentRow(0)
        
        # Recreate graph widgets and MDI subwindows for each loaded graph
        for graph_id in self.vm.get_graph_ids():
            graph_model = self.vm.get_graph(graph_id)
            if not graph_model:
                continue
            
            filtered_df = self.vm.apply_filters(graph_model.df_name, graph_model.filters)
            graph_widget = VGraph(graph_id=graph_model.graph_id)
            self._configure_graph_from_model(graph_widget, graph_model)
            graph_widget.create_plot_widget(graph_model.dpi)
            self._render_plot(graph_widget, filtered_df, graph_model)
            
            sub_window = self._create_mdi_subwindow(graph_widget, graph_model)
            graph_dialog = self._wrap_graph_in_dialog(graph_widget)
            sub_window.setWidget(graph_dialog)
            
            self.graph_widgets[graph_model.graph_id] = (graph_widget, graph_dialog, sub_window)
            self.mdi_area.addSubWindow(sub_window)
            sub_window.show()
    
    def clear_workspace(self):
        """Clear workspace."""
        # Close and remove all MDI subwindows
        for sub_window in self.mdi_area.subWindowList():
            sub_window.close()
            self.mdi_area.removeSubWindow(sub_window)
        
        self.graph_widgets.clear()
        self.df_listbox.clear()
        self.cbb_x.clear()
        self.cbb_y.clear()
        self.cbb_z.clear()
        self.cbb_graph_list.clear()
        self._on_clear_limits()
        self.edit_plot_title.clear()
        self.edit_xlabel.clear()
        self.edit_ylabel.clear()
        self.edit_zlabel.clear()
        self.lbl_plot_size.setText("(480x420)")
        self.vm.clear_workspace()

class MdiSubWindow(QMdiSubWindow):
    """Custom MDI subwindow."""
    closed = Signal(int)

    def __init__(self, graph_id, figsize_label, mdi_area, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.graph_id = graph_id
        self.figsize_label = figsize_label
        self.mdi_area = mdi_area  # Reference to the parent QMdiArea

    def closeEvent(self, event):
        """Override close event."""
        self.mdi_area.clearFocus()
        self.mdi_area.setActiveSubWindow(None)
        self.closed.emit(self.graph_id)
        super().closeEvent(event)

    def resizeEvent(self, event):
        """Override resize event."""
        new_size = self.size()
        self.figsize_label.setText(f"({new_size.width()}x{new_size.height()})")
        super().resizeEvent(event)

    def focusInEvent(self, event):
        """Override focus event."""
        if not self.mdi_area.activeSubWindow():
            self.mdi_area.setActiveSubWindow(None)
        super().focusInEvent(event)