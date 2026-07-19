"""View for Graphs Workspace - main UI coordinator for graph plotting and visualization."""
import os
import copy
from typing import Optional
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QListWidget,
    QComboBox, QSpinBox, QCheckBox, QLineEdit, QSplitter,
    QMdiArea, QMdiSubWindow, QTabWidget, QGroupBox, QMessageBox, QFrame, QScrollArea,
    QDialog, QGridLayout, QApplication, QListWidgetItem, QCompleter, QInputDialog
)
from PySide6.QtCore import Qt, Signal, QSize, QUrl
from PySide6.QtGui import QIcon, QDesktopServices, QBrush, QFont, QShortcut, QKeySequence, QPalette

from spectroview import ICON_DIR, PLOT_STYLES, AXIS_LABELS
from spectroview.model.m_settings import MSettings
from spectroview.model.m_plot_recipe_store import MPlotRecipeStore
from spectroview.model.m_style_template_store import MStyleTemplateStore
from spectroview.model.graph_style import extract_style, apply_style_dict, default_style
from spectroview.view.components.v_data_filter import VDataFilter
from spectroview.view.components.v_plot_recipe_dialog import VPlotRecipeDialog
from spectroview.view.components.v_style_template_dialog import VStyleTemplateDialog
from spectroview.view.components.v_dataframe_table import VDataframeTable
from spectroview.view.components.v_graph import VGraph
from spectroview.viewmodel.vm_workspace_graphs import VMWorkspaceGraphs
from spectroview.viewmodel.utils import show_toast_notification, get_tinted_icon
from spectroview.view.components.customized_widgets import CustomizedPalette
from spectroview.view.components.customize_graph.customize_graph_dialog import CustomizeGraphDialog
from spectroview.view.components.v_export_dialog import VExportDialog, VBatchExportDialog
from spectroview.view.components.v_multipanel_dialog import VMultiPanelDialog
from spectroview.view.components.graph_commit import snapshot, diff

class VWorkspaceGraphs(QWidget):
    """View for Graphs Workspace."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.m_settings = MSettings()
        self.vm = VMWorkspaceGraphs(self.m_settings)

        # Just a valid starting store; every handler that actually opens or
        # saves to one rebuilds it from the current setting first
        # (_refresh_recipe_and_style_stores()), or a Working Folder
        # configured after this widget is built would never take effect.
        self.recipe_store = MPlotRecipeStore(self.m_settings.get_plot_recipe_folder())
        self.style_template_store = MStyleTemplateStore(self.m_settings.get_plot_style_folder())

        # Session-level "copied style" clipboard for Copy Style/Paste Style
        # (not persisted -- for a reusable named style, see "Save Style"
        # instead, backed by style_template_store above).
        self._copied_style: Optional[dict] = None

        # Graph storage: {graph_id: (Graph widget, QDialog, QMdiSubWindow)}
        self.graph_widgets = {}
        
        # Singleton customize dialog
        self._customize_dialog = None
        
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
        self.mdi_area.setBackground(QBrush(Qt.transparent))
        self.mdi_area.setStyleSheet("QMdiArea { border: 1px solid rgba(128, 128, 128, 0.2); border-radius: 6px; }")
        left_layout.addWidget(self.mdi_area)
        
        # Bottom Toolbar
        bottom_toolbar = self._create_bottom_toolbar()
        left_layout.addWidget(bottom_toolbar)
        
        return left_panel
    
    def _create_bottom_toolbar(self):
        """Create bottom toolbar."""
        bottom_toolbar = QFrame()
        bottom_toolbar.setObjectName("bottomToolbarPanel")
        bottom_toolbar.setFrameShape(QFrame.StyledPanel)
        bottom_toolbar.setMaximumHeight(44)
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
        self.lbl_plot_size = QLabel("(480x420)")
        self.lbl_plot_size.setMinimumWidth(70)
        toolbar_layout.addWidget(self.lbl_plot_size)
        
        # X label rotation
        toolbar_layout.addWidget(QLabel("X label rotation:"))
        self.spin_xlabel_rotation = QSpinBox()
        self.spin_xlabel_rotation.setRange(0, 90)
        self.spin_xlabel_rotation.setValue(0)
        self.spin_xlabel_rotation.setSingleStep(10)
        self.spin_xlabel_rotation.setMaximumWidth(60)
        toolbar_layout.addWidget(self.spin_xlabel_rotation)
        
        # Grid checkbox
        self.cb_grid_toolbar = QCheckBox("Grid")
        toolbar_layout.addWidget(self.cb_grid_toolbar)
        
        toolbar_layout.addStretch()
        
        return bottom_toolbar
    
    def _setup_right_panel(self):
        """Setup right panel."""
        right_panel = QFrame()
        right_panel.setObjectName("workspaceRightPanel")
        right_panel.setFrameShape(QFrame.NoFrame)
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
        group_box = QGroupBox("Loaded dataframes:")
        group_layout = QVBoxLayout(group_box)
        group_layout.setContentsMargins(2, 0, 2, 0)
        
        # DataFrame listbox and buttons side by side
        df_section_layout = QHBoxLayout()
        df_section_layout.setContentsMargins(0, 0, 0, 0)
        
        # DataFrame listbox
        self.df_listbox = QListWidget()
        self.df_listbox.setMaximumHeight(110)
        # self.df_listbox.setAcceptDrops(True)  # Disabled to allow global drop
        
        # Connect selection change
        
        self._has_df_placeholder = False  # Track placeholder state
        self._update_df_placeholder()  # Show placeholder initially
        
        df_section_layout.addWidget(self.df_listbox)
        
        # DataFrame buttons (vertical layout on the right)
        df_buttons_layout = QVBoxLayout()
        
        self.btn_view_df = QPushButton()
        self.btn_view_df.setIconSize(QSize(18, 18))
        self.btn_view_df.setToolTip("View DataFrame\nCtrl+Click: Open source file")
        self.btn_view_df.setMaximumWidth(35)
        self.btn_view_df.setMaximumHeight(22)
        
        self.btn_remove_df = QPushButton()
        self.btn_remove_df.setIconSize(QSize(18, 18))
        self.btn_remove_df.setToolTip("Remove DataFrame")
        self.btn_remove_df.setMaximumWidth(35)
        self.btn_remove_df.setMaximumHeight(22)
        
        self.btn_save_df = QPushButton()
        self.btn_save_df.setIconSize(QSize(18, 18))
        self.btn_save_df.setToolTip("Save DataFrame to Excel")
        self.btn_save_df.setMaximumWidth(35)
        self.btn_save_df.setMaximumHeight(22)
        
        self.btn_refresh_df = QPushButton()
        self.btn_refresh_df.setIconSize(QSize(18, 18))
        self.btn_refresh_df.setToolTip("Refresh DataFrame from source file")
        self.btn_refresh_df.setMaximumWidth(35)
        self.btn_refresh_df.setMaximumHeight(22)
        
        df_buttons_layout.addWidget(self.btn_view_df)
        df_buttons_layout.addWidget(self.btn_remove_df)
        df_buttons_layout.addWidget(self.btn_save_df)
        df_buttons_layout.addWidget(self.btn_refresh_df)
        df_buttons_layout.addStretch()
        
        df_section_layout.addLayout(df_buttons_layout)
        group_layout.addLayout(df_section_layout)
        parent_layout.addWidget(group_box)
    
    def _setup_filter_section(self, parent_layout):  
        """Setup filter section."""
        self.v_data_filter = VDataFilter()
        parent_layout.addWidget(self.v_data_filter, stretch=1.5)
    
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
        self.select_all_checkbox.setChecked(False)
        self.select_all_checkbox.setVisible(False)
        header_layout.addWidget(self.select_all_checkbox)
        header_layout.addStretch()
        
        slot_layout.addLayout(header_layout)
        
        # Grid layout for checkboxes
        self.slot_grid_layout = QGridLayout()
        self.slot_grid_layout.setSpacing(2)
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
        self.plot_tabs.addTab(more_options_tab, "Plot multiple axes (beta)")
        
        parent_layout.addWidget(self.plot_tabs, stretch=2)
    
    def _create_plot_tab(self):
        """Create plot tab."""
        tab_plot = QWidget()
        tab_plot_layout = QVBoxLayout(tab_plot)
        tab_plot_layout.setContentsMargins(4, 4, 4, 4)
        
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
        
        # Slot selector section (for wafer plots)
        self._setup_slot_selector_section(tab_plot_layout)
        
        tab_plot_layout.addStretch()
        
        return tab_plot
    
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
            
            if axis_name == 'z':
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
        group_layout.setContentsMargins(5, 6, 5, 6)
        group_layout.setSpacing(6)
        
        # Create all label inputs
        labels = [
            ('plot_title', 'Plot title:', 'Type to modify the plot title'),
            ('plot_subtitle', 'Subtitle:', 'Optional text shown under the title'),
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
            
            # Add QCompleter for x, y, z labels (not for plot_title)
            if attr_name in ['xlabel', 'ylabel', 'zlabel']:
                completer = QCompleter(AXIS_LABELS)
                completer.setCaseSensitivity(Qt.CaseInsensitive)
                completer.setCompletionMode(QCompleter.PopupCompletion)
                line_edit.setCompleter(completer)
            
            setattr(self, f'edit_{attr_name}', line_edit)
            h_layout.addWidget(line_edit)
            group_layout.addLayout(h_layout)
        
        parent_layout.addWidget(title_labels_group)
    

    
    def _create_more_options_tab(self):
        """Create the 'Plot multiple Axes' tab with Y2, Y3, and X2 controls."""
        tab_more = QWidget()
        tab_more_layout = QVBoxLayout(tab_more)
        tab_more_layout.setContentsMargins(4, 4, 4, 4)
        tab_more_layout.setSpacing(6)
        
        # ── Secondary Y axis ──
        y2_group = QGroupBox("Secondary Y axis (right, red):")
        y2_layout = QVBoxLayout(y2_group)
        y2_layout.setContentsMargins(5, 2, 5, 2)
        y2_layout.setSpacing(2)
        
        y2_col_layout = QHBoxLayout()
        y2_col_layout.addWidget(QLabel("Y2:"))
        self.cbb_y2 = QComboBox()
        self.cbb_y2.setFixedWidth(200)
        y2_col_layout.addWidget(self.cbb_y2)
        self.cb_y2log = QCheckBox("Log scale")
        y2_col_layout.addWidget(self.cb_y2log)
        y2_col_layout.addStretch()
        y2_layout.addLayout(y2_col_layout)
        
        y2_label_layout = QHBoxLayout()
        y2_label_layout.addWidget(QLabel("Y2 label:"))
        self.edit_y2label = QLineEdit()
        self.edit_y2label.setPlaceholderText("Y2 axis label")
        completer_y2 = QCompleter(AXIS_LABELS)
        completer_y2.setCaseSensitivity(Qt.CaseInsensitive)
        completer_y2.setCompletionMode(QCompleter.PopupCompletion)
        self.edit_y2label.setCompleter(completer_y2)
        y2_label_layout.addWidget(self.edit_y2label)
        y2_layout.addLayout(y2_label_layout)
        
        tab_more_layout.addWidget(y2_group)
        
        # ── Tertiary Y axis ──
        y3_group = QGroupBox("Tertiary Y axis (right offset, green):")
        y3_layout = QVBoxLayout(y3_group)
        y3_layout.setContentsMargins(5, 2, 5, 2)
        y3_layout.setSpacing(2)
        
        y3_col_layout = QHBoxLayout()
        y3_col_layout.addWidget(QLabel("Y3:"))
        self.cbb_y3 = QComboBox()
        self.cbb_y3.setFixedWidth(200)
        y3_col_layout.addWidget(self.cbb_y3)
        self.cb_y3log = QCheckBox("Log scale")
        y3_col_layout.addWidget(self.cb_y3log)
        y3_col_layout.addStretch()
        y3_layout.addLayout(y3_col_layout)
        
        y3_label_layout = QHBoxLayout()
        y3_label_layout.addWidget(QLabel("Y3 label:"))
        self.edit_y3label = QLineEdit()
        self.edit_y3label.setPlaceholderText("Y3 axis label")
        completer_y3 = QCompleter(AXIS_LABELS)
        completer_y3.setCaseSensitivity(Qt.CaseInsensitive)
        completer_y3.setCompletionMode(QCompleter.PopupCompletion)
        self.edit_y3label.setCompleter(completer_y3)
        y3_label_layout.addWidget(self.edit_y3label)
        y3_layout.addLayout(y3_label_layout)
        
        tab_more_layout.addWidget(y3_group)
        
        # ── Secondary X axis ──
        x2_group = QGroupBox("Secondary X axis (top, purple):")
        x2_layout = QVBoxLayout(x2_group)
        x2_layout.setContentsMargins(5, 2, 5, 2)
        x2_layout.setSpacing(2)
        
        x2_col_layout = QHBoxLayout()
        x2_col_layout.addWidget(QLabel("X2:"))
        self.cbb_x2 = QComboBox()
        self.cbb_x2.setFixedWidth(200)
        x2_col_layout.addWidget(self.cbb_x2)
        self.cb_x2log = QCheckBox("Log scale")
        x2_col_layout.addWidget(self.cb_x2log)
        x2_col_layout.addStretch()
        x2_layout.addLayout(x2_col_layout)
        
        x2_label_layout = QHBoxLayout()
        x2_label_layout.addWidget(QLabel("X2 label:"))
        self.edit_x2label = QLineEdit()
        self.edit_x2label.setPlaceholderText("X2 axis label")
        completer_x2 = QCompleter(AXIS_LABELS)
        completer_x2.setCaseSensitivity(Qt.CaseInsensitive)
        completer_x2.setCompletionMode(QCompleter.PopupCompletion)
        self.edit_x2label.setCompleter(completer_x2)
        x2_label_layout.addWidget(self.edit_x2label)
        x2_layout.addLayout(x2_label_layout)
        
        info_x2 = QLabel("Applicable plot styles: point, scatter, line")
        info_x2.setStyleSheet("color: gray; font-style: italic; font-size: 10px;")
        x2_layout.addWidget(info_x2)
        
        tab_more_layout.addWidget(x2_group)
        
        tab_more_layout.addStretch()
        
        return tab_more
    
    def _setup_action_buttons(self, parent_layout):
        """Setup action buttons."""
        action_buttons_layout = QHBoxLayout()
        
        self.btn_add_plot = QPushButton("Add plot")
        self.btn_add_plot.setIconSize(QSize(20, 20))
        self.btn_add_plot.setToolTip("Add new plot with current configuration")
        self.btn_add_plot.setMinimumHeight(25)
        
        self.btn_update_plot = QPushButton("Update plot")
        self.btn_update_plot.setIconSize(QSize(20, 20))
        self.btn_update_plot.setToolTip("Update selected plot with current configuration")
        self.btn_update_plot.setMinimumHeight(25)
        
        # Multi-wafer plot button (shown when Slot column exists)
        self.btn_add_multi_wafer = QPushButton("Add Multi-Wafer")
        self.btn_add_multi_wafer.setIconSize(QSize(20, 20))
        self.btn_add_multi_wafer.setToolTip("Create wafer plots for selected slots")
        self.btn_add_multi_wafer.setMinimumHeight(25)
        self.btn_add_multi_wafer.setVisible(False)  # Initially hidden

        action_buttons_layout.addWidget(self.btn_add_plot)
        action_buttons_layout.addWidget(self.btn_update_plot)
        action_buttons_layout.addWidget(self.btn_add_multi_wafer)

        parent_layout.addLayout(action_buttons_layout)

        # Plot recipe buttons — browse/apply and save-all, second row
        template_buttons_layout = QHBoxLayout()

        self.btn_apply_recipe = QPushButton("📊 Plot Recipes")
        self.btn_apply_recipe.setToolTip("Browse & apply saved plot recipes")
        self.btn_apply_recipe.setMinimumHeight(25)

        self.btn_save_as_recipe = QPushButton("💾 Save Plot Recipe")
        self.btn_save_as_recipe.setToolTip("Save all currently open plots as a Plot Recipe")
        self.btn_save_as_recipe.setMinimumHeight(25)

        self.btn_export_all = QPushButton()
        self.btn_export_all.setIcon(QIcon(os.path.join(ICON_DIR, "save-all.png")))
        self.btn_export_all.setText(" Export All")
        self.btn_export_all.setToolTip("Export every open graph to a folder")
        self.btn_export_all.setMinimumHeight(25)

        self.btn_compose_figure = QPushButton("🖼️ Compose Figure")
        self.btn_compose_figure.setToolTip("Combine several open graphs into one multi-panel exported figure")
        self.btn_compose_figure.setMinimumHeight(25)

        template_buttons_layout.addWidget(self.btn_apply_recipe)
        template_buttons_layout.addWidget(self.btn_save_as_recipe)
        template_buttons_layout.addWidget(self.btn_export_all)
        template_buttons_layout.addWidget(self.btn_compose_figure)

        parent_layout.addLayout(template_buttons_layout)

        # Undo/Redo -- third row, own line since they're used far more
        # often than the template/export actions above and deserve to
        # always be visible rather than competing for space in a crowded row.
        undo_redo_layout = QHBoxLayout()

        self.btn_undo = QPushButton("↶ Undo")
        self.btn_undo.setToolTip("Undo the last action (Ctrl+Z)")
        self.btn_undo.setMinimumHeight(25)
        self.btn_undo.setEnabled(False)

        self.btn_redo = QPushButton("↷ Redo")
        self.btn_redo.setToolTip("Redo the last undone action (Ctrl+Shift+Z)")
        self.btn_redo.setMinimumHeight(25)
        self.btn_redo.setEnabled(False)

        undo_redo_layout.addWidget(self.btn_undo)
        undo_redo_layout.addWidget(self.btn_redo)

        parent_layout.addLayout(undo_redo_layout)
    
    def apply_theme(self, theme: str):
        """Propagate theme changes to all child graphs and update sidebar icons."""
        for graph_widget, _, _ in self.graph_widgets.values():
            graph_widget.update_icon_colors(theme)
        
        icon_color = "#404040" if theme != "dark" else "#F0F0F0"
        self.btn_view_df.setIcon(get_tinted_icon(os.path.join(ICON_DIR, "view.png"), icon_color))
        self.btn_remove_df.setIcon(get_tinted_icon(os.path.join(ICON_DIR, "trash.png"), icon_color))
        self.btn_save_df.setIcon(get_tinted_icon(os.path.join(ICON_DIR, "save.png"), icon_color))
        self.btn_refresh_df.setIcon(get_tinted_icon(os.path.join(ICON_DIR, "update.png"), icon_color))
        self.btn_add_plot.setIcon(get_tinted_icon(os.path.join(ICON_DIR, "add.png"), icon_color))
        self.btn_update_plot.setIcon(get_tinted_icon(os.path.join(ICON_DIR, "update.png"), icon_color))
        self.btn_add_multi_wafer.setIcon(get_tinted_icon(os.path.join(ICON_DIR, "add.png"), icon_color))
        self.btn_export_all.setIcon(get_tinted_icon(os.path.join(ICON_DIR, "save-all.png"), icon_color))

        # v_data_filter buttons
        self.v_data_filter.btn_add.setIcon(get_tinted_icon(os.path.join(ICON_DIR, "add.png"), icon_color))
        self.v_data_filter.btn_remove.setIcon(get_tinted_icon(os.path.join(ICON_DIR, "close.png"), icon_color))
        self.v_data_filter.btn_apply.setIcon(get_tinted_icon(os.path.join(ICON_DIR, "done.png"), icon_color))

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
        self.btn_export_all.clicked.connect(self._on_export_all_clicked)
        self.btn_compose_figure.clicked.connect(self._on_compose_figure_clicked)
        self.btn_undo.clicked.connect(self._on_undo_clicked)
        self.btn_redo.clicked.connect(self._on_redo_clicked)

        # Plot style connection
        self.cbb_plot_style.currentTextChanged.connect(self._on_plot_style_changed)
        
        # Bottom toolbar connections
        self.cbb_graph_list.currentIndexChanged.connect(self._on_graph_selected_toolbar)
        self.btn_minimize_all.clicked.connect(self._on_minimize_all)
        self.btn_delete_all.clicked.connect(self._on_delete_all)
        self.cb_grid_toolbar.stateChanged.connect(self._on_grid_changed_toolbar)

        # MDI area connections
        self.mdi_area.subWindowActivated.connect(self._on_subwindow_activated)

        # ViewModel → View signal connections
        vm.dataframes_changed.connect(self._update_df_list)
        vm.dataframe_columns_changed.connect(self._update_column_combos)
        vm.dataframe_columns_changed.connect(self._update_slot_selector)
        vm.graphs_changed.connect(self._update_graph_list)
        vm.undo_state_changed.connect(self._on_undo_state_changed)
        vm.notify.connect(self._show_toast_notification)

        self._setup_keyboard_shortcuts()

    def _setup_keyboard_shortcuts(self):
        """Undo/redo/copy-style/paste-style accelerators, scoped to the MDI
        area (WidgetWithChildrenShortcut on self.mdi_area) rather than the
        whole workspace widget -- so they fire while a graph window has
        focus but don't shadow a side-panel QLineEdit's own native Ctrl+Z/
        Ctrl+C/Ctrl+V while the user is typing a title/label there."""
        undo_sc = QShortcut(QKeySequence("Ctrl+Z"), self.mdi_area)
        undo_sc.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        undo_sc.activated.connect(self._on_undo_clicked)

        redo_sc = QShortcut(QKeySequence("Ctrl+Shift+Z"), self.mdi_area)
        redo_sc.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        redo_sc.activated.connect(self._on_redo_clicked)

        copy_style_sc = QShortcut(QKeySequence("Ctrl+C"), self.mdi_area)
        copy_style_sc.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        copy_style_sc.activated.connect(self._on_copy_style_shortcut)

        paste_style_sc = QShortcut(QKeySequence("Ctrl+V"), self.mdi_area)
        paste_style_sc.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        paste_style_sc.activated.connect(self._on_paste_style_shortcut)

    def _get_active_graph_id(self) -> Optional[int]:
        """graph_id of the currently active MDI subwindow, or None if no
        graph window is active."""
        active_subwindow = self.mdi_area.activeSubWindow()
        if not active_subwindow:
            return None
        for gid, (_gw, _gd, sw) in self.graph_widgets.items():
            if sw == active_subwindow:
                return gid
        return None

    def _on_copy_style_shortcut(self):
        gid = self._get_active_graph_id()
        if gid is not None:
            self._on_style_action_requested(gid, "copy")

    def _on_paste_style_shortcut(self):
        gid = self._get_active_graph_id()
        if gid is not None:
            self._on_style_action_requested(gid, "paste")

    def _update_df_placeholder(self):
        """Update placeholder text for dataframe list based on state."""
        if self.df_listbox.count() == 0:
            # Add 2 empty lines before placeholder for spacing
            for _ in range(2):
                spacer = QListWidgetItem("")
                spacer.setFlags(Qt.NoItemFlags)
                self.df_listbox.addItem(spacer)
            
            # Add the centered placeholder item with larger text
            placeholder = QListWidgetItem("📂 Drag and drop file(s) anywhere to open")
            placeholder.setFlags(Qt.NoItemFlags)  # Make it non-selectable and non-editable
            placeholder.setForeground(Qt.gray)
            placeholder.setTextAlignment(Qt.AlignCenter)  # Center the text horizontally
            
            # Set larger font size
            font = QFont()
            font.setPointSize(11)
            placeholder.setFont(font)
            
            self.df_listbox.addItem(placeholder)
            
            self._has_df_placeholder = True
        else:
            # Remove all placeholder items if they exist
            if self._has_df_placeholder:
                # Clear all items with NoItemFlags (placeholders and spacers)
                i = 0
                while i < self.df_listbox.count():
                    if self.df_listbox.item(i).flags() == Qt.NoItemFlags:
                        self.df_listbox.takeItem(i)
                    else:
                        i += 1
                self._has_df_placeholder = False
    

    
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
        """View DataFrame in table, or open source file if Ctrl is held."""
        current_item = self.df_listbox.currentItem()
        if not current_item:
            QMessageBox.warning(self, "No DataFrame", "Please select a DataFrame to view.")
            return
        
        df_name = current_item.text()
        
        # Check if Ctrl key is held down
        modifiers = QApplication.keyboardModifiers()
        
        if modifiers & Qt.ControlModifier:
            # Ctrl is held - open source file
            if df_name in self.vm.dataframe_sources:
                source_path = self.vm.dataframe_sources[df_name]
                try:
                    # Use Qt's QDesktopServices for cross-platform file opening
                    file_url = QUrl.fromLocalFile(source_path)
                    if QDesktopServices.openUrl(file_url):
                        self.vm.notify.emit(f"Opening file: {os.path.basename(source_path)}")
                    else:
                        QMessageBox.warning(self, "Error", f"Could not open file: {source_path}")
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Could not open file: {str(e)}")
            else:
                QMessageBox.warning(
                    self, 
                    "No Source File", 
                    f"No source file found for '{df_name}'."
                )
            return
        
        # Normal click - show DataFrame in dialog
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
        layout.setContentsMargins(4, 4, 4, 4)
        
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
        """Save DataFrame to Excel or CSV."""
        current_item = self.df_listbox.currentItem()
        if current_item:
            df_name = current_item.text()
            self.vm.save_dataframe(df_name)
    
    def _on_refresh_df(self):
        """Refresh DataFrame from source file."""
        current_item = self.df_listbox.currentItem()
        if not current_item:
            QMessageBox.warning(self, "No DataFrame", "Please select a DataFrame to refresh.")
            return
        
        df_name = current_item.text()
        
        # Get current graph before refresh to restore properties after
        current_graph_id = None
        current_graph_index = self.cbb_graph_list.currentIndex()
        if current_graph_index >= 0:
            current_graph_id = self.cbb_graph_list.currentData()
        
        success = self.vm.refresh_dataframe(df_name)
        
        if success:
            self.vm.notify.emit(f"DataFrame '{df_name}' refreshed successfully")
            
            # Restore current graph properties after df refresh
            if current_graph_id:
                graph_model = self.vm.get_graph(current_graph_id)
                if graph_model and graph_model.df_name == df_name:
                    self._sync_gui_from_graph(graph_model)
        else:
            QMessageBox.warning(
                self, 
                "Refresh Failed", 
                f"Could not refresh DataFrame '{df_name}'. Source file may not exist or be accessible."
            )
    
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
        
        plot_config = self._collect_plot_config(include_labels=False)
        if not plot_config['x'] or not plot_config['y']:
            QMessageBox.warning(self, "Missing Axes", "Please select X and Y axes.")
            return
        self._apply_default_style_to_config(plot_config)

        self._create_and_display_plot(plot_config, filters=current_filters)

    def _apply_default_style_to_config(self, plot_config: dict) -> None:
        """Merge the user's "Set as Default Style" baseline (if any) under
        a freshly-collected plot_config, for a graph being built from
        scratch (Add Plot / Add Multi-Wafer) -- never for
        _on_replicate_graph() or recipe-apply, whose configs already carry
        an intentional, fully-specified style that must not be silently
        overwritten. plot_config's own keys always win: the default style
        only ever contains graph_style.STYLE_FIELD_NAMES keys, so there's
        no collision with plot_config's data-identity fields (x/y/z/
        df_name/plot_style) in practice, but `update` (not overwrite)
        keeps that guarantee explicit rather than assumed."""
        default_style = self.m_settings.get_default_graph_style()
        if default_style:
            # Prevent the default style from forcing top/right/bottom spines onto a wafer plot
            if plot_config.get('plot_style') == 'wafer':
                default_style.pop('spines_visible', None)
                
            merged = dict(default_style)
            merged.update(plot_config)
            plot_config.clear()
            plot_config.update(merged)
            
        # Default wafer plot spines to left-only if not explicitly set by a saved default style
        if 'spines_visible' not in plot_config:
            if plot_config.get('plot_style') == 'wafer':
                plot_config['spines_visible'] = {'top': False, 'right': False, 'bottom': False, 'left': True}
            else:
                plot_config['spines_visible'] = {'top': True, 'right': True, 'bottom': True, 'left': True}

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
    
    def _build_graph_widget(self, graph_model, filtered_df, on_render_error) -> Optional[VGraph]:
        """Instantiate a VGraph for `graph_model`, wire its signals, render
        it, and register it into `graph_widgets` + the MDI area.

        On a render failure, the graph is removed from the ViewModel,
        `on_render_error(exc)` is called so the caller can report it however
        it likes (blocking dialog vs. a toast vs. warn-and-continue in a
        batch), and None is returned — nothing is added to the MDI area.

        Shared by _create_and_display_plot, _on_plot_multi_wafer, and
        load_workspace, which previously duplicated this exact sequence.
        """
        graph_widget = VGraph(graph_id=graph_model.graph_id)
        graph_widget.replicate_requested.connect(self._on_replicate_graph)
        graph_widget.customize_requested.connect(self._show_or_switch_customize_dialog)
        graph_widget.export_requested.connect(self._on_export_graph_requested)
        graph_widget.style_action_requested.connect(self._on_style_action_requested)
        graph_widget.properties_changed.connect(self._on_graph_properties_changed)
        graph_widget.notify.connect(self._show_toast_notification)

        self._configure_graph_from_model(graph_widget, graph_model)
        graph_widget.create_plot_widget(graph_model.dpi)

        try:
            self._render_plot(graph_widget, filtered_df, graph_model)
        except Exception as e:
            self.vm.delete_graph(graph_model.graph_id)
            on_render_error(e)
            return None

        # Write directly onto the model (not vm.update_graph()): derived
        # bookkeeping, not a user edit, so it must never be undo-tracked.
        graph_model.legend_properties = graph_widget.legend_properties

        sub_window = self._create_mdi_subwindow(graph_widget, graph_model)
        graph_dialog = self._wrap_graph_in_dialog(graph_widget)
        sub_window.setWidget(graph_dialog)

        self.graph_widgets[graph_model.graph_id] = (graph_widget, graph_dialog, sub_window)
        self.mdi_area.addSubWindow(sub_window)
        sub_window.show()

        return graph_widget

    def _create_and_display_plot(self, plot_config: dict, select_in_list: bool = True, filters: list = None):
        """Create and display a plot from configuration."""
        self.vm.begin_undo_batch()
        try:
            # Use provided filters or get current ones
            if filters is None:
                filters = self.v_data_filter.get_filters()

            graph_model = self.vm.create_graph(plot_config)
            # CRITICAL: Use df_name from plot_config, not vm.selected_df_name
            # because vm.selected_df_name can be changed by window activation events
            filtered_df = self.vm.apply_filters(plot_config['df_name'], filters)

            def _on_render_error(e):
                QMessageBox.critical(self, "Plot Error", f"Error rendering plot: {str(e)}")

            if self._build_graph_widget(graph_model, filtered_df, _on_render_error) is None:
                return

            self._update_graph_list(self.vm.get_graph_ids())

            if select_in_list:
                for i in range(self.cbb_graph_list.count()):
                    if self.cbb_graph_list.itemData(i) == graph_model.graph_id:
                        self.cbb_graph_list.setCurrentIndex(i)
                        break
        finally:
            self.vm.end_undo_batch()
    
    def _wrap_graph_in_dialog(self, graph_widget: VGraph) -> QDialog:
        """Wrap graph widget in dialog."""
        graph_dialog = QDialog(self)
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(graph_widget)
        graph_dialog.setLayout(layout)
        return graph_dialog
    
    def create_plot_from_config(self, df_name: str, plot_config: dict) -> bool:
        """API to create a plot directly from a configuration dict."""
        self.vm.select_dataframe(df_name)
        filters = plot_config.get('filters', [])
        if filters is None:
            filters = []
        self._create_and_display_plot(plot_config, select_in_list=False, filters=filters)
        return True

    def _refresh_recipe_and_style_stores(self) -> None:
        """Rebuild both stores from the *current* Working Folder setting.

        Fixes a real bug: self.recipe_store/self.style_template_store used
        to be built once in __init__ and never touched again, so a Working
        Folder configured (or changed) after this widget already existed
        -- i.e. almost always, since it's built at app startup -- silently
        never took effect until restart, even though the setting itself
        saved correctly. Reconstructing here (cheap: the constructor is
        just a folder scan) guarantees correctness at the point of use
        without needing a signal connection from VMSettings all the way
        back to this unrelated ViewModel/View."""
        self.recipe_store = MPlotRecipeStore(self.m_settings.get_plot_recipe_folder())
        self.style_template_store = MStyleTemplateStore(self.m_settings.get_plot_style_folder())

    def _on_apply_recipe_clicked(self) -> None:
        self._refresh_recipe_and_style_stores()
        dialog = VPlotRecipeDialog(self.recipe_store, self)
        dialog.recipe_applied.connect(self._on_recipe_applied)
        dialog.exec()

    def _on_save_as_recipe_clicked(self) -> None:
        """Save every currently open graph's live configuration as a new
        Plot Recipe. Self-contained — no AI Chat panel dependency."""
        self._refresh_recipe_and_style_stores()
        configs = []
        for graph_model in self.vm.graphs.values():
            cfg = graph_model.save()
            cfg.pop('graph_id', None)
            configs.append(cfg)

        if not configs:
            QMessageBox.information(
                self, "No Graphs", "There are no open graphs to save as a recipe."
            )
            return

        name, ok = QInputDialog.getText(self, "Save Plot Recipe", "Recipe name:")
        if ok and name:
            self.recipe_store.save_recipe(name, configs)
            QMessageBox.information(
                self, "Recipe Saved",
                f"Saved '{name}' with {len(configs)} plot{'s' if len(configs) > 1 else ''}."
            )

    def _on_export_all_clicked(self) -> None:
        """Export every currently open graph to a folder in one pass."""
        if not self.graph_widgets:
            QMessageBox.information(
                self, "No Graphs", "There are no open graphs to export."
            )
            return

        widgets = {gid: gw for gid, (gw, _, _) in self.graph_widgets.items()}
        dialog = VBatchExportDialog(widgets, parent=self)
        dialog.exec()

    def _on_compose_figure_clicked(self) -> None:
        """Open the multi-panel composer to combine several open graphs
        into one exported figure."""
        if not self.graph_widgets:
            QMessageBox.information(
                self, "No Graphs", "There are no open graphs to compose."
            )
            return

        widgets = {gid: gw for gid, (gw, _, _) in self.graph_widgets.items()}
        dialog = VMultiPanelDialog(widgets, parent=self)
        dialog.exec()

    def _on_style_action_requested(self, graph_id: int, action: str) -> None:
        """Central dispatch for VGraph's per-graph "🎨 Style" menu (save/
        apply template, copy/paste, reset-to-default) -- one connection per
        graph widget, same pattern as customize_requested/export_requested.
        See model/graph_style.py for the style/data field partition every
        branch here shares."""
        entry = self.graph_widgets.get(graph_id)
        if not entry:
            return
        gw = entry[0]

        if action == "save_template":
            self._save_style_template(gw)
        elif action == "apply_template":
            self._apply_style_template_dialog(gw)
        elif action == "copy":
            self._copied_style = extract_style(vars(gw))
            self._show_toast_notification(f"Style copied from graph {graph_id}.")
        elif action == "paste":
            if not self._copied_style:
                QMessageBox.information(self, "Nothing to Paste", "Copy a graph's style first.")
                return
            self._apply_style_to_graph(gw, self._copied_style)
        elif action == "reset":
            self._apply_style_to_graph(gw, default_style())
        elif action == "set_default":
            self.m_settings.set_default_graph_style(extract_style(vars(gw)))
            self._show_toast_notification(
                f"Graph {graph_id}'s style is now the default for new plots."
            )

    def _save_style_template(self, gw: VGraph) -> None:
        self._refresh_recipe_and_style_stores()
        name, ok = QInputDialog.getText(self, "Save Style", "Style name:")
        if not (ok and name):
            return
        style = extract_style(vars(gw))
        tpl_id = self.style_template_store.save_template(name, style)
        if tpl_id:
            self._show_toast_notification(f"Saved style '{name}'.")
        else:
            QMessageBox.warning(
                self, "Could Not Save",
                "No working folder is configured yet (set one in Settings)."
            )

    def _apply_style_template_dialog(self, gw: VGraph) -> None:
        self._refresh_recipe_and_style_stores()
        dialog = VStyleTemplateDialog(self.style_template_store, self)
        dialog.style_applied.connect(lambda style, target=gw: self._apply_style_to_graph(target, style))
        dialog.exec()

    def _apply_style_to_graph(self, gw: VGraph, style: dict) -> None:
        """Write `style` onto `gw`, replot, and persist via the same
        properties_changed path every other Apply handler in this codebase
        uses (so undo/save-workspace/etc. all see the change)."""
        applied = apply_style_dict(gw, style)
        if gw.df is not None:
            gw.plot(gw.df)
        gw.properties_changed.emit(gw.graph_id, applied)

    def _on_recipe_applied(self, configs: list) -> None:
        """Apply every plot config in a saved recipe against the
        currently selected DataFrame (not each plot's originally-saved
        df_name). A plot whose required axis columns (x, y, z, y2, y3, x2)
        aren't all present in that DataFrame is skipped — rather than
        aborting the whole batch or raising — and reported in one summary
        once every config has been processed."""
        from spectroview.ai_agent.utils.plot_utils import normalize_plot_config

        df_name = self.vm.selected_df_name
        if not df_name:
            QMessageBox.warning(
                self, "No DataFrame Selected",
                "Please select a DataFrame before applying a recipe."
            )
            return

        df = self.vm.get_dataframe(df_name)
        available_columns = set(df.columns) if df is not None else set()

        applied = 0
        skipped = []
        self.vm.begin_undo_batch()  # one undo step for the whole recipe, not one per plot
        try:
            for i, raw_cfg in enumerate(configs, start=1):
                cfg = copy.deepcopy(raw_cfg)
                normalize_plot_config(cfg)

                missing = sorted(self._required_plot_columns(cfg) - available_columns)
                if missing:
                    skipped.append((i, missing))
                    continue

                cfg['df_name'] = df_name
                self.create_plot_from_config(df_name, cfg)
                applied += 1
        finally:
            self.vm.end_undo_batch()

        if skipped:
            lines = [
                f"Plot {i} can not be plotted since the selected dataframe "
                f"does not have columns: {', '.join(cols)}."
                for i, cols in skipped
            ]
            QMessageBox.warning(
                self, "Some Plots Skipped",
                f"{applied} of {len(configs)} plot(s) applied.\n\n" + "\n".join(lines)
            )
        elif applied:
            self.vm.notify.emit(f"Applied {applied} plot(s) from recipe.")

    @staticmethod
    def _required_plot_columns(cfg: dict) -> set:
        """Column names a plot config's axes reference (x/y/z/y2/y3/x2)."""
        cols = set()
        for key in ('x', 'z', 'y2', 'y3', 'x2'):
            val = cfg.get(key)
            if val:
                cols.add(val)
        y = cfg.get('y')
        if isinstance(y, list):
            cols.update(v for v in y if v)
        elif y:
            cols.add(y)
        return cols

    def _on_replicate_graph(self, graph_id: int):
        """Replicate a graph."""
        graph_model = self.vm.get_graph(graph_id)
        if not graph_model:
            return
        
        # Get properties and remove graph_id to let ViewModel create a new one
        plot_config = graph_model.save()
        del plot_config['graph_id']
        
        # Capture current widget size so user-resizing is retained
        if graph_id in self.graph_widgets:
            _, _, sub_window = self.graph_widgets[graph_id]
            size = sub_window.size()
            plot_config['plot_width'] = size.width()
            plot_config['plot_height'] = size.height()
        
        # We need to make sure df_name is correct
        if not plot_config.get('df_name'):
            QMessageBox.warning(self, "Error", "Cannot replicate a plot without a valid DataFrame.")
            return
            
        self._create_and_display_plot(plot_config, select_in_list=True, filters=plot_config.get('filters', []))
        
    def _on_graph_properties_changed(self, graph_id: int, properties: dict):
        """Update graph model when properties change directly from graph widget."""
        self.vm.update_graph(graph_id, properties)
    
    def _show_or_switch_customize_dialog(self, graph_id: int):
        """Show, create, or switch the singleton CustomizeGraphDialog."""
        if graph_id not in self.graph_widgets:
            return
        
        graph_widget, _, _ = self.graph_widgets[graph_id]
        
        if self._customize_dialog is None:
            # Create the dialog for the first time
            self._customize_dialog = CustomizeGraphDialog(graph_widget, graph_id, parent=self)
            self._customize_dialog.show()
        elif self._customize_dialog.isVisible():
            # Dialog is open — switch to the new graph
            self._customize_dialog.switch_graph(graph_widget, graph_id)
            self._customize_dialog.raise_()
            self._customize_dialog.activateWindow()
        else:
            # Dialog exists but was closed/hidden — switch and re-show
            self._customize_dialog.switch_graph(graph_widget, graph_id)
            self._customize_dialog.show()
            self._customize_dialog.raise_()
            self._customize_dialog.activateWindow()

    def _on_export_graph_requested(self, graph_id: int):
        """Open the (modal, one-shot) export dialog for a single graph."""
        if graph_id not in self.graph_widgets:
            return
        graph_widget, _, _ = self.graph_widgets[graph_id]
        dialog = VExportDialog(graph_widget, parent=self)
        dialog.exec()

    def _on_update_plot(self):
        """Update selected plot."""
        self.vm.begin_undo_batch()
        try:
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

            # Preserve the graph's own df_name -- the side panel's selected
            # DataFrame can drift from it (browsed without reactivating this
            # plot's window), and "Update plot" must never silently rebind.
            plot_config['df_name'] = graph_model.df_name

            # Preserve limits and annotations that are managed outside main GUI
            plot_config['xmin'] = graph_widget.xmin
            plot_config['xmax'] = graph_widget.xmax
            plot_config['ymin'] = graph_widget.ymin
            plot_config['ymax'] = graph_widget.ymax
            plot_config['zmax'] = graph_widget.zmax
            plot_config['annotations'] = getattr(graph_widget, 'annotations', [])
            plot_config['axis_breaks'] = getattr(graph_widget, 'axis_breaks', {'x': None, 'y': None})

            # Check if plot style changed to/from wafer
            if plot_config['plot_style'] != graph_model.plot_style:
                if plot_config['plot_style'] == 'wafer':
                    plot_config['spines_visible'] = {'top': False, 'right': False, 'bottom': False, 'left': True}
                elif graph_model.plot_style == 'wafer':
                    plot_config['spines_visible'] = {'top': True, 'right': True, 'bottom': True, 'left': True}
            else:
                # Auto-correct existing wafer plots that still have the old default 4-spines
                if plot_config['plot_style'] == 'wafer' and getattr(graph_model, 'spines_visible', None) == {'top': True, 'right': True, 'bottom': True, 'left': True}:
                    plot_config['spines_visible'] = {'top': False, 'right': False, 'bottom': False, 'left': True}

            # Check if Z-axis has changed (reset legend properties if so)
            z_changed = plot_config['z'] != graph_model.z
            if z_changed:
                graph_widget.legend_properties = []

            # Check if filters have changed
            current_filters = self.v_data_filter.get_filters()
            if current_filters != graph_model.filters:
                graph_widget.legend_properties = []

            # Update graph model
            self.vm.update_graph(graph_model.graph_id, plot_config)

            # Capture current legend position from matplotlib before syncing
            graph_widget._save_legend_position()

            # Sync current legend properties to model BEFORE reconfiguring
            # (so customizations from the dialog are not lost)
            self.vm.update_graph(graph_model.graph_id, {
                'legend_properties': graph_widget.legend_properties,
                'legend_bbox': graph_widget.legend_bbox
            })

            # Get updated model
            graph_model = self.vm.get_graph(graph_model.graph_id)

            # Apply filters against the graph's own DataFrame, not whatever is
            # currently selected in the side list (they can differ if the user
            # switched DataFrame selection after activating this plot).
            filtered_df = self.vm.apply_filters(graph_model.df_name, current_filters)

            # Reconfigure and re-render
            self._configure_graph_from_model(graph_widget, graph_model)
            graph_widget.create_plot_widget(graph_model.dpi)

            try:
                self._render_plot(graph_widget, filtered_df, graph_model)
            except Exception as e:
                QMessageBox.critical(
                    self, "Plot Update Error",
                    f"Error updating plot: {str(e)}"
                )
                return

            # Save legend properties back to model after rendering
            self.vm.update_graph(graph_model.graph_id, {
                'legend_properties': graph_widget.legend_properties,
                'legend_bbox': graph_widget.legend_bbox
            })

            # If Z changed and CustomizeGraphDialog is open, reload its content
            if z_changed and self._customize_dialog is not None and self._customize_dialog.isVisible():
                if self._customize_dialog.graph_id == graph_model.graph_id:
                    self._customize_dialog.legend_widget.load_legend_properties()

            # Update window title
            title = f"{graph_model.graph_id}-{graph_model.plot_style}: [{graph_model.x}] vs [{graph_model.y[0] if graph_model.y else 'None'}]"
            if graph_model.z:
                title += f" - [{graph_model.z}]"
            active_subwindow.setWindowTitle(title)
        finally:
            self.vm.end_undo_batch()
    
    def _update_df_list(self, df_names: list):
        """Update DataFrame list from ViewModel."""
        self.df_listbox.blockSignals(True)
        
        # Remember current selection
        current_selection = self.df_listbox.currentItem()
        selected_name = current_selection.text() if current_selection else None
        
        self.df_listbox.clear()
        self._has_df_placeholder = False  # Reset placeholder flag
        
        for name in df_names:
            self.df_listbox.addItem(name)
        
        # Restore selection if possible
        if selected_name:
            items = self.df_listbox.findItems(selected_name, Qt.MatchExactly)
            if items:
                self.df_listbox.setCurrentItem(items[0])
        elif df_names:
            self.df_listbox.setCurrentRow(0)
        
        self.df_listbox.blockSignals(False)
        
        # Trigger selection changed to update UI
        self._on_df_selected()
        
        # Update placeholder only if list is empty
        if len(df_names) == 0:
            self._update_df_placeholder()
    
    def _update_column_combos(self, columns: list):
        """Update column comboboxes."""
        self.cbb_x.clear()
        self.cbb_y.clear()
        self.cbb_z.clear()
        self.cbb_y2.clear()
        self.cbb_y3.clear()
        self.cbb_x2.clear()
        
        self.cbb_x.addItems(columns)
        self.cbb_y.addItems(columns)
        self.cbb_z.addItem("None")
        self.cbb_z.addItems(columns)
        
        # Multiple axes comboboxes (with "None" default)
        self.cbb_y2.addItem("None")
        self.cbb_y2.addItems(columns)
        self.cbb_y3.addItem("None")
        self.cbb_y3.addItems(columns)
        self.cbb_x2.addItem("None")
        self.cbb_x2.addItems(columns)
        
        # Update filter autocomplete with DataFrame columns
        self.v_data_filter.update_column_list(columns)
        
    def _update_slot_selector(self, columns: list):
        """Update slot selector."""
        # Save current checkbox states before clearing
        saved_states = {}
        for cb in self.slot_checkboxes:
            saved_states[cb.text()] = cb.isChecked()
        
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
            # Restore previous state if it existed, otherwise default to unchecked
            slot_str = str(slot)
            cb.setChecked(saved_states.get(slot_str, False))
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

        checked_slots = [int(float(cb.text())) for cb in self.slot_checkboxes if cb.isChecked()]

        if not checked_slots:
            QMessageBox.warning(self, "No Slots Selected", "Please select at least one slot.")
            return

        if not self._validate_plot_request():
            return

        self.vm.begin_undo_batch()
        try:
            # Force wafer plot style
            wafer_index = self.cbb_plot_style.findText('wafer')
            if wafer_index >= 0:
                self.cbb_plot_style.setCurrentIndex(wafer_index)

            plot_config = self._collect_plot_config(include_labels=False)
            plot_config['plot_style'] = 'wafer'
            self._apply_default_style_to_config(plot_config)

            created_graphs = self.vm.create_multi_wafer_graphs(
                self.vm.selected_df_name,
                checked_slots,
                plot_config,
                current_filters
            )

            successfully_created = 0
            for graph_model in created_graphs:
                filtered_df = self.vm.apply_filters(self.vm.selected_df_name, graph_model.filters)

                def _on_render_error(e, graph_model=graph_model):
                    slot_expr = graph_model.filters[-1].get('expression', 'unknown') if graph_model.filters else 'unknown'
                    QMessageBox.warning(
                        self, "Plot Error",
                        f"Error rendering plot for {slot_expr}: {str(e)}"
                    )

                if self._build_graph_widget(graph_model, filtered_df, _on_render_error) is None:
                    continue
                successfully_created += 1

            # Update graph list
            self._update_graph_list(self.vm.get_graph_ids())

            if successfully_created > 0:
                self.vm.notify.emit(f"Created {successfully_created} wafer plot(s)")
        finally:
            self.vm.end_undo_batch()
    
    def _update_graph_list(self, graph_ids: list):
        """Update graph list."""
        # Save current selection so we don't randomly switch
        current_id = self.cbb_graph_list.currentData()
        
        # Block signals so adding items doesn't trigger UI focus changes
        self.cbb_graph_list.blockSignals(True)
        
        self.cbb_graph_list.clear()
        for gid in graph_ids:
            graph = self.vm.get_graph(gid)
            if graph:
                display_name = graph.get_display_name()
                self.cbb_graph_list.addItem(display_name, gid)
                
        # Restore previous selection visually
        if current_id is not None:
            for i in range(self.cbb_graph_list.count()):
                if self.cbb_graph_list.itemData(i) == current_id:
                    self.cbb_graph_list.setCurrentIndex(i)
                    break
                    
        self.cbb_graph_list.blockSignals(False)
    
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
            # Close singleton customize dialog if open
            if self._customize_dialog is not None:
                self._customize_dialog.close()
                self._customize_dialog = None

            # Disconnect `closed` first: a programmatic batch teardown must
            # not cascade into _on_graph_closed()'s per-window delete_graph()
            # (which would push one undo step per graph instead of one).
            for sub_window in self.mdi_area.subWindowList():
                sub_window.closed.disconnect()
                self.mdi_area.removeSubWindow(sub_window)
                sub_window.close()

            # Delete graphs from ViewModel (one undo step for the whole batch)
            self.vm.begin_undo_batch()
            try:
                for graph_id in list(self.graph_widgets.keys()):
                    self.vm.delete_graph(graph_id)
            finally:
                self.vm.end_undo_batch()

            # Clear graph widgets storage
            self.graph_widgets.clear()

            # Update graph list combobox
            self._update_graph_list([])
    

    
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
        
        # Auto-switch the singleton customize dialog if it's open
        if self._customize_dialog is not None and self._customize_dialog.isVisible():
            graph_id = graph_model.graph_id
            if graph_id in self.graph_widgets:
                gw, _, _ = self.graph_widgets[graph_id]
                self._customize_dialog.switch_graph(gw, graph_id)
        
        # Update graph list combobox selection (block signals to prevent conflicts)
        self.cbb_graph_list.blockSignals(True)
        for i in range(self.cbb_graph_list.count()):
            if self.cbb_graph_list.itemData(i) == graph_model.graph_id:
                self.cbb_graph_list.setCurrentIndex(i)
                break
        self.cbb_graph_list.blockSignals(False)
    
    def _sync_gui_from_graph(self, model):
        """Sync GUI from graph model."""
        # Block signals for all controls during sync
        self.cbb_plot_style.blockSignals(True)
        self.cbb_x.blockSignals(True)
        self.cbb_y.blockSignals(True)
        self.cbb_z.blockSignals(True)
        self.df_listbox.blockSignals(True)
        
        # Block checkbox signals to prevent expensive signal handlers during sync
        self.cb_grid_toolbar.blockSignals(True)
        
        try:
            # Dataframe selection
            if model.df_name:
                # Only select in VM if dataframe actually changed to avoid expensive updates
                if self.vm.selected_df_name != model.df_name:
                    self.vm.select_dataframe(model.df_name)
                
                # Find and select the dataframe in the list
                for i in range(self.df_listbox.count()):
                    if self.df_listbox.item(i).text() == model.df_name:
                        self.df_listbox.setCurrentRow(i)
                        break
            
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
            
            # Color palette
            if model.color_palette:
                idx = self.cbb_colormap.findText(model.color_palette)
                if idx >= 0:
                    self.cbb_colormap.setCurrentIndex(idx)
            
            # Checkboxes
            self.cb_grid_toolbar.setChecked(model.grid)
            
            # Text inputs
            self.edit_plot_title.setText(model.plot_title or "")
            self.edit_plot_subtitle.setText(getattr(model, 'plot_subtitle', None) or "")
            self.edit_xlabel.setText(model.xlabel or "")
            self.edit_ylabel.setText(model.ylabel or "")
            self.edit_zlabel.setText(model.zlabel or "")
            
            # Toolbar controls
            self.spin_xlabel_rotation.setValue(model.x_rot)
            
            # Filters
            self.v_data_filter.set_filters(model.filters)
            
            # Multiple axes (Y2, Y3, X2)
            y2_val = model.y2 or "None"
            idx = self.cbb_y2.findText(y2_val)
            if idx >= 0:
                self.cbb_y2.setCurrentIndex(idx)
            
            y3_val = model.y3 or "None"
            idx = self.cbb_y3.findText(y3_val)
            if idx >= 0:
                self.cbb_y3.setCurrentIndex(idx)
            
            x2_val = getattr(model, 'x2', None) or "None"
            idx = self.cbb_x2.findText(x2_val)
            if idx >= 0:
                self.cbb_x2.setCurrentIndex(idx)
            
            self.edit_y2label.setText(model.y2label or "")
            self.edit_y3label.setText(model.y3label or "")
            self.edit_x2label.setText(getattr(model, 'x2label', '') or "")
            
            self.cb_y2log.setChecked(getattr(model, 'y2logscale', False))
            self.cb_y3log.setChecked(getattr(model, 'y3logscale', False))
            self.cb_x2log.setChecked(getattr(model, 'x2logscale', False))
        finally:
            self.cbb_plot_style.blockSignals(False)
            self.cbb_x.blockSignals(False)
            self.cbb_y.blockSignals(False)
            self.cbb_z.blockSignals(False)
            self.df_listbox.blockSignals(False)
            
            # Unblock checkbox signals
            self.cb_grid_toolbar.blockSignals(False)
    
    # ═════════════════════════════════════════════════════════════════════
    # Plotting Helper Methods
    # ═════════════════════════════════════════════════════════════════════
    
    def _collect_plot_config(self, include_labels: bool = True) -> dict:
        """Collect plot configuration.
        
        Args:
            include_labels: If True, includes custom title and axis labels from GUI.
                          If False, sets them to None so graph auto-generates them.
        """
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
            'xlogscale': False,
            'ylogscale': False,
            'plot_title': (self.edit_plot_title.text() or None) if include_labels else None,
            'plot_subtitle': (self.edit_plot_subtitle.text() or None) if include_labels else None,
            'xlabel': (self.edit_xlabel.text() or None) if include_labels else None,
            'ylabel': (self.edit_ylabel.text() or None) if include_labels else None,
            'zlabel': (self.edit_zlabel.text() or None) if include_labels else None,
            'color_palette': self.cbb_colormap.currentText() if use_palette else 'jet',
            'wafer_size': float(self.cbb_wafer_size.currentText()),
            'x_rot': self.spin_xlabel_rotation.value(),
            'legend_visible': True,
            'grid': self.cb_grid_toolbar.isChecked(),
            'filters': self.v_data_filter.get_filters(),
            # Multiple axes
            'y2': self.cbb_y2.currentText() if self.cbb_y2.currentText() != "None" else None,
            'y3': self.cbb_y3.currentText() if self.cbb_y3.currentText() != "None" else None,
            'x2': self.cbb_x2.currentText() if self.cbb_x2.currentText() != "None" else None,
            'y2label': (self.edit_y2label.text() or None) if include_labels else None,
            'y3label': (self.edit_y3label.text() or None) if include_labels else None,
            'x2label': (self.edit_x2label.text() or None) if include_labels else None,
            'y2logscale': self.cb_y2log.isChecked(),
            'y3logscale': self.cb_y3log.isChecked(),
            'x2logscale': self.cb_x2log.isChecked(),
        }
    
    def _configure_graph_from_model(self, graph_widget: VGraph, model):
        """Configure a graph widget from its model's fields.

        Copies every MGraph field onto the widget generically: `model` is
        always a real MGraph instance (from vm.create_graph()/
        vm.get_graph()), so every field declared in MGraph.__init__ always
        exists on it -- the getattr(..., default)/hasattr(...) guards this
        used to have per-field were vestigial. A short list of fields need
        real transformation instead of a plain copy (independent-copy
        safety for mutable containers, stale-value sanitization); those are
        applied as overrides below.

        Fixes a real bug found while collapsing this method: `axis_breaks`
        was never copied model -> widget at all (only ever the other way,
        widget -> model, in _on_update_plot/save_workspace) -- so a graph
        with a configured axis break silently lost it whenever a *new*
        widget was built from its model, i.e. on workspace reload or
        "Replicate graph".
        """
        for key, value in vars(model).items():
            if key == 'graph_id':
                continue
            setattr(graph_widget, key, value)

        # Independent copies for mutable containers the widget can mutate
        # in place (dragging a legend entry, editing an axis break, dragging
        # an annotation, editing filters, toggling a spine) -- aliasing the
        # model's own object would leak unsaved edits into it.
        graph_widget.y = model.y.copy() if model.y else []
        graph_widget.legend_properties = model.legend_properties.copy() if model.legend_properties else []
        graph_widget.axis_breaks = copy.deepcopy(model.axis_breaks) if model.axis_breaks else {'x': None, 'y': None}
        graph_widget.annotations = copy.deepcopy(model.annotations) if model.annotations else []
        graph_widget.filters = copy.deepcopy(model.filters) if model.filters else []
        graph_widget.spines_visible = (
            copy.deepcopy(model.spines_visible) if model.spines_visible
            else {'top': True, 'right': True, 'bottom': True, 'left': True}
        )

        # scatter_edgecolor must always be a valid, non-empty color string.
        edge_c = model.scatter_edgecolor
        if not edge_c or not isinstance(edge_c, str) or edge_c.strip() in ("", "None", "none", "null"):
            edge_c = 'black'
        graph_widget.scatter_edgecolor = edge_c
    
    def _render_plot(self, graph_widget: VGraph, filtered_df, model):
        """Render plot."""
        # The Graph class has a single plot() method that handles all plot types
        graph_widget.plot(filtered_df)
    
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
        
        # Block ESC key — Qt internally hides the widget on ESC
        shortcut = QShortcut(QKeySequence(Qt.Key_Escape), sub_window)
        shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        shortcut.activated.connect(lambda: None)  # Consume ESC, do nothing
        
        # Connect close signal to cleanup
        sub_window.closed.connect(lambda gid=model.graph_id: self._on_graph_closed(gid))
        
        return sub_window
    
    def _on_grid_changed_toolbar(self, state: int):
        """Handle grid checkbox change."""
        graph_id = self.cbb_graph_list.currentData()
        if graph_id is not None and graph_id in self.graph_widgets:
            graph_widget, _, _ = self.graph_widgets[graph_id]
            before = snapshot(graph_widget)
            # bool(state), not `state == Qt.Checked`: the stateChanged
            # signal carries a plain int (0/2), not the CheckState enum.
            graph_widget.grid = bool(state)
            # Replot from the widget's own already-filtered df, not a fresh
            # fetch keyed on the side list's current selection.
            if graph_widget.df is not None:
                graph_widget.plot(graph_widget.df)
            # Commit through the same path every customize-dialog tab uses,
            # so this persists to the model and survives a save.
            graph_widget.properties_changed.emit(
                graph_widget.graph_id, diff(graph_widget, before)
            )
    
    def _on_graph_closed(self, graph_id: int):
        """Handle graph closing."""
        # Close singleton customize dialog if it's showing this graph
        if self._customize_dialog is not None and self._customize_dialog.graph_id == graph_id:
            self._customize_dialog.close()
            self._customize_dialog = None
        
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
        # Generic snapshot/diff against MGraph's full field schema (see
        # graph_commit.py), not a hand-picked field list, so nothing new is
        # ever silently left out of what gets synced back before save.
        for gid, (gw, gd, sw) in self.graph_widgets.items():
            before = snapshot(gw)
            size = sw.size()
            gw.plot_width, gw.plot_height = size.width(), size.height()
            # Save legend position if it was dragged
            gw._save_legend_position()
            patch = diff(gw, before)
            if patch:
                self.vm.update_graph(gid, patch)

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

        # Recreate graph widgets and MDI subwindows for each loaded graph.
        self._rebuild_all_graph_widgets()

    def _rebuild_all_graph_widgets(self):
        """Tear down every VGraph widget/MDI subwindow and rebuild them
        from the ViewModel's current self.vm.graphs.

        The graph-widget-lifecycle portion of what load_workspace() does,
        factored out so undo()/redo() can reuse it without touching
        dataframes/selected_df_name/filter UI (those aren't part of
        undo/redo -- only the graph set is). A graph that fails to render
        (e.g. a stale column reference) is skipped with a toast rather
        than aborting the rest of the rebuild.
        """
        if self._customize_dialog is not None:
            self._customize_dialog.close()
            self._customize_dialog = None

        for sub_window in self.mdi_area.subWindowList():
            # Disconnect first: a programmatic teardown must not cascade
            # into _on_graph_closed()'s delete_graph(), which would delete
            # the very graphs this method is about to rebuild.
            sub_window.closed.disconnect()
            self.mdi_area.removeSubWindow(sub_window)
            sub_window.close()
        self.graph_widgets.clear()

        for graph_id in self.vm.get_graph_ids():
            graph_model = self.vm.get_graph(graph_id)
            if not graph_model:
                continue

            filtered_df = self.vm.apply_filters(graph_model.df_name, graph_model.filters)

            def _on_render_error(e, graph_model=graph_model):
                self.vm.notify.emit(
                    f"Skipped graph {graph_model.graph_id} ({graph_model.plot_style}): could not render ({e})"
                )

            self._build_graph_widget(graph_model, filtered_df, _on_render_error)

        self._update_graph_list(self.vm.get_graph_ids())

    def _on_undo_clicked(self):
        """Revert the last undo-tracked action (see
        VMWorkspaceGraphs.begin_undo_batch()/undo() for what counts as
        "one action")."""
        if self.vm.undo():
            self._rebuild_all_graph_widgets()
            self.vm.notify.emit("Undid last action.")

    def _on_redo_clicked(self):
        """Re-apply the last undone action."""
        if self.vm.redo():
            self._rebuild_all_graph_widgets()
            self.vm.notify.emit("Redid last action.")

    def _on_undo_state_changed(self):
        """Keep the Undo/Redo buttons' enabled state in sync with the
        ViewModel's undo/redo stacks."""
        self.btn_undo.setEnabled(self.vm.can_undo)
        self.btn_redo.setEnabled(self.vm.can_redo)

    def clear_workspace(self):
        """Clear workspace."""
        # Close singleton customize dialog
        if self._customize_dialog is not None:
            self._customize_dialog.close()
            self._customize_dialog = None
        
        # Disconnect `closed` first so this programmatic teardown doesn't
        # cascade into _on_graph_closed()'s delete_graph() (harmless here
        # since clear_workspace() below wipes everything anyway).
        for sub_window in self.mdi_area.subWindowList():
            sub_window.closed.disconnect()
            sub_window.close()
            self.mdi_area.removeSubWindow(sub_window)
        
        self.graph_widgets.clear()
        self.df_listbox.clear()
        self.cbb_x.clear()
        self.cbb_y.clear()
        self.cbb_z.clear()
        self.cbb_y2.clear()
        self.cbb_y3.clear()
        self.cbb_x2.clear()
        self.cbb_graph_list.clear()
        self.edit_plot_title.clear()
        self.edit_plot_subtitle.clear()
        self.edit_xlabel.clear()
        self.edit_ylabel.clear()
        self.edit_zlabel.clear()
        self.edit_y2label.clear()
        self.edit_y3label.clear()
        self.edit_x2label.clear()
        self.v_data_filter.clear_filters()
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