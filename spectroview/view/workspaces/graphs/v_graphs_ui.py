"""UI construction, theme synchronization, and keyboard shortcut setup mixin for VWorkspaceGraphs."""

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QFont, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QCompleter,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMdiArea,
    QPushButton,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from spectroview import AXIS_LABELS
from spectroview.i18n import (
    bind_combo_items,
    bind_placeholder,
    bind_tab_text,
    bind_text,
    bind_title,
    bind_tooltip,
    tr,
)
from spectroview.view.components.common.customized_widgets import CustomizedPalette
from spectroview.view.components.graphs.v_data_filter import VDataFilter
from spectroview.view.theme import Icons, current_tokens


class GraphsUIMixin:
    """Mixin responsible for building the Graphs workspace layout, controls, theme, and shortcuts."""

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
        self.mdi_area.setStyleSheet("QMdiArea { border: none; }")
        from PySide6.QtGui import QBrush
        self.mdi_area.setBackground(QBrush(Qt.transparent))
        left_layout.addWidget(self.mdi_area)

        # Bottom Toolbar
        bottom_toolbar = self._create_bottom_toolbar()
        left_layout.addWidget(bottom_toolbar)

        return left_panel

    def _create_bottom_toolbar(self):
        """Create bottom toolbar."""
        bottom_toolbar = QGroupBox()
        bottom_toolbar.setObjectName("bottomToolbarPanel")

        bottom_toolbar.setMaximumHeight(44)
        toolbar_layout = QHBoxLayout(bottom_toolbar)
        toolbar_layout.setContentsMargins(4, 2, 4, 2)
        toolbar_layout.setSpacing(8)

        # Delete all button
        self.btn_delete_all = QPushButton()
        self.btn_delete_all.setIcon(Icons.trash(color="#EF4444"))
        self.btn_delete_all.setIconSize(QSize(22, 22))
        self.btn_delete_all.setFixedSize(30, 30)
        bind_tooltip(self.btn_delete_all, "Delete all graphs from workspace")
        toolbar_layout.addWidget(self.btn_delete_all)

        # Minimize all button -- icon only, no label.
        self.btn_minimize_all = QPushButton()
        self.btn_minimize_all.setIcon(Icons.minimize_windows())
        self.btn_minimize_all.setIconSize(QSize(26, 26))
        self.btn_minimize_all.setFixedSize(30, 30)
        bind_tooltip(self.btn_minimize_all, "Minimize All")
        toolbar_layout.addWidget(self.btn_minimize_all)

        # Undo/Redo -- workspace-global actions living in the fixed toolbar
        self.btn_undo = QPushButton()
        self.btn_undo.setIcon(Icons.undo())
        self.btn_undo.setIconSize(QSize(26, 26))
        self.btn_undo.setFixedSize(30, 30)
        bind_tooltip(self.btn_undo, "Undo the last action (Ctrl+Z)")
        self.btn_undo.setEnabled(False)
        toolbar_layout.addWidget(self.btn_undo)

        self.btn_redo = QPushButton()
        self.btn_redo.setIcon(Icons.redo())
        self.btn_redo.setIconSize(QSize(26, 26))
        self.btn_redo.setFixedSize(30, 30)
        bind_tooltip(self.btn_redo, "Redo the last undone action (Ctrl+Shift+Z)")
        self.btn_redo.setEnabled(False)
        toolbar_layout.addWidget(self.btn_redo)

        # Compose Figure -- workspace-global (combines several open graphs)
        self.btn_compose_figure = QPushButton()
        self.btn_compose_figure.setIcon(Icons.multipanel())
        self.btn_compose_figure.setIconSize(QSize(22, 22))
        self.btn_compose_figure.setFixedSize(30, 30)
        bind_tooltip(
            self.btn_compose_figure,
            "Combine several open graphs into one multi-panel exported figure",
        )
        toolbar_layout.addWidget(self.btn_compose_figure)

        # Toolbar buttons should never steal focus from the MDI subwindow/canvas
        for btn in (
            self.btn_delete_all, self.btn_minimize_all,
            self.btn_undo, self.btn_redo, self.btn_compose_figure,
        ):
            btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        # Shared slot the active graph's toolbar_container is reparented into
        self.graph_toolbar_slot = QWidget()
        self._graph_toolbar_slot_layout = QHBoxLayout(self.graph_toolbar_slot)
        self._graph_toolbar_slot_layout.setContentsMargins(0, 0, 0, 0)
        self._graph_toolbar_slot_layout.setSpacing(0)
        toolbar_layout.addWidget(self.graph_toolbar_slot, 1)

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
        group_box = QGroupBox()
        bind_title(group_box, "Loaded dataframes:")
        group_layout = QVBoxLayout(group_box)
        group_layout.setContentsMargins(2, 0, 2, 0)

        # DataFrame listbox and buttons side by side
        df_section_layout = QHBoxLayout()
        df_section_layout.setContentsMargins(0, 0, 0, 0)

        # DataFrame listbox
        self.df_listbox = QListWidget()
        self.df_listbox.setMaximumHeight(110)

        self._has_df_placeholder = False  # Track placeholder state
        self._update_df_placeholder()  # Show placeholder initially

        df_section_layout.addWidget(self.df_listbox)

        # DataFrame buttons (vertical layout on the right)
        df_buttons_layout = QVBoxLayout()

        self.btn_view_df = QPushButton()
        self.btn_view_df.setIcon(Icons.search())
        self.btn_view_df.setIconSize(QSize(18, 18))
        bind_tooltip(self.btn_view_df, "View DataFrame\nCtrl+Click: Open source file")
        self.btn_view_df.setMaximumWidth(35)
        self.btn_view_df.setMaximumHeight(22)

        self.btn_remove_df = QPushButton()
        self.btn_remove_df.setIcon(Icons.trash())
        self.btn_remove_df.setIconSize(QSize(18, 18))
        bind_tooltip(self.btn_remove_df, "Remove DataFrame")
        self.btn_remove_df.setMaximumWidth(35)
        self.btn_remove_df.setMaximumHeight(22)

        self.btn_save_df = QPushButton()
        self.btn_save_df.setIcon(Icons.save())
        self.btn_save_df.setIconSize(QSize(18, 18))
        bind_tooltip(self.btn_save_df, "Save DataFrame to Excel")
        self.btn_save_df.setMaximumWidth(35)
        self.btn_save_df.setMaximumHeight(22)

        self.btn_refresh_df = QPushButton()
        self.btn_refresh_df.setIcon(Icons.refresh())
        self.btn_refresh_df.setIconSize(QSize(18, 18))
        bind_tooltip(self.btn_refresh_df, "Refresh DataFrame from source file")
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
        self.slot_selector_widget = QWidget()
        self.slot_selector_widget.setMaximumHeight(120)
        slot_layout = QVBoxLayout(self.slot_selector_widget)
        slot_layout.setContentsMargins(2, 2, 2, 2)
        slot_layout.setSpacing(2)

        # Label and Select All checkbox in same row
        header_layout = QHBoxLayout()
        lbl_slot = QLabel()
        bind_text(lbl_slot, "Select wafer slot(s):")
        header_layout.addWidget(lbl_slot)

        self.select_all_checkbox = QCheckBox()
        bind_text(self.select_all_checkbox, "Select All")
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

        plot_tab = self._create_plot_tab()
        more_options_tab = self._create_more_options_tab()

        self.plot_tabs.addTab(plot_tab, "")
        bind_tab_text(self.plot_tabs, plot_tab, "Plot")
        self.plot_tabs.addTab(more_options_tab, "")
        bind_tab_text(self.plot_tabs, more_options_tab, "Plot multiple axes (beta)")

        parent_layout.addWidget(self.plot_tabs, stretch=2)

    def _create_plot_tab(self):
        """Create plot tab."""
        tab_plot = QWidget()
        tab_plot_layout = QVBoxLayout(tab_plot)
        tab_plot_layout.setContentsMargins(4, 4, 4, 4)

        # Plot style and color palette
        plot_style_layout = QHBoxLayout()
        lbl_ps = QLabel()
        bind_text(lbl_ps, "Plot style:")
        lbl_ps.setFixedWidth(60)
        plot_style_layout.addWidget(lbl_ps)
        self.cbb_plot_style = QComboBox()
        self.cbb_plot_style.setIconSize(QSize(30, 30))
        bind_combo_items(
            self.cbb_plot_style,
            [
                ("point", "point"),
                ("scatter", "scatter"),
                ("box", "box"),
                ("bar", "bar"),
                ("line", "line"),
                ("trendline", "trendline"),
                ("histogram", "histogram"),
                ("wafer", "wafer"),
                ("2Dmap", "2Dmap"),
            ],
        )
        for i in range(self.cbb_plot_style.count()):
            style = self.cbb_plot_style.itemData(i)
            self.cbb_plot_style.setItemIcon(i, Icons.get(style))
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
                lbl_wafer = QLabel()
                bind_text(lbl_wafer, "Wafer size:")
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
        title_labels_group = QGroupBox()
        bind_title(title_labels_group, "Title and labels:")
        group_layout = QVBoxLayout(title_labels_group)
        group_layout.setContentsMargins(5, 6, 5, 6)
        group_layout.setSpacing(6)

        # Note: static calls use tr literal keys via bind_* below
        labels = [
            ("plot_title", "Plot title:", "Type to modify the plot title"),
            ("plot_subtitle", "Subtitle:", "Optional text shown under the title"),
            ("xlabel", "X label:", "X axis label"),
            ("ylabel", "Y label:", "Y axis label"),
            ("zlabel", "Z label:", "Z axis label"),
        ]

        for attr_name, label_text, placeholder in labels:
            h_layout = QHBoxLayout()
            lbl = QLabel()
            bind_text(lbl, label_text)
            lbl.setFixedWidth(60)
            h_layout.addWidget(lbl)

            line_edit = QLineEdit()
            bind_placeholder(line_edit, placeholder)

            if attr_name in ['xlabel', 'ylabel', 'zlabel']:
                completer = QCompleter(AXIS_LABELS)
                completer.setCaseSensitivity(Qt.CaseInsensitive)
                completer.setCompletionMode(QCompleter.PopupCompletion)
                line_edit.setCompleter(completer)

            setattr(self, f'edit_{attr_name}', line_edit)
            h_layout.addWidget(line_edit)
            group_layout.addLayout(h_layout)

        parent_layout.addWidget(title_labels_group)

    def _build_secondary_axis_group(self, key: str, title: str) -> QGroupBox:
        """One secondary-axis group box (column combo + log toggle + label
        edit); creates self.cbb_<key>, self.cb_<key>log, self.edit_<key>label."""
        name = key.upper()
        group = QGroupBox()
        bind_title(group, title)
        layout = QVBoxLayout(group)
        layout.setContentsMargins(5, 2, 5, 2)
        layout.setSpacing(2)

        col_layout = QHBoxLayout()
        col_layout.addWidget(QLabel(f"{name}:"))
        combo = QComboBox()
        combo.setFixedWidth(200)
        setattr(self, f'cbb_{key}', combo)
        col_layout.addWidget(combo)
        log_cb = QCheckBox()
        bind_text(log_cb, "Log scale")
        setattr(self, f'cb_{key}log', log_cb)
        col_layout.addWidget(log_cb)
        col_layout.addStretch()
        layout.addLayout(col_layout)

        label_layout = QHBoxLayout()
        lbl_ax_label = QLabel()
        bind_text(lbl_ax_label, "{name} label:", name=name)
        label_layout.addWidget(lbl_ax_label)
        edit = QLineEdit()
        bind_placeholder(edit, "{name} axis label", name=name)
        completer = QCompleter(AXIS_LABELS)
        completer.setCaseSensitivity(Qt.CaseInsensitive)
        completer.setCompletionMode(QCompleter.PopupCompletion)
        edit.setCompleter(completer)
        setattr(self, f'edit_{key}label', edit)
        label_layout.addWidget(edit)
        layout.addLayout(label_layout)

        return group

    def _create_more_options_tab(self):
        """Create the 'Plot multiple Axes' tab with Y2, Y3, and X2 controls."""
        tab_more = QWidget()
        tab_more_layout = QVBoxLayout(tab_more)
        tab_more_layout.setContentsMargins(4, 4, 4, 4)
        tab_more_layout.setSpacing(6)

        tab_more_layout.addWidget(
            self._build_secondary_axis_group("y2", "Secondary Y axis (right, red):"))
        tab_more_layout.addWidget(
            self._build_secondary_axis_group("y3", "Tertiary Y axis (right offset, green):"))

        x2_group = self._build_secondary_axis_group("x2", "Secondary X axis (top, purple):")
        self.lbl_info_x2 = QLabel()
        bind_text(self.lbl_info_x2, "Applicable plot styles: point, scatter, line")
        self.lbl_info_x2.setStyleSheet(
            f"color: {current_tokens().text_muted}; font-style: italic; font-size: 10px;"
        )
        x2_group.layout().addWidget(self.lbl_info_x2)
        tab_more_layout.addWidget(x2_group)

        tab_more_layout.addStretch()

        return tab_more

    def _setup_action_buttons(self, parent_layout):
        """Setup action buttons."""
        graph_list_layout = QHBoxLayout()
        lbl_active_plot = QLabel()
        bind_text(lbl_active_plot, "Active plot:")
        graph_list_layout.addWidget(lbl_active_plot)
        self.cbb_graph_list = QComboBox()
        self.cbb_graph_list.setMinimumWidth(150)
        graph_list_layout.addWidget(self.cbb_graph_list)
        parent_layout.addLayout(graph_list_layout)

        action_buttons_layout = QHBoxLayout()

        self.btn_add_plot = QPushButton()
        bind_text(self.btn_add_plot, "Add plot")
        self.btn_add_plot.setIcon(Icons.plus())
        self.btn_add_plot.setIconSize(QSize(20, 20))
        bind_tooltip(self.btn_add_plot, "Add new plot with current configuration")
        self.btn_add_plot.setMinimumHeight(25)

        self.btn_update_plot = QPushButton()
        bind_text(self.btn_update_plot, "Update plot")
        self.btn_update_plot.setIcon(Icons.refresh())
        self.btn_update_plot.setIconSize(QSize(20, 20))
        bind_tooltip(self.btn_update_plot, "Update selected plot with current configuration")
        self.btn_update_plot.setMinimumHeight(25)

        self.btn_add_multi_wafer = QPushButton()
        bind_text(self.btn_add_multi_wafer, "Add Multi-Wafer")
        self.btn_add_multi_wafer.setIcon(Icons.plus())
        self.btn_add_multi_wafer.setIconSize(QSize(20, 20))
        bind_tooltip(self.btn_add_multi_wafer, "Create wafer plots for selected slots")
        self.btn_add_multi_wafer.setMinimumHeight(25)
        self.btn_add_multi_wafer.setVisible(False)  # Initially hidden

        action_buttons_layout.addWidget(self.btn_add_plot)
        action_buttons_layout.addWidget(self.btn_update_plot)
        action_buttons_layout.addWidget(self.btn_add_multi_wafer)

        parent_layout.addLayout(action_buttons_layout)

        template_buttons_layout = QHBoxLayout()

        self.btn_apply_recipe = QPushButton()
        bind_text(self.btn_apply_recipe, "Plot Recipes")
        self.btn_apply_recipe.setIcon(Icons.get("recipe"))
        self.btn_apply_recipe.setIconSize(QSize(18, 18))
        bind_tooltip(self.btn_apply_recipe, "Browse & apply saved plot recipes")
        self.btn_apply_recipe.setMinimumHeight(25)

        self.btn_save_as_recipe = QPushButton()
        bind_text(self.btn_save_as_recipe, "Save Plot Recipe")
        self.btn_save_as_recipe.setIcon(Icons.save())
        self.btn_save_as_recipe.setIconSize(QSize(18, 18))
        bind_tooltip(
            self.btn_save_as_recipe,
            "Save all currently open plots as a Plot Recipe",
        )
        self.btn_save_as_recipe.setMinimumHeight(25)

        template_buttons_layout.addWidget(self.btn_apply_recipe)
        template_buttons_layout.addWidget(self.btn_save_as_recipe)

        parent_layout.addLayout(template_buttons_layout)

    def apply_theme(self, theme: str):
        """Update sidebar icon colors for the current application theme."""
        tok = current_tokens()
        icon_color = tok.text_primary

        self.btn_view_df.setIcon(Icons.search(color=icon_color))
        self.btn_remove_df.setIcon(Icons.trash(color=icon_color))
        self.btn_save_df.setIcon(Icons.save(color=icon_color))
        self.btn_refresh_df.setIcon(Icons.refresh(color=icon_color))
        self.btn_add_plot.setIcon(Icons.plus(color=icon_color))
        self.btn_update_plot.setIcon(Icons.refresh(color=icon_color))
        self.btn_add_multi_wafer.setIcon(Icons.plus(color=icon_color))
        if hasattr(self, 'btn_apply_recipe'):
            self.btn_apply_recipe.setIcon(Icons.get('recipe', color=icon_color))
        if hasattr(self, 'btn_save_as_recipe'):
            self.btn_save_as_recipe.setIcon(Icons.save(color=icon_color))

        # v_data_filter buttons
        self.v_data_filter.btn_add.setIcon(Icons.plus(color=icon_color))
        self.v_data_filter.btn_remove.setIcon(Icons.close(color=icon_color))
        self.v_data_filter.btn_apply.setIcon(Icons.check(color=icon_color))

        # Bottom toolbar buttons
        if hasattr(self, 'btn_delete_all'):
            self.btn_delete_all.setIcon(Icons.trash(color=tok.error))
        if hasattr(self, 'btn_minimize_all'):
            self.btn_minimize_all.setIcon(Icons.minimize_windows(color=icon_color))
        if hasattr(self, 'btn_undo'):
            self.btn_undo.setIcon(Icons.undo(color=icon_color))
        if hasattr(self, 'btn_redo'):
            self.btn_redo.setIcon(Icons.redo(color=icon_color))
        if hasattr(self, 'btn_compose_figure'):
            self.btn_compose_figure.setIcon(Icons.multipanel(color=icon_color))

        # Refresh plot style item icons
        if hasattr(self, 'cbb_plot_style'):
            for i in range(self.cbb_plot_style.count()):
                style = self.cbb_plot_style.itemData(i) or self.cbb_plot_style.itemText(i)
                self.cbb_plot_style.setItemIcon(i, Icons.get(style, color=icon_color))

        if hasattr(self, 'lbl_info_x2'):
            self.lbl_info_x2.setStyleSheet(
                f"color: {tok.text_muted}; font-style: italic; font-size: 10px;"
            )

        for entry in getattr(self, 'graph_widgets', {}).values():
            gw = entry[0] if isinstance(entry, (tuple, list)) and entry else entry
            if hasattr(gw, 'apply_global_theme'):
                gw.apply_global_theme(theme)

    def _setup_keyboard_shortcuts(self):
        """Undo/redo/copy-figure/paste-style/customize accelerators scoped to self.mdi_area."""
        undo_sc = QShortcut(QKeySequence("Ctrl+Z"), self.mdi_area)
        undo_sc.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        undo_sc.activated.connect(self._on_undo_clicked)

        redo_sc = QShortcut(QKeySequence("Ctrl+Shift+Z"), self.mdi_area)
        redo_sc.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        redo_sc.activated.connect(self._on_redo_clicked)

        copy_figure_sc = QShortcut(QKeySequence("Ctrl+C"), self.mdi_area)
        copy_figure_sc.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        copy_figure_sc.activated.connect(self._on_copy_figure_shortcut)

        paste_style_sc = QShortcut(QKeySequence("Ctrl+V"), self.mdi_area)
        paste_style_sc.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        paste_style_sc.activated.connect(self._on_paste_style_shortcut)

        customize_sc = QShortcut(QKeySequence("Ctrl+E"), self.mdi_area)
        customize_sc.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        customize_sc.activated.connect(self._on_customize_shortcut)

        rescale_sc = QShortcut(QKeySequence("Ctrl+R"), self.mdi_area)
        rescale_sc.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        rescale_sc.activated.connect(self._on_rescale_shortcut)

    def _update_df_placeholder(self):
        """Update placeholder text for dataframe list based on state."""
        if self.df_listbox.count() == 0:
            for _ in range(2):
                spacer = QListWidgetItem("")
                spacer.setFlags(Qt.NoItemFlags)
                self.df_listbox.addItem(spacer)

            placeholder = QListWidgetItem(
                tr("Drag and drop file(s) anywhere to open")
            )
            placeholder.setFlags(Qt.NoItemFlags)
            placeholder.setForeground(Qt.gray)
            placeholder.setTextAlignment(Qt.AlignCenter)

            font = QFont()
            font.setPointSize(11)
            placeholder.setFont(font)

            self.df_listbox.addItem(placeholder)

            self._has_df_placeholder = True
        else:
            if self._has_df_placeholder:
                i = 0
                while i < self.df_listbox.count():
                    if self.df_listbox.item(i).flags() == Qt.NoItemFlags:
                        self.df_listbox.takeItem(i)
                    else:
                        i += 1
                self._has_df_placeholder = False
