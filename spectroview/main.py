# main.py
import sys
import os
from pathlib import Path
import matplotlib as mpl
mpl.use('qtagg')

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*Glyph.*")

import pandas as pd

from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QTabWidget, QFileDialog, QMessageBox
from PySide6.QtCore import Qt, QFileInfo, QUrl
from PySide6.QtGui import QIcon, QDesktopServices

from spectroview.model.m_file_converter import MFileConverter
from spectroview.model.m_spc import SpcReader
from spectroview.model.m_settings import MSettings
from spectroview.model.m_update_checker import UpdateCheckerWorker

from spectroview.viewmodel.vm_settings import VMSettings

from spectroview.view.components.v_settings import VSettingsDialog
from spectroview.view.components.v_about import VAboutDialog
from spectroview.view.components.v_update_banner import VUpdateBanner
from spectroview.view.components.v_quick_calculators import MQuickCalc
from spectroview.view.components.v_menubar import VMenuBar
from spectroview.view.components.v_user_manual import VUserManualDialog
from spectroview.view.v_workspace_spectra import VWorkspaceSpectra
from spectroview.view.v_workspace_maps import VWorkspaceMaps
from spectroview.view.v_workspace_graphs import VWorkspaceGraphs
from spectroview.view.theme import ThemeManager

from spectroview import LOGO_APPLI, USER_MANUAL_DIR

try:
    from renishawWiRE import WDFReader
    WDF_AVAILABLE = True
except ImportError:
    WDF_AVAILABLE = False

try:
    from spectroview.ai_agent.v_chat_panel import VChatPanel
    LLM_AVAILABLE = True
    LLM_ERROR_MSG = ""
except ImportError as e:
    VChatPanel = None   # type: ignore[assignment,misc]
    LLM_AVAILABLE = False
    LLM_ERROR_MSG = str(e)

class Main(QMainWindow):
    def __init__(self):
        super().__init__()
        self.settings = MSettings()
        self.theme_mgr = ThemeManager(self.settings)

        # Apply global application style BEFORE creating widgets to avoid expensive unpolish/polish
        self.theme_mgr.apply(self.settings.get_theme())
        
        self.init_ui()
        self._propagate_theme(self.settings.get_theme())
        
        self.setup_connections()
        self.tabWidget.setCurrentWidget(self.v_maps_workspace)
        
        # Lazy chat panel (created on first use)
        self._chat_panel = None


    def init_ui(self):
        self.setWindowTitle(
            "SPECTROview (Tool for Spectroscopic Data Processing and Visualization)"
        )
        self.setGeometry(100, 100, 1400, 930)
        self.setWindowIcon(QIcon(LOGO_APPLI))

        # Central widget
        central = QWidget(self)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(0)

        # ── Update notification banner (created lazily when an update is detected) ──
        self._update_banner = None

        # Main Tab Widget
        self.tabWidget = QTabWidget(central)

        self.v_spectra_workspace = VWorkspaceSpectra()
        self.v_graphs_workspace = VWorkspaceGraphs()
        self.v_maps_workspace = VWorkspaceMaps()

        self.tabWidget.addTab(self.v_spectra_workspace, "Spectra")
        self.tabWidget.addTab(self.v_maps_workspace, "Maps")
        self.tabWidget.addTab(self.v_graphs_workspace, "Graphs")
        
        # Enable Drag & Drop
        self.setAcceptDrops(True)

        layout.addWidget(self.tabWidget)
        self.setCentralWidget(central)

        # Toolbar
        self.menu_bar = VMenuBar()
        self.addToolBar(Qt.TopToolBarArea, self.menu_bar)
        
    def setup_connections(self):
        self.menu_bar.open_requested.connect(self.open_files)
        self.menu_bar.save_requested.connect(self.save)
        self.menu_bar.clear_requested.connect(self.clear_workspace)
        self.menu_bar.settings_requested.connect(self._open_settings)
        self.menu_bar.convert_requested.connect(self.file_converter)
        self.menu_bar.calc_requested.connect(self.quick_calc)

        self.menu_bar.about_requested.connect(self.about)
        self.menu_bar.manual_requested.connect(self.manual) 
        self.menu_bar.github_requested.connect(self.open_github_repo)
        self.menu_bar.version_requested.connect(self.open_releases)
        self.menu_bar.check_update_requested.connect(self._manual_update_check)
        self.menu_bar.theme_selected.connect(self.toggle_theme)
        self.menu_bar.ai_chat_requested.connect(self.open_ai_chat)
        
        # Inject Graphs workspace into Maps ViewModel for cross-workspace communication
        self.v_maps_workspace.vm.set_graphs_workspace(self.v_graphs_workspace)

        # Maps → Spectra: ingest spectra sent from the Maps workspace
        self.v_maps_workspace.vm.send_spectra_to_workspace.connect(
            self.v_spectra_workspace.vm.receive_spectra
        )

        # Connect Maps to Main: tab switching
        self.v_maps_workspace.vm.switch_to_graphs_tab.connect(
            lambda: self.tabWidget.setCurrentWidget(self.v_graphs_workspace)
        )
        
        # Synchronize plot style between SpectraViewer and MapViewer
        def update_map_viewers():
            style_name = self.v_maps_workspace.v_spectra_viewer.cbb_theme.currentText()
            self.v_maps_workspace.v_map_viewer.apply_plot_style(style_name)
            for dialog in getattr(self.v_maps_workspace, 'viewer_dialogs', []):
                if hasattr(dialog, 'map_viewer'):
                    dialog.map_viewer.apply_plot_style(style_name)
        
        if hasattr(self.v_maps_workspace, 'v_spectra_viewer'):
            self.v_maps_workspace.v_spectra_viewer.plotStyleChanged.connect(update_map_viewers)
            
        # Synchronize options between Spectra and Maps workspaces
        if hasattr(self.v_spectra_workspace, 'v_spectra_viewer') and hasattr(self.v_maps_workspace, 'v_spectra_viewer'):
            self.v_spectra_workspace.v_spectra_viewer.allOptionsSyncChanged.connect(
                self.v_maps_workspace.v_spectra_viewer.set_options_state
            )
            self.v_maps_workspace.v_spectra_viewer.allOptionsSyncChanged.connect(
                self.v_spectra_workspace.v_spectra_viewer.set_options_state
            )
            
            # Load persisted view options and apply them
            persisted_view_options = self.settings.load_view_options()
            self.v_spectra_workspace.v_spectra_viewer.set_options_state(persisted_view_options)
            # The above will sync to the maps workspace automatically, but we can do it explicitly just in case
            self.v_maps_workspace.v_spectra_viewer.set_options_state(persisted_view_options)
            
            # Save view options whenever they change
            self.v_spectra_workspace.v_spectra_viewer.allOptionsSyncChanged.connect(self.settings.save_view_options)
        

    def open_files(self):
        """Universal file opener supporting all SPECTROview formats."""
        last_dir = self.settings.get_last_directory()
        paths, _ = QFileDialog.getOpenFileNames(
            None,
            "Open file(s)",
            last_dir,
            "SPECTROview formats (*.csv *.txt *.dat *.wdf *.spc *.spectra *.maps *.graphs *.xlsx)"
        )
        
        if not paths:
            return
        
        self._load_files_by_paths(paths)
    
    def _load_files_by_paths(self, paths: list):
        """Load files from a list of paths into appropriate workspaces.
        
        This method is called by both the file dialog (open_files) and drag-and-drop operations.
        """
        if not paths:
            return
        
        # Save last directory
        last_dir = QFileInfo(paths[0]).absolutePath()
        self.settings.set_last_directory(last_dir)
        
        # Categorize files by type
        spectra_files = []
        hyperspectral_files = []
        dataframes = []
        spectra_work_file = None
        maps_work_file = None
        graphs_work_file = None
        
        for file_path in paths:
            path = Path(file_path)
            ext = path.suffix.lower()
            
            # Saved workspace files
            if ext == '.spectra':
                spectra_work_file = str(path)
            elif ext == '.maps':
                maps_work_file = str(path)
            elif ext == '.graphs':
                graphs_work_file = str(path)
            elif ext == '.dat':
                # TRPL time-resolved data
                spectra_files.append(str(path))
            elif ext == '.wdf':
                # Renishaw WiRE native format - detect if it's a map or single spectrum
                try:
                    if not WDF_AVAILABLE:
                        raise ImportError("renishawWiRE library is not installed.")
                    reader = WDFReader(str(path))
                    # Check measurement type: Mapping = hyperspectral, Single = spectrum
                    # measurement_type is an enum, so convert to string for comparison
                    if str(reader.measurement_type) == 'Mapping' or 'Mapping' in str(reader.measurement_type):
                        hyperspectral_files.append(str(path))
                    else:
                        spectra_files.append(str(path))
                    reader.close()
                except Exception as e:
                    QMessageBox.warning(self, "WDF Error", f"Failed to read WDF file {path.name}: {e}")
            elif ext == '.spc':
                # Galactic SPC format
                try:
                    reader = SpcReader(str(path))
                    # Check if multifile (fnsub > 1) -> likely a map or series
                    if reader.header['fnsub'] > 1:
                        hyperspectral_files.append(str(path))
                    else:
                        spectra_files.append(str(path))
                except Exception as e:
                     QMessageBox.warning(self, "SPC Error", f"Failed to read SPC file {path.name}: {e}")
            elif ext == '.xlsx':
                dataframes.append(str(path))
            elif ext in ['.csv', '.txt']:
                # Detect if it's a dataframe, spectrum, or hyperspectral map data
                try:
                    # Read first line to determine file type
                    with open(path, 'r') as f:
                        first_line = f.readline().strip()
                    
                    # Determine if this is a saved dataframe CSV vs map/spectrum data
                    is_dataframe_csv = False
                    is_wafer_map = False
                    
                    if ext == '.csv':
                        # Check for wafer map CSV signature
                        if "Dynamic Sitebased Spectral" in first_line:
                            is_wafer_map = True
                        # Check if it's a saved dataframe (has semicolons and text header)
                        elif ';' in first_line:
                            first_values = first_line.split(';')
                            # Try to parse first value as float
                            try:
                                float(first_values[0])
                                # Numeric header, not a saved dataframe
                                is_dataframe_csv = False
                            except (ValueError, AttributeError):
                                # Text header = saved dataframe
                                is_dataframe_csv = True
                    
                    if is_dataframe_csv:
                        # CSV with dataframe header format
                        dataframes.append(str(path))
                    elif is_wafer_map:
                        # Wafer map CSV
                        hyperspectral_files.append(str(path))
                    else:
                        # Spectroscopic data (map or spectrum) - need to check structure
                        if ext == '.csv':
                            # CSV files use semicolon delimiter and have 3 header rows for maps
                            delimiter = ";"
                            skiprows = 3
                            engine = 'c'
                        else:  # .txt
                            # Auto-detect delimiter by reading first lines
                            with open(path, 'r') as f:
                                first_line = next(f, None)
                                second_line = next(f, None)
                            
                            test_line = second_line if second_line else first_line
                            if test_line:
                                if ';' in test_line:
                                    delimiter = ';'
                                    engine = 'c'
                                elif '\t' in test_line:
                                    delimiter = '\t'
                                    engine = 'c'
                                else:
                                    delimiter = r'\s+'  # space/whitespace
                                    engine = 'python'
                            else:
                                delimiter = '\t'
                                engine = 'c'
                            skiprows = 1  # TXT files typically have 1 header row
                        
                        df = pd.read_csv(path, delimiter=delimiter, header=None, 
                                       skiprows=skiprows, nrows=5, engine=engine)
                        
                        if df.shape[1] == 2:
                            spectra_files.append(str(path))
                        elif df.shape[1] > 3:
                            hyperspectral_files.append(str(path))
                        else:
                            QMessageBox.warning(self, "Invalid File", f"Invalid number of columns in {path.name}")
                except Exception as e:
                    QMessageBox.warning(self, "Read Error", f"Failed to read {path.name}: {e}")
            else:
                QMessageBox.warning(self, "Unsupported Format", f"Unsupported file format: {ext}")
        
        # Load files into appropriate workspaces
        if spectra_files:
            self.v_spectra_workspace.vm.load_files(spectra_files)
            self.tabWidget.setCurrentWidget(self.v_spectra_workspace)
        
        if hyperspectral_files:
            self.v_maps_workspace.vm.load_map_files(hyperspectral_files)
            self.tabWidget.setCurrentWidget(self.v_maps_workspace)
        
        if dataframes:
            self.v_graphs_workspace.vm.load_dataframes(dataframes)
            self.tabWidget.setCurrentWidget(self.v_graphs_workspace)
        
        # Load saved work files
        if spectra_work_file:
            self.v_spectra_workspace.load_work(spectra_work_file)
            self.tabWidget.setCurrentWidget(self.v_spectra_workspace)
        
        if maps_work_file:
            self.v_maps_workspace.load_work(maps_work_file)
            self.tabWidget.setCurrentWidget(self.v_maps_workspace)
        
        if graphs_work_file:
            self.v_graphs_workspace.load_workspace(graphs_work_file)
            self.tabWidget.setCurrentWidget(self.v_graphs_workspace)

    def save(self):
        """Save current workspace based on active tab."""
        current_tab = self.tabWidget.currentWidget()
        
        if current_tab == self.v_spectra_workspace:
            self.v_spectra_workspace.save_work()
        elif current_tab == self.v_maps_workspace:
            self.v_maps_workspace.save_work()
        elif current_tab == self.v_graphs_workspace:
            self.v_graphs_workspace.save_workspace()
        else:
            QMessageBox.warning(self, "No Tab Selected", "No valid tab is selected for saving.")

    def clear_workspace(self):
        """Clear current workspace based on active tab without confirmation."""
        current_tab = self.tabWidget.currentWidget()
        
        if current_tab == self.v_spectra_workspace:
            self.v_spectra_workspace.clear_workspace()
        elif current_tab == self.v_maps_workspace:
            self.v_maps_workspace.clear_workspace()
        elif current_tab == self.v_graphs_workspace:
            self.v_graphs_workspace.clear_workspace()
        else:
            QMessageBox.warning(self, "No Tab Selected", "Nothing to clear.")

    def _open_settings(self):
        """   Open settings dialog. """
        vm = VMSettings()
        dlg = VSettingsDialog(vm, self)
        if dlg.exec():
            # Refresh viewers to reflect settings changes (e.g. coef_noise)
            if hasattr(self, 'v_spectra_workspace') and hasattr(self.v_spectra_workspace, 'v_spectra_viewer'):
                self.v_spectra_workspace.v_spectra_viewer._plot()
            if hasattr(self, 'v_maps_workspace') and hasattr(self.v_maps_workspace, 'v_spectra_viewer'):
                self.v_maps_workspace.v_spectra_viewer._plot()

    def file_converter(self):
        """Open file converter dialog for hyperspectral data."""
        dlg = MFileConverter(self.settings, self)
        dlg.exec()

    def quick_calc(self):
        """Open quick calculation dialog."""
        if not hasattr(self, '_quick_calc_dlg'):
            self._quick_calc_dlg = MQuickCalc(self)
            self._quick_calc_dlg.setWindowFlags(self._quick_calc_dlg.windowFlags() | Qt.WindowStaysOnTopHint)
        self._quick_calc_dlg.show()
        self._quick_calc_dlg.raise_()
        self._quick_calc_dlg.activateWindow()

    def open_ai_chat(self):
        """Open (or raise) the SPECTROview AI Agent panel.

        The panel is created lazily on first use and then kept alive so
        the conversation history is preserved across multiple open/close
        cycles.  The active DataFrame from the Graphs workspace is
        injected each time the panel is shown.
        """
        if not LLM_AVAILABLE:
            QMessageBox.information(
                self,
                "SPECTROview AI Agent — Not Available",
                f"The AI Chat module could not be imported.\nError: {LLM_ERROR_MSG}\n\n"
                "Please install the optional dependencies:\n"
                "    pip install ollama mcp\n\n"
                "Then restart SPECTROview.",
            )
            return

        # Lazy creation
        if self._chat_panel is None:
            self._chat_panel = VChatPanel(self)
            # When the AI suggests a plot, configure the Graphs workspace
            self._chat_panel.plot_requested.connect(self._on_chat_plot_requested)

            # Keep chat panel in sync with workspace dataframes
            def sync_chat_dfs_full(*args):
                """Called when dataframes are added/removed."""
                vm_graphs = self.v_graphs_workspace.vm
                self._chat_panel.set_dataframes(vm_graphs.dataframes, vm_graphs.selected_df_name or "")
                self._chat_panel.vm.set_graphs(vm_graphs.graphs)

            def sync_chat_active(*args):
                """Called when the user selects a different dataframe — preserve history."""
                vm_graphs = self.v_graphs_workspace.vm
                active = vm_graphs.selected_df_name or ""
                self._chat_panel.vm.update_active_df_name(active)

            def sync_chat_graphs(*args):
                """Called when graphs are added/removed/updated."""
                vm_graphs = self.v_graphs_workspace.vm
                self._chat_panel.vm.set_graphs(vm_graphs.graphs)

            self.v_graphs_workspace.vm.dataframes_changed.connect(sync_chat_dfs_full)
            self.v_graphs_workspace.vm.dataframe_columns_changed.connect(sync_chat_active)
            self.v_graphs_workspace.vm.graphs_changed.connect(sync_chat_graphs)

        # Toggle: clicking the toolbar button again while the panel is
        # already open closes it, instead of just re-focusing it.
        if self._chat_panel.isVisible():
            self._chat_panel.hide()
            return

        # Force a sync right now when opening
        vm_graphs = self.v_graphs_workspace.vm
        self._chat_panel.set_dataframes(vm_graphs.dataframes, vm_graphs.selected_df_name or "")
        self._chat_panel.vm.set_graphs(vm_graphs.graphs)

        self._chat_panel.show()
        self._chat_panel.raise_()
        self._chat_panel.activateWindow()

    def _on_chat_plot_requested(self, plot_config: dict):
        """Apply the AI-suggested plot configuration or graph update to the Graphs workspace.

        Normalizes the AI's JSON output so it matches the MGraph model's
        expected types (e.g. y must be a list, limits must be float|None).
        """
        # ── Handle graph UPDATE (existing graph by ID) ───────────────
        if "_graph_update" in plot_config:
            self._apply_graph_update(plot_config["_graph_update"])
            return

        # ── Handle graph DELETE ──────────────────────────────────────
        if "_graph_delete" in plot_config:
            self._apply_graph_delete(plot_config["_graph_delete"])
            return
        import copy
        ws = self.v_graphs_workspace
        # Switch to Graphs tab
        self.tabWidget.setCurrentWidget(ws)

        df_name = plot_config.get("df_name")
        if not df_name:
            df_name = ws.vm.selected_df_name
        if not df_name:
            return

        # Deep copy so we don't mutate the original
        cfg = copy.deepcopy(plot_config)

        # ── Normalise types and structures ───────────────────────────
        # Use the single source of truth from plot_tool to ensure
        # filters (list of dicts), lists, and numeric types are correct.
        from spectroview.ai_agent.utils.plot_utils import normalize_plot_config
        normalize_plot_config(cfg)

        # ── Ensure df_name is set ────────────────────────────────────
        cfg['df_name'] = df_name

        # Create the plot directly via the workspace API
        ws.create_plot_from_config(df_name, cfg)

        # Update the sidebar combo boxes to reflect last plot config
        def _set_combo(cbb, value):
            if value:
                idx = cbb.findText(str(value))
                if idx >= 0:
                    cbb.setCurrentIndex(idx)

        if hasattr(ws, 'cbb_x'): _set_combo(ws.cbb_x, plot_config.get("x"))
        if hasattr(ws, 'cbb_y'): _set_combo(ws.cbb_y, plot_config.get("y"))
        if hasattr(ws, 'cbb_z'): _set_combo(ws.cbb_z, plot_config.get("z"))
        if hasattr(ws, 'cbb_plot_style'): _set_combo(ws.cbb_plot_style, plot_config.get("plot_style"))

    def _apply_graph_update(self, update_payload: dict):
        """Update an existing graph by ID with new properties from the AI."""
        import copy
        ws = self.v_graphs_workspace
        graph_id = update_payload.get("graph_id")
        properties = update_payload.get("properties", {})

        if graph_id is None or not isinstance(properties, dict):
            return

        graph_id = int(graph_id)
        model = ws.vm.get_graph(graph_id)
        if model is None:
            return

        # Normalize types using single source of truth
        props = copy.deepcopy(properties)
        from spectroview.ai_agent.utils.plot_utils import normalize_plot_config
        normalize_plot_config(props)

        # Apply to model
        ws.vm.update_graph(graph_id, props)

        # Re-render the existing graph widget
        if graph_id in ws.graph_widgets:
            graph_widget, _, sub_window = ws.graph_widgets[graph_id]
            updated_model = ws.vm.get_graph(graph_id)
            filtered_df = ws.vm.apply_filters(updated_model.df_name, updated_model.filters)
            ws._configure_graph_from_model(graph_widget, updated_model)
            graph_widget.create_plot_widget(updated_model.dpi)
            try:
                ws._render_plot(graph_widget, filtered_df, updated_model)
            except Exception as e:
                QMessageBox.warning(self, "Graph Update Error", f"Could not re-render graph {graph_id}:\n{e}")
            # create_plot_widget() rebuilt this graph's toolbar_container --
            # re-sync in case this graph happens to be the active one.
            ws._sync_active_graph_toolbar()

        # Switch to Graphs tab to show the result
        self.tabWidget.setCurrentWidget(ws)

    def _apply_graph_delete(self, delete_payload: dict):
        """Delete requested graphs based on the AI instructions."""
        ws = self.v_graphs_workspace
        delete_all = delete_payload.get("delete_all", False)
        target_ids = delete_payload.get("graph_ids", [])
        
        # Collect IDs to close
        ids_to_close = []
        open_ids = list(ws.graph_widgets.keys())
        
        if delete_all:
            ids_to_close = open_ids
            # If they meant "delete all except [1,2,3]"
            if target_ids:
                ids_to_close = [gid for gid in open_ids if gid not in target_ids]
        else:
            ids_to_close = [gid for gid in target_ids if gid in open_ids]
            
        # Close the subwindows (this triggers the closed signal which cleans up the model)
        for gid in ids_to_close:
            _, _, sub_window = ws.graph_widgets[gid]
            sub_window.close()

    def about(self):
        """Show About dialog."""
        dlg = VAboutDialog(self)
        dlg.exec()

    def manual(self):
        """Open integrated user manual MD viewer or web documentation."""
        modifiers = QApplication.keyboardModifiers()
        if modifiers & Qt.ControlModifier:
            url = QUrl("https://cea-metrocarac.github.io/SPECTROview/")
            QDesktopServices.openUrl(url)
            return

        if not os.path.exists(USER_MANUAL_DIR):
            QMessageBox.warning(
                self, 
                "Manual Not Found", 
                f"User manual not found at:\n{USER_MANUAL_DIR}"
            )
            return
            
        if not hasattr(self, '_manual_dlg') or self._manual_dlg is None:
            self._manual_dlg = VUserManualDialog(USER_MANUAL_DIR, self)
        self._manual_dlg.show()
        self._manual_dlg.raise_()
        self._manual_dlg.activateWindow()

    def open_github_repo(self):
        """Open the project's GitHub repository."""
        url = QUrl("https://github.com/CEA-MetroCarac/SPECTROview/")
        QDesktopServices.openUrl(url)

    def open_releases(self):
        """Open the project's releases page."""
        url = QUrl("https://cea-metrocarac.github.io/SPECTROview/changelog/")
        QDesktopServices.openUrl(url)

    def toggle_theme(self, theme=None):
        if theme is None:
            theme = "light" if self.settings.get_theme() == "dark" else "dark"

        # Apply palette + QSS + Fusion refresh via the manager
        self.theme_mgr.apply(theme)
        self._propagate_theme(theme)

    def _propagate_theme(self, theme):
        # Derived helpers
        viewer_theme = self.theme_mgr.viewer_theme_name
        ws_theme     = self.theme_mgr.workspace_theme

        # Keep menubar checkmark in sync
        if hasattr(self, 'menu_bar'):
            self.menu_bar.set_current_theme(theme)

        # Propagate to workspaces (icons, plot backgrounds)
        for ws in (getattr(self, 'v_spectra_workspace', None),
                   getattr(self, 'v_maps_workspace', None)):
            if ws is None:
                continue
            if hasattr(ws, 'v_spectra_viewer'):
                ws.v_spectra_viewer.cbb_theme.setCurrentText(viewer_theme)
                if hasattr(ws.v_spectra_viewer, 'apply_global_theme'):
                    ws.v_spectra_viewer.apply_global_theme(ws_theme)
            if hasattr(ws, 'apply_theme'):
                ws.apply_theme(ws_theme)

        if hasattr(self, 'v_graphs_workspace'):
            self.v_graphs_workspace.apply_theme(ws_theme)

        # Keep update banner in sync with the current theme
        if self._update_banner is not None:
            self._update_banner.apply_theme(theme)

    def dragEnterEvent(self, event):
        """Accept dragging files into the application."""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        """Handle dropped files."""
        if event.mimeData().hasUrls():
            paths = [url.toLocalFile() for url in event.mimeData().urls()]
            self._load_files_by_paths(paths)
            event.acceptProposedAction()

    # ── Update checker ────────────────────────────────────────────────────────
    def _start_update_check(self):
        """Launch a background thread to query GitHub for the latest release."""
        if getattr(self, '_has_checked_for_updates', False):
            return
            
        if not self.settings.get_check_for_updates():
            return
            
        self._has_checked_for_updates = True

        from spectroview import VERSION
        self._checker = UpdateCheckerWorker(current_version=VERSION)
        self._checker.update_available.connect(self._on_update_available)
        self._checker.start()

    def _on_update_available(self, tag: str, notes: str, html_url: str):
        """Show the update banner when a newer version is found on GitHub."""
        # Never show if the user already skipped this exact version
        if self.settings.get_skipped_version() == tag:
            return

        if self._update_banner is not None:
            return   # already showing

        banner = VUpdateBanner(
            tag=tag,
            html_url=html_url,
            on_skip=self.settings.set_skipped_version,
            on_dismiss=self._hide_banner,
            parent=self.centralWidget(),
        )
        # Apply current theme
        banner.apply_theme(self.settings.get_theme())

        # Insert banner into the layout at position 0 (above tab widget)
        self.centralWidget().layout().insertWidget(0, banner)
        self._update_banner = banner

    def _hide_banner(self):
        """Reset the banner reference (the widget removes itself via deleteLater)."""
        self._update_banner = None

    def _manual_update_check(self):
        """User clicked 'Check for updates' in the menu bar — always runs (no throttle)."""
        from spectroview import VERSION
        self._manual_checker = UpdateCheckerWorker(current_version=VERSION)
        self._manual_check_found_update = False
        self._manual_checker.update_available.connect(self._on_manual_update_found)
        self._manual_checker.check_finished.connect(self._on_manual_check_done)
        self._manual_checker.start()

    def _on_manual_update_found(self, tag: str, notes: str, html_url: str):
        """A newer version was found during a user-initiated check."""
        self._manual_check_found_update = True
        # Show banner even if user previously skipped this version
        if self._update_banner is not None:
            return

        banner = VUpdateBanner(
            tag=tag,
            html_url=html_url,
            on_skip=self.settings.set_skipped_version,
            on_dismiss=self._hide_banner,
            parent=self.centralWidget(),
        )
        banner.apply_theme(self.settings.get_theme())
        self.centralWidget().layout().insertWidget(0, banner)
        self._update_banner = banner

    def _on_manual_check_done(self):
        """Show 'up to date' message if the manual check found nothing new."""
        from spectroview import VERSION
        if not self._manual_check_found_update:
            QMessageBox.information(
                self,
                "No Update Available",
                f"You are already using the latest version of SPECTROview (v{VERSION}).",
            )

    def showEvent(self, event):
        """Start the update check after the window has been displayed."""
        super().showEvent(event)
        # Use a short single-shot timer so the UI paints before the thread starts
        from PySide6.QtCore import QTimer
        QTimer.singleShot(2000, self._start_update_check)

    def closeEvent(self, event):
        """Clean up on application exit to prevent Matplotlib C++ threading crashes."""
        try:
            import matplotlib.pyplot as plt
            plt.close('all')
        except Exception:
            pass
        event.accept()

def launcher():
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon(LOGO_APPLI))
    app.setStyle("Fusion")

    window = Main()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    launcher()
