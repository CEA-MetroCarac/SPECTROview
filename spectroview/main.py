# main.py
import sys
import os
import importlib.util
import threading
from pathlib import Path

# ── Windows taskbar identity ─────────────────────────────────────────────────
# Must run BEFORE any PySide6/Qt import. When Qt is first imported it
# initialises the Windows platform plugin which registers the process with
# the Shell.  If we haven't set our own AppUserModelID by that point,
# Windows associates the process with python.exe's default icon and the
# taskbar button will never show the SPECTROview logo.
# This mirrors the pattern used in PLUME (see plume/__main__.py).
from spectroview.winapi import apply_windows_taskbar_icon, set_current_process_app_id
set_current_process_app_id()
# ─────────────────────────────────────────────────────────────────────────────

import matplotlib as mpl
mpl.use('qtagg')

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*Glyph.*")

import pandas as pd

from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QTabWidget, QFileDialog, QMessageBox
from PySide6.QtCore import Qt, QFileInfo, QUrl, QTimer
from PySide6.QtGui import QIcon, QDesktopServices

from spectroview.model.m_file_converter import MFileConverter
from spectroview.model.m_spc import SpcReader
from spectroview.model.m_settings import MSettings
from spectroview.model.m_update_checker import (
    UpdateCheckerWorker,
    UpdateDownloadWorker,
    UpdateInstallationError,
    get_update_python_executable,
    install_update_and_restart,
)

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

from spectroview import LOGO_APPLI, USER_MANUAL_DIR, get_app_icon

try:
    from renishawWiRE import WDFReader
    WDF_AVAILABLE = True
except ImportError:
    WDF_AVAILABLE = False

# The AI chat module drags in the provider SDKs (~4.5 s of import time), so it is
# only probed here (find_spec locates the module without executing it) and
# imported for real when the user opens the panel.
try:
    LLM_AVAILABLE = importlib.util.find_spec("spectroview.ai_agent.v_chat_panel") is not None
    LLM_ERROR_MSG = "" if LLM_AVAILABLE else "spectroview.ai_agent.v_chat_panel not found"
except ImportError as e:      # the ai_agent package itself is absent
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

        # One domain-oriented facade over the running application. The AI
        # Chat's graph commands and the optional external MCP endpoint both
        # reuse it; all external-thread calls are queued onto this GUI thread.
        from spectroview.application.dispatch import QtMainThreadDispatcher
        from spectroview.application.service import SpectroviewApplicationAPI
        self._application_dispatcher = QtMainThreadDispatcher(self)
        self.application_api = SpectroviewApplicationAPI(
            self, dispatcher=self._application_dispatcher
        )
        self._mcp_runtime = None
        self._mcp_error = ""
        self._sync_mcp_runtime()

        # Pre-warm AI chat panel in background after main window is idle
        if LLM_AVAILABLE:
            QTimer.singleShot(1500, self._prewarm_ai_chat)


    def init_ui(self):
        self.setWindowTitle(
            "SPECTROview (Tool for Spectroscopic Data Processing and Visualization)"
        )
        self.setGeometry(100, 100, 1400, 930)
        self.setWindowIcon(get_app_icon())

        # Central widget
        central = QWidget(self)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(0)

        # ── Update notification banner (created lazily when an update is detected) ──
        self._update_banner = None
        self._update_download_worker = None

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
            self._sync_mcp_runtime()

    def _sync_mcp_runtime(self):
        """Apply the opt-in localhost MCP setting without restarting the app."""
        config = self.settings.load_mcp_settings()
        enabled = bool(config["mcp_enabled"])
        port = int(config["mcp_port"])

        if not enabled:
            if self._mcp_runtime is not None:
                self._mcp_runtime.stop()
                self._mcp_runtime = None
            self._mcp_error = ""
            return

        if (self._mcp_runtime is not None
                and self._mcp_runtime.port == port
                and self._mcp_runtime.running):
            return

        if self._mcp_runtime is not None:
            self._mcp_runtime.stop()
            self._mcp_runtime = None

        try:
            # Lazy: mcp + uvicorn stay off the normal startup import path.
            from spectroview.ai_agent.mcp.runtime import LocalMCPRuntime
            runtime = LocalMCPRuntime(self.application_api, port=port)
            runtime.start()
            self._mcp_runtime = runtime
            self._mcp_error = ""
            self.statusBar().showMessage(
                f"Local MCP server enabled: {runtime.endpoint}", 8000
            )
        except Exception as exc:
            self._mcp_error = str(exc)
            self.statusBar().showMessage(f"Local MCP server failed: {exc}", 15000)

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

    def _ensure_chat_panel(self) -> bool:
        """Create and wire the chat panel if not already created."""
        if self._chat_panel is not None:
            return True
        try:
            if not LLM_AVAILABLE:
                raise ImportError(LLM_ERROR_MSG)
            from spectroview.ai_agent.v_chat_panel import VChatPanel
        except ImportError:
            return False

        self._chat_panel = VChatPanel(self)
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
        self.v_graphs_workspace.vm.graph_state_changed.connect(sync_chat_graphs)
        return True

    def _prewarm_ai_chat(self):
        """Asynchronously pre-create the AI Chat panel when the app is idle
        so that when the user clicks the AI Chat toolbar button, it opens instantly."""
        try:
            self._ensure_chat_panel()
        except Exception:
            pass

    def open_ai_chat(self):
        """Open (or raise) the SPECTROview AI Agent panel.

        The panel is created lazily on first use (or pre-warmed during idle)
        and then kept alive so the conversation history is preserved across
        multiple open/close cycles. The active DataFrame from the Graphs
        workspace is injected each time the panel is shown.
        """
        if not self._ensure_chat_panel():
            QMessageBox.information(
                self,
                "SPECTROview AI Agent — Not Available",
                f"The AI Chat module could not be imported.\nError: {LLM_ERROR_MSG}\n\n"
                "Please install the optional dependencies:\n"
                "    pip install ollama mcp\n\n"
                "Then restart SPECTROview.",
            )
            return

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

        # Deep copy so we don't mutate the original. Types/structures were
        # already normalised by VMChat before the config was emitted.
        cfg = copy.deepcopy(plot_config)
        cfg['df_name'] = df_name

        # Create the plot directly via the workspace API
        try:
            self.application_api.create_graph(cfg)
        except Exception as exc:
            ws.vm.notify.emit(f"AI graph creation failed: {exc}")
            return

        # Update the sidebar combo boxes to reflect last plot config
        def _set_combo(cbb, value):
            if value:
                idx = cbb.findText(str(value))
                if idx >= 0:
                    cbb.setCurrentIndex(idx)

        if hasattr(ws, 'cbb_x'):
            _set_combo(ws.cbb_x, plot_config.get("x"))
        if hasattr(ws, 'cbb_y'):
            _set_combo(ws.cbb_y, plot_config.get("y"))
        if hasattr(ws, 'cbb_z'):
            _set_combo(ws.cbb_z, plot_config.get("z"))
        if hasattr(ws, 'cbb_plot_style'):
            _set_combo(ws.cbb_plot_style, plot_config.get("plot_style"))

    def _apply_graph_update(self, update_payload: dict):
        """Apply one validated AI/MCP patch through the Graph workspace API."""
        ws = self.v_graphs_workspace
        graph_id = update_payload.get("graph_id")
        properties = update_payload.get("properties", {})

        if graph_id is None or not isinstance(properties, dict):
            return

        try:
            self.application_api.update_graph(graph_id, properties)
        except Exception as exc:
            ws.vm.notify.emit(f"AI graph update failed: {exc}")
            return

        # Switch to Graphs tab to show the result
        self.tabWidget.setCurrentWidget(ws)

    def _apply_graph_delete(self, delete_payload: dict):
        """Delete requested graphs based on the AI instructions."""
        delete_all = delete_payload.get("delete_all", False)
        target_ids = delete_payload.get("graph_ids", [])
        try:
            self.application_api.delete_graphs(delete_all, target_ids)
        except Exception as exc:
            self.v_graphs_workspace.vm.notify.emit(f"AI graph deletion failed: {exc}")

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
        url = QUrl("https://github.com/CEA-MetroCarac/SPECTROview/releases")
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

    def _on_update_available(
        self, tag: str, notes: str, html_url: str, wheel_url: str, wheel_sha256: str
    ):
        """Show the update banner when a newer version is found on GitHub."""
        # Never show if the user already skipped this exact version
        if self.settings.get_skipped_version() == tag:
            return

        if self._update_banner is not None:
            return   # already showing

        self._show_update_banner(tag, html_url, wheel_url, wheel_sha256)

    def _show_update_banner(
        self, tag: str, html_url: str, wheel_url: str, wheel_sha256: str
    ) -> None:
        """Create and insert the single update banner above the workspace tabs."""
        banner = VUpdateBanner(
            tag=tag,
            html_url=html_url,
            on_skip=self.settings.set_skipped_version,
            on_dismiss=self._hide_banner,
            on_update=self._download_and_install_update,
            wheel_url=wheel_url,
            wheel_sha256=wheel_sha256,
            parent=self.centralWidget(),
        )
        banner.apply_theme(self.settings.get_theme())
        self.centralWidget().layout().insertWidget(0, banner)
        self._update_banner = banner

    def _hide_banner(self):
        """Remove and dispose of the current update banner."""
        banner = self._update_banner
        self._update_banner = None
        if banner is not None:
            self.centralWidget().layout().removeWidget(banner)
            banner.deleteLater()

    def _manual_update_check(self):
        """User clicked 'Check for updates' in the menu bar — always runs (no throttle)."""
        from spectroview import VERSION
        self._manual_checker = UpdateCheckerWorker(current_version=VERSION)
        self._manual_check_found_update = False
        self._manual_checker.update_available.connect(self._on_manual_update_found)
        self._manual_checker.check_finished.connect(self._on_manual_check_done)
        self._manual_checker.start()

    def _on_manual_update_found(
        self, tag: str, notes: str, html_url: str, wheel_url: str, wheel_sha256: str
    ):
        """A newer version was found during a user-initiated check."""
        self._manual_check_found_update = True
        # Show banner even if user previously skipped this version
        if self._update_banner is not None:
            return

        self._show_update_banner(tag, html_url, wheel_url, wheel_sha256)

    def _download_and_install_update(
        self, _tag: str, wheel_url: str, wheel_sha256: str
    ) -> None:
        """Download the announced release wheel, then hand installation to the helper."""
        if self._update_download_worker is not None:
            return
        try:
            get_update_python_executable()
        except UpdateInstallationError as error:
            QMessageBox.warning(self, "Automatic update unavailable", str(error))
            return

        if self._update_banner is not None:
            self._update_banner.set_download_progress(-1)
        worker = UpdateDownloadWorker(wheel_url, wheel_sha256, self)
        worker.progress_changed.connect(self._on_update_download_progress)
        worker.download_finished.connect(self._on_update_download_finished)
        worker.download_failed.connect(self._on_update_download_failed)
        self._update_download_worker = worker
        worker.start()

    def _on_update_download_progress(self, percent: int) -> None:
        if self._update_banner is not None:
            self._update_banner.set_download_progress(percent)

    def _on_update_download_finished(self, wheel_path: str) -> None:
        """Schedule installation after the Qt process exits, then close this instance."""
        self._release_update_download_worker()
        try:
            install_update_and_restart(Path(wheel_path))
        except (OSError, UpdateInstallationError) as error:
            Path(wheel_path).unlink(missing_ok=True)
            self._show_update_download_error(str(error))
            return
        self.close()

    def _on_update_download_failed(self, error: str) -> None:
        self._release_update_download_worker()
        self._show_update_download_error(error)

    def _release_update_download_worker(self) -> None:
        worker = self._update_download_worker
        self._update_download_worker = None
        if worker is not None:
            worker.deleteLater()

    def _show_update_download_error(self, error: str) -> None:
        if self._update_banner is not None:
            self._update_banner.set_update_error()
        QMessageBox.warning(
            self,
            "Update download failed",
            f"SPECTROview could not download the update. Please try again or install it manually.\n\n{error}",
        )

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
        """Start the update check and the import prewarm once the window is up."""
        super().showEvent(event)
        apply_windows_taskbar_icon(int(self.winId()))
        # Use a short single-shot timer so the UI paints before the thread starts
        from PySide6.QtCore import QTimer
        QTimer.singleShot(2000, self._start_update_check)
        QTimer.singleShot(0, self._prewarm_heavy_imports)

    _prewarmed = False

    def _prewarm_heavy_imports(self):
        """Load scipy into ``sys.modules`` on a background thread.

        scipy is deliberately off the startup import path — it would add ~1.3 s
        before the first window paint. But the first histogram / wafer render
        must not pay for it either (that made opening a saved .graphs workspace
        noticeably slower), so it is warmed here while the user reads the UI.
        Importing is thread-safe; the worst case is that a render arriving mid
        prewarm simply waits for the same import it would have done itself.
        """
        if Main._prewarmed:
            return
        Main._prewarmed = True

        def _warm():
            for name in ("scipy.stats", "scipy.interpolate", "scipy.linalg"):
                try:
                    importlib.import_module(name)
                except Exception:           # noqa: BLE001 - prewarm is best-effort
                    pass

        threading.Thread(target=_warm, name="prewarm-scipy", daemon=True).start()

    def closeEvent(self, event):
        """Clean up on application exit to prevent Matplotlib C++ threading crashes."""
        try:
            import matplotlib.pyplot as plt
            plt.close('all')
        except Exception:
            pass
        # Reject new local requests, then stop the HTTP endpoint before Qt's
        # objects disappear.
        if hasattr(self, "application_api"):
            self.application_api.close()
        if self._mcp_runtime is not None:
            try:
                self._mcp_runtime.stop()
            except Exception:
                pass
            self._mcp_runtime = None

        # Close the AI agent's MCP sessions and stop its event-loop thread —
        # a stdio server would otherwise leave an orphaned child process.
        if self._chat_panel is not None:
            try:
                self._chat_panel.vm.shutdown()
            except Exception:
                pass
        event.accept()

def launcher():
    app = QApplication(sys.argv)
    app.setApplicationName("SPECTROview")
    app.setWindowIcon(get_app_icon())
    app.setStyle("Fusion")

    window = Main()
    window.show()
    code = app.exec()
    os._exit(code)

if __name__ == "__main__":
    launcher()
