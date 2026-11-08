# main.py
import sys
import os
import importlib.util
import threading
from pathlib import Path

# ── Windows taskbar identity ─────────────────────────────────────────────────
from spectroview.winapi import apply_windows_taskbar_icon, set_current_process_app_id
set_current_process_app_id()
# ─────────────────────────────────────────────────────────────────────────────

# ── macOS Matplotlib font discovery fix ──────────────────────────────────────
from spectroview.model.m_macos_font_patch import apply_macos_font_cache_fix
apply_macos_font_cache_fix()
# ─────────────────────────────────────────────────────────────────────────────

import matplotlib as mpl

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*Glyph.*")

from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QTabWidget, QFileDialog, QMessageBox, QSplitter
from PySide6.QtCore import Qt, QFileInfo, QUrl, QTimer
from PySide6.QtGui import QDesktopServices


def _configure_matplotlib_qt_backend() -> None:
    """Configure the ``qtagg`` backend only when Qt or a GUI display is active.

    Prevents ``ImportError: Cannot load backend 'qtagg' ... as 'headless' is
    currently running`` on headless Linux CI when ``matplotlib.pyplot`` was
    already imported with ``Agg`` (or will be imported before a ``QApplication``
    exists).
    """
    if "matplotlib.pyplot" in sys.modules:
        return
    if QApplication.instance() is None:
        try:
            from matplotlib import _c_internal_utils
            if not _c_internal_utils.display_is_valid():
                return
        except Exception:
            if sys.platform.startswith("linux") and not (
                os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
            ):
                return
    mpl.use("qtagg", force=False)


_configure_matplotlib_qt_backend()

from spectroview.model.m_io import detect_file_type
from spectroview.settings import MSettings, VMSettings, VSettingsDialog
from spectroview.model.m_update_manager import (
    UnifiedUpdateCheckWorker,
    UnifiedUpdateDownloadWorker,
    UpdateCheckResult,
    UpdateManager,
    WindowsPortableUpdater,
)


from spectroview.view.components.v_update_banner import VUpdateBanner
from spectroview.view.components.v_quick_calculators import MQuickCalc
from spectroview.view.components.v_menubar import VMenuBar
from spectroview.view.components.v_notification import NotificationManager
from spectroview.view.v_workspace_spectra import VWorkspaceSpectra
from spectroview.view.v_workspace_maps import VWorkspaceMaps
from spectroview.view.v_workspace_graphs import VWorkspaceGraphs
from spectroview.view.theme import ThemeManager, setup_application_font

from spectroview import get_app_icon

try:
    LLM_AVAILABLE = importlib.util.find_spec("spectroview.ai_agent.v_chat_panel") is not None
    LLM_ERROR_MSG = "" if LLM_AVAILABLE else "spectroview.ai_agent.v_chat_panel not found"
except ImportError as e:      # the ai_agent package itself is absent
    LLM_AVAILABLE = False
    LLM_ERROR_MSG = str(e)

class Main(QMainWindow):
    def __init__(self):
        super().__init__()
        setup_application_font()
        self.settings = MSettings()
        # Restore configuration from working folder if config.json exists
        self.settings.import_config()
        self.theme_mgr = ThemeManager(self.settings)

        # Apply global application style BEFORE creating widgets to avoid expensive unpolish/polish
        self.theme_mgr.apply(self.settings.get_theme())
        
        self.init_ui()
        self.setup_connections()
        self._propagate_theme(self.settings.get_theme())
        self.tabWidget.setCurrentWidget(self.v_maps_workspace)
        
        # Lazy chat panel (created on first use)
        self._chat_panel = None

        # One domain-oriented facade over the running application. 
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
            QTimer.singleShot(1500, self, self._prewarm_ai_chat)


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
        self._update_manager = None
        self._pending_release = None
        self._manual_check_worker = None
        self._startup_check_worker = None

        # Horizontal Splitter between Workspaces and Embedded AI Chat Panel
        self.workspace_splitter = QSplitter(Qt.Horizontal, central)
        self.workspace_splitter.setChildrenCollapsible(False)

        # Main Tab Widget
        self.tabWidget = QTabWidget(self.workspace_splitter)

        self.v_spectra_workspace = VWorkspaceSpectra()
        self.v_graphs_workspace = VWorkspaceGraphs()
        self.v_maps_workspace = VWorkspaceMaps()

        self.tabWidget.addTab(self.v_spectra_workspace, "Spectra")
        self.tabWidget.addTab(self.v_maps_workspace, "Maps")
        self.tabWidget.addTab(self.v_graphs_workspace, "Graphs")
        
        self.workspace_splitter.addWidget(self.tabWidget)

        # Enable Drag & Drop
        self.setAcceptDrops(True)

        layout.addWidget(self.workspace_splitter)
        self.setCentralWidget(central)

        # Toolbar
        self.menu_bar = VMenuBar()
        self.addToolBar(Qt.TopToolBarArea, self.menu_bar)
        NotificationManager.instance().set_area(self.menu_bar.notification_area)
        
    def setup_connections(self):
        self.menu_bar.open_requested.connect(self.open_files)
        self.menu_bar.save_requested.connect(self.save)
        self.menu_bar.clear_requested.connect(self.clear_workspace)
        self.menu_bar.settings_requested.connect(self._open_settings)
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
            update_map_viewers()
            
            # Save view options whenever they change in either workspace
            self.v_spectra_workspace.v_spectra_viewer.allOptionsSyncChanged.connect(self.settings.save_view_options)
            self.v_maps_workspace.v_spectra_viewer.allOptionsSyncChanged.connect(self.settings.save_view_options)
        

    def show_notification(
        self,
        message: str,
        notification_type: str | None = None,
        duration: int = 3500,
        title: str | None = None,
    ):
        """Display a floating notification in the application toolbar."""
        return NotificationManager.instance().show(
            message=message,
            notification_type=notification_type,
            duration=duration,
            title=title,
        )

    def open_files(self):
        """Universal file opener supporting all SPECTROview formats."""
        last_dir = self.settings.get_last_directory()
        paths, _ = QFileDialog.getOpenFileNames(
            None,
            "Open file(s)",
            last_dir,
            "SPECTROview formats (*.csv *.txt *.dat *.wdf *.spc *.l6s *.l6m *.ngs *.ngc *.spectra *.maps *.graphs *.xlsx *.xls *.json)"
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
        
        # Categorize files by type via Model-layer header/sample inspection
        spectra_files = []
        hyperspectral_files = []
        dataframes = []
        fit_model_files = []
        spectra_work_file = None
        maps_work_file = None
        graphs_work_file = None
        
        for file_path in paths:
            path = Path(file_path)
            try:
                file_kind = detect_file_type(path)
            except Exception as e:
                QMessageBox.warning(self, "File Load Error", f"Failed to read {path.name}: {e}")
                continue

            if file_kind == "spectra":
                spectra_files.append(str(path))
            elif file_kind == "map":
                hyperspectral_files.append(str(path))
            elif file_kind == "dataframe":
                dataframes.append(str(path))
            elif file_kind == "fit_model":
                fit_model_files.append(str(path))
            elif file_kind == "spectra_workspace":
                spectra_work_file = str(path)
            elif file_kind == "maps_workspace":
                maps_work_file = str(path)
            elif file_kind == "graphs_workspace":
                graphs_work_file = str(path)
        
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
        
        # Load fit model files
        if fit_model_files:
            current_tab = self.tabWidget.currentWidget()
            target_workspace = (
                current_tab
                if current_tab in (self.v_spectra_workspace, self.v_maps_workspace)
                else self.v_spectra_workspace
            )
            for f in fit_model_files:
                target_workspace.load_fit_model_file(f)
            self.tabWidget.setCurrentWidget(target_workspace)
        
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

    def _open_settings(self, section: str = "general"):
        """Open settings dialog, optionally selecting a specific section."""
        if not isinstance(section, str):
            section = "general"
        vm = VMSettings()
        dlg = VSettingsDialog(vm, self)
        dlg.select_section(section)
        dlg.check_updates_requested.connect(self._manual_update_check)
        dlg.theme_selected.connect(self.toggle_theme)
        initial_theme = self.settings.get_theme()
        if dlg.exec():
            # Refresh viewers to reflect settings changes (e.g. coef_noise)
            if hasattr(self, 'v_spectra_workspace') and hasattr(self.v_spectra_workspace, 'v_spectra_viewer'):
                self.v_spectra_workspace.v_spectra_viewer._plot()
            if hasattr(self, 'v_maps_workspace') and hasattr(self.v_maps_workspace, 'v_spectra_viewer'):
                self.v_maps_workspace.v_spectra_viewer._plot()
            self._sync_mcp_runtime()
            if self._chat_panel is not None:
                self._chat_panel.reload_settings()
        else:
            if self.settings.get_theme() != initial_theme:
                self.toggle_theme(initial_theme)

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

    def quick_calc(self):
        """Open quick calculation dialog."""
        if not hasattr(self, '_quick_calc_dlg'):
            self._quick_calc_dlg = MQuickCalc(self)
            self._quick_calc_dlg.setWindowFlags(self._quick_calc_dlg.windowFlags() | Qt.WindowStaysOnTopHint)
        self._quick_calc_dlg.show()
        self._quick_calc_dlg.raise_()
        self._quick_calc_dlg.activateWindow()

    def _ensure_chat_panel(self) -> bool:
        """Create and wire the embedded chat panel if not already created."""
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
        self.workspace_splitter.addWidget(self._chat_panel)
        self._chat_panel.hide()

        def _on_splitter_moved(pos, index):
            if self._chat_panel is not None and self._chat_panel.isVisible():
                sizes = self.workspace_splitter.sizes()
                if len(sizes) >= 2 and sizes[1] > 100:
                    self._chat_panel_width = sizes[1]

        self.workspace_splitter.splitterMoved.connect(_on_splitter_moved)

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
        """Open (or toggle) the embedded SPECTROview AI Agent panel on the far right.

        The panel is created lazily on first use (or pre-warmed during idle)
        and embedded directly alongside the main workspace tab widget.
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
        # already open closes it.
        if self._chat_panel.isVisible():
            self._chat_panel.hide()
            return

        # Force a sync right now when opening
        self._chat_panel.reload_settings()
        vm_graphs = self.v_graphs_workspace.vm
        self._chat_panel.set_dataframes(vm_graphs.dataframes, vm_graphs.selected_df_name or "")
        self._chat_panel.vm.set_graphs(vm_graphs.graphs)

        self._chat_panel.show()
        total_w = self.workspace_splitter.width()
        chat_w = getattr(self, "_chat_panel_width", 420)
        if total_w > 600:
            chat_w = max(340, min(chat_w, int(total_w * 0.4)))
            self.workspace_splitter.setSizes([total_w - chat_w, chat_w])
        if hasattr(self._chat_panel, "focus_input"):
            self._chat_panel.focus_input()

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
        """Show About section within the Settings dialog."""
        self._open_settings(section="about")

    def manual(self):
        """Open online User Manual web documentation."""
        url = QUrl("https://cea-metrocarac.github.io/SPECTROview/user_manual/")
        QDesktopServices.openUrl(url)

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
        # Keep menubar checkmark and icons in sync
        if hasattr(self, 'menu_bar'):
            self.menu_bar.set_current_theme(theme)
            if hasattr(self.menu_bar, 'apply_theme'):
                self.menu_bar.apply_theme(theme)

        # Propagate to workspaces (icons, stylesheets)
        for ws in (getattr(self, 'v_spectra_workspace', None),
                   getattr(self, 'v_maps_workspace', None)):
            if ws is None:
                continue
            if hasattr(ws, 'v_spectra_viewer') and hasattr(ws.v_spectra_viewer, 'apply_global_theme'):
                ws.v_spectra_viewer.apply_global_theme(theme)
            if hasattr(ws, 'apply_theme'):
                ws.apply_theme(theme)

        if hasattr(self, 'v_graphs_workspace'):
            self.v_graphs_workspace.apply_theme(theme)

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
    def _get_update_manager(self) -> UpdateManager:
        """Return (and lazily create) the centralized multi-platform UpdateManager."""
        if self._update_manager is None:
            from spectroview import VERSION
            self._update_manager = UpdateManager(
                current_version=VERSION,
                settings=self.settings,
            )
        return self._update_manager

    def _start_update_check(self):
        """Launch a background thread to query GitHub for the latest release."""
        if getattr(self, "_has_checked_for_updates", False):
            return
        if not self.settings.get_check_for_updates():
            return

        self._has_checked_for_updates = True
        worker = UnifiedUpdateCheckWorker(self._get_update_manager(), force=False, parent=self)
        worker.check_succeeded.connect(self._on_startup_check_succeeded)
        worker.check_finished.connect(worker.deleteLater)
        self._startup_check_worker = worker
        worker.start()

    def _on_startup_check_succeeded(self, result: UpdateCheckResult) -> None:
        """Show the update banner on startup only if newer and not skipped."""
        if not result.update_available or result.release is None:
            return
        if self.settings.get_skipped_version() == result.release.tag:
            return
        if self._update_banner is not None:
            return
        self._show_update_banner_from_result(result)

    def _show_update_banner_from_result(self, result: UpdateCheckResult) -> None:
        """Create and insert the single update banner above the workspace tabs."""
        release = result.release
        if release is None:
            return
        self._pending_release = release
        asset = result.selected_asset
        asset_url = asset.url if asset is not None else ""
        asset_sha256 = asset.sha256 if asset is not None else ""

        # Allow action whenever there is a downloadable platform asset
        on_update_cb = self._download_and_install_update if asset_url else None
        self._show_update_banner(
            release.tag,
            release.html_url,
            asset_url,
            asset_sha256,
            on_update=on_update_cb,
            action_label=result.action_label,
            action_tooltip=result.action_tooltip or result.message,
            current_version=result.current_version,
        )

    def _show_update_banner(
        self,
        tag: str,
        html_url: str,
        wheel_url: str,
        wheel_sha256: str,
        *,
        on_update=None,
        action_label: str = "Update",
        action_tooltip: str = "",
        current_version: str = "",
    ) -> None:
        """Create and insert the single update banner above the workspace tabs."""
        banner = VUpdateBanner(
            tag=tag,
            html_url=html_url,
            on_skip=self.settings.set_skipped_version,
            on_dismiss=self._hide_banner,
            on_update=on_update if on_update is not None else self._download_and_install_update,
            wheel_url=wheel_url,
            wheel_sha256=wheel_sha256,
            parent=self.centralWidget(),
            action_label=action_label,
            action_tooltip=action_tooltip,
            current_version=current_version,
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
        """User clicked 'Check for updates' — forces an uncached check with clear feedback."""
        if self._manual_check_worker is not None and self._manual_check_worker.isRunning():
            return
        self._manual_check_found_update = False
        self._manual_check_error = ""
        worker = UnifiedUpdateCheckWorker(self._get_update_manager(), force=True, parent=self)
        worker.check_succeeded.connect(self._on_manual_check_succeeded)
        worker.check_failed.connect(self._on_manual_check_failed)
        worker.check_finished.connect(self._on_manual_check_done)
        worker.check_finished.connect(worker.deleteLater)
        self._manual_check_worker = worker
        worker.start()

    def _on_manual_check_succeeded(self, result: UpdateCheckResult) -> None:
        if result.update_available and result.release is not None:
            self._manual_check_found_update = True
            if self._update_banner is None:
                self._show_update_banner_from_result(result)

    def _on_manual_check_failed(self, error: str) -> None:
        self._manual_check_error = error

    def _download_and_install_update(
        self, _tag: str, _asset_url: str, _asset_sha256: str
    ) -> None:
        """Download and verify the active platform update package."""
        if self._update_download_worker is not None:
            return
        if self._pending_release is None:
            QMessageBox.warning(
                self,
                "Update unavailable",
                "Release metadata is no longer available. Please run Check for updates again.",
            )
            return

        if self._update_banner is not None:
            self._update_banner.set_download_progress(-1)
        worker = UnifiedUpdateDownloadWorker(
            self._get_update_manager(),
            self._pending_release,
            parent=self,
        )
        worker.progress_changed.connect(self._on_update_download_progress)
        worker.download_finished.connect(self._on_update_download_finished)
        worker.download_failed.connect(self._on_update_download_failed)
        self._update_download_worker = worker
        worker.start()

    def _on_update_download_progress(self, percent: int) -> None:
        if self._update_banner is not None:
            self._update_banner.set_download_progress(percent)

    def _on_update_download_finished(self, artifact_path: str) -> None:
        """Apply automatic install + restart, or guide user through downloaded installer."""
        self._release_update_download_worker()
        mgr = self._get_update_manager()
        try:
            install_result = mgr.install_update(Path(artifact_path), release=self._pending_release)
        except Exception as error:
            self._show_update_download_error(str(error))
            return

        if install_result.restart_required:
            try:
                self.settings.settings.sync()
            except Exception:
                pass
            mgr.restart_application()
            self.close()
            return

        if self._update_banner is not None:
            self._update_banner.hide()
            self._hide_banner()

        if install_result.requires_manual_action:
            self._show_guided_update_dialog(install_result)
        elif install_result.message:
            QMessageBox.information(self, "SPECTROview Update", install_result.message)

    def _show_guided_update_dialog(self, install_result) -> None:
        """Present platform-specific instructions and convenience actions for guided updates."""
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Information)
        box.setWindowTitle("SPECTROview Update Ready")
        box.setText(install_result.instructions or install_result.message)

        artifact = Path(install_result.artifact_path) if install_result.artifact_path else None
        open_file_btn = None
        open_folder_btn = None
        extract_btn = None
        copy_cmd_btn = None

        if artifact is not None and artifact.exists():
            if artifact.suffix.lower() in (".exe", ".dmg"):
                open_file_btn = box.addButton("Open Installer", QMessageBox.AcceptRole)
            open_folder_btn = box.addButton("Open Folder", QMessageBox.ActionRole)
        if getattr(install_result, "can_extract_zip", False) and artifact is not None:
            extract_btn = box.addButton("Extract to Folder...", QMessageBox.ActionRole)
        if getattr(install_result, "command", ""):
            copy_cmd_btn = box.addButton("Copy Command", QMessageBox.ActionRole)
        box.addButton(QMessageBox.Close)
        box.exec()

        clicked = box.clickedButton()
        if clicked is None:
            return
        if clicked is open_file_btn and artifact is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(artifact)))
        elif clicked is open_folder_btn and artifact is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(artifact.parent)))
        elif clicked is extract_btn and artifact is not None:
            dest = QFileDialog.getExistingDirectory(
                self,
                "Choose Folder to Extract Portable SPECTROview",
                str(artifact.parent),
            )
            if dest:
                try:
                    extracted_root = WindowsPortableUpdater.extract_portable_archive(
                        artifact, Path(dest) / artifact.stem
                    )
                    QMessageBox.information(
                        self,
                        "Extraction Complete",
                        f"SPECTROview was extracted to:\n{extracted_root}\n\n"
                        "You can close this instance and launch the new SPECTROview.exe directly.",
                    )
                    QDesktopServices.openUrl(QUrl.fromLocalFile(str(extracted_root)))
                except Exception as exc:
                    QMessageBox.warning(self, "Extraction Failed", str(exc))
        elif clicked is copy_cmd_btn and install_result.command:
            cb = QApplication.clipboard()
            if cb is not None:
                cb.setText(install_result.command)
                self.show_notification("Update command copied to clipboard.", notification_type="info")

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
        """Show result after a user-initiated check finishes."""
        from spectroview import VERSION
        if self._manual_check_error:
            QMessageBox.warning(
                self,
                "Update Check Failed",
                f"SPECTROview could not check for updates:\n\n{self._manual_check_error}",
            )
        elif not self._manual_check_found_update:
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
        QTimer.singleShot(2000, self, self._start_update_check)
        QTimer.singleShot(0, self, self._prewarm_heavy_imports)

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
            try:
                from spectroview.settings.ui.tab_about import get_runtime_package_versions
                get_runtime_package_versions()
            except Exception:
                pass

        threading.Thread(target=_warm, name="prewarm-scipy", daemon=True).start()

    def closeEvent(self, event):
        """Clean up on application exit to prevent Matplotlib C++ threading crashes."""
        try:
            import matplotlib.pyplot as plt
            plt.close('all')
            if hasattr(self, 'v_spectra_workspace') and hasattr(self.v_spectra_workspace.vm, '_cancel_fit_thread'):
                self.v_spectra_workspace.vm._cancel_fit_thread()
            if hasattr(self, 'v_maps_workspace') and hasattr(self.v_maps_workspace.vm, '_cancel_fit_thread'):
                self.v_maps_workspace.vm._cancel_fit_thread()
        except Exception:
            pass

        # Persist view options on application close
        try:
            if hasattr(self, 'v_spectra_workspace') and hasattr(self.v_spectra_workspace, 'v_spectra_viewer'):
                self.settings.save_view_options(self.v_spectra_workspace.v_spectra_viewer.get_options_state())
            elif hasattr(self, 'v_maps_workspace') and hasattr(self.v_maps_workspace, 'v_spectra_viewer'):
                self.settings.save_view_options(self.v_maps_workspace.v_spectra_viewer.get_options_state())
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

def _run_velopack_bootstrap() -> None:
    """Execute Velopack install/update hooks before QApplication when frozen under Velopack."""
    try:
        from spectroview.model.m_installation import InstallationManager
        if not InstallationManager.should_run_velopack_hook():
            return
        import velopack
        velopack.App().run()
    except Exception:
        pass


def launcher():
    _run_velopack_bootstrap()

    if "--version" in sys.argv:
        from spectroview import VERSION
        print(f"SPECTROview {VERSION}")
        sys.exit(0)

    app = QApplication(sys.argv)
    _configure_matplotlib_qt_backend()
    app.setApplicationName("SPECTROview")
    app.setWindowIcon(get_app_icon())
    app.setStyle("Fusion")
    setup_application_font(app)

    window = Main()
    if "--smoke-test" in sys.argv:
        # Offscreen / CI smoke test: verify full UI and workspace hierarchy load cleanly, then exit 0
        QTimer.singleShot(500, app.quit)
        code = app.exec()
        sys.exit(code)

    window.show()
    code = app.exec()
    os._exit(code)

if __name__ == "__main__":
    launcher()
