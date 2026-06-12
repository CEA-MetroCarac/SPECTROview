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
from spectroview.model.m_quick_calc import MQuickCalc
from spectroview.model.m_spc import SpcReader
from spectroview.model.m_settings import MSettings

from spectroview.viewmodel.vm_settings import VMSettings
from spectroview.view.components.v_settings import VSettingsDialog
from spectroview.view.components.v_about import VAboutDialog

from spectroview.view.components.v_menubar import VMenuBar
from spectroview.view.v_workspace_spectra import VWorkspaceSpectra
from spectroview.view.v_workspace_maps import VWorkspaceMaps
from spectroview.view.v_workspace_graphs import VWorkspaceGraphs

from spectroview.view.theme import ThemeManager

from spectroview import LOGO_APPLI, USER_MANUAL_DIR
from spectroview.view.components.v_user_manual import VUserManualDialog

try:
    from renishawWiRE import WDFReader
    WDF_AVAILABLE = True
except ImportError:
    WDF_AVAILABLE = False

class Main(QMainWindow):
    def __init__(self):
        super().__init__()
        self.settings = MSettings()
        self.theme_mgr = ThemeManager(self.settings)

        self.init_ui()
        self.toggle_theme(self.settings.get_theme())
        self.setup_connections()
        self.tabWidget.setCurrentWidget(self.v_maps_workspace)

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
        self.menu_bar.theme_selected.connect(self.toggle_theme)
        
        # Inject Graphs workspace into Maps ViewModel for cross-workspace communication
        self.v_maps_workspace.vm.set_graphs_workspace(self.v_graphs_workspace)
        
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
