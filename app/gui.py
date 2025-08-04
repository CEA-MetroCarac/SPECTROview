# app/gui.py
from functools import partial
import logging

logger = logging.getLogger(__name__)

class Gui:
    def __init__(self, app_settings, ui, main_app):
        """
        Centralize all signal wiring / UI connections here.
        main_app is the instance of Main so that callbacks that live there can be used.
        """
        self.app_settings = app_settings
        self.ui = ui
        self.main_app = main_app

        # sync helper
        self.sync_settings = partial(self.app_settings.sync_app_settings, self.ui)
        self.connect_all()

    def _watch(self, widget):
        """Attach sync_settings to common signal names if they exist on widget."""
        for sig_name in ("valueChanged", "stateChanged", "textChanged", "currentIndexChanged", "toggled"):
            sig = getattr(widget, sig_name, None)
            if sig:
                sig.connect(self.sync_settings)
               

    def connect_ui_main_tab(self):
        """Wiring for toolbar / global actions."""
        self.ui.actionOpen.triggered.connect(lambda: self.main_app.open())
        self.ui.actionSave.triggered.connect(self.main_app.save)
        self.ui.actionClear_env.triggered.connect(self.main_app.clear_env)
        self.ui.actionDarkMode.triggered.connect(self.main_app.toggle_dark_mode)
        self.ui.actionLightMode.triggered.connect(self.main_app.toggle_light_mode)
        self.ui.actionAbout.triggered.connect(self.main_app.show_about)
        self.ui.actionHelps.triggered.connect(self.main_app.open_manual)
        
        ### DRAG & DROP to open files 
        self.ui.spectrums_listbox.files_dropped.connect(self.main_app.open)
        self.ui.spectra_listbox.files_dropped.connect(self.main_app.open)


    def connect_ui_maps_tab(self):
        for widget_name in (
            "ncpu",
            "cb_fit_negative",
            "max_iteration",
            "cbb_fit_methods",
            "xtol",
            "cb_attached",
            "rbtn_linear",
            "rbtn_polynomial",
            "noise",
            "degre",
        ):
            widget = getattr(self.ui, widget_name, None)
            if widget is not None:
                self._watch(widget)
            else:
                logger.warning("Maps tab widget '%s' not found on UI", widget_name)


    def connect_ui_spectra_tab(self):
        """Spectra-related watchers/buttons."""
        for widget_name in (
            "ncpu_2",
            "cb_fit_negative_2",
            "max_iteration_2",
            "cbb_fit_methods_2",
            "xtol_2",
            "cb_attached_2",
            "rbtn_linear_2",
            "rbtn_polynomial_2",
            "noise_2",
            "degre_2",
        ):
            widget = getattr(self.ui, widget_name, None)
            if widget is not None:
                self._watch(widget)

    def connect_ui_visu_tab(self):
        """Visualization/graph watchers."""
        
        for widget_name in (
            "cb_grid",
        ):
            widget = getattr(self.ui, widget_name, None)
            if widget is not None:
                self._watch(widget)

    def connect_all(self):
        """Convenience to hook everything in one call."""
        self.connect_ui_main_tab()
        self.connect_ui_maps_tab()
        self.connect_ui_spectra_tab()
        self.connect_ui_visu_tab()
