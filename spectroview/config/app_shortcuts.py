from functools import partial
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtCore import Qt

def switch_to(main_app, widget):
    main_app.ui.tabWidget.setCurrentWidget(widget)

def rescale_shortcut(main_app):
    """Dispatch Ctrl+R based on current tab."""
    current_tab = main_app.ui.tabWidget.currentWidget()
    if current_tab == main_app.ui.tab_spectra:
        main_app.spectrums.spectra_viewer.rescale()
    elif current_tab == main_app.ui.tab_maps:
        main_app.maps.spectra_viewer.rescale()

def fitting_shortcut(main_app):
    """Dispatch Ctrl+F based on current tab."""
    current_tab = main_app.ui.tabWidget.currentWidget()
    if current_tab == main_app.ui.tab_spectra:
        main_app.spectrums.fit()
    elif current_tab == main_app.ui.tab_maps:
        main_app.maps.fit()

# def copy_shortcut(main_app):
#     """Dispatch Ctrl+C based on current tab."""
#     current_tab = main_app.ui.tabWidget.currentWidget()
#     if current_tab == main_app.ui.tab_graphs:
#         main_app.visu.copy_fig_to_clb()

#     elif current_tab == main_app.ui.tab_maps:
#         main_app.maps.spectra_widget.copy_fig()
#     elif current_tab == main_app.ui.tab_spectra:
#         main_app.spectrums.spectra_widget.copy_fig()

def setup_shortcuts(main_app):
    """
    Wire keyboard shortcuts for the main application.
    Expects main_obj to have:
      - ui with tabWidget and tab_* attributes
      - spectrums.spectra_widget
      - maps.spectra_widget
    """
    # Tab switching shortcuts
    QShortcut(QKeySequence(Qt.ControlModifier | Qt.Key_1), main_app.ui).activated.connect(
        partial(switch_to, main_app, main_app.ui.tab_spectra)
    )
    QShortcut(QKeySequence(Qt.ControlModifier | Qt.Key_Ampersand), main_app.ui).activated.connect(
        partial(switch_to, main_app, main_app.ui.tab_spectra)
    )

    QShortcut(QKeySequence(Qt.ControlModifier | Qt.Key_2), main_app.ui).activated.connect(
        partial(switch_to, main_app, main_app.ui.tab_maps)
    )
    QShortcut(QKeySequence(Qt.ControlModifier | Qt.Key_Eacute), main_app.ui).activated.connect(
        partial(switch_to, main_app, main_app.ui.tab_maps)
    )

    QShortcut(QKeySequence(Qt.ControlModifier | Qt.Key_3), main_app.ui).activated.connect(
        partial(switch_to, main_app, main_app.ui.tab_graphs)
    )
    QShortcut(QKeySequence(Qt.ControlModifier | Qt.Key_QuoteDbl), main_app.ui).activated.connect(
        partial(switch_to, main_app, main_app.ui.tab_graphs)
    )

   

    # Global rescale (Ctrl+R)
    shortcut_rescale = QShortcut(QKeySequence(Qt.ControlModifier | Qt.Key_R), main_app.ui)
    shortcut_rescale.setContext(Qt.ShortcutContext.ApplicationShortcut)
    shortcut_rescale.activated.connect(lambda: rescale_shortcut(main_app))


    # FITTING action (Ctrl+F)
    shortcut_fitting = QShortcut(QKeySequence(Qt.ControlModifier | Qt.Key_F), main_app.ui)
    shortcut_fitting.setContext(Qt.ShortcutContext.ApplicationShortcut)
    shortcut_fitting.activated.connect(lambda: fitting_shortcut(main_app))


    # # COPY action (Ctrl+C)
    # shortcut_copy = QShortcut(QKeySequence(Qt.ControlModifier | Qt.Key_C), main_app.ui)
    # shortcut_copy.setContext(Qt.ShortcutContext.ApplicationShortcut)
    # shortcut_copy.activated.connect(lambda: copy_shortcut(main_app))