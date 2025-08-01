# app/app_shortcuts.py
from functools import partial
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtCore import Qt


def switch_to(main_obj, widget):
    main_obj.ui.tabWidget.setCurrentWidget(widget)


def handle_rescale_shortcut(main_obj):
    """Dispatch Ctrl+R rescale based on current tab."""
    current_tab = main_obj.ui.tabWidget.currentWidget()
    if current_tab == main_obj.ui.tab_spectra:
        main_obj.spectrums.spectra_widget.rescale()
    elif current_tab == main_obj.ui.tab_maps:
        main_obj.maps.spectra_widget.rescale()


def setup_shortcuts(main_obj):
    """
    Wire keyboard shortcuts for the main application.
    Expects main_obj to have:
      - ui with tabWidget and tab_* attributes
      - spectrums.spectra_widget
      - maps.spectra_widget
    """
    # Tab switching shortcuts
    QShortcut(QKeySequence(Qt.ControlModifier | Qt.Key_1), main_obj.ui).activated.connect(
        partial(switch_to, main_obj, main_obj.ui.tab_spectra)
    )
    QShortcut(QKeySequence(Qt.ControlModifier | Qt.Key_Ampersand), main_obj.ui).activated.connect(
        partial(switch_to, main_obj, main_obj.ui.tab_spectra)
    )

    QShortcut(QKeySequence(Qt.ControlModifier | Qt.Key_2), main_obj.ui).activated.connect(
        partial(switch_to, main_obj, main_obj.ui.tab_maps)
    )
    QShortcut(QKeySequence(Qt.ControlModifier | Qt.Key_Eacute), main_obj.ui).activated.connect(
        partial(switch_to, main_obj, main_obj.ui.tab_maps)
    )

    QShortcut(QKeySequence(Qt.ControlModifier | Qt.Key_3), main_obj.ui).activated.connect(
        partial(switch_to, main_obj, main_obj.ui.tab_graphs)
    )
    QShortcut(QKeySequence(Qt.ControlModifier | Qt.Key_QuoteDbl), main_obj.ui).activated.connect(
        partial(switch_to, main_obj, main_obj.ui.tab_graphs)
    )

    QShortcut(QKeySequence(Qt.ControlModifier | Qt.Key_4), main_obj.ui).activated.connect(
        partial(switch_to, main_obj, main_obj.ui.tab_fileconvert)
    )
    QShortcut(QKeySequence(Qt.ControlModifier | Qt.Key_Apostrophe), main_obj.ui).activated.connect(
        partial(switch_to, main_obj, main_obj.ui.tab_fileconvert)
    )

    # Global rescale (Ctrl+R)
    shortcut_rescale = QShortcut(QKeySequence(Qt.ControlModifier | Qt.Key_R), main_obj.ui)
    shortcut_rescale.setContext(Qt.ShortcutContext.ApplicationShortcut)
    shortcut_rescale.activated.connect(lambda: handle_rescale_shortcut(main_obj))
