import os

from spectroview import PEAK_MODELS, ICON_DIR

from PySide6.QtWidgets import  QVBoxLayout, QHBoxLayout, QLabel, QSizePolicy,\
    QLineEdit, QWidget, QPushButton, QComboBox, QCheckBox, QWidget, QSpacerItem
from PySide6.QtCore import  Qt, QTimer
from PySide6.QtGui import  QIcon, Qt 

from PySide6.QtGui import QPalette, QColor

class PeakTable:
    """Class dedicated to show fit parameters of Spectrum objects in the GUI"""
    def __init__(self, main_app, main_layout, cbb_layout):
        self.main_app = main_app 
        self.main_layout = main_layout # layout where the peak_table are placed
        self.cbb_layout = cbb_layout  # layout where comboboxes are placed
        self.sel_spectrum = None

        # Initialize Checkboxes
        self.cb_limits = QCheckBox("Limits")
        self.cb_expr = QCheckBox("Expression")
        self.cbb_layout.addWidget(self.cb_limits)
        self.cbb_layout.addWidget(self.cb_expr)
        self.cb_limits.stateChanged.connect(self.refresh_gui)
        self.cb_expr.stateChanged.connect(self.refresh_gui)

    def clear_layout(self, layout):
        """To clear a given layout"""
        if layout is not None:
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                if widget:
                    widget.deleteLater()
                else:
                    self.clear_layout(item.layout())

    def show(self, sel_spectrum=None):
        """To show all fitted parameters in GUI"""
        if sel_spectrum is None:
            self.clear()
            return
        
        self.sel_spectrum = sel_spectrum
        self.clear_layout(self.main_layout)
        header_labels = ["  ", "Label", "Model"]
        param_hint_order = ['x0', 'fwhm', 'fwhm_l', 'fwhm_r', 'ampli', 'alpha']

        # Create and add headers to list
        for param_hint_key in param_hint_order:
            if any(param_hint_key in peak_model.param_hints for peak_model in
                   self.sel_spectrum.peak_models):
                header_labels.append(param_hint_key.title())
                header_labels.append(f"fix {param_hint_key.title()}")
                if self.cb_limits.isChecked():
                    header_labels.append(f"min {param_hint_key.title()}")
                    header_labels.append(f"max {param_hint_key.title()}")
                if self.cb_expr.isChecked():
                    header_labels.append(f"expression {param_hint_key.title()}")

        # Create vertical layouts for each column type
        delete_layout = QVBoxLayout()
        label_layout = QVBoxLayout()
        model_layout = QVBoxLayout()
        param_hint_layouts = {param_hint: {var: QVBoxLayout() for var in [ 'min', 'value','max', 'expr','vary']} for
                              param_hint in param_hint_order}

        # Add header labels to each layout
        for header_label in header_labels:
            label = QLabel(header_label)
            label.setAlignment(Qt.AlignCenter)
            if header_label == "  ":
                delete_layout.addWidget(label)
            elif header_label == "Label":
                label_layout.addWidget(label)
            elif header_label == "Model":
                model_layout.addWidget(label)
            elif header_label.startswith("fix"):
                param_hint_key = header_label.split()[1].lower()
                param_hint_layouts[param_hint_key]['vary'].addWidget(label)
            elif "min" in header_label:
                param_hint_key = header_label.split()[1].lower()
                param_hint_layouts[param_hint_key]['min'].addWidget(label)
            elif "max" in header_label:
                param_hint_key = header_label.split()[1].lower()
                param_hint_layouts[param_hint_key]['max'].addWidget(label)
            elif "expression" in header_label:
                param_hint_key = header_label.split()[1].lower()
                param_hint_layouts[param_hint_key]['expr'].addWidget(label)
            else:
                param_hint_key = header_label.lower()
                param_hint_layouts[param_hint_key]['value'].addWidget(label)

        for i, peak_model in enumerate(self.sel_spectrum.peak_models):
            # Button to delete peak_model
            delete = QPushButton(peak_model.prefix)
            icon = QIcon()
            icon.addFile(os.path.join(ICON_DIR, "close.png"))
            delete.setIcon(icon)
            delete.setFixedWidth(50)
            delete.clicked.connect(self.delete_helper(self.sel_spectrum, i))
            delete_layout.addWidget(delete)

            # Peak_label
            label = QLineEdit(self.sel_spectrum.peak_labels[i])
            label.setFixedWidth(80)
            label.textChanged.connect(
                lambda text, idx=i, spectrum=self.sel_spectrum: self.update_peak_label(spectrum,idx, text))
            label_layout.addWidget(label)

            # Peak model : Lorentizan, Gaussian, etc...
            model = QComboBox()
            model.addItems(PEAK_MODELS)
            current_model_index = PEAK_MODELS.index(
                peak_model.name2) if peak_model.name2 in PEAK_MODELS else 0
            model.setCurrentIndex(current_model_index)
            model.setFixedWidth(120)
            model.currentIndexChanged.connect(
                lambda index, spectrum=self.sel_spectrum, idx=i,
                       combo=model: self.update_model_name(spectrum, index, idx, combo.currentText()))
            model_layout.addWidget(model)

            # variables of peak_model
            param_hints = peak_model.param_hints
            for param_hint_key in param_hint_order:
                if param_hint_key in param_hints:
                    param_hint_value = param_hints[param_hint_key]

                    # 4.1 VALUE
                    value_val = round(param_hint_value.get('value', 0.0), 3)
                    value = QLineEdit(str(value_val))
                    value.setFixedWidth(70)
                    value.setFixedHeight(24)
                    value.setAlignment(Qt.AlignRight)
                    value.textChanged.connect(lambda text, pm=peak_model, key=param_hint_key: self.update_param_hint_value(pm, key, text))
                    param_hint_layouts[param_hint_key]['value'].addWidget(value)

                    # 4.2 FIXED or NOT
                    vary = QCheckBox()
                    vary.setChecked(not param_hint_value.get('vary', False))
                    vary.setFixedHeight(24)

                    # Create container widget with horizontal layout to center the checkbox
                    checkbox_container = QWidget()
                    checkbox_layout = QHBoxLayout()
                    checkbox_layout.setContentsMargins(0, 0, 0, 0)
                    checkbox_layout.setAlignment(Qt.AlignCenter)
                    checkbox_layout.addWidget(vary)
                    checkbox_container.setLayout(checkbox_layout)

                    vary.stateChanged.connect(
                        lambda state, pm=peak_model,
                            key=param_hint_key: self.update_param_hint_vary(pm, key, not state))

                    param_hint_layouts[param_hint_key]['vary'].addWidget(checkbox_container)

                    # 4.3 MIN MAX
                    if self.cb_limits.isChecked():
                        min_val = round(param_hint_value.get('min', 0.0), 2)
                        min_lineedit = QLineEdit(str(min_val))
                        min_lineedit.setFixedWidth(70)
                        min_lineedit.setFixedHeight(24)
                        min_lineedit.setAlignment(Qt.AlignRight)
                        
                        palette = min_lineedit.palette()
                        palette.setColor(QPalette.Text, QColor("red"))
                        min_lineedit.setPalette(palette)
                        
                        min_lineedit.textChanged.connect(
                            lambda text, pm=peak_model,key=param_hint_key:
                            self.update_param_hint_min(pm, key, text))
                        param_hint_layouts[param_hint_key]['min'].addWidget(min_lineedit)

                        max_val = round(param_hint_value.get('max', 0.0), 2)
                        max_lineedit = QLineEdit(str(max_val))
                        max_lineedit.setFixedWidth(70)
                        max_lineedit.setFixedHeight(24)
                        max_lineedit.setAlignment(Qt.AlignRight)
                        
                        palette = max_lineedit.palette()
                        palette.setColor(QPalette.Text, QColor("red"))
                        max_lineedit.setPalette(palette)
                        
                        max_lineedit.textChanged.connect(
                            lambda text, pm=peak_model, key=param_hint_key:
                            self.update_param_hint_max(pm, key, text))
                        param_hint_layouts[param_hint_key]['max'].addWidget(max_lineedit)

                    # 4.4 EXPRESSION
                    if self.cb_expr.isChecked():
                        expr_val = str(param_hint_value.get('expr', ''))
                        expr = QLineEdit(expr_val)
                        expr.setFixedWidth(150)
                        expr.setFixedHeight(
                            24)  # Set a fixed height for QLineEdit
                        expr.setAlignment(Qt.AlignRight)
                        expr.textChanged.connect(
                            lambda text, pm=peak_model,
                                   key=param_hint_key:
                            self.update_param_hint_expr(
                                pm, key, text))
                        param_hint_layouts[param_hint_key]['expr'].addWidget(expr)
                else:
                    # Add empty labels for alignment
                    empty_label = QLabel()
                    empty_label.setFixedHeight(24)
                    param_hint_layouts[param_hint_key]['value'].addWidget(empty_label)
                    param_hint_layouts[param_hint_key]['vary'].addWidget(empty_label)
                    if self.cb_limits.isChecked():
                        param_hint_layouts[param_hint_key]['min'].addWidget(empty_label)
                        param_hint_layouts[param_hint_key]['max'].addWidget(empty_label)
                    if self.cb_expr.isChecked():
                        param_hint_layouts[param_hint_key]['expr'].addWidget(empty_label)

        # Add vertical layouts to main layout
        self.main_layout.addLayout(delete_layout)
        self.main_layout.addLayout(label_layout)
        self.main_layout.addLayout(model_layout)

        for param_hint_key, param_hint_layout in param_hint_layouts.items():
            for var_layout in param_hint_layout.values():
                self.main_layout.addLayout(var_layout)
                
        # Add a horizontal spacer to absorb any remaining space
        spacer = QSpacerItem(20, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.main_layout.addItem(spacer)

    def update_model_name(self, spectrum, index, idx, new_model):
        """ Update the model function (Lorentizan, Gaussian...) related to the ith-model """
        old_model_name = spectrum.peak_models[idx].name2
        new_model_name = new_model
        if new_model_name != old_model_name:
            ampli = spectrum.peak_models[idx].param_hints['ampli']['value']
            x0 = spectrum.peak_models[idx].param_hints['x0']['value']
            peak_model = spectrum.create_peak_model(idx + 1, new_model_name, x0=x0, ampli=ampli, dx0=(20., 20.))
            spectrum.peak_models[idx] = peak_model
            spectrum.result_fit = lambda: None
            self.refresh_gui()  # To update in GUI of main application.

    def delete_helper(self, spectrum, idx):
        """Helper method"""
        return lambda: self.delete_peak_model(spectrum, idx)
    
    def delete_peak_model(self, spectrum, idx):
        """To delete a peak model"""
        del spectrum.peak_models[idx]
        del spectrum.peak_labels[idx]
        self.refresh_gui()  # To update in GUI of main application.
        
    def refresh_gui(self):
        """Call the refresh_gui method of the main application."""
        if hasattr(self.main_app, 'refresh_gui'):
            self.main_app.refresh_gui()
        else:
            print("Main application does not have upd_spectra_list method.")

    def update_peak_label(self, spectrum, idx, text):
        spectrum.peak_labels[idx] = text

    def update_param_hint_value(self, pm, key, text):
        pm.param_hints[key]['value'] = float(text)
        # Update GUI / spectraviewer after changing a parameter value
        QTimer.singleShot(1000, self.refresh_gui)

    def update_param_hint_min(self, pm, key, text):
        pm.param_hints[key]['min'] = float(text)

    def update_param_hint_max(self, pm, key, text):
        pm.param_hints[key]['max'] = float(text)

    def update_param_hint_vary(self, pm, key, state):
        pm.param_hints[key]['vary'] = state

    def update_param_hint_expr(self, pm, key, text):
        pm.param_hints[key]['expr'] = text

    def clear(self):
        """Clears all data from the main layout."""
        self.clear_layout(self.main_layout)
