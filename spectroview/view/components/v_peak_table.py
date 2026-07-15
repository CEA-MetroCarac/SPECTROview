# spectroview/view/components/v_peak_table.py

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QComboBox, QPushButton, QCheckBox, QSizePolicy, QSpacerItem
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QIcon, QPalette, QColor

from spectroview import PEAK_MODELS, ICON_DIR
from spectroview.viewmodel.utils import fano_display_amplitude, fano_internal_amplitude

ROW_HEIGHT = 28
class VPeakTable(QWidget):
    # ───── View → ViewModel signals ─────
    peak_label_changed = Signal(int, str)
    peak_model_changed = Signal(int, str)
    peak_param_changed = Signal(int, str, str, object)  # idx, key, field, value
    peak_deleted = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)

        self._spectrum = None
        self._show_limits = False
        self._show_expr = False

        self._build_ui()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.main_layout = QHBoxLayout()
        layout.addLayout(self.main_layout)

        layout.addItem(
            QSpacerItem(10, 10, QSizePolicy.Expanding, QSizePolicy.Minimum)
        )

    # ------------------------------------------------------------------
    # Public API (called from VFitModelBuilder)
    # ------------------------------------------------------------------
    def set_spectrum(self, spectrum):
        self._spectrum = spectrum
        self._rebuild()

    def set_show_limits(self, state: bool):
        self._show_limits = state
        self._rebuild()

    def set_show_expr(self, state: bool):
        self._show_expr = state
        self._rebuild()

    def clear(self):
        self._spectrum = None
        self._clear_layout(self.main_layout)

    # ------------------------------------------------------------------
    # Table build
    # ------------------------------------------------------------------
    def _rebuild(self):
        self._clear_layout(self.main_layout)

        fit_model = None
        if hasattr(self._spectrum, "fit_model"):
            fit_model = self._spectrum.fit_model
        elif isinstance(self._spectrum, dict):
            fit_model = self._spectrum.get("fit_model")
            if not fit_model and self._spectrum.get("fit_models"):
                fit_models = self._spectrum.get("fit_models")
                if fit_models and len(fit_models) > 0:
                    fit_model = fit_models[0]
            
        if not fit_model or not fit_model.get("peak_models"):
            return

        class DummyPeak:
            def __init__(self, key, shape, params, prefix):
                self.key = key
                self.name2 = shape
                self.param_hints = params
                self.prefix = prefix

        peaks = []
        for k, pdict in fit_model["peak_models"].items():
            shape = list(pdict.keys())[0]
            peaks.append(DummyPeak(k, shape, pdict[shape], f"P{int(k)+1}_"))

        labels = fit_model.get("peak_labels", [])

        param_order = ["x0", "fwhm", "fwhm_l", "fwhm_r", "ampli", "alpha", "q", "tau", "tau1", "tau2", "A", "A1", "A2", "B"]
        
        # determine which parameters are present in at least one peak
        active_params = {
            p for p in param_order
            if any(p in pm.param_hints for pm in peaks)
        }

        def _make_col():
            lay = QVBoxLayout()
            lay.setSpacing(4)
            lay.setContentsMargins(0, 0, 0, 0)
            lay.setAlignment(Qt.AlignTop)
            return lay

        # ── Column containers
        cols = {
            "del": _make_col(),
            "label": _make_col(),
            "model": _make_col(),
        }

        param_cols = {}
        for p in param_order:
            if p not in active_params:
                continue

            param_cols[p] = {}
            # Target order: [min] [value] [max] [fix]
            if self._show_limits:
                param_cols[p]["min"] = _make_col()

            param_cols[p]["value"] = _make_col()

            if self._show_limits:
                param_cols[p]["max"] = _make_col()

            param_cols[p]["vary"] = _make_col()
            
            if self._show_expr:
                param_cols[p]["expr"] = _make_col()


        # ── Headers
        self._add_header(cols["del"], "")
        self._add_header(cols["label"], "Label")
        self._add_header(cols["model"], "Model")

        for p, d in param_cols.items():
            self._add_header(d["value"], p)
            self._add_header(d["vary"], "fix")
            if "min" in d:
                self._add_header(d["min"], "min")
                self._add_header(d["max"], "max")
            if "expr" in d:
                self._add_header(d["expr"], "expr")

        # ── Rows
        for i, pm in enumerate(peaks):
            # delete
            btn = QPushButton(pm.prefix)
            btn.setIcon(QIcon(f"{ICON_DIR}/close.png"))
            btn.setFixedSize(50, ROW_HEIGHT)
            btn.clicked.connect(self._make_delete_callback(int(pm.key)))
            cols["del"].addWidget(btn)

            # label
            k_int = int(pm.key)
            le_text = labels[k_int] if k_int < len(labels) and labels[k_int] is not None else f"Peak{k_int+1}"
            le = QLineEdit(le_text)
            le.setFixedSize(60, ROW_HEIGHT)
            le.editingFinished.connect(
                lambda key=pm.key, w=le: self.peak_label_changed.emit(int(key), w.text())
            )
            cols["label"].addWidget(le)

            # model
            cmb = QComboBox()
            cmb.addItems(PEAK_MODELS)
            cmb.setCurrentText(pm.name2)
            cmb.setFixedSize(100, ROW_HEIGHT)
            cmb.currentTextChanged.connect(
                lambda txt, key=pm.key: self.peak_model_changed.emit(int(key), txt)
            )
            cols["model"].addWidget(cmb)

            # parameters
            for p, d in param_cols.items():
                hint = pm.param_hints.get(p)

                if hint is None:
                    self._add_empty(d)
                    continue

                # value
                display_val = hint.get('value', 0)
                
                # Fano model correction: show actual peak height for 'ampli'
                if pm.name2 == "Fano" and p == "ampli":
                    q = pm.param_hints.get("q", {}).get("value", 50.0)
                    display_val = fano_display_amplitude(display_val, q)

                val = QLineEdit(f"{display_val:.3f}")
                val.setAlignment(Qt.AlignRight)
                val.setFixedSize(72, ROW_HEIGHT)

                def make_editing_finished_cb(i_idx, p_key, w_val, model_name, peak_model):
                    def cb():
                        text = w_val.text()
                        try:
                            num_val = float(text)
                            # If they edit Fano's ampli, we must scale it back to internal representation
                            if model_name == "Fano" and p_key == "ampli":
                                current_q = peak_model.param_hints.get("q", {}).get("value", 50.0)
                                num_val = fano_internal_amplitude(num_val, current_q)
                            self.peak_param_changed.emit(i_idx, p_key, "value", num_val)
                        except ValueError:
                            pass
                    return cb

                val.editingFinished.connect(make_editing_finished_cb(int(pm.key), p, val, pm.name2, pm))
                d["value"].addWidget(val)

                # vary
                chk = QCheckBox()
                chk.setChecked(not hint.get("vary", True))
                chk.toggled.connect(
                    lambda st, key=pm.key, k=p:
                    self.peak_param_changed.emit(int(key), k, "vary", not st)
                )
                d["vary"].addWidget(self._center_checkbox(chk))

                # limits
                if "min" in d:
                    self._add_limit(d["min"], int(pm.key), p, "min", hint, pm.name2, pm)
                    self._add_limit(d["max"], int(pm.key), p, "max", hint, pm.name2, pm)

                # expr
                if "expr" in d:
                    expr = QLineEdit(str(hint.get("expr", "")))
                    # expr.setFixedWidth(150)
                    expr.setFixedSize(150, ROW_HEIGHT)
                    expr.editingFinished.connect(
                        lambda key=pm.key, k=p, w=expr:
                        self.peak_param_changed.emit(int(key), k, "expr", w.text())
                    )
                    d["expr"].addWidget(expr)

        # ── Add vertical stretch to absorb remaining space
        def add_vstretch(layout):
            layout.addStretch(1)

        # ── Assemble
        # add stretch to fixed columns
        for c in cols.values():
            add_vstretch(c)
            self.main_layout.addLayout(c)

        # add stretch to parameter columns
        for d in param_cols.values():
            for col in d.values():
                add_vstretch(col)
                self.main_layout.addLayout(col)

        self.main_layout.addStretch()
        
    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _make_delete_callback(self, idx):
        """Create a callback that properly captures the peak index."""
        return lambda: self.peak_deleted.emit(idx)

    def _add_limit(self, layout, idx, key, field, hint, pm_name=None, peak_model=None):
        display_val = hint.get(field, 0)
        
        # Fano model correction: show actual peak height for 'ampli' limits
        if pm_name == "Fano" and key == "ampli" and peak_model:
            q = peak_model.param_hints.get("q", {}).get("value", 50.0)
            display_val = fano_display_amplitude(display_val, q)


        le = QLineEdit(f"{display_val:.3f}")
        # le.setFixedWidth(60)
        le.setFixedSize(72, ROW_HEIGHT)
        le.setAlignment(Qt.AlignRight)

        pal = le.palette()
        pal.setColor(QPalette.Text, QColor("red"))
        le.setPalette(pal)

        def make_editing_finished_limit_cb(i_idx, p_key, p_field, w_val, model_name, p_model):
            def cb():
                text = w_val.text()
                try:
                    num_val = float(text)
                    if model_name == "Fano" and p_key == "ampli":
                        current_q = p_model.param_hints.get("q", {}).get("value", 50.0)
                        num_val = fano_internal_amplitude(num_val, current_q)
                    self.peak_param_changed.emit(i_idx, p_key, p_field, num_val)
                except ValueError:
                    pass
            return cb

        le.editingFinished.connect(make_editing_finished_limit_cb(idx, key, field, le, pm_name, peak_model))
        layout.addWidget(le)

    def _add_header(self, layout, text):
        lbl = QLabel(text)
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setFixedHeight(ROW_HEIGHT)
        layout.addWidget(lbl)

    def _add_empty(self, d):
        """Add empty labels for alignment"""
        for col in d.values():
            empty_label = QLabel()
            empty_label.setFixedHeight(ROW_HEIGHT)
            col.addWidget(empty_label)
            

    def _center_checkbox(self, chk):
        c = QWidget()
        l = QHBoxLayout(c)
        l.setContentsMargins(0, 0, 0, 0)
        l.setAlignment(Qt.AlignCenter)

        c.setFixedHeight(ROW_HEIGHT)

        l.addWidget(chk)
        return c


    def _clear_layout(self, layout):
        if not layout:
            return
        while layout.count():
            item = layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                self._clear_layout(item.layout())
