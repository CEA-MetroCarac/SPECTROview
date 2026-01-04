# spectroview/view/components/v_peak_table.py

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QComboBox, QPushButton, QCheckBox, QSizePolicy, QSpacerItem
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QIcon, QPalette, QColor

from spectroview import PEAK_MODELS, ICON_DIR


class VPeakTable(QWidget):
    """
    Peak table VIEW (MVVM compliant)

    - Displays peak parameters
    - Emits signals only
    - No direct model mutation
    """

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

        if (
            self._spectrum is None
            or not getattr(self._spectrum, "peak_models", None)
            or not self._spectrum.peak_models
        ):
            return

        peaks = self._spectrum.peak_models
        labels = self._spectrum.peak_labels

        param_order = ["x0", "fwhm", "fwhm_l", "fwhm_r", "ampli", "alpha"]

        # ── Column containers
        cols = {
            "del": QVBoxLayout(),
            "label": QVBoxLayout(),
            "model": QVBoxLayout(),
        }

        param_cols = {}
        for p in param_order:
            param_cols[p] = {
                "value": QVBoxLayout(),
                "vary": QVBoxLayout(),
            }
            if self._show_limits:
                param_cols[p]["min"] = QVBoxLayout()
                param_cols[p]["max"] = QVBoxLayout()
            if self._show_expr:
                param_cols[p]["expr"] = QVBoxLayout()

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
            btn = QPushButton()
            btn.setIcon(QIcon(f"{ICON_DIR}/close.png"))
            btn.setFixedWidth(28)
            btn.clicked.connect(lambda _, idx=i: self.peak_deleted.emit(idx))
            cols["del"].addWidget(btn)

            # label
            le = QLineEdit(labels[i])
            le.editingFinished.connect(
                lambda idx=i, w=le: self.peak_label_changed.emit(idx, w.text())
            )
            cols["label"].addWidget(le)

            # model
            cmb = QComboBox()
            cmb.addItems(PEAK_MODELS)
            cmb.setCurrentText(pm.name2)
            cmb.currentTextChanged.connect(
                lambda txt, idx=i: self.peak_model_changed.emit(idx, txt)
            )
            cols["model"].addWidget(cmb)

            # parameters
            for p, d in param_cols.items():
                hint = pm.param_hints.get(p)

                if hint is None:
                    self._add_empty(d)
                    continue

                # value
                val = QLineEdit(f"{hint.get('value', 0):.6g}")
                val.setAlignment(Qt.AlignRight)
                val.editingFinished.connect(
                    lambda idx=i, k=p, w=val:
                    self._emit_value(idx, k, "value", w.text())
                )
                d["value"].addWidget(val)

                # vary
                chk = QCheckBox()
                chk.setChecked(not hint.get("vary", True))
                chk.toggled.connect(
                    lambda st, idx=i, k=p:
                    self.peak_param_changed.emit(idx, k, "vary", not st)
                )
                d["vary"].addWidget(self._center(chk))

                # limits
                if "min" in d:
                    self._add_limit(d["min"], i, p, "min", hint)
                    self._add_limit(d["max"], i, p, "max", hint)

                # expr
                if "expr" in d:
                    expr = QLineEdit(str(hint.get("expr", "")))
                    expr.editingFinished.connect(
                        lambda idx=i, k=p, w=expr:
                        self.peak_param_changed.emit(idx, k, "expr", w.text())
                    )
                    d["expr"].addWidget(expr)

        # ── Assemble
        self.main_layout.addLayout(cols["del"])
        self.main_layout.addLayout(cols["label"])
        self.main_layout.addLayout(cols["model"])

        for d in param_cols.values():
            for col in d.values():
                self.main_layout.addLayout(col)

        self.main_layout.addStretch()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _emit_value(self, idx, key, field, text):
        try:
            val = float(text)
        except ValueError:
            return
        self.peak_param_changed.emit(idx, key, field, val)

    def _add_limit(self, layout, idx, key, field, hint):
        le = QLineEdit(f"{hint.get(field, 0):.6g}")
        le.setAlignment(Qt.AlignRight)

        pal = le.palette()
        pal.setColor(QPalette.Text, QColor("red"))
        le.setPalette(pal)

        le.editingFinished.connect(
            lambda idx=idx, k=key, f=field, w=le:
            self._emit_value(idx, k, f, w.text())
        )
        layout.addWidget(le)

    def _add_header(self, layout, text):
        lbl = QLabel(text)
        lbl.setAlignment(Qt.AlignCenter)
        layout.addWidget(lbl)

    def _add_empty(self, d):
        for col in d.values():
            col.addWidget(QLabel(""))

    def _center(self, w):
        c = QWidget()
        l = QHBoxLayout(c)
        l.setContentsMargins(0, 0, 0, 0)
        l.setAlignment(Qt.AlignCenter)
        l.addWidget(w)
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
