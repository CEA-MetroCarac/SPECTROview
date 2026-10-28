# spectroview/settings/ui/tab_fit.py
"""Fit settings tab: peak polarity, solver convergence, and global peak bounds."""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QSpacerItem,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


def _highlight_group_style() -> str:
    return (
        "QGroupBox { border: 1.5px solid #3B82F6; border-radius: 6px; margin-top: 10px; font-weight: bold; } "
        "QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 0 4px; color: #3B82F6; }"
    )


def _highlight_label_style() -> str:
    return "background-color: rgba(59, 130, 246, 0.2); border: 1px solid #3B82F6; border-radius: 4px; padding: 2px 4px; color: #3B82F6; font-weight: bold;"


class FitSettingsTabPage(QWidget):
    """The Fit Settings tab: solver convergence parameters, peak bounds, and options."""

    KEYWORDS = [
        "fit", "fitting", "solver", "convergence", "negative", "absorption",
        "iterations", "tolerance", "xtol", "ftol", "noise", "threshold",
        "loss", "scale", "f_scale", "bounds", "limits", "fwhm", "shift"
    ]

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._searchable_rows: list[tuple[list[str], QLabel, QGroupBox]] = []
        self._init_ui()

    def _init_ui(self) -> None:
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QScrollArea.NoFrame)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        # ───── Negative Peak Fitting ─────
        self.grp_neg = QGroupBox("Peak Polarity")
        neg_layout = QVBoxLayout(self.grp_neg)
        neg_layout.setContentsMargins(8, 8, 8, 8)
        neg_layout.setSpacing(6)

        self.chk_fit_negative = QCheckBox("Fit Negative Peaks (absorption/dips)")
        self.chk_fit_negative.setToolTip("Allow peak fitting models to fit inverted/negative peak profiles")
        neg_layout.addWidget(self.chk_fit_negative)
        layout.addWidget(self.grp_neg)

        # ───── Solver Convergence ─────
        self.grp_solver = QGroupBox("Fitting Convergence")
        solver_layout = QVBoxLayout(self.grp_solver)
        solver_layout.setContentsMargins(8, 8, 8, 8)
        solver_layout.setSpacing(6)

        def add_row(lbl_text: str, widget: QWidget, tooltip: str = "", keywords: list[str] | None = None) -> QLabel:
            row = QHBoxLayout()
            lbl = QLabel(lbl_text)
            lbl.setMinimumWidth(180)
            lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
            if tooltip:
                lbl.setToolTip(tooltip)
                widget.setToolTip(tooltip)
            row.addWidget(lbl)
            row.addWidget(widget, 1)
            solver_layout.addLayout(row)
            terms = [lbl_text.lower(), tooltip.lower()]
            if keywords:
                terms.extend(keywords)
            self._searchable_rows.append((terms, lbl, self.grp_solver))
            return lbl

        self.spin_max_iter = QSpinBox()
        self.spin_max_iter.setRange(10, 10000)
        self.spin_max_iter.setSingleStep(50)
        add_row("Max iterations:", self.spin_max_iter, "Maximum number of Levenberg-Marquardt optimizer iterations", ["max_ite", "iterations"])

        self.spin_x_tol = QDoubleSpinBox()
        self.spin_x_tol.setRange(1e-10, 1.0)
        self.spin_x_tol.setDecimals(6)
        add_row("X tolerance (xtol):", self.spin_x_tol, "Relative error desired in the approximate solution", ["xtol", "tolerance"])

        self.spin_f_tol = QDoubleSpinBox()
        self.spin_f_tol.setRange(1e-10, 1.0)
        self.spin_f_tol.setDecimals(6)
        add_row("Function tolerance (ftol):", self.spin_f_tol, "Relative error desired in the sum of squares", ["ftol", "tolerance"])

        self.spin_coef_noise = QDoubleSpinBox()
        self.spin_coef_noise.setRange(0.0, 10.0)
        self.spin_coef_noise.setSingleStep(0.5)
        self.spin_coef_noise.setDecimals(2)
        add_row("Noise threshold (× noise):", self.spin_coef_noise, "Residual weight factor relative to baseline noise", ["noise", "threshold", "coef_noise"])

        self.combo_loss = QComboBox()
        self.combo_loss.addItems(["linear", "soft_l1", "huber", "cauchy", "arctan"])
        add_row("Loss function:", self.combo_loss, "Robust loss function used for outlier rejection", ["loss", "robust", "huber", "cauchy"])

        self.spin_f_scale = QDoubleSpinBox()
        self.spin_f_scale.setRange(0.01, 100.0)
        self.spin_f_scale.setSingleStep(0.1)
        self.spin_f_scale.setDecimals(3)
        add_row("Loss scale (f_scale):", self.spin_f_scale, "Soft margin between inlier and outlier residuals", ["f_scale", "scale"])

        layout.addWidget(self.grp_solver)

        # ───── Global Peak Limits ─────
        self.grp_bounds = QGroupBox("Global Peak Limits")
        bounds_layout = QVBoxLayout(self.grp_bounds)
        bounds_layout.setContentsMargins(8, 8, 8, 8)
        bounds_layout.setSpacing(6)

        def add_bounds_row(lbl_text: str, widget: QWidget, tooltip: str = "", keywords: list[str] | None = None) -> QLabel:
            row = QHBoxLayout()
            lbl = QLabel(lbl_text)
            lbl.setMinimumWidth(180)
            lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
            if tooltip:
                lbl.setToolTip(tooltip)
                widget.setToolTip(tooltip)
            row.addWidget(lbl)
            row.addWidget(widget, 1)
            bounds_layout.addLayout(row)
            terms = [lbl_text.lower(), tooltip.lower()]
            if keywords:
                terms.extend(keywords)
            self._searchable_rows.append((terms, lbl, self.grp_bounds))
            return lbl

        self.spin_maxshift = QDoubleSpinBox()
        self.spin_maxshift.setRange(0, 100)
        self.spin_maxshift.setSingleStep(5)
        self.spin_maxshift.setDecimals(2)
        add_bounds_row("Max peak shift:", self.spin_maxshift, "Maximum allowable center position shift during fit", ["shift", "maxshift", "position"])

        self.spin_minfwhm = QDoubleSpinBox()
        self.spin_minfwhm.setRange(0.01, 500)
        self.spin_minfwhm.setSingleStep(1)
        self.spin_minfwhm.setDecimals(2)
        add_bounds_row("Min peak FWHM:", self.spin_minfwhm, "Minimum allowable full-width at half-maximum", ["fwhm", "minfwhm", "width"])

        self.spin_maxfwhm = QDoubleSpinBox()
        self.spin_maxfwhm.setRange(1, 1000)
        self.spin_maxfwhm.setSingleStep(20)
        self.spin_maxfwhm.setDecimals(2)
        add_bounds_row("Max peak FWHM:", self.spin_maxfwhm, "Maximum allowable full-width at half-maximum", ["fwhm", "maxfwhm", "width"])

        layout.addWidget(self.grp_bounds)

        layout.addSpacerItem(QSpacerItem(20, 10, QSizePolicy.Minimum, QSizePolicy.Expanding))

        self._scroll.setWidget(container)
        outer_layout.addWidget(self._scroll)

    def apply_settings(self, data: dict) -> None:
        self.chk_fit_negative.setChecked(data.get("fit_negative", False))
        self.spin_max_iter.setValue(data.get("max_ite", 200))
        self.spin_x_tol.setValue(data.get("xtol", 1e-4))
        self.spin_f_tol.setValue(data.get("ftol", 1e-4))
        self.spin_coef_noise.setValue(data.get("coef_noise", 1.0))

        loss = str(data.get("loss", "linear"))
        idx = self.combo_loss.findText(loss)
        if idx >= 0:
            self.combo_loss.setCurrentIndex(idx)
        self.spin_f_scale.setValue(float(data.get("f_scale", 1.0)))

        self.spin_maxshift.setValue(data.get("maxshift", 20.0))
        self.spin_minfwhm.setValue(data.get("minfwhm", 0.1))
        self.spin_maxfwhm.setValue(data.get("maxfwhm", 200.0))

    def get_settings(self) -> dict:
        return {
            "fit_negative": self.chk_fit_negative.isChecked(),
            "max_ite": self.spin_max_iter.value(),
            "xtol": self.spin_x_tol.value(),
            "ftol": self.spin_f_tol.value(),
            "coef_noise": self.spin_coef_noise.value(),
            "loss": self.combo_loss.currentText(),
            "f_scale": self.spin_f_scale.value(),
            "maxshift": self.spin_maxshift.value(),
            "minfwhm": self.spin_minfwhm.value(),
            "maxfwhm": self.spin_maxfwhm.value(),
        }

    # ───── Search Highlighting & Scrolling ─────
    def highlight_search(self, query: str) -> None:
        """Highlight matching options/groupboxes and scroll first match into view."""
        q = query.strip().lower()

        # Clear existing styles
        self.grp_neg.setStyleSheet("")
        self.grp_solver.setStyleSheet("")
        self.grp_bounds.setStyleSheet("")
        for _terms, lbl, _grp in self._searchable_rows:
            lbl.setStyleSheet("")

        if not q:
            return

        first_match: Optional[QWidget] = None

        if "negative" in q or "polarity" in q or "absorption" in q or "dip" in q:
            self.grp_neg.setStyleSheet(_highlight_group_style())
            first_match = self.grp_neg

        for terms, lbl, grp in self._searchable_rows:
            if any(q in t or t in q for t in terms):
                lbl.setStyleSheet(_highlight_label_style())
                if first_match is None:
                    first_match = lbl

        if first_match:
            self._scroll.ensureWidgetVisible(first_match, 0, 30)
