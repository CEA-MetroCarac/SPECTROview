# spectroview/settings/ui/tab_fit.py
"""Fit settings tab: peak polarity, solver convergence, and global peak bounds."""

from __future__ import annotations

from typing import Any, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpacerItem,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from spectroview.i18n import (
    bind_text,
    bind_title,
    bind_tooltip,
    combo_data_or_text,
    on_language_changed,
    tr,
)
from spectroview.settings.m_settings import BUILTIN_PEAK_PROFILES
from spectroview.view.theme import Icons

_CUSTOM_PROFILE_KEY = "__custom__"
_PROFILE_MATCH_TOL = 1e-6


DEFAULT_FIT_SETTINGS: dict[str, Any] = {
    "fit_negative": False,
    "max_ite": 200,
    "xtol": 1e-4,
    "ftol": 1e-4,
    "coef_noise": 0.0,
    "loss": "linear",
    "f_scale": 1.0,
    "maxshift": 20.0,
    "minfwhm": 0.1,
    "maxfwhm": 200.0,
}


def _highlight_group_style() -> str:
    return (
        "QGroupBox { border: 1.5px solid #3B82F6; border-radius: 6px; margin-top: 10px; font-weight: bold; } "
        "QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 0 4px; color: #3B82F6; }"
    )


def _highlight_label_style() -> str:
    return "background-color: rgba(59, 130, 246, 0.2); border: 1px solid #3B82F6; border-radius: 4px; padding: 2px 4px; color: #3B82F6; font-weight: bold;"


class FitSettingsTabPage(QWidget):
    """The Fit Settings tab: solver convergence parameters, peak bounds, and options."""

    save_profile_requested = Signal(float, float, float, str)
    delete_profile_requested = Signal(str)

    KEYWORDS = [
        "fit", "fitting", "ajustement", "solver", "solveur", "convergence", "negative", "négatif",
        "absorption", "iterations", "itérations", "tolerance", "tolérance", "xtol", "ftol",
        "noise", "bruit", "threshold", "seuil", "loss", "perte", "scale", "échelle", "f_scale",
        "bounds", "bornes", "limits", "limites", "fwhm", "shift", "décalage",
        "profile", "profil", "preset", "raman", "pl", "xrd",
        "reset", "réinitialiser", "default", "defaults", "défaut",
    ]

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._searchable_rows: list[tuple[list[str], QLabel, QGroupBox]] = []
        self._peak_profiles: dict[str, dict[str, float]] = {
            name: dict(vals) for name, vals in BUILTIN_PEAK_PROFILES.items()
        }
        self._user_peak_profiles: set[str] = set()
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

        # ───── Reset to Defaults Action ─────
        top_bar = QHBoxLayout()
        top_bar.addStretch()
        self.btn_reset_defaults = QPushButton()
        bind_text(self.btn_reset_defaults, "Reset all fit settings to default")
        self.btn_reset_defaults.setIcon(Icons.refresh())
        self.btn_reset_defaults.setCursor(Qt.PointingHandCursor)
        bind_tooltip(self.btn_reset_defaults, "Reset all fit parameters and peak limits to default values")
        self.btn_reset_defaults.clicked.connect(self.reset_to_defaults)
        top_bar.addWidget(self.btn_reset_defaults)
        layout.addLayout(top_bar)

        # ───── Negative Peak Fitting ─────
        self.grp_neg = QGroupBox()
        bind_title(self.grp_neg, "Peak Polarity")
        neg_layout = QVBoxLayout(self.grp_neg)
        neg_layout.setContentsMargins(8, 8, 8, 8)
        neg_layout.setSpacing(6)

        self.chk_fit_negative = QCheckBox()
        bind_text(self.chk_fit_negative, "Fit Negative Peaks (absorption/dips)")
        bind_tooltip(self.chk_fit_negative, "Allow peak fitting models to fit inverted/negative peak profiles")
        neg_layout.addWidget(self.chk_fit_negative)
        layout.addWidget(self.grp_neg)

        # ───── Solver Convergence ─────
        self.grp_solver = QGroupBox()
        bind_title(self.grp_solver, "Fitting Convergence")
        solver_layout = QVBoxLayout(self.grp_solver)
        solver_layout.setContentsMargins(8, 8, 8, 8)
        solver_layout.setSpacing(6)

        def add_row(lbl_text: str, widget: QWidget, tooltip: str = "", keywords: list[str] | None = None) -> QLabel:
            row = QHBoxLayout()
            lbl = QLabel()
            bind_text(lbl, lbl_text)
            lbl.setMinimumWidth(180)
            lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
            if tooltip:
                bind_tooltip(lbl, tooltip)
                bind_tooltip(widget, tooltip)
            row.addWidget(lbl)
            row.addWidget(widget, 1)
            solver_layout.addLayout(row)
            terms = [lbl_text.lower(), tr(lbl_text).lower(), tooltip.lower(), tr(tooltip).lower()]
            if keywords:
                terms.extend(keywords)
            self._searchable_rows.append((terms, lbl, self.grp_solver))
            return lbl

        self.spin_max_iter = QSpinBox()
        self.spin_max_iter.setRange(10, 10000)
        self.spin_max_iter.setSingleStep(50)
        add_row("Max iterations:", self.spin_max_iter, "Maximum number of Levenberg-Marquardt optimizer iterations", ["max_ite", "iterations", "itérations"])

        self.spin_x_tol = QDoubleSpinBox()
        self.spin_x_tol.setRange(1e-10, 1.0)
        self.spin_x_tol.setDecimals(6)
        add_row("X tolerance (xtol):", self.spin_x_tol, "Relative error desired in the approximate solution", ["xtol", "tolerance", "tolérance"])

        self.spin_f_tol = QDoubleSpinBox()
        self.spin_f_tol.setRange(1e-10, 1.0)
        self.spin_f_tol.setDecimals(6)
        add_row("Function tolerance (ftol):", self.spin_f_tol, "Relative error desired in the sum of squares", ["ftol", "tolerance", "tolérance"])

        self.spin_coef_noise = QDoubleSpinBox()
        self.spin_coef_noise.setRange(0.0, 10.0)
        self.spin_coef_noise.setSingleStep(0.5)
        self.spin_coef_noise.setDecimals(2)
        add_row("Noise threshold (× noise):", self.spin_coef_noise, "Residual weight factor relative to baseline noise", ["noise", "bruit", "threshold", "seuil", "coef_noise"])

        self.combo_loss = QComboBox()
        self.combo_loss.addItems(["linear", "soft_l1", "huber", "cauchy", "arctan"])
        add_row("Loss function:", self.combo_loss, "Robust loss function used for outlier rejection", ["loss", "perte", "robust", "huber", "cauchy"])

        self.spin_f_scale = QDoubleSpinBox()
        self.spin_f_scale.setRange(0.01, 100.0)
        self.spin_f_scale.setSingleStep(0.1)
        self.spin_f_scale.setDecimals(3)
        add_row("Loss scale (f_scale):", self.spin_f_scale, "Soft margin between inlier and outlier residuals", ["f_scale", "scale", "échelle"])

        layout.addWidget(self.grp_solver)

        # ───── Global Peak Limits ─────
        self.grp_bounds = QGroupBox()
        bind_title(self.grp_bounds, "Global Peak Limits")
        bounds_layout = QVBoxLayout(self.grp_bounds)
        bounds_layout.setContentsMargins(8, 8, 8, 8)
        bounds_layout.setSpacing(6)

        def add_bounds_row(lbl_text: str, widget: QWidget, tooltip: str = "", keywords: list[str] | None = None) -> QLabel:
            row = QHBoxLayout()
            lbl = QLabel()
            bind_text(lbl, lbl_text)
            lbl.setMinimumWidth(180)
            lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
            if tooltip:
                bind_tooltip(lbl, tooltip)
                bind_tooltip(widget, tooltip)
            row.addWidget(lbl)
            row.addWidget(widget, 1)
            bounds_layout.addLayout(row)
            terms = [lbl_text.lower(), tr(lbl_text).lower(), tooltip.lower(), tr(tooltip).lower()]
            if keywords:
                terms.extend(keywords)
            self._searchable_rows.append((terms, lbl, self.grp_bounds))
            return lbl

        # Peak profile row
        profile_row = QHBoxLayout()
        profile_row.setSpacing(6)
        lbl_profile = QLabel()
        bind_text(lbl_profile, "Peak profile:")
        lbl_profile.setMinimumWidth(180)
        lbl_profile.setTextInteractionFlags(Qt.TextSelectableByMouse)
        profile_tooltip = "Select a technique preset or user-saved profile for global peak limits"
        bind_tooltip(lbl_profile, profile_tooltip)

        self.cbb_peak_profile = QComboBox()
        bind_tooltip(self.cbb_peak_profile, profile_tooltip)
        self.cbb_peak_profile.currentIndexChanged.connect(self._on_peak_profile_changed)

        self.btn_save_profile = QPushButton()
        self.btn_save_profile.setIcon(Icons.save())
        self.btn_save_profile.setFixedSize(28, 24)
        self.btn_save_profile.setCursor(Qt.PointingHandCursor)
        bind_tooltip(self.btn_save_profile, "Save current peak limits as a profile")
        self.btn_save_profile.clicked.connect(self._on_save_profile_clicked)

        self.btn_delete_profile = QPushButton()
        self.btn_delete_profile.setIcon(Icons.trash())
        self.btn_delete_profile.setFixedSize(28, 24)
        self.btn_delete_profile.setCursor(Qt.PointingHandCursor)
        bind_tooltip(self.btn_delete_profile, "Delete selected custom peak profile")
        self.btn_delete_profile.clicked.connect(self._on_delete_profile_clicked)

        profile_row.addWidget(lbl_profile)
        profile_row.addWidget(self.cbb_peak_profile, 1)
        profile_row.addWidget(self.btn_save_profile)
        profile_row.addWidget(self.btn_delete_profile)
        bounds_layout.addLayout(profile_row)
        self._searchable_rows.append((
            ["peak profile", "profile", "profil", "preset", "raman", "pl", "xrd", tr("Peak profile:").lower()],
            lbl_profile,
            self.grp_bounds,
        ))
        on_language_changed(self, self._retranslate_custom_profile_item)

        self.spin_maxshift = QDoubleSpinBox()
        self.spin_maxshift.setRange(0.0, 10000.0)
        self.spin_maxshift.setSingleStep(1.0)
        self.spin_maxshift.setDecimals(4)
        add_bounds_row("Max peak shift:", self.spin_maxshift, "Maximum allowable center position shift during fit", ["shift", "décalage", "maxshift", "position"])

        self.spin_minfwhm = QDoubleSpinBox()
        self.spin_minfwhm.setRange(0.0001, 10000.0)
        self.spin_minfwhm.setSingleStep(0.1)
        self.spin_minfwhm.setDecimals(4)
        add_bounds_row("Min peak FWHM:", self.spin_minfwhm, "Minimum allowable full-width at half-maximum", ["fwhm", "minfwhm", "width", "largeur"])

        self.spin_maxfwhm = QDoubleSpinBox()
        self.spin_maxfwhm.setRange(0.001, 10000.0)
        self.spin_maxfwhm.setSingleStep(10.0)
        self.spin_maxfwhm.setDecimals(4)
        add_bounds_row("Max peak FWHM:", self.spin_maxfwhm, "Maximum allowable full-width at half-maximum", ["fwhm", "maxfwhm", "width", "largeur"])

        self.spin_maxshift.valueChanged.connect(self._on_bounds_spin_changed)
        self.spin_minfwhm.valueChanged.connect(self._on_bounds_spin_changed)
        self.spin_maxfwhm.valueChanged.connect(self._on_bounds_spin_changed)

        self.set_peak_profiles(self._peak_profiles, [])

        layout.addWidget(self.grp_bounds)

        layout.addSpacerItem(QSpacerItem(20, 10, QSizePolicy.Minimum, QSizePolicy.Expanding))

        self._scroll.setWidget(container)
        outer_layout.addWidget(self._scroll)

    def _retranslate_custom_profile_item(self) -> None:
        """Retranslate the trailing 'Custom' entry in cbb_peak_profile."""
        idx = self.cbb_peak_profile.findData(_CUSTOM_PROFILE_KEY)
        if idx >= 0:
            self.cbb_peak_profile.setItemText(idx, tr("Custom"))

    @staticmethod
    def _values_match_profile(
        maxshift: float,
        minfwhm: float,
        maxfwhm: float,
        profile: dict[str, float],
    ) -> bool:
        return (
            abs(float(maxshift) - float(profile["maxshift"])) <= _PROFILE_MATCH_TOL
            and abs(float(minfwhm) - float(profile["minfwhm"])) <= _PROFILE_MATCH_TOL
            and abs(float(maxfwhm) - float(profile["maxfwhm"])) <= _PROFILE_MATCH_TOL
        )

    def _find_matching_profile(
        self,
        maxshift: float,
        minfwhm: float,
        maxfwhm: float,
        preferred: str = "",
    ) -> str:
        if preferred and preferred in self._peak_profiles:
            if self._values_match_profile(maxshift, minfwhm, maxfwhm, self._peak_profiles[preferred]):
                return preferred
        for name, vals in self._peak_profiles.items():
            if self._values_match_profile(maxshift, minfwhm, maxfwhm, vals):
                return name
        return _CUSTOM_PROFILE_KEY

    def _update_delete_profile_button(self) -> None:
        key = combo_data_or_text(self.cbb_peak_profile)
        self.btn_delete_profile.setEnabled(key in self._user_peak_profiles)

    def set_peak_profiles(
        self,
        profiles: dict[str, dict[str, float]],
        user_profiles: list[str] | set[str] | None = None,
        select_name: Optional[str] = None,
    ) -> None:
        """Populate the peak profile combobox and sync delete button state."""
        self._peak_profiles = {name: dict(vals) for name, vals in profiles.items()}
        self._user_peak_profiles = set(user_profiles or [])

        target_key = select_name if select_name is not None else combo_data_or_text(self.cbb_peak_profile)

        was_blocked = self.cbb_peak_profile.blockSignals(True)
        try:
            self.cbb_peak_profile.clear()
            for name in self._peak_profiles:
                self.cbb_peak_profile.addItem(name, name)
            self.cbb_peak_profile.addItem(tr("Custom"), _CUSTOM_PROFILE_KEY)

            idx = self.cbb_peak_profile.findData(target_key)
            if idx < 0:
                idx = self.cbb_peak_profile.findData(
                    self._find_matching_profile(
                        self.spin_maxshift.value(),
                        self.spin_minfwhm.value(),
                        self.spin_maxfwhm.value(),
                    )
                )
            if idx >= 0:
                self.cbb_peak_profile.setCurrentIndex(idx)
        finally:
            self.cbb_peak_profile.blockSignals(was_blocked)
        self._update_delete_profile_button()

    def _on_peak_profile_changed(self, _index: int) -> None:
        key = combo_data_or_text(self.cbb_peak_profile)
        if key in self._peak_profiles:
            vals = self._peak_profiles[key]
            for spin, val in (
                (self.spin_maxshift, float(vals["maxshift"])),
                (self.spin_minfwhm, float(vals["minfwhm"])),
                (self.spin_maxfwhm, float(vals["maxfwhm"])),
            ):
                spin.blockSignals(True)
                spin.setValue(val)
                spin.blockSignals(False)
        self._update_delete_profile_button()

    def _on_bounds_spin_changed(self, _value: float) -> None:
        matched = self._find_matching_profile(
            self.spin_maxshift.value(),
            self.spin_minfwhm.value(),
            self.spin_maxfwhm.value(),
            preferred=combo_data_or_text(self.cbb_peak_profile),
        )
        idx = self.cbb_peak_profile.findData(matched)
        if idx >= 0 and idx != self.cbb_peak_profile.currentIndex():
            self.cbb_peak_profile.blockSignals(True)
            self.cbb_peak_profile.setCurrentIndex(idx)
            self.cbb_peak_profile.blockSignals(False)
        self._update_delete_profile_button()

    def _on_save_profile_clicked(self) -> None:
        key = combo_data_or_text(self.cbb_peak_profile)
        suggested = key if key in self._user_peak_profiles else ""
        self.save_profile_requested.emit(
            self.spin_maxshift.value(),
            self.spin_minfwhm.value(),
            self.spin_maxfwhm.value(),
            suggested,
        )

    def _on_delete_profile_clicked(self) -> None:
        key = combo_data_or_text(self.cbb_peak_profile)
        if key in self._user_peak_profiles:
            self.delete_profile_requested.emit(key)

    def apply_settings(self, data: dict) -> None:
        if "peak_profiles" in data and isinstance(data["peak_profiles"], dict):
            self.set_peak_profiles(
                data["peak_profiles"],
                data.get("user_peak_profiles", []),
                select_name=data.get("peak_profile") or None,
            )

        self.chk_fit_negative.setChecked(data.get("fit_negative", DEFAULT_FIT_SETTINGS["fit_negative"]))
        self.spin_max_iter.setValue(data.get("max_ite", DEFAULT_FIT_SETTINGS["max_ite"]))
        self.spin_x_tol.setValue(data.get("xtol", DEFAULT_FIT_SETTINGS["xtol"]))
        self.spin_f_tol.setValue(data.get("ftol", DEFAULT_FIT_SETTINGS["ftol"]))
        self.spin_coef_noise.setValue(data.get("coef_noise", DEFAULT_FIT_SETTINGS["coef_noise"]))

        loss = str(data.get("loss", DEFAULT_FIT_SETTINGS["loss"]))
        idx = self.combo_loss.findText(loss)
        if idx >= 0:
            self.combo_loss.setCurrentIndex(idx)
        self.spin_f_scale.setValue(float(data.get("f_scale", DEFAULT_FIT_SETTINGS["f_scale"])))

        maxshift = float(data.get("maxshift", DEFAULT_FIT_SETTINGS["maxshift"]))
        minfwhm = float(data.get("minfwhm", DEFAULT_FIT_SETTINGS["minfwhm"]))
        maxfwhm = float(data.get("maxfwhm", DEFAULT_FIT_SETTINGS["maxfwhm"]))
        for spin, val in (
            (self.spin_maxshift, maxshift),
            (self.spin_minfwhm, minfwhm),
            (self.spin_maxfwhm, maxfwhm),
        ):
            spin.blockSignals(True)
            spin.setValue(val)
            spin.blockSignals(False)

        matched = self._find_matching_profile(
            maxshift,
            minfwhm,
            maxfwhm,
            preferred=str(data.get("peak_profile", "") or ""),
        )
        p_idx = self.cbb_peak_profile.findData(matched)
        if p_idx >= 0:
            self.cbb_peak_profile.blockSignals(True)
            self.cbb_peak_profile.setCurrentIndex(p_idx)
            self.cbb_peak_profile.blockSignals(False)
        self._update_delete_profile_button()

    def reset_to_defaults(self) -> None:
        """Reset all fit settings and peak limits to factory default values."""
        defaults = dict(DEFAULT_FIT_SETTINGS)
        defaults["peak_profile"] = "Raman (cm⁻¹)"
        self.apply_settings(defaults)

    def get_settings(self) -> dict:
        profile_key = combo_data_or_text(self.cbb_peak_profile)
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
            "peak_profile": profile_key if profile_key and profile_key != _CUSTOM_PROFILE_KEY else "",
        }

    def highlight_peak_limits(self) -> None:
        """Highlight the Global Peak Limits groupbox and scroll it into view."""
        self.highlight_search("")
        self.grp_bounds.setStyleSheet(_highlight_group_style())
        self._scroll.ensureWidgetVisible(self.grp_bounds, 0, 30)

    # ───── Search Highlighting & Scrolling ─────
    def highlight_search(self, query: str) -> None:
        """Highlight matching options/groupboxes and scroll first match into view."""
        q = query.strip().lower()

        # Clear existing styles
        self.btn_reset_defaults.setStyleSheet("")
        self.grp_neg.setStyleSheet("")
        self.grp_solver.setStyleSheet("")
        self.grp_bounds.setStyleSheet("")
        for _terms, lbl, _grp in self._searchable_rows:
            lbl.setStyleSheet("")

        if not q:
            return

        first_match: Optional[QWidget] = None

        if any(k in q for k in ("reset", "default", "réinitialiser", "défaut")):
            self.btn_reset_defaults.setStyleSheet("border: 1.5px solid #3B82F6; font-weight: bold;")
            first_match = self.btn_reset_defaults

        if any(k in q for k in ("negative", "négatif", "polarity", "polarité", "absorption", "dip")):
            self.grp_neg.setStyleSheet(_highlight_group_style())
            if first_match is None:
                first_match = self.grp_neg

        for terms, lbl, grp in self._searchable_rows:
            if any(q in t or t in q for t in terms):
                lbl.setStyleSheet(_highlight_label_style())
                if first_match is None:
                    first_match = lbl

        if first_match:
            self._scroll.ensureWidgetVisible(first_match, 0, 30)
