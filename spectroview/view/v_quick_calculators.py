import math
import os
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QPainter
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QDoubleSpinBox, 
    QGroupBox, QLineEdit, QLabel, QFrame, QSizePolicy
)
from spectroview import ICON_DIR
from spectroview.model.m_quick_calculators import calc_spot_size, calc_penetration_depth, convert_absolute_units, convert_relative_units


class MQuickCalc(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Quick Calculators")
        self.resize(850, 650)
        
        dialog_layout = QVBoxLayout(self)
        
        # Instantiate the separated calculators
        self.spot_size_calc = SpotSizeCalculator(self)
        self.penetration_depth_calc = PenetrationDepthCalculator(self)
        self.converter_calc = UnitConverterCalculator(self)
        
        bottom_layout = QHBoxLayout()
        bottom_layout.addWidget(self.penetration_depth_calc)
        bottom_layout.addWidget(self.converter_calc)
        
        dialog_layout.addWidget(self.spot_size_calc)
        dialog_layout.addLayout(bottom_layout)


class SpotSizeCalculator(QGroupBox):
    def __init__(self, parent=None):
        super().__init__("1. Calculation of laser spot size, DOF and power density", parent)
        self._create_ui()
        self._connect_signals()
        self._update_calculations()

    def _create_ui(self):
        frame_layout = QVBoxLayout(self)
        
        main_layout = QHBoxLayout()
        left_layout = QVBoxLayout()

        # Inputs
        group_inputs = QGroupBox("Inputs")
        group_inputs.setFixedWidth(260)
        form_inputs = QFormLayout(group_inputs)

        self.spin_wave = QDoubleSpinBox()
        self.spin_wave.setRange(0.1, 10000)
        self.spin_wave.setDecimals(1)
        self.spin_wave.setValue(532.0)
        self.spin_wave.setSuffix(" nm")

        self.spin_na = QDoubleSpinBox()
        self.spin_na.setRange(0.01, 2.0)
        self.spin_na.setDecimals(2)
        self.spin_na.setSingleStep(0.05)
        self.spin_na.setValue(0.9)

        self.spin_wd = QDoubleSpinBox()
        self.spin_wd.setRange(0.01, 1000)
        self.spin_wd.setDecimals(2)
        self.spin_wd.setValue(1.0)
        self.spin_wd.setSuffix(" mm")

        self.spin_index = QDoubleSpinBox()
        self.spin_index.setRange(1.0, 5.0)
        self.spin_index.setDecimals(2)
        self.spin_index.setSingleStep(0.1)
        self.spin_index.setValue(1.0)

        self.spin_power = QDoubleSpinBox()
        self.spin_power.setRange(0.01, 10000)
        self.spin_power.setDecimals(2)
        self.spin_power.setValue(1.00)
        self.spin_power.setSuffix(" mW")

        self.spin_wave.setToolTip("Laser Wavelength (λ)")
        self.spin_na.setToolTip("Objective Numerical Aperture (NA) = n × sin(θ)")
        self.spin_wd.setToolTip("Working Distance (WD)")
        self.spin_index.setToolTip("Refractive Index (n)")
        self.spin_power.setToolTip("Laser Power")

        form_inputs.addRow("Laser Wavelength:", self.spin_wave)
        form_inputs.addRow("Objective NA:", self.spin_na)
        form_inputs.addRow("Working Distance (WD):", self.spin_wd)
        form_inputs.addRow("Refractive Index:", self.spin_index)
        form_inputs.addRow("Laser Power:", self.spin_power)

        left_layout.addWidget(group_inputs)

        # Outputs
        group_outputs = QGroupBox("Outputs")
        group_outputs.setFixedWidth(260)
        form_outputs = QFormLayout(group_outputs)

        self.out_spot = QLineEdit()
        self.out_spot.setReadOnly(True)
        
        self.out_depth = QLineEdit()
        self.out_depth.setReadOnly(True)

        self.out_angle = QLineEdit()
        self.out_angle.setReadOnly(True)

        self.out_lens_dia = QLineEdit()
        self.out_lens_dia.setReadOnly(True)

        self.out_pd_kw_cm2 = QLineEdit()
        self.out_pd_kw_cm2.setReadOnly(True)

        self.out_pd_mw_um2 = QLineEdit()
        self.out_pd_mw_um2.setReadOnly(True)

        self.out_pd_w_m2 = QLineEdit()
        self.out_pd_w_m2.setReadOnly(True)

        self.out_spot.setToolTip("Diffraction-Limited Spot Size = 1.22 × λ / NA")
        self.out_depth.setToolTip("Depth of Focus (DOF) = 4 × n × λ / NA²")
        self.out_angle.setToolTip("Angle of View (2θ) = 2 × arcsin(NA)")
        self.out_lens_dia.setToolTip("Lens Diameter = 2 × WD × tan(arcsin(NA))")
        self.out_pd_kw_cm2.setToolTip("Power Density = Power / Area")
        self.out_pd_mw_um2.setToolTip("Power Density = Power / Area")
        self.out_pd_w_m2.setToolTip("Power Density = Power / Area")

        form_outputs.addRow("Focused Spot Size (μm):", self.out_spot)
        form_outputs.addRow("Depth of Focus (μm):", self.out_depth)
        form_outputs.addRow("Angle of View (°):", self.out_angle)
        form_outputs.addRow("Lens Diameter (mm):", self.out_lens_dia)
        form_outputs.addRow("Power Density (kW/cm²):", self.out_pd_kw_cm2)
        form_outputs.addRow("Power Density (mW/μm²):", self.out_pd_mw_um2)
        form_outputs.addRow("Power Density (W/m²):", self.out_pd_w_m2)

        for widget in [self.spin_wave, self.spin_na, self.spin_wd, self.spin_index, self.spin_power,
                       self.out_spot, self.out_depth, self.out_angle, self.out_lens_dia, 
                       self.out_pd_kw_cm2, self.out_pd_mw_um2, self.out_pd_w_m2]:
            widget.setFixedWidth(100)

        left_layout.addWidget(group_outputs)
        main_layout.addLayout(left_layout)

        # Image
        self.lbl_image = ResizableImageLabel(os.path.join(ICON_DIR, "dof_scheme.png"))
        main_layout.addWidget(self.lbl_image)

        frame_layout.addLayout(main_layout)

    def _connect_signals(self):
        self.spin_wave.valueChanged.connect(self._update_calculations)
        self.spin_na.valueChanged.connect(self._update_calculations)
        self.spin_wd.valueChanged.connect(self._update_calculations)
        self.spin_index.valueChanged.connect(self._update_calculations)
        self.spin_power.valueChanged.connect(self._update_calculations)

    def _update_calculations(self):
        wave = self.spin_wave.value()
        na = self.spin_na.value()
        wd = self.spin_wd.value()
        index = self.spin_index.value()
        power = self.spin_power.value()

        if na <= 0:
            return

        res = calc_spot_size(wave, na, wd, index, power)

        self.out_spot.setText(f"{res['spot_diameter_um']:.4f}")
        self.out_depth.setText(f"{res['dof_um']:.4f}")

        if math.isnan(res['angle_deg']):
            self.out_angle.setText("N/A")
            self.out_lens_dia.setText("N/A")
        else:
            self.out_angle.setText(f"{res['angle_deg']:.2f}")
            self.out_lens_dia.setText(f"{res['lens_diameter_mm']:.4f}")

        self.out_pd_mw_um2.setText(f"{res['power_density_mw_um2']:.4f}")
        self.out_pd_kw_cm2.setText(f"{res['power_density_kw_cm2']:.4f}")
        self.out_pd_w_m2.setText(f"{res['power_density_w_m2']:.2f}")

class PenetrationDepthCalculator(QGroupBox):
    def __init__(self, parent=None):
        super().__init__("2. Calculation of penetration depth", parent)
        self._create_ui()
        self._connect_signals()
        self._update_calculations()

    def _create_ui(self):
        depth_layout = QVBoxLayout(self)
        
        depth_main_layout = QHBoxLayout()
        depth_form = QFormLayout()
        
        self.spin_wave = QDoubleSpinBox()
        self.spin_wave.setRange(0.1, 10000)
        self.spin_wave.setDecimals(1)
        self.spin_wave.setValue(363.0)
        self.spin_wave.setSuffix(" nm")
        self.spin_wave.setToolTip("Laser Wavelength (λ)")

        self.spin_k = QDoubleSpinBox()
        self.spin_k.setRange(0.0, 100.0)
        self.spin_k.setDecimals(5)
        self.spin_k.setSingleStep(0.01)
        self.spin_k.setValue(2.842)
        self.spin_k.setToolTip("Extinction Coefficient (k)")
        
        depth_form.addRow("Laser Wavelength (λ):", self.spin_wave)
        depth_form.addRow("Extinction Coefficient (k):", self.spin_k)
        
        self.out_alpha = QLineEdit()
        self.out_alpha.setReadOnly(True)
        self.out_alpha.setToolTip("Absorption Coefficient α = 4πk / (λ × 10⁻⁷)")
        
        self.out_penetration_depth = QLineEdit()
        self.out_penetration_depth.setReadOnly(True)
        self.out_penetration_depth.setToolTip("Penetration Depth d = λ / (4πk)")
        
        depth_form.addRow("Absorption Coeff α (cm⁻¹):", self.out_alpha)
        depth_form.addRow("Penetration Depth d (nm):", self.out_penetration_depth)
        
        for widget in [self.spin_wave, self.spin_k, self.out_alpha, self.out_penetration_depth]:
            widget.setFixedWidth(100)
            
        lbl_link = QLabel('<a href="https://refractiveindex.info/?shelf=main&book=Si&page=Franta-300K">Find n and k on refractiveindex.info</a>')
        lbl_link.setOpenExternalLinks(True)
        depth_form.addRow(lbl_link)
        
        depth_main_layout.addLayout(depth_form)
        
        self.lbl_eq_image = ResizableImageLabel(os.path.join(ICON_DIR, "penetration_eq.png"))
        depth_main_layout.addWidget(self.lbl_eq_image)
        
        depth_layout.addLayout(depth_main_layout)

    def _connect_signals(self):
        self.spin_wave.valueChanged.connect(self._update_calculations)
        self.spin_k.valueChanged.connect(self._update_calculations)

    def _update_calculations(self):
        wave = self.spin_wave.value()
        k_val = self.spin_k.value()
        
        res = calc_penetration_depth(wave, k_val)
        
        if k_val > 0:
            self.out_alpha.setText(f"{res['absorption_coeff_cm1']:.2f}")
            self.out_penetration_depth.setText(f"{res['penetration_depth_nm']:.2f}")
        else:
            self.out_alpha.setText("0.00")
            self.out_penetration_depth.setText("Infinite")

class UnitConverterCalculator(QGroupBox):
    def __init__(self, parent=None):
        super().__init__("3. Unit Converter", parent)
        self._create_ui()
        self._connect_signals()

    def _create_ui(self):
        main_vbox = QVBoxLayout(self)
        
        # Absolute Conversion
        form_abs = QFormLayout()
        
        self.spin_wave = QDoubleSpinBox()
        self.spin_wave.setRange(0.1, 100000)
        self.spin_wave.setDecimals(2)
        self.spin_wave.setValue(532.0)
        self.spin_wave.setSuffix(" nm")
        self.spin_wave.setToolTip("Wavelength (nm) = 1239.84193 / E(eV) = 10⁷ / Wavenumber(cm⁻¹)")
        
        self.spin_energy = QDoubleSpinBox()
        self.spin_energy.setRange(0.0001, 100000)
        self.spin_energy.setDecimals(4)
        self.spin_energy.setValue(1239.84193 / 532.0)
        self.spin_energy.setSuffix(" eV")
        self.spin_energy.setToolTip("Energy (eV) = 1239.84193 / λ(nm)")
        
        self.spin_wavenumber = QDoubleSpinBox()
        self.spin_wavenumber.setRange(0.1, 1000000)
        self.spin_wavenumber.setDecimals(2)
        self.spin_wavenumber.setValue(1e7 / 532.0)
        self.spin_wavenumber.setSuffix(" cm⁻¹")
        self.spin_wavenumber.setToolTip("Wavenumber (cm⁻¹) = 10⁷ / λ(nm)")
        
        form_abs.addRow("Absolute Wavelength:", self.spin_wave)
        form_abs.addRow("Energy:", self.spin_energy)
        form_abs.addRow("Absolute Wavenumber:", self.spin_wavenumber)
        
        # Relative Conversion (Raman Shift)
        form_rel = QFormLayout()
        
        self.spin_laser_wave = QDoubleSpinBox()
        self.spin_laser_wave.setRange(0.1, 100000)
        self.spin_laser_wave.setDecimals(2)
        self.spin_laser_wave.setValue(532.0)
        self.spin_laser_wave.setSuffix(" nm")
        self.spin_laser_wave.setToolTip("Laser Wavelength λ₀ (nm)")
        
        self.spin_shift = QDoubleSpinBox()
        self.spin_shift.setRange(-10000, 10000)
        self.spin_shift.setDecimals(2)
        self.spin_shift.setValue(0.0)
        self.spin_shift.setSuffix(" cm⁻¹")
        self.spin_shift.setToolTip("Raman Shift Δw (cm⁻¹) = 10⁷ / λ₀(nm) - 10⁷ / λ(nm)")
        
        self.spin_scattered_wave = QDoubleSpinBox()
        self.spin_scattered_wave.setRange(0.1, 100000)
        self.spin_scattered_wave.setDecimals(2)
        self.spin_scattered_wave.setValue(532.0)
        self.spin_scattered_wave.setSuffix(" nm")
        self.spin_scattered_wave.setToolTip("Scattered Wavelength λ (nm) = 10⁷ / (10⁷ / λ₀(nm) - Δw(cm⁻¹))")
        
        form_rel.addRow("Laser Wavelength (λ₀):", self.spin_laser_wave)
        form_rel.addRow("Raman Shift (Δw):", self.spin_shift)
        form_rel.addRow("Scattered Wavelength (λ):", self.spin_scattered_wave)
        
        for w in [self.spin_wave, self.spin_energy, self.spin_wavenumber, 
                  self.spin_laser_wave, self.spin_shift, self.spin_scattered_wave]:
            w.setFixedWidth(100)
            
        main_vbox.addLayout(form_abs)
        main_vbox.addSpacing(10)
        main_vbox.addLayout(form_rel)
        
        main_vbox.addStretch()

    def _connect_signals(self):
        self.spin_wave.valueChanged.connect(self._on_wave_changed)
        self.spin_energy.valueChanged.connect(self._on_energy_changed)
        self.spin_wavenumber.valueChanged.connect(self._on_wavenumber_changed)
        
        self.spin_laser_wave.valueChanged.connect(self._on_relative_changed)
        self.spin_shift.valueChanged.connect(self._on_shift_changed)
        self.spin_scattered_wave.valueChanged.connect(self._on_scattered_changed)

    def _on_wave_changed(self, val):
        if val <= 0: return
        self.spin_energy.blockSignals(True)
        self.spin_wavenumber.blockSignals(True)
        res = convert_absolute_units(val, 'nm')
        self.spin_energy.setValue(res['energy_ev'])
        self.spin_wavenumber.setValue(res['wavenumber_cm1'])
        self.spin_energy.blockSignals(False)
        self.spin_wavenumber.blockSignals(False)
        
    def _on_energy_changed(self, val):
        if val <= 0: return
        self.spin_wave.blockSignals(True)
        self.spin_wavenumber.blockSignals(True)
        res = convert_absolute_units(val, 'eV')
        self.spin_wave.setValue(res['wavelength_nm'])
        self.spin_wavenumber.setValue(res['wavenumber_cm1'])
        self.spin_wave.blockSignals(False)
        self.spin_wavenumber.blockSignals(False)
        
    def _on_wavenumber_changed(self, val):
        if val <= 0: return
        self.spin_wave.blockSignals(True)
        self.spin_energy.blockSignals(True)
        res = convert_absolute_units(val, 'cm-1')
        self.spin_wave.setValue(res['wavelength_nm'])
        self.spin_energy.setValue(res['energy_ev'])
        self.spin_wave.blockSignals(False)
        self.spin_energy.blockSignals(False)
        
    def _on_relative_changed(self):
        self._on_shift_changed(self.spin_shift.value())
        
    def _on_shift_changed(self, shift):
        laser = self.spin_laser_wave.value()
        if laser <= 0: return
        
        self.spin_scattered_wave.blockSignals(True)
        res = convert_relative_units(laser_wavelength_nm=laser, shift_cm1=shift)
        self.spin_scattered_wave.setValue(res['scattered_wavelength_nm'])
        self.spin_scattered_wave.blockSignals(False)
        
    def _on_scattered_changed(self, scattered):
        laser = self.spin_laser_wave.value()
        if laser <= 0 or scattered <= 0: return
        
        self.spin_shift.blockSignals(True)
        res = convert_relative_units(laser_wavelength_nm=laser, scattered_wavelength_nm=scattered)
        self.spin_shift.setValue(res['shift_cm1'])
        self.spin_shift.blockSignals(False)



class ResizableImageLabel(QLabel):
    def __init__(self, img_path):
        super().__init__()
        self._pixmap = QPixmap(img_path)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(100, 100)

    def paintEvent(self, event):
        if not self._pixmap.isNull():
            painter = QPainter(self)
            rect = self.contentsRect()
            scaled_pix = self._pixmap.scaled(rect.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            x = rect.x() + (rect.width() - scaled_pix.width()) // 2
            y = rect.y() + (rect.height() - scaled_pix.height()) // 2
            painter.drawPixmap(x, y, scaled_pix)
        else:
            super().paintEvent(event)