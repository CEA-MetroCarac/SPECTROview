# Калькуляторы { #quick-calculators }

Модуль `spectroview.api.calculators` предоставляет чистые математические функции для типовых спектроскопических преобразований и оптических расчётов.

---

## 1. Размер пятна и оптика { #1-spot-size-optics }

Расчёт теоретического дифракционно-ограниченного размера пятна, глубины резкости (DOF) и плотности мощности лазера для заданной оптической системы.

```python
from spectroview.api import calculators

optics = calculators.calc_spot_size(
    wavelength_nm=532.0, 
    na=0.9, 
    working_distance_mm=1.0, 
    refractive_index=1.0, 
    laser_power_mw=5.0
)

print(f"Spot Diameter: {optics['spot_diameter_um']:.2f} µm")
print(f"Depth of Focus: {optics['dof_um']:.2f} µm")
print(f"Power Density: {optics['power_density_mw_um2']:.2f} mW/µm²")
```

---

## 2. Глубина проникновения { #2-penetration-depth }

Расчёт теоретической оптической глубины проникновения лазера в конкретный материал по его коэффициенту экстинкции ($k$).

```python
depth_res = calculators.calc_penetration_depth(
    wavelength_nm=363.0, 
    k=2.842  # Extinction coefficient for Silicon at 363nm
)

print(f"Absorption Coefficient (alpha): {depth_res['absorption_coeff_cm1']:.2f} cm-1")
print(f"Penetration Depth (d): {depth_res['penetration_depth_nm']:.2f} nm")
```

---

## 3. Преобразование абсолютных единиц { #3-absolute-unit-conversion }

Прямое преобразование между абсолютными спектроскопическими единицами (длиной волны в `nm`, энергией в `eV` и волновым числом в `cm-1`).

```python
# Convert 520 cm-1 to other units
conversion = calculators.convert_absolute_units(value=520.0, from_unit="cm-1")

print(f"Wavenumber: {conversion['wavenumber_cm1']:.2f} cm-1")
print(f"Wavelength: {conversion['wavelength_nm']:.2f} nm")
print(f"Energy: {conversion['energy_ev']:.4f} eV")
```

---

## 4. Преобразование рамановского сдвига (относительные единицы) { #4-raman-shift-relative-conversion }

Преобразование между длиной волны возбуждения (лазера), длиной волны рассеянного излучения и рамановским сдвигом.

```python
# Given a 532 nm laser, calculate the scattered wavelength for a 520 cm-1 Raman shift
raman = calculators.convert_relative_units(
    laser_wavelength_nm=532.0, 
    shift_cm1=520.0
)

print(f"Laser: 532.0 nm")
print(f"Raman Shift: {raman['shift_cm1']:.2f} cm-1")
print(f"Scattered Wavelength: {raman['scattered_wavelength_nm']:.2f} nm")
```
