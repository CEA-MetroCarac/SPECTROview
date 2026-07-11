# Quick Calculators

The `spectroview.api.calculators` module provides pure mathematical functions for common spectroscopic conversions and optics calculations.

---

## 1. Spot Size & Optics

Calculate the theoretical diffraction-limited spot size, depth of focus (DOF), and laser power density for a given optical setup.

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

## 2. Penetration Depth

Calculate the theoretical optical penetration depth of a laser into a specific material using its extinction coefficient ($k$).

```python
depth_res = calculators.calc_penetration_depth(
    wavelength_nm=363.0, 
    k=2.842  # Extinction coefficient for Silicon at 363nm
)

print(f"Absorption Coefficient (alpha): {depth_res['absorption_coeff_cm1']:.2f} cm-1")
print(f"Penetration Depth (d): {depth_res['penetration_depth_nm']:.2f} nm")
```

---

## 3. Absolute Unit Conversion

Convert seamlessly between absolute spectroscopic units (Wavelength in `nm`, Energy in `eV`, and Wavenumber in `cm-1`).

```python
# Convert 520 cm-1 to other units
conversion = calculators.convert_absolute_units(value=520.0, from_unit="cm-1")

print(f"Wavenumber: {conversion['wavenumber_cm1']:.2f} cm-1")
print(f"Wavelength: {conversion['wavelength_nm']:.2f} nm")
print(f"Energy: {conversion['energy_ev']:.4f} eV")
```

---

## 4. Raman Shift (Relative) Conversion

Convert between excitation (laser) wavelength, scattered emission wavelength, and Raman shift.

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
