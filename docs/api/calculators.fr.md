# Calculateurs rapides

Le module `spectroview.api.calculators` fournit des fonctions mathématiques pures pour les conversions spectroscopiques courantes et les calculs optiques.

---

## 1. Taille de spot et optique

Calcule la taille de spot théorique limitée par la diffraction, la profondeur de foyer (DOF) et la densité de puissance laser pour un montage optique donné.

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

## 2. Profondeur de pénétration

Calcule la profondeur de pénétration optique théorique d'un laser dans un matériau spécifique à partir de son coefficient d'extinction ($k$).

```python
depth_res = calculators.calc_penetration_depth(
    wavelength_nm=363.0, 
    k=2.842  # Coefficient d'extinction du silicium à 363 nm
)

print(f"Absorption Coefficient (alpha): {depth_res['absorption_coeff_cm1']:.2f} cm-1")
print(f"Penetration Depth (d): {depth_res['penetration_depth_nm']:.2f} nm")
```

---

## 3. Conversion d'unités absolues

Convertit directement entre unités spectroscopiques absolues (longueur d'onde en `nm`, énergie en `eV` et nombre d'onde en `cm-1`).

```python
# Convertir 520 cm-1 vers d'autres unités
conversion = calculators.convert_absolute_units(value=520.0, from_unit="cm-1")

print(f"Wavenumber: {conversion['wavenumber_cm1']:.2f} cm-1")
print(f"Wavelength: {conversion['wavelength_nm']:.2f} nm")
print(f"Energy: {conversion['energy_ev']:.4f} eV")
```

---

## 4. Conversion de déplacement Raman (unités relatives)

Convertit entre longueur d'onde d'excitation (laser), longueur d'onde diffusée et déplacement Raman.

```python
# Avec un laser à 532 nm, calculer la longueur d'onde diffusée pour un déplacement Raman de 520 cm-1
raman = calculators.convert_relative_units(
    laser_wavelength_nm=532.0, 
    shift_cm1=520.0
)

print(f"Laser: 532.0 nm")
print(f"Raman Shift: {raman['shift_cm1']:.2f} cm-1")
print(f"Scattered Wavelength: {raman['scattered_wavelength_nm']:.2f} nm")
```
