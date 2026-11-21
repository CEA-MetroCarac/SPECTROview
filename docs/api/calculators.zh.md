# 快捷计算器 { #quick-calculators }

`spectroview.api.calculators` 模块提供用于常见光谱换算和光学计算的纯数学函数。

---

## 1. 光斑尺寸与光学参数 { #1-spot-size-optics }

计算给定光学配置下的理论衍射极限光斑尺寸、景深 (DOF) 和激光功率密度。

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

## 2. 穿透深度 { #2-penetration-depth }

根据特定材料的消光系数 ($k$)，计算激光在该材料中的理论光学穿透深度。

```python
depth_res = calculators.calc_penetration_depth(
    wavelength_nm=363.0, 
    k=2.842  # Extinction coefficient for Silicon at 363nm
)

print(f"Absorption Coefficient (alpha): {depth_res['absorption_coeff_cm1']:.2f} cm-1")
print(f"Penetration Depth (d): {depth_res['penetration_depth_nm']:.2f} nm")
```

---

## 3. 绝对单位换算 { #3-absolute-unit-conversion }

在绝对光谱单位之间无缝换算（波长单位为 `nm`、能量单位为 `eV`、波数单位为 `cm-1`）。

```python
# Convert 520 cm-1 to other units
conversion = calculators.convert_absolute_units(value=520.0, from_unit="cm-1")

print(f"Wavenumber: {conversion['wavenumber_cm1']:.2f} cm-1")
print(f"Wavelength: {conversion['wavelength_nm']:.2f} nm")
print(f"Energy: {conversion['energy_ev']:.4f} eV")
```

---

## 4. 拉曼位移（相对单位）换算 { #4-raman-shift-relative-conversion }

在激发（激光）波长、散射发射波长与拉曼位移之间进行换算。

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
