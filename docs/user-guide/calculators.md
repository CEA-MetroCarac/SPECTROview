# Quick Calculators

Accessible from the **Quick Calc** button in the toolbar or **Tools → Quick Calculators**.

## Laser Spot Size Calculator

Calculates diffraction-limited spot size, depth of focus, and power density.

### Inputs

| Parameter | Default | Unit |
|-----------|---------|------|
| Laser Wavelength (λ) | 532.0 | nm |
| Objective NA | 0.90 | — |
| Working Distance (WD) | 1.00 | mm |
| Refractive Index (n) | 1.00 | — |
| Laser Power | 1.00 | mW |

### Outputs

| Output | Formula |
|--------|---------|
| Spot Size | `1.22 × λ / NA` (μm) |
| Depth of Focus | `4 × n × λ / NA²` (μm) |
| Angle of View | `2 × arcsin(NA)` (°) |
| Lens Diameter | `2 × WD × tan(arcsin(NA))` (mm) |
| Power Density | `Power / Area` (kW/cm², mW/μm², W/m²) |

## Penetration Depth Calculator

Calculates optical penetration depth from the extinction coefficient.

### Inputs

| Parameter | Default | Unit |
|-----------|---------|------|
| Laser Wavelength (λ) | 363.0 | nm |
| Extinction Coefficient (k) | 2.842 | — |

### Outputs

| Output | Formula |
|--------|---------|
| Absorption Coefficient α | `4πk / (λ × 10⁻⁷)` (cm⁻¹) |
| Penetration Depth d | `λ / (4πk)` (nm) |

!!! tip "Finding k values"
    Look up extinction coefficients at [refractiveindex.info](https://refractiveindex.info/?shelf=main&book=Si&page=Franta-300K).

## Unit Converter

### Absolute Conversion

| From | To | Formula |
|------|-----|---------|
| Wavelength (nm) | Energy (eV) | `E = 1239.84193 / λ` |
| Wavelength (nm) | Wavenumber (cm⁻¹) | `ν = 10⁷ / λ` |

All three fields are bidirectionally linked — change any one to update the others.

### Relative Conversion (Raman Shift)

| From | To | Formula |
|------|-----|---------|
| Laser λ₀ + Raman Shift Δω | Scattered λ | `λ = 10⁷ / (10⁷/λ₀ - Δω)` |

Enter the laser wavelength and either the Raman shift or the scattered wavelength to compute the other.
