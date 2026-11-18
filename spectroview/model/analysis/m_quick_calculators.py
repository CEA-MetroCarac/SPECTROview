import math
from typing import Dict

def calc_spot_size(wavelength_nm: float, na: float, working_distance_mm: float, 
                   refractive_index: float, laser_power_mw: float) -> Dict[str, float]:
    """Calculate laser spot size, depth of focus, and power density."""
    if na <= 0:
        return {
            "spot_diameter_um": 0.0,
            "dof_um": 0.0,
            "angle_deg": 0.0,
            "lens_diameter_mm": 0.0,
            "power_density_mw_um2": 0.0,
            "power_density_kw_cm2": 0.0,
            "power_density_w_m2": 0.0
        }

    # Spot size (µm)
    spot_size = 1.22 * wavelength_nm / na / 1000.0

    # Depth of focus (µm)
    depth = 4 * refractive_index * wavelength_nm / (na**2) / 1000.0

    # Angle of view and Lens diameter
    try:
        angle = 2 * math.degrees(math.asin(na))
        lens_dia = 2 * working_distance_mm * math.tan(math.asin(na))
    except ValueError:
        angle = float('nan')
        lens_dia = float('nan')

    # Power density
    area_um2 = (math.pi / 4) * (spot_size**2)
    pd_mw_um2 = laser_power_mw / area_um2 if area_um2 > 0 else 0.0

    area_cm2 = (math.pi / 4) * ((spot_size * 0.0001)**2)
    pd_kw_cm2 = (laser_power_mw * 0.001) / area_cm2 / 1000.0 if area_cm2 > 0 else 0.0

    area_m2 = (math.pi / 4) * ((spot_size * 0.000001)**2)
    pd_w_m2 = (laser_power_mw * 0.001) / area_m2 if area_m2 > 0 else 0.0

    return {
        "spot_diameter_um": spot_size,
        "dof_um": depth,
        "angle_deg": angle,
        "lens_diameter_mm": lens_dia,
        "power_density_mw_um2": pd_mw_um2,
        "power_density_kw_cm2": pd_kw_cm2,
        "power_density_w_m2": pd_w_m2
    }

def calc_penetration_depth(wavelength_nm: float, k: float) -> Dict[str, float]:
    """Calculate absorption coefficient and penetration depth."""
    if k > 0:
        alpha = (4 * math.pi * k) / (wavelength_nm * 1e-7)
        d_nm = wavelength_nm / (4 * math.pi * k)
    else:
        alpha = 0.0
        d_nm = float('inf')

    return {
        "absorption_coeff_cm1": alpha,
        "penetration_depth_nm": d_nm
    }

def convert_absolute_units(value: float, from_unit: str) -> Dict[str, float]:
    """Convert absolute spectroscopic units (nm, eV, cm-1).
    from_unit must be 'nm', 'eV', or 'cm-1'
    """
    if value <= 0:
        return {"wavelength_nm": 0.0, "energy_ev": 0.0, "wavenumber_cm1": 0.0}

    if from_unit == "nm":
        wave = value
    elif from_unit == "eV":
        wave = 1239.84193 / value
    elif from_unit == "cm-1":
        wave = 1e7 / value
    else:
        raise ValueError(f"Unknown unit: {from_unit}")

    return {
        "wavelength_nm": wave,
        "energy_ev": 1239.84193 / wave,
        "wavenumber_cm1": 1e7 / wave
    }

def convert_relative_units(laser_wavelength_nm: float, shift_cm1: float = None, 
                           scattered_wavelength_nm: float = None) -> Dict[str, float]:
    """Convert relative spectroscopic units (Raman shift).
    Provide laser_wavelength_nm and exactly one of (shift_cm1, scattered_wavelength_nm)
    """
    if laser_wavelength_nm <= 0:
        return {"shift_cm1": 0.0, "scattered_wavelength_nm": 0.0}
        
    if shift_cm1 is not None and scattered_wavelength_nm is None:
        inv_scat = (1e7 / laser_wavelength_nm) - shift_cm1
        scat = 1e7 / inv_scat if inv_scat > 0 else 0.0
        return {"shift_cm1": shift_cm1, "scattered_wavelength_nm": scat}
        
    elif scattered_wavelength_nm is not None and shift_cm1 is None:
        if scattered_wavelength_nm <= 0:
            return {"shift_cm1": 0.0, "scattered_wavelength_nm": 0.0}
        shift = (1e7 / laser_wavelength_nm) - (1e7 / scattered_wavelength_nm)
        return {"shift_cm1": shift, "scattered_wavelength_nm": scattered_wavelength_nm}
        
    else:
        raise ValueError("Provide exactly one of shift_cm1 or scattered_wavelength_nm")
