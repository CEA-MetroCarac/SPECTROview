"""Tests for spectroview.api.calculators. This module is the only public
surface over spectroview.model.m_quick_calculators, so these tests carry
the real math assertions (no separate model-layer test file exists)."""
import math

from spectroview.api import calculators


class TestCalcSpotSize:
    def test_known_values(self):
        result = calculators.calc_spot_size(
            wavelength_nm=532.0, na=0.9, working_distance_mm=1.0,
            refractive_index=1.0, laser_power_mw=10.0,
        )
        expected_spot = 1.22 * 532.0 / 0.9 / 1000.0
        assert math.isclose(result["spot_diameter_um"], expected_spot, rel_tol=1e-9)

    def test_zero_na_returns_zeros(self):
        result = calculators.calc_spot_size(500.0, 0.0, 1.0, 1.0, 10.0)
        assert result["spot_diameter_um"] == 0.0
        assert result["power_density_mw_um2"] == 0.0


class TestCalcPenetrationDepth:
    def test_known_values(self):
        result = calculators.calc_penetration_depth(wavelength_nm=500.0, k=0.1)
        expected_alpha = (4 * math.pi * 0.1) / (500.0 * 1e-7)
        assert math.isclose(result["absorption_coeff_cm1"], expected_alpha, rel_tol=1e-9)

    def test_zero_k_gives_infinite_depth(self):
        result = calculators.calc_penetration_depth(500.0, 0.0)
        assert result["penetration_depth_nm"] == float("inf")


class TestConvertAbsoluteUnits:
    def test_round_trip_nm_to_wavenumber(self):
        result = calculators.convert_absolute_units(500.0, "nm")
        assert math.isclose(result["wavenumber_cm1"], 1e7 / 500.0, rel_tol=1e-6)

    def test_unknown_unit_raises(self):
        import pytest
        with pytest.raises(ValueError):
            calculators.convert_absolute_units(500.0, "furlongs")


class TestConvertRelativeUnits:
    def test_shift_to_scattered_wavelength(self):
        result = calculators.convert_relative_units(laser_wavelength_nm=532.0, shift_cm1=520.0)
        assert result["shift_cm1"] == 520.0
        assert result["scattered_wavelength_nm"] > 532.0

    def test_requires_exactly_one_of_shift_or_wavelength(self):
        import pytest
        with pytest.raises(ValueError):
            calculators.convert_relative_units(532.0, shift_cm1=100.0, scattered_wavelength_nm=550.0)
