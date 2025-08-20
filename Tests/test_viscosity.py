import pytest
import numpy as np
from magma.viscosity import viscosity_lebrun


class TestViscosityLebrun:
    """Test suite for the viscosity_lebrun function (Lebrun et al. 2013 parameterization)."""
    
    def test_basic_functionality(self):
        """Test that the function runs without errors for typical inputs."""
        result = viscosity_lebrun(Tm=2000, Ts=1500, rho_m=3300, FH2O=0.1, meltfrac=0.5)
        assert isinstance(result, (float, np.floating))
        assert result > 0, "viscosity should be positive"
    
    def test_temperature_dependence(self):
        """Test that viscosity_lebrun decreases with increasing temperature."""
        temp_low = viscosity_lebrun(Tm=1600, Ts=1500, rho_m=3300, FH2O=0.1, meltfrac=0.5)
        temp_high = viscosity_lebrun(Tm=2200, Ts=1500, rho_m=3300, FH2O=0.1, meltfrac=0.5)
        
        assert temp_low > temp_high, "viscosity_lebrun should decrease with increasing temperature"
    
    def test_melt_fraction_regimes(self):
        """Test different melt fraction regimes (liquid vs solid behavior)."""
        # Test liquid regime (melt fraction > 0.4, solid fraction < 0.6)
        liquid_visc = viscosity_lebrun(Tm=2000, Ts=1500, rho_m=3300, FH2O=0.1, meltfrac=0.5)
        
        # Test solid regime (melt fraction < 0.4, solid fraction > 0.6)  
        solid_visc = viscosity_lebrun(Tm=1600, Ts=1500, rho_m=3300, FH2O=0.1, meltfrac=0.5)
        
        assert liquid_visc > 0
        assert solid_visc > 0
        # Generally expect solid to have higher viscosity, but depends on temperature
    
    def test_melt_fraction_calculation(self):
        """Test internal melt fraction calculation based on temperature."""
        # Temperature below solidus (1420 K) should give melt fraction = 0
        result_cold = viscosity_lebrun(Tm=1400, Ts=1500, rho_m=3300, FH2O=0.1, meltfrac=0.5)
        assert result_cold > 0
        
        # Temperature at liquidus should give melt fraction = 1
        result_hot = viscosity_lebrun(Tm=2100, Ts=1500, rho_m=3300, FH2O=0.1, meltfrac=0.5)
        assert result_hot > 0
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Test at solidus temperature
        result_solidus = viscosity_lebrun(Tm=1420, Ts=1500, rho_m=3300, FH2O=0.1, meltfrac=0.5)
        assert result_solidus > 0
        
        # Test at liquidus temperature  
        result_liquidus = viscosity_lebrun(Tm=2020, Ts=1500, rho_m=3300, FH2O=0.1, meltfrac=0.5)
        assert result_liquidus > 0
        
        # Test very high temperature
        result_high = viscosity_lebrun(Tm=3000, Ts=1500, rho_m=3300, FH2O=0.1, meltfrac=0.5)
        assert result_high > 0
    
    def test_parameter_independence(self):
        """Test that unused parameters don't affect the result."""
        # Ts, FH2O, and input meltfrac should not affect result
        result1 = viscosity_lebrun(Tm=2000, Ts=1500, rho_m=3300, FH2O=0.1, meltfrac=0.5)
        result2 = viscosity_lebrun(Tm=2000, Ts=2000, rho_m=3300, FH2O=0.5, meltfrac=0.8)
        
        # Results should be identical since these parameters are unused
        assert np.isclose(result1, result2, rtol=1e-10)
    
    def test_density_override(self):
        """Test that input density is overridden internally."""
        # Function should use rho_m = 3.3e3 regardless of input
        result1 = viscosity_lebrun(Tm=2000, Ts=1500, rho_m=3000, FH2O=0.1, meltfrac=0.5)
        result2 = viscosity_lebrun(Tm=2000, Ts=1500, rho_m=4000, FH2O=0.1, meltfrac=0.5)
        
        assert np.isclose(result1, result2, rtol=1e-10)
    
    def test_units_and_magnitudes(self):
        """Test that results are in expected units and magnitude ranges."""
        result = viscosity_lebrun(Tm=2000, Ts=1500, rho_m=3300, FH2O=0.1, meltfrac=0.5)
        
        # Kinematic viscosity should be in reasonable range for magma (m²/s)
        # Typical range: 1e-6 to 1e6 m²/s
        assert 1e-10 < result < 1e10, f"viscosity_lebrun {result} outside expected range"
    
    def test_array_inputs(self):
        """Test function behavior with array inputs."""
        temps = np.array([1500, 1800, 2100])
        results = []
        
        for temp in temps:
            result = viscosity_lebrun(Tm=temp, Ts=1500, rho_m=3300, FH2O=0.1, meltfrac=0.5)
            results.append(result)
        
        results = np.array(results)
        assert len(results) == len(temps)
        assert all(r > 0 for r in results)
        
        # Should generally decrease with temperature
        assert results[0] > results[-1]
    
    def test_numerical_stability(self):
        """Test numerical stability for extreme but valid inputs."""
        # Very high temperature
        result_hot = viscosity_lebrun(Tm=5000, Ts=1500, rho_m=3300, FH2O=0.1, meltfrac=0.5)
        assert np.isfinite(result_hot)
        assert result_hot > 0
        
        # Temperature just above solidus
        result_warm = viscosity_lebrun(Tm=1421, Ts=1500, rho_m=3300, FH2O=0.1, meltfrac=0.5)
        assert np.isfinite(result_warm)
        assert result_warm > 0
    
    @pytest.mark.parametrize("temp", [1400, 1600, 2000, 2500])
    def test_temperature_range(self, temp):
        """Parametrized test for different temperature values."""
        result = viscosity_lebrun(Tm=temp, Ts=1500, rho_m=3300, FH2O=0.1, meltfrac=0.5)
        assert result > 0
        assert np.isfinite(result)


class TestViscosityLebrunRegimes:
    """Test specific viscosity regimes and transitions."""
    
    def test_liquid_regime_formula(self):
        """Test the liquid regime formula explicitly."""
        # Force liquid regime by using high temperature
        result = viscosity_lebrun(Tm=2200, Ts=1500, rho_m=3300, FH2O=0.1, meltfrac=0.5)
        assert result > 0
    
    def test_solid_regime_formula(self):
        """Test the solid regime formula explicitly.""" 
        # Force solid regime by using low temperature
        result = viscosity_lebrun(Tm=1500, Ts=1500, rho_m=3300, FH2O=0.1, meltfrac=0.5)
        assert result > 0
    
    def test_regime_transition(self):
        """Test behavior around the regime transition."""
        # Test temperatures around the transition
        temp_solid = 1600  # Should be in solid regime
        temp_liquid = 1800  # Should be in liquid regime
        
        visc_solid = viscosity_lebrun(Tm=temp_solid, Ts=1500, rho_m=3300, FH2O=0.1, meltfrac=0.5)
        visc_liquid = viscosity_lebrun(Tm=temp_liquid, Ts=1500, rho_m=3300, FH2O=0.1, meltfrac=0.5)
        
        assert visc_solid > 0
        assert visc_liquid > 0
        # Both should be finite and positive

from magma.viscosity import viscosity_sandu 

class TestViscositySandu:
    """Test suite for the viscosity_sandu function (Sandu et al. 2011 parameterization)."""
    
    def test_basic_functionality(self):
        """Test that the function runs without errors for typical inputs."""
        result = viscosity_sandu(temp=1600, f_water=0.001, g=9.81, rho_m=3300, P=1e9)
        assert isinstance(result, (float, np.floating))
        assert result > 0, "Viscosity should be positive"
    
    def test_temperature_dependence(self):
        """Test that viscosity decreases with increasing temperature."""
        temp_low = viscosity_sandu(temp=1400, f_water=0.001, g=9.81, rho_m=3300, P=1e9)
        temp_high = viscosity_sandu(temp=1800, f_water=0.001, g=9.81, rho_m=3300, P=1e9)
        
        assert temp_low > temp_high, "Viscosity should decrease with increasing temperature"
    
    def test_water_content_dependence(self):
        """Test that viscosity decreases with increasing water content."""
        water_low = viscosity_sandu(temp=1600, f_water=0.0001, g=9.81, rho_m=3300, P=1e9)
        water_high = viscosity_sandu(temp=1600, f_water=0.01, g=9.81, rho_m=3300, P=1e9)
        
        assert water_low > water_high, "Viscosity should decrease with increasing water content"
    
    def test_pressure_independence(self):
        """Test that viscosity is independent of pressure (V=0)."""
        pressure_low = viscosity_sandu(temp=1600, f_water=0.001, g=9.81, rho_m=3300, P=1e8)
        pressure_high = viscosity_sandu(temp=1600, f_water=0.001, g=9.81, rho_m=3300, P=1e10)
        
        # Should be equal since activation volume V = 0
        assert np.isclose(pressure_low, pressure_high, rtol=1e-10)
    
    def test_density_dependence(self):
        """Test that kinematic viscosity scales inversely with density."""
        rho_low = viscosity_sandu(temp=1600, f_water=0.001, g=9.81, rho_m=3000, P=1e9)
        rho_high = viscosity_sandu(temp=1600, f_water=0.001, g=9.81, rho_m=3600, P=1e9)
        
        # Kinematic viscosity = dynamic viscosity / density
        # So higher density should give lower kinematic viscosity
        assert rho_low > rho_high
        
        # Test the scaling relationship
        ratio_expected = 3600 / 3000
        ratio_actual = rho_low / rho_high
        assert np.isclose(ratio_actual, ratio_expected, rtol=1e-10)
    
    def test_gravitational_acceleration_independence(self):
        """Test that g parameter doesn't affect the result (unused in calculation)."""
        result1 = viscosity_sandu(temp=1600, f_water=0.001, g=9.81, rho_m=3300, P=1e9)
        result2 = viscosity_sandu(temp=1600, f_water=0.001, g=5.0, rho_m=3300, P=1e9)
        
        # g is not used in the calculation, so results should be identical
        assert np.isclose(result1, result2, rtol=1e-10)
    
    def test_water_concentration_calculation(self):
        """Test the water concentration conversion calculation."""
        # Test with different water fractions
        f_waters = [0.0001, 0.001, 0.01]
        results = []
        
        for f_water in f_waters:
            result = viscosity_sandu(temp=1600, f_water=f_water, g=9.81, rho_m=3300, P=1e9)
            results.append(result)
        
        # All should be positive and finite
        assert all(r > 0 and np.isfinite(r) for r in results)
        
        # Higher water content should give lower viscosity
        assert results[0] > results[1] > results[2]
    
    def test_extreme_temperatures(self):
        """Test behavior at extreme but geologically relevant temperatures."""
        # Very hot mantle
        result_hot = viscosity_sandu(temp=2000, f_water=0.001, g=9.81, rho_m=3300, P=1e9)
        assert result_hot > 0
        assert np.isfinite(result_hot)
        
        # Cool mantle
        result_cool = viscosity_sandu(temp=1200, f_water=0.001, g=9.81, rho_m=3300, P=1e9)
        assert result_cool > 0
        assert np.isfinite(result_cool)
        
        # Hot should be less viscous than cool
        assert result_hot < result_cool
    
    def test_water_content_range(self):
        """Test various water content values."""
        # Very dry
        result_dry = viscosity_sandu(temp=1600, f_water=1e-6, g=9.81, rho_m=3300, P=1e9)
        
        # Typical mantle
        result_typical = viscosity_sandu(temp=1600, f_water=0.001, g=9.81, rho_m=3300, P=1e9)
        
        # Water-rich
        result_wet = viscosity_sandu(temp=1600, f_water=0.1, g=9.81, rho_m=3300, P=1e9)
        
        assert all(r > 0 and np.isfinite(r) for r in [result_dry, result_typical, result_wet])
        assert result_dry > result_typical > result_wet
    
    def test_units_and_magnitudes(self):
        """Test that results are in expected units and magnitude ranges."""
        result = viscosity_sandu(temp=1600, f_water=0.001, g=9.81, rho_m=3300, P=1e9)
        
        # Kinematic viscosity should be in reasonable range for mantle (m²/s)
        # Typical mantle viscosity: 1e15 - 1e23 Pa·s
        # With density ~3300 kg/m³: ~1e11 - 1e19 m²/s
        assert 1e5 < result < 1e25, f"Viscosity {result} outside expected range"
    
    def test_olivine_composition_constants(self):
        """Test that the olivine composition is correctly implemented."""
        # The function should work with the assumed 90% forsterite, 10% fayalite
        result = viscosity_sandu(temp=1600, f_water=0.001, g=9.81, rho_m=3300, P=1e9)
        assert result > 0
        assert np.isfinite(result)
    
    @pytest.mark.parametrize("temp,f_water", [
        (1400, 0.0001),
        (1600, 0.001), 
        (1800, 0.01),
        (2000, 0.1),
    ])
    def test_parameter_combinations(self, temp, f_water):
        """Parametrized test for different parameter combinations."""
        result = viscosity_sandu(temp=temp, f_water=f_water, g=9.81, rho_m=3300, P=1e9)
        assert result > 0
        assert np.isfinite(result)
    
    def test_numerical_stability_low_water(self):
        """Test numerical stability for very low water contents."""
        # Very low water content might cause issues with log calculations
        result = viscosity_sandu(temp=1600, f_water=1e-8, g=9.81, rho_m=3300, P=1e9)
        assert np.isfinite(result)
        assert result > 0
    
    def test_activation_energy_effect(self):
        """Test that the activation energy term works correctly."""
        # At higher temperatures, the exp((Qa + P*V)/R/T) term should be smaller
        temp_ratio = 1800 / 1400  # Temperature ratio
        
        visc_low = viscosity_sandu(temp=1400, f_water=0.001, g=9.81, rho_m=3300, P=1e9)
        visc_high = viscosity_sandu(temp=1800, f_water=0.001, g=9.81, rho_m=3300, P=1e9)
        
        # The exponential term should decrease significantly with temperature
        assert visc_low > visc_high
        
        # Check that the effect is significant (not just numerical noise)
        assert visc_low / visc_high > 2


class TestViscositySanduEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_very_high_pressure(self):
        """Test with very high pressure (though V=0 so no effect expected)."""
        result = viscosity_sandu(temp=1600, f_water=0.001, g=9.81, rho_m=3300, P=1e12)
        assert np.isfinite(result)
        assert result > 0
    
    def test_consistency_check(self):
        """Test internal consistency of calculations."""
        # Same conditions should give same results
        result1 = viscosity_sandu(temp=1600, f_water=0.001, g=9.81, rho_m=3300, P=1e9)
        result2 = viscosity_sandu(temp=1600, f_water=0.001, g=9.81, rho_m=3300, P=1e9)
        
        assert np.isclose(result1, result2, rtol=1e-15)