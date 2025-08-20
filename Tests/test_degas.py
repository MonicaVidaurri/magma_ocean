import pytest
import numpy as np
from magma.degas import (
    degas_method_1,
    degas_method_2
)


class TestDegasMethod1:
    """Test suite for degas_method_1 function."""
    
    def test_basic_functionality(self):
        """Test that the function runs without errors for typical inputs."""
        Dmelt, rmor, avgfmelt, avgXmelt = degas_method_1(
            Tm=1800, Db=50e3, qm=0.1, FH2O=0.001,
            Rp=6.371e6, g=9.81, Tsurf=1500
        )
        
        # Check that all outputs are non-negative and finite
        assert all(val >= 0 and np.isfinite(val) for val in [Dmelt, rmor, avgfmelt, avgXmelt])
        
        # Check output types
        assert all(isinstance(val, (float, np.floating)) for val in [Dmelt, rmor, avgfmelt, avgXmelt])
    
    def test_no_melt_conditions(self):
        """Test behavior when no melting occurs."""
        # Very low temperature - should produce no melt
        Dmelt, rmor, avgfmelt, avgXmelt = degas_method_1(
            Tm=1200, Db=50e3, qm=0.05, FH2O=0.001,
            Rp=6.371e6, g=9.81, Tsurf=300
        )
        # Should have no melt
        assert Dmelt == 0.0
        assert rmor == 0.0
        assert avgfmelt == 0.0
        assert avgXmelt == 0.0
    
    def test_high_temperature_melting(self):
        """Test behavior with high temperatures causing extensive melting."""
        Dmelt, rmor, avgfmelt, avgXmelt = degas_method_1(
            Tm=2200, Db=100e3, qm=0.2, FH2O=0.01,
            Rp=6.371e6, g=9.81, Tsurf=1800
        )
        
        # Should have significant melt
        assert Dmelt > 0
        assert rmor > 0
        assert avgfmelt > 0
        assert avgXmelt > 0
        assert avgfmelt <= 1.0  # Melt fraction should not exceed 1
    
    def test_water_content_effect(self):
        """Test effect of water content on degassing."""
        # Low water content
        Dmelt1, rmor1, avgfmelt1, avgXmelt1 = degas_method_1(
            Tm=1800, Db=50e3, qm=0.1, FH2O=0.0001,
            Rp=6.371e6, g=9.81, Tsurf=1500
        )
        
        # High water content
        Dmelt2, rmor2, avgfmelt2, avgXmelt2 = degas_method_1(
            Tm=1800, Db=50e3, qm=0.1, FH2O=0.01,
            Rp=6.371e6, g=9.81, Tsurf=1500
        )
        
        # Higher water content should give higher water concentration in melt
        if avgXmelt1 > 0 and avgXmelt2 > 0:
            assert avgXmelt2 > avgXmelt1
        
        # Degassing rate should increase with water content
        if rmor1 > 0 and rmor2 > 0:
            assert rmor2 > rmor1
    
    def test_temperature_scaling(self):
        """Test scaling with mantle temperature."""
        # Lower temperature
        Dmelt1, rmor1, _, _ = degas_method_1(
            Tm=1600, Db=50e3, qm=0.1, FH2O=0.001,
            Rp=6.371e6, g=9.81, Tsurf=1200
        )
        
        # Higher temperature
        Dmelt2, rmor2, _, _ = degas_method_1(
            Tm=2000, Db=50e3, qm=0.1, FH2O=0.001,
            Rp=6.371e6, g=9.81, Tsurf=1200
        )
        
        # Higher temperature should generally increase melt thickness
        assert Dmelt2 >= Dmelt1
    
    def test_planetary_size_effect(self):
        """Test effect of planetary size."""
        # Smaller planet
        Dmelt1, rmor1, _, _ = degas_method_1(
            Tm=1800, Db=30e3, qm=0.1, FH2O=0.001,
            Rp=3.4e6, g=3.7, Tsurf=1500  # Mars-like
        )
        
        # Larger planet
        Dmelt2, rmor2, _, _ = degas_method_1(
            Tm=1800, Db=50e3, qm=0.1, FH2O=0.001,
            Rp=6.371e6, g=9.81, Tsurf=1500  # Earth-like
        )
        
        # Both should be valid
        assert all(val >= 0 for val in [Dmelt1, rmor1, Dmelt2, rmor2])
    
    def test_boundary_layer_effect(self):
        """Test effect of boundary layer depth."""
        # Thin boundary layer
        Dmelt1, rmor1, _, _ = degas_method_1(
            Tm=1800, Db=20e3, qm=0.1, FH2O=0.001,
            Rp=6.371e6, g=9.81, Tsurf=1500
        )
        
        # Thick boundary layer
        Dmelt2, rmor2, _, _ = degas_method_1(
            Tm=1800, Db=100e3, qm=0.1, FH2O=0.001,
            Rp=6.371e6, g=9.81, Tsurf=1500
        )
        
        # Different boundary layer depths should affect temperature profile
        # and potentially melt distribution
        assert all(val >= 0 for val in [Dmelt1, rmor1, Dmelt2, rmor2])
    
    def test_heat_flux_scaling(self):
        """Test scaling with heat flux."""
        # Low heat flux
        Dmelt1, rmor1, _, _ = degas_method_1(
            Tm=1800, Db=50e3, qm=0.05, FH2O=0.001,
            Rp=6.371e6, g=9.81, Tsurf=1500
        )
        
        # High heat flux
        Dmelt2, rmor2, _, _ = degas_method_1(
            Tm=1800, Db=50e3, qm=0.2, FH2O=0.001,
            Rp=6.371e6, g=9.81, Tsurf=1500
        )
        
        # Higher heat flux should affect temperature profile
        assert all(val >= 0 for val in [Dmelt1, rmor1, Dmelt2, rmor2])
    
    @pytest.mark.parametrize("Tm,FH2O", [
        (1600, 0.0001),
        (1800, 0.001),
        (2000, 0.01),
        (2200, 0.1),
    ])
    def test_parameter_combinations(self, Tm, FH2O):
        """Parametrized test for different parameter combinations."""
        Dmelt, rmor, avgfmelt, avgXmelt = degas_method_1(
            Tm=Tm, Db=50e3, qm=0.1, FH2O=FH2O,
            Rp=6.371e6, g=9.81, Tsurf=1500
        )
        
        assert all(val >= 0 and np.isfinite(val) for val in [Dmelt, rmor, avgfmelt, avgXmelt])
        if avgfmelt > 0:
            assert avgfmelt <= 1.0


class TestDegasMethod2:
    """Test suite for degas_method_2 function."""
    
    def test_basic_functionality(self):
        """Test that the function runs without errors for typical inputs."""
        Dmelt, rmor, meltfrac, avgXmelt = degas_method_2(
            Tm=1800, Db=50e3, qm=0.1, FH2O=0.001,
            Rp=6.371e6, g=9.81, Tsurf=1500
        )
        
        assert all(val >= 0 and np.isfinite(val) for val in [Dmelt, rmor, meltfrac, avgXmelt])
        assert all(isinstance(val, (float, np.floating)) for val in [Dmelt, rmor, meltfrac, avgXmelt])
    
    def test_no_melt_conditions(self):
        """Test behavior when no melting occurs."""
        Dmelt, rmor, meltfrac, avgXmelt = degas_method_2(
            Tm=1200, Db=50e3, qm=0.05, FH2O=0.001,
            Rp=6.371e6, g=9.81, Tsurf=300
        )
        
        assert Dmelt == 0.0
        assert rmor == 0.0
        assert meltfrac == 0.0
        assert avgXmelt == 0.0
    
    def test_melt_fraction_constraint(self):
        """Test that melt fraction is properly constrained to [0, 1]."""
        Dmelt, rmor, meltfrac, avgXmelt = degas_method_2(
            Tm=2500, Db=100e3, qm=0.3, FH2O=0.1,
            Rp=6.371e6, g=9.81, Tsurf=2000
        )
        
        if meltfrac > 0:
            assert 0.0 <= meltfrac <= 1.0
    
    def test_water_content_scaling(self):
        """Test scaling with water content."""
        # Test multiple water contents
        water_contents = [0.0001, 0.001, 0.01]
        results = []
        
        for FH2O in water_contents:
            result = degas_method_2(
                Tm=1800, Db=50e3, qm=0.1, FH2O=FH2O,
                Rp=6.371e6, g=9.81, Tsurf=1500
            )
            results.append(result)
        
        # Higher water content should generally increase avgXmelt and rmor
        for i in range(len(results) - 1):
            _, rmor1, _, avgXmelt1 = results[i]
            _, rmor2, _, avgXmelt2 = results[i + 1]
            
            if avgXmelt1 > 0 and avgXmelt2 > 0:
                assert avgXmelt2 > avgXmelt1
    
    def test_temperature_scaling(self):
        """Test scaling with temperature."""
        # Low temperature
        Dmelt1, rmor1, meltfrac1, _ = degas_method_2(
            Tm=1600, Db=50e3, qm=0.1, FH2O=0.001,
            Rp=6.371e6, g=9.81, Tsurf=1200
        )
        
        # High temperature
        Dmelt2, rmor2, meltfrac2, _ = degas_method_2(
            Tm=2000, Db=50e3, qm=0.1, FH2O=0.001,
            Rp=6.371e6, g=9.81, Tsurf=1200
        )
        
        # Higher temperature should increase melt thickness and melt fraction
        if Dmelt1 > 0 and Dmelt2 > 0:
            assert Dmelt2 >= Dmelt1

            # meltfrac is volume averaged so by having a larger volume even if there is more overall melt, the fraction
            # may be lower. So the below check is turned off.
            # assert meltfrac2 >= meltfrac1
    
    @pytest.mark.parametrize("Tm,Db,qm", [
        (1600, 30e3, 0.05),
        (1800, 50e3, 0.1),
        (2000, 70e3, 0.2),
        (2200, 100e3, 0.3),
    ])
    def test_parameter_combinations(self, Tm, Db, qm):
        """Parametrized test for different parameter combinations."""
        Dmelt, rmor, meltfrac, avgXmelt = degas_method_2(
            Tm=Tm, Db=Db, qm=qm, FH2O=0.001,
            Rp=6.371e6, g=9.81, Tsurf=1200
        )
        
        [print(val) for val in [Dmelt, rmor, meltfrac, avgXmelt]]
        assert all(val >= 0 and np.isfinite(val) for val in [Dmelt, rmor, meltfrac, avgXmelt])
        if meltfrac > 0:
            assert meltfrac <= 1.0


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_very_high_temperature(self):
        """Test with very high temperatures."""
        # Method 1
        Dmelt1, rmor1, avgfmelt1, avgXmelt1 = degas_method_1(
            Tm=3000, Db=100e3, qm=0.5, FH2O=0.01,
            Rp=6.371e6, g=9.81, Tsurf=2000
        )
        
        # Method 2
        Dmelt2, rmor2, meltfrac2, avgXmelt2 = degas_method_2(
            Tm=3000, Db=100e3, qm=0.5, FH2O=0.01,
            Rp=6.371e6, g=9.81, Tsurf=2000
        )
        
        # Both methods should handle extreme temperatures
        assert all(np.isfinite(val) for val in [Dmelt1, rmor1, avgfmelt1, avgXmelt1])
        assert all(np.isfinite(val) for val in [Dmelt2, rmor2, meltfrac2, avgXmelt2])
        
        # Should have significant melt
        assert Dmelt1 > 0
        assert Dmelt2 > 0
    
    def test_zero_gravity(self):
        """Test with zero gravity (edge case)."""
        # Method 1
        Dmelt1, rmor1, avgfmelt1, avgXmelt1 = degas_method_1(
            Tm=1800, Db=50e3, qm=0.1, FH2O=0.001,
            Rp=6.371e6, g=0.0, Tsurf=1500
        )
        
        # Method 2
        Dmelt2, rmor2, meltfrac2, avgXmelt2 = degas_method_2(
            Tm=1800, Db=50e3, qm=0.1, FH2O=0.001,
            Rp=6.371e6, g=0.0, Tsurf=1500
        )
        
        # Should handle zero gravity without errors
        assert all(np.isfinite(val) for val in [Dmelt1, rmor1, avgfmelt1, avgXmelt1])
        assert all(np.isfinite(val) for val in [Dmelt2, rmor2, meltfrac2, avgXmelt2])
    
    def test_surface_equal_to_mantle_temp(self):
        """Test when surface temperature equals mantle temperature."""
        # Method 1
        Dmelt1, rmor1, avgfmelt1, avgXmelt1 = degas_method_1(
            Tm=1800, Db=50e3, qm=0.1, FH2O=0.001,
            Rp=6.371e6, g=9.81, Tsurf=1800
        )
        
        # Method 2
        Dmelt2, rmor2, meltfrac2, avgXmelt2 = degas_method_2(
            Tm=1800, Db=50e3, qm=0.1, FH2O=0.001,
            Rp=6.371e6, g=9.81, Tsurf=1800
        )
        
        # Should handle this case without errors
        assert all(np.isfinite(val) for val in [Dmelt1, rmor1, avgfmelt1, avgXmelt1])
        assert all(np.isfinite(val) for val in [Dmelt2, rmor2, meltfrac2, avgXmelt2])
    
    def test_zero_water_content(self):
        """Test with zero water content."""
        # Method 1
        Dmelt1, rmor1, avgfmelt1, avgXmelt1 = degas_method_1(
            Tm=1800, Db=50e3, qm=0.1, FH2O=0.0,
            Rp=6.371e6, g=9.81, Tsurf=1500
        )
        
        # Method 2
        Dmelt2, rmor2, meltfrac2, avgXmelt2 = degas_method_2(
            Tm=1800, Db=50e3, qm=0.1, FH2O=0.0,
            Rp=6.371e6, g=9.81, Tsurf=1500
        )
        
        # With zero water, should still calculate melt thickness but no degassing
        assert all(np.isfinite(val) for val in [Dmelt1, rmor1, avgfmelt1, avgXmelt1])
        assert all(np.isfinite(val) for val in [Dmelt2, rmor2, meltfrac2, avgXmelt2])
        
        # No water to degas
        assert rmor1 == 0.0
        assert rmor2 == 0.0


class TestIntegration:
    """Integration tests combining multiple function calls."""
    
    def test_cooling_sequence(self):
        """Test a sequence of cooling temperatures."""
        temperatures = np.linspace(2200, 1400, 5)
        results1 = []
        results2 = []
        
        for temp in temperatures:
            r1 = degas_method_1(
                Tm=temp, Db=50e3, qm=0.1, FH2O=0.001,
                Rp=6.371e6, g=9.81, Tsurf=300
            )
            results1.append(r1)
            
            r2 = degas_method_2(
                Tm=temp, Db=50e3, qm=0.1, FH2O=0.001,
                Rp=6.371e6, g=9.81, Tsurf=300
            )
            results2.append(r2)
        
        # Convert to arrays for easier analysis
        results1 = np.array(results1)
        results2 = np.array(results2)
        
        # As temperature decreases, melt thickness should generally decrease
        for i in range(len(results1) - 1):
            if results1[i, 0] > 0 and results1[i+1, 0] > 0:
                assert results1[i, 0] >= results1[i+1, 0]
            
            if results2[i, 0] > 0 and results2[i+1, 0] > 0:
                assert results2[i, 0] >= results2[i+1, 0]
    
    def test_earth_like_conditions(self):
        """Test with Earth-like conditions."""
        # Earth-like parameters
        params = {
            'Tm': 1600,
            'Db': 100e3,
            'qm': 0.08,
            'FH2O': 0.0005,
            'Rp': 6.371e6,
            'g': 9.81,
            'Tsurf': 300
        }
        
        # Try both methods
        Dmelt1, rmor1, avgfmelt1, avgXmelt1 = degas_method_1(**params)
        Dmelt2, rmor2, meltfrac2, avgXmelt2 = degas_method_2(**params)
        
        # Check results are reasonable for Earth conditions
        assert all(np.isfinite(val) for val in [Dmelt1, rmor1, avgfmelt1, avgXmelt1])
        assert all(np.isfinite(val) for val in [Dmelt2, rmor2, meltfrac2, avgXmelt2])