import pytest
import numpy as np
from magma.thermal import (
    mantle_heat_flux, 
    calculate_thermal_diffusivity,
    calculate_rayleigh_number
)


class TestMantleHeatFlux:
    """Test suite for the mantle_heat_flux function."""
    
    def test_basic_functionality(self):
        """Test that the function runs without errors for typical inputs."""
        qm, Db, uc, Ra, nu = mantle_heat_flux(
            Tm=1600, Ts=300, rs=6.0e6, Rp=6.371e6, Rc=3.485e6,
            g=9.81, rho_m=3300, FH2O=0.001, meltfrac=0.2
        )
        
        # Check that all outputs are positive and finite
        assert all(val > 0 and np.isfinite(val) for val in [qm, Db, uc, Ra, nu])
        
        # Check output types
        assert all(isinstance(val, (float, np.floating)) for val in [qm, Db, uc, Ra, nu])
    
    def test_melt_fraction_regimes(self):
        """Test different convection regimes based on melt fraction."""
        # Magma ocean regime (meltfrac >= 0.4)
        qm1, Db1, uc1, Ra1, nu1 = mantle_heat_flux(
            Tm=2000, Ts=1500, rs=6.0e6, Rp=6.371e6, Rc=3.485e6,
            g=9.81, rho_m=3300, FH2O=0.001, meltfrac=0.5
        )
        
        # Solid mantle regime (meltfrac < 0.4)
        qm2, Db2, uc2, Ra2, nu2 = mantle_heat_flux(
            Tm=1600, Ts=300, rs=6.0e6, Rp=6.371e6, Rc=3.485e6,
            g=9.81, rho_m=3300, FH2O=0.001, meltfrac=0.2
        )
        
        # Both should produce valid results
        assert all(val > 0 for val in [qm1, Db1, uc1, Ra1, nu1])
        assert all(val > 0 for val in [qm2, Db2, uc2, Ra2, nu2])
        
        # Different regimes should give different results
        assert not np.isclose(Ra1, Ra2)  # Different convection depths
    
    def test_temperature_dependence(self):
        """Test that heat flux increases with temperature difference."""
        # Lower temperature difference
        qm_low, _, _, Ra_low, _ = mantle_heat_flux(
            Tm=1400, Ts=300, rs=6.0e6, Rp=6.371e6, Rc=3.485e6,
            g=9.81, rho_m=3300, FH2O=0.001, meltfrac=0.2
        )
        
        # Higher temperature difference
        qm_high, _, _, Ra_high, _ = mantle_heat_flux(
            Tm=1800, Ts=300, rs=6.0e6, Rp=6.371e6, Rc=3.485e6,
            g=9.81, rho_m=3300, FH2O=0.001, meltfrac=0.2
        )
        
        # Higher temperature should give higher heat flux and Rayleigh number
        assert qm_high > qm_low
        assert Ra_high > Ra_low
    
    def test_planetary_size_dependence(self):
        """Test dependence on planetary size."""
        # Smaller planet
        qm_small, _, _, Ra_small, _ = mantle_heat_flux(
            Tm=1600, Ts=300, rs=3.0e6, Rp=3.0e6, Rc=1.5e6,
            g=5.0, rho_m=3300, FH2O=0.001, meltfrac=0.2
        )
        
        # Larger planet (Earth-like)
        qm_large, _, _, Ra_large, _ = mantle_heat_flux(
            Tm=1600, Ts=300, rs=6.0e6, Rp=6.371e6, Rc=3.485e6,
            g=9.81, rho_m=3300, FH2O=0.001, meltfrac=0.2
        )
        
        # Larger planet should have higher Rayleigh number (Z³ scaling)
        assert Ra_large > Ra_small
    
    def test_gravity_dependence(self):
        """Test dependence on gravitational acceleration."""
        # Lower gravity
        qm_low_g, _, _, Ra_low_g, _ = mantle_heat_flux(
            Tm=1600, Ts=300, rs=6.0e6, Rp=6.371e6, Rc=3.485e6,
            g=3.71, rho_m=3300, FH2O=0.001, meltfrac=0.2  # Mars-like gravity
        )
        
        # Higher gravity
        qm_high_g, _, _, Ra_high_g, _ = mantle_heat_flux(
            Tm=1600, Ts=300, rs=6.0e6, Rp=6.371e6, Rc=3.485e6,
            g=9.81, rho_m=3300, FH2O=0.001, meltfrac=0.2  # Earth gravity
        )
        
        # Higher gravity should give higher Rayleigh number and heat flux
        assert Ra_high_g > Ra_low_g
        assert qm_high_g > qm_low_g
    
    def test_density_dependence(self):
        """Test dependence on mantle density."""
        # Different densities should affect thermal diffusivity and viscosity
        qm1, _, _, _, _ = mantle_heat_flux(
            Tm=1600, Ts=300, rs=6.0e6, Rp=6.371e6, Rc=3.485e6,
            g=9.81, rho_m=3000, FH2O=0.001, meltfrac=0.2
        )
        
        qm2, _, _, _, _ = mantle_heat_flux(
            Tm=1600, Ts=300, rs=6.0e6, Rp=6.371e6, Rc=3.485e6,
            g=9.81, rho_m=3600, FH2O=0.001, meltfrac=0.2
        )
        
        # Should produce different results
        assert not np.isclose(qm1, qm2)
    
    def test_physical_units_and_ranges(self):
        """Test that outputs are in reasonable physical ranges."""
        qm, Db, uc, Ra, nu = mantle_heat_flux(
            Tm=1600, Ts=300, rs=6.0e6, Rp=6.371e6, Rc=3.485e6,
            g=9.81, rho_m=3300, FH2O=0.001, meltfrac=0.2
        )
        
        # Heat flux: typical Earth values 10-100 mW/m²
        assert 0.001 < qm < 1.0, f"Heat flux {qm} W/m² outside expected range"
        
        # Boundary layer depth: should be much smaller than mantle thickness
        assert 1e3 < Db < 1e6, f"Boundary layer depth {Db} m outside expected range"
        
        # Spreading velocity: typical mantle convection cm/year scale
        assert 1e-11 < uc < 1e-5, f"Spreading velocity {uc} m/s outside expected range"
        
        # Rayleigh number: vigorous convection Ra > 1e6
        assert Ra > 1e6, f"Rayleigh number {Ra} too low for vigorous convection"
        
        # Kinematic viscosity: typical mantle values
        assert 1e10 < nu < 1e25, f"Viscosity {nu} m²/s outside expected range"
    
    def test_scaling_relationships(self):
        """Test expected scaling relationships."""
        # Test Ra^(1/3) scaling in heat flux
        base_conditions = {
            'Tm': 1600, 'Ts': 300, 'rs': 6.371e6, 'Rp': 6.371e6, 'Rc': 3.485e6,
            'g': 9.81, 'rho_m': 3300, 'FH2O': 0.001, 'meltfrac': 0.2
        }
        
        qm1, _, _, Ra1, _ = mantle_heat_flux(**base_conditions)
        
        # Double the temperature difference
        qm2, _, _, Ra2, _ = mantle_heat_flux(Tm=2200, **{k: v for k, v in base_conditions.items() if k != 'Tm'})
        
        # Check approximate scaling (allowing for viscosity changes)
        expected_ratio = (Ra2 / Ra1)**(1/3)
        actual_ratio = qm2 / qm1
        
        # Should be roughly consistent (within factor of 2-3 due to viscosity effects)
        assert 0.5 < actual_ratio / expected_ratio < 3.0
    
    @pytest.mark.parametrize("meltfrac", [0.1, 0.3, 0.5, 0.8])
    def test_melt_fraction_range(self, meltfrac):
        """Parametrized test for different melt fractions."""
        qm, Db, uc, Ra, nu = mantle_heat_flux(
            Tm=1800, Ts=300, rs=6.0e6, Rp=6.371e6, Rc=3.485e6,
            g=9.81, rho_m=3300, FH2O=0.001, meltfrac=meltfrac
        )
        
        assert all(val > 0 and np.isfinite(val) for val in [qm, Db, uc, Ra, nu])
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Very small temperature difference
        qm_small, _, _, _, _ = mantle_heat_flux(
            Tm=310, Ts=300, rs=6.0e6, Rp=6.371e6, Rc=3.485e6,
            g=9.81, rho_m=3300, FH2O=0.001, meltfrac=0.2
        )
        assert qm_small > 0 and np.isfinite(qm_small)
        
        # Very large temperature difference
        qm_large, _, _, _, _ = mantle_heat_flux(
            Tm=3000, Ts=300, rs=6.0e6, Rp=6.371e6, Rc=3.485e6,
            g=9.81, rho_m=3300, FH2O=0.001, meltfrac=0.2
        )
        assert qm_large > 0 and np.isfinite(qm_large)
        
        # Large temperature difference should give much higher heat flux
        assert qm_large > qm_small * 10


class TestThermalDiffusivity:
    """Test suite for calculate_thermal_diffusivity function."""
    
    def test_basic_calculation(self):
        """Test basic thermal diffusivity calculation."""
        kappa = calculate_thermal_diffusivity(km=4.2, rho_m=3300, cp=1200)
        expected = 4.2 / (3300 * 1200)
        assert np.isclose(kappa, expected)
    
    def test_units_and_magnitude(self):
        """Test that thermal diffusivity is in reasonable range."""
        kappa = calculate_thermal_diffusivity(km=4.2, rho_m=3300, cp=1200)
        # Typical rock thermal diffusivity: 1e-7 to 1e-5 m²/s
        assert 1e-8 < kappa < 1e-4
    
    @pytest.mark.parametrize("km,rho_m,cp", [
        (3.0, 3000, 1000),
        (4.2, 3300, 1200),
        (5.0, 3500, 1300),
    ])
    def test_parameter_variations(self, km, rho_m, cp):
        """Parametrized test for different parameter combinations."""
        kappa = calculate_thermal_diffusivity(km, rho_m, cp)
        assert kappa > 0
        assert np.isfinite(kappa)


class TestRayleighNumber:
    """Test suite for calculate_rayleigh_number function."""
    
    def test_basic_calculation(self):
        """Test basic Rayleigh number calculation."""
        Ra = calculate_rayleigh_number(
            g=9.81, alpha=2e-5, delta_T=1300, Z=2.9e6, nu=1e17, kappa=1e-6
        )
        
        expected = (9.81 * 2e-5 * 1300 * (2.9e6)**3) / (1e17 * 1e-6)
        assert np.isclose(Ra, expected)
    
    def test_parameter_scaling(self):
        """Test scaling relationships."""
        base_Ra = calculate_rayleigh_number(
            g=9.81, alpha=2e-5, delta_T=1000, Z=1e6, nu=1e15, kappa=1e-6
        )
        
        # Double temperature difference
        Ra_2T = calculate_rayleigh_number(
            g=9.81, alpha=2e-5, delta_T=2000, Z=1e6, nu=1e15, kappa=1e-6
        )
        assert np.isclose(Ra_2T, 2 * base_Ra)
        
        # Double layer thickness (Z³ scaling)
        Ra_2Z = calculate_rayleigh_number(
            g=9.81, alpha=2e-5, delta_T=1000, Z=2e6, nu=1e15, kappa=1e-6
        )
        assert np.isclose(Ra_2Z, 8 * base_Ra)  # 2³ = 8
    
    def test_absolute_value_temperature(self):
        """Test that absolute value is taken for temperature difference."""
        Ra_pos = calculate_rayleigh_number(
            g=9.81, alpha=2e-5, delta_T=1000, Z=1e6, nu=1e15, kappa=1e-6
        )
        Ra_neg = calculate_rayleigh_number(
            g=9.81, alpha=2e-5, delta_T=-1000, Z=1e6, nu=1e15, kappa=1e-6
        )
        assert np.isclose(Ra_pos, Ra_neg)
    
    def test_typical_values(self):
        """Test with typical mantle values."""
        Ra = calculate_rayleigh_number(
            g=9.81, alpha=2e-5, delta_T=1300, Z=2.9e6, nu=1e17, kappa=1e-6
        )
        
        # Should be much greater than critical Ra for vigorous convection
        assert Ra > 1e6
        
        # Typical mantle Ra should be 1e6 to 1e9
        assert 1e6 < Ra < 1e10


class TestIntegration:
    """Integration tests combining multiple functions."""
    
    def test_earth_like_conditions(self):
        """Test with Earth-like parameters."""
        results = mantle_heat_flux(
            Tm=1600, Ts=300, rs=6.0e6, Rp=6.371e6, Rc=3.485e6,
            g=9.81, rho_m=3300, FH2O=0.001, meltfrac=0.2
        )
        
        qm, Db, uc, Ra, nu = results
        
        # Earth's surface heat flux is ~87 mW/m² total
        # Mantle contribution should be significant fraction
        assert 0.01 < qm < 1.0  # 10-200 mW/m²
        
        # Check that results are self-consistent
        km = 4.2
        kappa = km / (3300 * 1200)
        
        # Verify boundary layer calculation
        Db_expected = km * (1600 - 300) / qm
        assert np.isclose(Db, Db_expected, rtol=1e-10)
        
        # Verify spreading time and velocity relationship
        ts = Db**2 / (5.38 * kappa)
        Z = 6.371e6 - 3.485e6  # Mantle thickness
        uc_expected = Z / ts
        assert np.isclose(uc, uc_expected, rtol=1e-10)
    
    def test_mars_like_conditions(self):
        """Test with Mars-like parameters."""
        results = mantle_heat_flux(
            Tm=1400, Ts=220, rs=3.0e6, Rp=3.39e6, Rc=1.7e6,
            g=3.71, rho_m=3300, FH2O=0.0005, meltfrac=0.1
        )
        
        qm, Db, uc, Ra, nu = results
        
        # Mars is smaller and cooler, should have lower heat flux
        assert 0.001 < qm < 0.1  # Lower than Earth
        assert Ra > 3e4  # Still convecting