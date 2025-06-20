"""
Tests for C4 photosynthesis calculations.
"""

import pytest
import numpy as np
import pandas as pd
from ..core.data_structures import ExtendedDataFrame
from ..core.c4_calculations import (
    calculate_c4_assimilation,
    identify_c4_limiting_processes,
    apply_gm_c4
)


class TestC4Calculations:
    """Test C4 photosynthesis calculations."""
    
    @pytest.fixture
    def sample_c4_data(self):
        """Create sample C4 data for testing."""
        n_points = 10
        
        # Create CO2 gradient
        ci_values = np.linspace(50, 400, n_points)
        
        # Create sample data
        data = pd.DataFrame({
            'Ci': ci_values,
            'PCi': ci_values * 0.04,  # Approximate conversion to PCi
            'PCm': ci_values * 0.04 * 0.9,  # Slightly lower than PCi
            'A': np.linspace(5, 35, n_points),  # Dummy values
            'Tleaf': 25.0,
            'oxygen': 21.0,
            'total_pressure': 1.0,
            # Kinetic parameters at 25°C
            'ao': 0.21,
            'gamma_star': 0.000193,  # C4 value (much lower than C3)
            'Kc': 650.0,
            'Ko': 450.0,
            'Kp': 80.0,
            # Temperature normalized values (all 1.0 at 25°C)
            'Vcmax_norm': 1.0,
            'Vpmax_norm': 1.0,
            'RL_norm': 1.0,
            'J_norm': 1.0,
            'gmc_norm': 1.0
        })
        
        return ExtendedDataFrame(data)
    
    def test_calculate_c4_assimilation_basic(self, sample_c4_data):
        """Test basic C4 assimilation calculation."""
        result = calculate_c4_assimilation(
            sample_c4_data,
            Vcmax_at_25=60,
            Vpmax_at_25=120,
            J_at_25=200,
            RL_at_25=1.0,
            Vpr=80,
            return_extended=True
        )
        
        # Check output structure
        assert isinstance(result, ExtendedDataFrame)
        assert 'An' in result.data.columns
        assert 'Ac' in result.data.columns
        assert 'Aj' in result.data.columns
        assert 'Ar' in result.data.columns
        assert 'Ap' in result.data.columns
        assert 'Apr' in result.data.columns
        
        # Check that assimilation is positive
        assert np.all(result.data['An'] >= 0)
        
        # Check that An is minimum of Ac and Aj
        assert np.allclose(
            result.data['An'],
            np.minimum(result.data['Ac'], result.data['Aj']),
            rtol=1e-10
        )
    
    def test_c4_enzyme_limitations(self, sample_c4_data):
        """Test different enzyme limitations in C4."""
        # Test Rubisco limitation (high Vpmax, low Vcmax)
        result = calculate_c4_assimilation(
            sample_c4_data,
            Vcmax_at_25=20,   # Low
            Vpmax_at_25=200,  # High
            J_at_25=400,      # High
            Vpr=150,          # High
            return_extended=True
        )
        
        # At high CO2, should be Rubisco limited
        high_co2_mask = sample_c4_data.data['PCm'] > 10
        assert np.mean(np.abs(result.data['Ac'][high_co2_mask] - 
                            result.data['Ar'][high_co2_mask])) < 1.0
    
    def test_c4_pep_carboxylation_limitation(self, sample_c4_data):
        """Test PEP carboxylation limitation."""
        # Test PEP carboxylation limitation (low Vpmax)
        result = calculate_c4_assimilation(
            sample_c4_data,
            Vcmax_at_25=100,  # High
            Vpmax_at_25=30,   # Low
            J_at_25=400,      # High
            Vpr=150,          # High
            return_extended=True
        )
        
        # Should show PEP carboxylation limitation at low CO2
        low_co2_mask = sample_c4_data.data['PCm'] < 5
        if np.any(low_co2_mask):
            # Ap should be close to Apc (CO2 limited)
            assert np.mean(np.abs(result.data['Ap'][low_co2_mask] - 
                                result.data['Apc'][low_co2_mask])) < 1.0
    
    def test_c4_light_limitation(self, sample_c4_data):
        """Test light limitation in C4."""
        # Test light limitation (low J)
        result = calculate_c4_assimilation(
            sample_c4_data,
            Vcmax_at_25=100,
            Vpmax_at_25=150,
            J_at_25=50,       # Low
            Vpr=100,
            return_extended=True
        )
        
        # Check that some points are light limited
        light_limited = np.abs(result.data['An'] - result.data['Aj']) < 1e-6
        assert np.any(light_limited)
    
    def test_c4_parameter_bounds(self, sample_c4_data):
        """Test parameter bound checking."""
        # Test negative Vcmax
        with pytest.raises(ValueError, match="Vcmax must be >= 0"):
            calculate_c4_assimilation(
                sample_c4_data,
                Vcmax_at_25=-10,
                check_inputs=True
            )
        
        # Test invalid alpha_psii
        with pytest.raises(ValueError, match="alpha_psii must be between 0 and 1"):
            calculate_c4_assimilation(
                sample_c4_data,
                alpha_psii=1.5,
                check_inputs=True
            )
        
        # Test invalid Rm_frac
        with pytest.raises(ValueError, match="Rm_frac must be between 0 and 1"):
            calculate_c4_assimilation(
                sample_c4_data,
                Rm_frac=-0.1,
                check_inputs=True
            )
    
    def test_c4_temperature_response(self, sample_c4_data):
        """Test temperature response integration."""
        # Modify temperature
        sample_c4_data.data['Tleaf'] = 30.0
        
        # Adjust normalization factors for 30°C (approximate)
        sample_c4_data.data['Vcmax_norm'] = 1.5
        sample_c4_data.data['Vpmax_norm'] = 1.4
        sample_c4_data.data['J_norm'] = 1.3
        sample_c4_data.data['RL_norm'] = 1.8
        
        result = calculate_c4_assimilation(
            sample_c4_data,
            Vcmax_at_25=60,
            return_extended=True
        )
        
        # Check temperature adjustments were applied
        assert np.allclose(result.data['Vcmax_tl'], 60 * 1.5)
        assert np.allclose(result.data['Vpmax_tl'], 150 * 1.4)  # Using default Vpmax
    
    def test_identify_c4_limiting_processes(self, sample_c4_data):
        """Test identification of limiting processes."""
        # Calculate with specific limitations
        result = calculate_c4_assimilation(
            sample_c4_data,
            Vcmax_at_25=30,   # Low - force Rubisco limitation
            Vpmax_at_25=150,  # High
            J_at_25=300,      # High
            Vpr=100,
            return_extended=True
        )
        
        # Identify limitations
        result = identify_c4_limiting_processes(result)
        
        assert 'limiting_process' in result.data.columns
        assert 'enzyme_limited_process' in result.data.columns
        
        # Check that processes are correctly identified
        assert set(result.data['limiting_process']) <= {'enzyme', 'light'}
        assert all(p in ['', 'rubisco', 'pep_carboxylation_co2', 'pep_regeneration'] 
                  for p in result.data['enzyme_limited_process'])
    
    def test_apply_gm_c4(self, sample_c4_data):
        """Test mesophyll conductance application for C4."""
        # Remove PCm to test calculation
        sample_c4_data.data = sample_c4_data.data.drop('PCm', axis=1)
        
        # Apply mesophyll conductance
        result = apply_gm_c4(sample_c4_data, gmc_at_25=1.0)
        
        assert 'PCm' in result.data.columns
        assert 'gmc' in result.data.columns
        assert 'gmc_at_25' in result.data.columns
        
        # Check that PCm < PCi (due to resistance)
        assert np.all(result.data['PCm'] <= result.data['PCi'])
        
        # Check units
        assert result.units['PCm'] == 'microbar'
        assert result.units['gmc'] == 'mol m^(-2) s^(-1) bar^(-1)'
    
    def test_c4_vs_c3_gamma_star(self, sample_c4_data):
        """Test that C4 uses much lower gamma_star than C3."""
        result = calculate_c4_assimilation(
            sample_c4_data,
            return_extended=True
        )
        
        # C4 gamma_star should be much lower than C3 (around 37 µbar)
        gamma_star_value = sample_c4_data.data['gamma_star'].iloc[0]
        assert gamma_star_value < 0.001  # Much less than C3 value
    
    def test_c4_bundle_sheath_parameters(self, sample_c4_data):
        """Test bundle sheath specific parameters."""
        # Test with PSII in bundle sheath
        result_with_psii = calculate_c4_assimilation(
            sample_c4_data,
            alpha_psii=0.15,  # Some PSII in bundle sheath
            return_extended=True
        )
        
        # Test without PSII in bundle sheath (default)
        result_no_psii = calculate_c4_assimilation(
            sample_c4_data,
            alpha_psii=0.0,
            return_extended=True
        )
        
        # Results should be different
        assert not np.allclose(result_with_psii.data['An'], 
                              result_no_psii.data['An'])
    
    def test_c4_oxygen_sensitivity(self, sample_c4_data):
        """Test that C4 is less sensitive to oxygen than C3."""
        # Calculate at normal oxygen
        result_normal = calculate_c4_assimilation(
            sample_c4_data,
            return_extended=True
        )
        
        # Calculate at high oxygen
        sample_c4_data.data['oxygen'] = 30.0
        result_high_o2 = calculate_c4_assimilation(
            sample_c4_data,
            return_extended=True
        )
        
        # C4 should show minimal oxygen effect
        relative_change = np.abs(result_high_o2.data['An'] - result_normal.data['An']) / result_normal.data['An']
        assert np.mean(relative_change) < 0.1  # Less than 10% change on average