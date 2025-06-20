"""
Tests for C4 ACI curve fitting.
"""

import pytest
import numpy as np
import pandas as pd
from ..core.data_structures import ExtendedDataFrame
from ..core.c4_calculations import calculate_c4_assimilation
from ..analysis.c4_fitting import initial_guess_c4_aci, fit_c4_aci


class TestC4Fitting:
    """Test C4 ACI curve fitting functions."""
    
    @pytest.fixture
    def synthetic_c4_data(self):
        """Create synthetic C4 ACI curve data with known parameters."""
        # True parameters
        true_params = {
            'Vcmax_at_25': 60.0,
            'Vpmax_at_25': 120.0,
            'J_at_25': 200.0,
            'RL_at_25': 1.5,
            'Vpr': 80.0,
            'alpha_psii': 0.0,
            'gbs': 0.003,
            'Rm_frac': 0.5
        }
        
        # Create CO2 gradient
        n_points = 15
        ci_values = np.array([50, 75, 100, 125, 150, 200, 250, 300, 
                             350, 400, 500, 600, 800, 1000, 1200])[:n_points]
        
        # Create base data
        data = pd.DataFrame({
            'Ci': ci_values,
            'Ca': ci_values * 1.2,  # Approximate Ca from Ci
            'PCi': ci_values * 0.04,  # Convert to partial pressure
            'PCm': ci_values * 0.04 * 0.95,  # Slightly lower due to gm
            'Tleaf': 25.0,
            'Tleaf_K': 298.15,
            'oxygen': 21.0,
            'total_pressure': 1.0,
            # Kinetic parameters
            'ao': 0.21,
            'gamma_star': 0.000193,  # C4 value
            'Kc': 650.0,
            'Ko': 450.0,
            'Kp': 80.0,
            # Temperature responses (all 1.0 at 25°C)
            'Vcmax_norm': 1.0,
            'Vpmax_norm': 1.0,
            'RL_norm': 1.0,
            'J_norm': 1.0,
            'gmc_norm': 1.0
        })
        
        exdf = ExtendedDataFrame(data)
        
        # Calculate true assimilation values
        result = calculate_c4_assimilation(
            exdf,
            **true_params,
            return_extended=True
        )
        
        # Add some noise
        np.random.seed(42)
        noise = np.random.normal(0, 0.5, n_points)
        result.data['A'] = result.data['An'] + noise
        result.data['A'] = np.maximum(result.data['A'], 0)  # No negative values
        
        # Store true parameters
        result.true_params = true_params
        
        return result
    
    def test_initial_guess_c4_aci(self, synthetic_c4_data):
        """Test initial parameter guessing for C4."""
        guesses = initial_guess_c4_aci(synthetic_c4_data)
        
        # Check that all required parameters are present
        required = ['RL_at_25', 'Vcmax_at_25', 'Vpmax_at_25', 'Vpr', 'J_at_25']
        assert all(param in guesses for param in required)
        
        # Check that guesses are positive
        assert all(guesses[param] > 0 for param in required)
        
        # Check that guesses are in reasonable ranges
        assert 0.1 <= guesses['RL_at_25'] <= 10.0
        assert 10 <= guesses['Vcmax_at_25'] <= 500.0
        assert 10 <= guesses['Vpmax_at_25'] <= 500.0
        assert 10 <= guesses['Vpr'] <= 200.0
        assert 50 <= guesses['J_at_25'] <= 1000.0
        
        # For synthetic data, guesses should be somewhat close to true values
        true_params = synthetic_c4_data.true_params
        # Allow 100% error in initial guesses
        assert abs(guesses['Vcmax_at_25'] - true_params['Vcmax_at_25']) / true_params['Vcmax_at_25'] < 1.0
    
    def test_fit_c4_aci_basic(self, synthetic_c4_data):
        """Test basic C4 ACI fitting."""
        result = fit_c4_aci(
            synthetic_c4_data,
            calculate_confidence=False  # Speed up test
        )
        
        # Check result structure
        assert hasattr(result, 'parameters')
        assert hasattr(result, 'rmse')
        assert hasattr(result, 'r_squared')
        assert hasattr(result, 'exdf')
        
        # Check that fitted parameters are present
        assert 'Vcmax_at_25' in result.parameters
        assert 'Vpmax_at_25' in result.parameters
        assert 'J_at_25' in result.parameters
        assert 'RL_at_25' in result.parameters
        assert 'Vpr' in result.parameters
        
        # Check fit quality
        assert result.rmse < 2.0  # Should fit well with small noise
        assert result.r_squared > 0.95
        
        # Check that parameters are reasonable
        assert 0 < result.parameters['RL_at_25'] < 10
        assert 10 < result.parameters['Vcmax_at_25'] < 200
        assert 10 < result.parameters['Vpmax_at_25'] < 300
        assert 10 < result.parameters['Vpr'] < 200
        assert 50 < result.parameters['J_at_25'] < 500
    
    def test_fit_c4_aci_fixed_parameters(self, synthetic_c4_data):
        """Test fitting with fixed parameters."""
        # Fix some parameters
        fixed = {
            'RL_at_25': 1.5,  # Fix at true value
            'Vpr': 80.0,      # Fix at true value
            'alpha_psii': 0.0,
            'gbs': 0.003,
            'Rm_frac': 0.5
        }
        
        result = fit_c4_aci(
            synthetic_c4_data,
            fixed_parameters=fixed,
            calculate_confidence=False
        )
        
        # Check that fixed parameters weren't changed
        assert result.parameters['RL_at_25'] == 1.5
        assert result.parameters['Vpr'] == 80.0
        
        # Check that other parameters were fitted
        assert result.parameters['Vcmax_at_25'] != fixed.get('Vcmax_at_25', 0)
        assert result.parameters['Vpmax_at_25'] != fixed.get('Vpmax_at_25', 0)
        
        # Should still get good fit
        assert result.rmse < 2.0
        assert result.r_squared > 0.95
    
    def test_fit_c4_aci_custom_bounds(self, synthetic_c4_data):
        """Test fitting with custom parameter bounds."""
        # Set tight bounds around true values
        true_params = synthetic_c4_data.true_params
        bounds = {
            'Vcmax_at_25': (true_params['Vcmax_at_25'] * 0.8, 
                           true_params['Vcmax_at_25'] * 1.2),
            'Vpmax_at_25': (true_params['Vpmax_at_25'] * 0.8,
                           true_params['Vpmax_at_25'] * 1.2)
        }
        
        result = fit_c4_aci(
            synthetic_c4_data,
            bounds=bounds,
            calculate_confidence=False
        )
        
        # Check that parameters respect bounds
        assert bounds['Vcmax_at_25'][0] <= result.parameters['Vcmax_at_25'] <= bounds['Vcmax_at_25'][1]
        assert bounds['Vpmax_at_25'][0] <= result.parameters['Vpmax_at_25'] <= bounds['Vpmax_at_25'][1]
    
    def test_fit_c4_aci_confidence_intervals(self, synthetic_c4_data):
        """Test confidence interval calculation."""
        result = fit_c4_aci(
            synthetic_c4_data,
            calculate_confidence=True,
            confidence_level=0.95
        )
        
        # Check that confidence intervals were calculated
        assert result.confidence_intervals is not None
        
        # Check structure
        for param in ['Vcmax_at_25', 'Vpmax_at_25', 'J_at_25', 'RL_at_25', 'Vpr']:
            if param in result.parameter_names:
                assert param in result.confidence_intervals
                ci = result.confidence_intervals[param]
                assert len(ci) == 2
                assert ci[0] < result.parameters[param] < ci[1]
    
    def test_fit_c4_aci_with_outliers(self, synthetic_c4_data):
        """Test fitting with outlier points."""
        # Add some outliers
        synthetic_c4_data.data.loc[5, 'A'] = 50  # Unrealistically high
        synthetic_c4_data.data.loc[10, 'A'] = -5  # Negative
        
        result = fit_c4_aci(
            synthetic_c4_data,
            calculate_confidence=False
        )
        
        # Should still converge despite outliers
        assert result.parameters is not None
        assert result.rmse < 10.0  # Higher due to outliers
    
    def test_fit_c4_aci_different_optimizers(self, synthetic_c4_data):
        """Test different optimization methods."""
        # Test differential evolution (default)
        result_de = fit_c4_aci(
            synthetic_c4_data,
            optimizer='differential_evolution',
            calculate_confidence=False
        )
        
        # Test Nelder-Mead
        result_nm = fit_c4_aci(
            synthetic_c4_data,
            optimizer='nelder-mead',
            calculate_confidence=False
        )
        
        # Both should give reasonable results
        assert result_de.rmse < 2.0
        assert result_nm.rmse < 3.0  # NM might be slightly worse
        
        # Parameters should be similar
        for param in ['Vcmax_at_25', 'Vpmax_at_25']:
            assert abs(result_de.parameters[param] - result_nm.parameters[param]) / \
                   result_de.parameters[param] < 0.3  # Within 30%
    
    def test_fit_c4_aci_result_exdf(self, synthetic_c4_data):
        """Test the extended data frame in fitting results."""
        result = fit_c4_aci(
            synthetic_c4_data,
            calculate_confidence=False
        )
        
        # Check that model predictions are included
        assert 'An_model' in result.exdf.data.columns
        assert 'residuals' in result.exdf.data.columns
        
        # Check process rates
        assert 'Ac' in result.exdf.data.columns
        assert 'Aj' in result.exdf.data.columns
        assert 'Ar' in result.exdf.data.columns
        assert 'Ap' in result.exdf.data.columns
        
        # Check that residuals match
        expected_residuals = synthetic_c4_data.data['A'] - result.exdf.data['An_model']
        assert np.allclose(result.exdf.data['residuals'], expected_residuals)
    
    def test_c4_limiting_processes_in_fit(self, synthetic_c4_data):
        """Test that we can identify limiting processes after fitting."""
        from ..core.c4_calculations import identify_c4_limiting_processes
        
        result = fit_c4_aci(
            synthetic_c4_data,
            calculate_confidence=False
        )
        
        # Identify limiting processes
        result_with_limits = identify_c4_limiting_processes(result.exdf)
        
        # Check that limitations make sense
        # At low CO2, expect PEP carboxylation or Rubisco limitation
        low_co2_mask = result_with_limits.data['PCm'] < 5
        if np.any(low_co2_mask):
            enzyme_limits = result_with_limits.data.loc[low_co2_mask, 'enzyme_limited_process']
            assert all(lim in ['pep_carboxylation_co2', 'rubisco', ''] 
                      for lim in enzyme_limits)
    
    def test_c4_vs_c3_parameter_ranges(self, synthetic_c4_data):
        """Test that C4 parameters are in expected ranges compared to C3."""
        result = fit_c4_aci(
            synthetic_c4_data,
            calculate_confidence=False
        )
        
        # C4 specific expectations
        # Vpmax should be substantial (PEP carboxylase activity)
        assert result.parameters['Vpmax_at_25'] > 50
        
        # Vpmax often > Vcmax in C4 plants
        assert result.parameters['Vpmax_at_25'] > result.parameters['Vcmax_at_25'] * 0.5
        
        # Bundle sheath parameters
        assert result.parameters['gbs'] > 0  # Should have some conductance
        assert 0 <= result.parameters['alpha_psii'] <= 1  # Fraction bounds