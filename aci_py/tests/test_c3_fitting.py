"""
Tests for C3 photosynthesis model fitting.

This module tests the complete C3 fitting workflow including:
- Initial parameter estimation
- Model optimization
- Result validation
"""

import pytest
import numpy as np
import pandas as pd
from aci_py.core.data_structures import ExtendedDataFrame
from aci_py.core.c3_calculations import calculate_c3_assimilation
from aci_py.analysis.c3_fitting import fit_c3_aci, summarize_c3_fit
from aci_py.analysis.initial_guess import (
    estimate_c3_initial_parameters,
    estimate_c3_parameter_bounds,
    estimate_vcmax_from_initial_slope,
    estimate_j_from_plateau,
    estimate_tp_from_high_ci,
    estimate_rd
)
from aci_py.analysis.optimization import (
    negative_log_likelihood,
    rmse,
    calculate_aic,
    calculate_bic,
    parameter_penalty
)


class TestInitialGuess:
    """Test initial parameter estimation functions."""
    
    def test_estimate_vcmax_from_slope(self):
        """Test Vcmax estimation from initial slope."""
        # Create synthetic data with known slope
        ci = np.linspace(50, 300, 10)
        a = 0.1 * ci - 2  # Slope = 0.1
        
        vcmax_est = estimate_vcmax_from_initial_slope(ci, a)
        
        # Should be positive and reasonable
        assert vcmax_est > 0
        assert 20 <= vcmax_est <= 200  # Allow equality for edge cases
    
    def test_estimate_j_from_plateau(self):
        """Test J estimation from plateau region."""
        # Create synthetic data with plateau
        ci = np.linspace(100, 800, 20)
        a = np.full_like(ci, 20.0)  # Plateau at A = 20
        a[ci < 300] = 0.1 * ci[ci < 300]  # Initial slope
        
        j_est = estimate_j_from_plateau(ci, a, vcmax_est=100)
        
        # J should be greater than Vcmax
        assert j_est > 100
        assert j_est < 500
    
    def test_estimate_tp_from_high_ci(self):
        """Test Tp estimation from high Ci region."""
        # Create data with TPU limitation (declining at high Ci)
        ci = np.linspace(100, 1200, 30)
        a = np.full_like(ci, 25.0)
        a[ci > 700] = 25.0 - 0.01 * (ci[ci > 700] - 700)  # Decline
        
        tp_est = estimate_tp_from_high_ci(ci, a)
        
        # Should detect TPU limitation
        assert tp_est > 0
        assert tp_est < 50
    
    def test_estimate_rd(self):
        """Test Rd estimation."""
        # Test with negative minimum
        a = np.array([-1.5, 0, 5, 10, 15, 20])
        rd_est = estimate_rd(a)
        assert np.isclose(rd_est, 1.5, rtol=0.1)
        
        # Test with all positive values
        a = np.array([2, 5, 10, 15, 20])
        rd_est = estimate_rd(a)
        assert 0.5 <= rd_est <= 2.0
    
    def test_estimate_c3_initial_parameters(self):
        """Test complete initial parameter estimation."""
        # Create realistic A-Ci curve data
        ci = np.linspace(50, 1000, 20)
        
        # Simulate realistic A values
        a = []
        for c in ci:
            if c < 300:  # Rubisco limited
                a.append(0.15 * c - 3)
            elif c < 700:  # RuBP limited
                a.append(35)
            else:  # TPU limited
                a.append(35 - 0.01 * (c - 700))
        
        a = np.array(a)
        
        # Create ExtendedDataFrame
        exdf = ExtendedDataFrame({
            'A': a,
            'Ci': ci,
            'Tleaf': np.full_like(ci, 25.0)
        })
        
        # Estimate parameters
        params = estimate_c3_initial_parameters(exdf)
        
        # Check all parameters exist
        assert 'Vcmax_at_25' in params
        assert 'J_at_25' in params
        assert 'Tp_at_25' in params
        assert 'RL_at_25' in params
        assert 'gmc' in params
        
        # Check reasonable values
        assert 10 < params['Vcmax_at_25'] < 300
        assert 20 < params['J_at_25'] < 500
        assert 5 < params['Tp_at_25'] < 50
        assert 0 < params['RL_at_25'] < 10
        assert params['gmc'] == 3.0  # Default value
    
    def test_parameter_bounds_generation(self):
        """Test parameter bounds generation."""
        initial_params = {
            'Vcmax_at_25': 100,
            'J_at_25': 200,
            'Tp_at_25': 15,
            'RL_at_25': 1.5,
            'gmc': 3.0
        }
        
        bounds = estimate_c3_parameter_bounds(initial_params)
        
        # Check all parameters have bounds
        assert len(bounds) == 5
        
        # Check bounds are reasonable
        assert bounds['Vcmax_at_25'][0] < initial_params['Vcmax_at_25'] < bounds['Vcmax_at_25'][1]
        assert bounds['J_at_25'][0] < initial_params['J_at_25'] < bounds['J_at_25'][1]
        
        # Test with fixed parameters
        bounds = estimate_c3_parameter_bounds(
            initial_params,
            fixed_params={'gmc': 3.0}
        )
        assert 'gmc' not in bounds


class TestOptimizationFunctions:
    """Test optimization utility functions."""
    
    def test_negative_log_likelihood(self):
        """Test negative log-likelihood calculation."""
        observed = np.array([1, 2, 3, 4, 5])
        predicted = np.array([1.1, 1.9, 3.1, 3.9, 5.1])
        
        nll = negative_log_likelihood(observed, predicted, sigma=1.0)
        
        # Should be positive
        assert nll > 0
        
        # Perfect predictions should have lower NLL
        nll_perfect = negative_log_likelihood(observed, observed, sigma=1.0)
        assert nll_perfect < nll
    
    def test_rmse(self):
        """Test RMSE calculation."""
        observed = np.array([1, 2, 3, 4, 5])
        predicted = np.array([1.1, 1.9, 3.1, 3.9, 5.1])
        
        error = rmse(observed, predicted)
        
        # Known RMSE
        expected = np.sqrt(np.mean([0.01, 0.01, 0.01, 0.01, 0.01]))
        assert np.isclose(error, expected)
    
    def test_information_criteria(self):
        """Test AIC and BIC calculations."""
        n_params = 5
        n_obs = 20
        nll = 30.0
        
        aic = calculate_aic(n_params, n_obs, nll)
        bic = calculate_bic(n_params, n_obs, nll)
        
        # AIC = 2k + 2*nll
        assert aic > 2 * n_params + 2 * nll  # With correction
        
        # BIC = ln(n)*k + 2*nll
        expected_bic = np.log(n_obs) * n_params + 2 * nll
        assert np.isclose(bic, expected_bic)
    
    def test_parameter_penalty(self):
        """Test parameter penalty function."""
        # Within bounds - no penalty
        penalty = parameter_penalty(5.0, 0.0, 10.0)
        assert penalty == 0.0
        
        # Below lower bound
        penalty = parameter_penalty(-1.0, 0.0, 10.0)
        assert penalty > 0
        
        # Above upper bound
        penalty = parameter_penalty(11.0, 0.0, 10.0)
        assert penalty > 0


class TestC3Fitting:
    """Test complete C3 fitting workflow."""
    
    @pytest.fixture
    def synthetic_aci_data(self):
        """Create synthetic A-Ci data with known parameters."""
        # Known parameters
        true_params = {
            'Vcmax_at_25': 120.0,
            'J_at_25': 250.0,
            'Tp_at_25': 20.0,
            'RL_at_25': 2.0,
            'gmc': 5.0,
            # Kinetic parameters at 25°C
            'Gamma_star_at_25': 36.94438,  # µmol/mol
            'Kc_at_25': 269.3391,  # µmol/mol
            'Ko_at_25': 163.7146   # mmol/mol
        }
        
        # Generate Ci values
        ci = np.concatenate([
            np.linspace(50, 300, 8),    # Rubisco limited
            np.linspace(350, 700, 8),   # RuBP limited  
            np.linspace(800, 1200, 8)   # TPU limited
        ])
        
        # Create data
        data = {
            'Ci': ci,
            'Tleaf': np.full_like(ci, 25.0),
            'Pa': np.full_like(ci, 101.3),
            'O': np.full_like(ci, 0.21)
        }
        
        # Create ExtendedDataFrame
        exdf = ExtendedDataFrame(data)
        exdf.calculate_gas_properties()
        
        # Add Cc column (same as Ci for simplicity when not using gm)
        exdf.set_variable('Cc', data['Ci'])
        
        # Calculate true A values
        result = calculate_c3_assimilation(
            exdf, 
            true_params,
            cc_column_name='Cc',
            oxygen=21.0  # percentage
        )
        
        # Add some noise
        np.random.seed(42)
        noise = np.random.normal(0, 0.5, len(ci))
        exdf.set_variable('A', result.An + noise)
        
        return exdf, true_params
    
    def test_fit_c3_aci_basic(self, synthetic_aci_data):
        """Test basic C3 fitting functionality."""
        exdf, true_params = synthetic_aci_data
        
        # Fit model
        result = fit_c3_aci(
            exdf,
            fixed_parameters={'gmc': 5.0},  # Fix gmc to true value
            seed=42,
            maxiter=100,
            verbose=False
        )
        
        # Check convergence
        assert result.success
        
        # Check parameter recovery (within 20% of true values)
        assert np.abs(result.parameters['Vcmax_at_25'] - true_params['Vcmax_at_25']) / true_params['Vcmax_at_25'] < 0.2
        assert np.abs(result.parameters['J_at_25'] - true_params['J_at_25']) / true_params['J_at_25'] < 0.2
        
        # Check fit quality
        assert result.r_squared > 0.90  # Good fit
        assert result.rmse < 5.0  # Reasonable RMSE given noise
    
    def test_fit_c3_aci_with_custom_bounds(self, synthetic_aci_data):
        """Test fitting with custom parameter bounds."""
        exdf, true_params = synthetic_aci_data
        
        # Set tight bounds around true values
        bounds = {
            'Vcmax_at_25': (100, 140),
            'J_at_25': (230, 270),
            'Tp_at_25': (15, 25),
            'RL_at_25': (1, 3)
        }
        
        result = fit_c3_aci(
            exdf,
            parameter_bounds=bounds,
            fixed_parameters={'gmc': 5.0},
            seed=42,
            maxiter=100
        )
        
        # Parameters should be within bounds
        for param, (lower, upper) in bounds.items():
            assert lower <= result.parameters[param] <= upper
    
    def test_fit_c3_aci_nelder_mead(self, synthetic_aci_data):
        """Test fitting with Nelder-Mead optimizer."""
        exdf, true_params = synthetic_aci_data
        
        # Use good initial guess for local optimizer
        initial_guess = {
            'Vcmax_at_25': 110,
            'J_at_25': 240,
            'Tp_at_25': 18,
            'RL_at_25': 2.5,
            'gmc': 5.0
        }
        
        result = fit_c3_aci(
            exdf,
            optimizer='nelder_mead',
            initial_guess=initial_guess,
            fixed_parameters={'gm': 5.0},
            maxiter=1000
        )
        
        # Should converge with good initial guess
        assert result.success
        assert result.r_squared > 0.9
    
    def test_limiting_process_identification(self, synthetic_aci_data):
        """Test identification of limiting processes."""
        exdf, true_params = synthetic_aci_data
        
        result = fit_c3_aci(
            exdf,
            fixed_parameters={'gm': 5.0},
            seed=42,
            maxiter=100
        )
        
        # Check limiting processes make sense
        ci = exdf.data['Ci'].values
        limiting = result.limiting_process
        
        # Low Ci should be Rubisco limited
        assert np.all(limiting[ci < 200] == 'Wc')
        
        # High Ci might be TPU limited (but not always if Tp is high)
        high_ci_limiting = limiting[ci > 900]
        # At least check that the limiting process changes with Ci
        assert len(np.unique(limiting)) >= 2  # At least 2 different limiting processes
    
    def test_summarize_c3_fit(self, synthetic_aci_data):
        """Test result summarization."""
        exdf, true_params = synthetic_aci_data
        
        result = fit_c3_aci(
            exdf,
            fixed_parameters={'gm': 5.0},
            seed=42,
            maxiter=100
        )
        
        # Create summary
        summary = summarize_c3_fit(result)
        
        # Check structure
        assert isinstance(summary, pd.DataFrame)
        assert 'Parameter' in summary.columns
        assert 'Value' in summary.columns
        assert 'Unit' in summary.columns
        
        # Check content
        params = summary[summary['Parameter'].isin(['Vcmax_at_25', 'J_at_25', 'Tp_at_25', 'RL_at_25'])]
        assert len(params) == 4
        
        stats = summary[summary['Parameter'].isin(['RMSE', 'R²', 'AIC', 'BIC'])]
        assert len(stats) == 4


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_missing_columns(self):
        """Test error handling for missing columns."""
        exdf = ExtendedDataFrame({
            'A': [1, 2, 3],
            'Ci': [100, 200, 300]
            # Missing Tleaf and Pa
        })
        
        with pytest.raises(ValueError, match="Missing required columns"):
            fit_c3_aci(exdf)
    
    def test_insufficient_data(self):
        """Test handling of insufficient data points."""
        exdf = ExtendedDataFrame({
            'A': [10, 15],  # Only 2 points
            'Ci': [100, 200],
            'Tleaf': [25, 25],
            'Pa': [101.3, 101.3]
        })
        exdf.calculate_gas_properties()  # Add T_leaf_K
        
        # Should still attempt to fit but may not converge well
        result = fit_c3_aci(exdf, maxiter=50)
        
        # May have warnings
        assert len(result.warnings) > 0 or not result.success
    
    def test_all_parameters_fixed(self):
        """Test when all parameters are fixed."""
        # Create simple data
        exdf = ExtendedDataFrame({
            'A': np.linspace(5, 25, 10),
            'Ci': np.linspace(100, 1000, 10),
            'Tleaf': np.full(10, 25.0),
            'Pa': np.full(10, 101.3)
        })
        exdf.calculate_gas_properties()  # Add T_leaf_K
        
        # Fix all parameters
        fixed = {
            'Vcmax_at_25': 100,
            'J_at_25': 200,
            'Tp_at_25': 15,
            'RL_at_25': 1.5,
            'gmc': 3.0
        }
        
        result = fit_c3_aci(exdf, fixed_parameters=fixed)
        
        # Should return fixed values
        for param, value in fixed.items():
            assert result.parameters[param] == value