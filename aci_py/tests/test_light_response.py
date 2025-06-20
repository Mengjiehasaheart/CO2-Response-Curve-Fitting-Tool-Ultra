"""
Tests for light response curve fitting module.
"""

import pytest
import numpy as np
import pandas as pd
from aci_py.core.data_structures import ExtendedDataFrame
from aci_py.analysis.light_response import (
    non_rectangular_hyperbola,
    rectangular_hyperbola,
    exponential_model,
    initial_guess_light_response,
    fit_light_response,
    compare_light_models
)


class TestLightResponseModels:
    """Test individual light response model functions."""
    
    def test_non_rectangular_hyperbola(self):
        """Test non-rectangular hyperbola calculations."""
        # Test single value
        I = 500
        phi = 0.05
        Amax = 30
        theta = 0.7
        Rd = 2
        
        A = non_rectangular_hyperbola(I, phi, Amax, theta, Rd)
        
        # Should be positive and less than Amax - Rd
        assert A > 0
        assert A < Amax - Rd
        
        # Test array
        I_array = np.array([0, 100, 500, 1000, 2000])
        A_array = non_rectangular_hyperbola(I_array, phi, Amax, theta, Rd)
        
        # Check shape
        assert A_array.shape == I_array.shape
        
        # Check monotonic increase (after accounting for Rd)
        A_gross = A_array + Rd
        assert np.all(np.diff(A_gross) > 0)
        
        # Check limits
        assert A_array[0] == pytest.approx(-Rd)  # A = -Rd at I = 0
        assert A_array[-1] < Amax  # Approaches but doesn't exceed Amax
    
    def test_rectangular_hyperbola(self):
        """Test rectangular hyperbola (theta = 0)."""
        I = np.array([0, 100, 500, 1000, 2000])
        phi = 0.05
        Amax = 30
        Rd = 2
        
        A = rectangular_hyperbola(I, phi, Amax, Rd)
        
        # Compare with non-rectangular hyperbola at theta = 0
        A_nrh = non_rectangular_hyperbola(I, phi, Amax, 0, Rd)
        
        # Should be very close (allowing for numerical precision)
        np.testing.assert_allclose(A, A_nrh, rtol=1e-10)
        
        # Check properties
        assert A[0] == pytest.approx(-Rd)
        assert np.all(np.diff(A) > 0)  # Monotonic increase
    
    def test_exponential_model(self):
        """Test exponential light response model."""
        I = np.array([0, 100, 500, 1000, 2000])
        phi = 0.05
        Amax = 30
        Rd = 2
        
        A = exponential_model(I, phi, Amax, Rd)
        
        # Check properties
        assert A[0] == pytest.approx(-Rd)
        assert np.all(np.diff(A) > 0)  # Monotonic increase
        assert np.all(A < Amax - Rd)  # Never exceeds Amax - Rd
        
        # Check diminishing returns - first derivative should decrease
        dA = np.diff(A)
        # For exponential, diminishing returns means smaller increases
        assert dA[0] > dA[-1]  # Early slope > late slope


class TestInitialGuess:
    """Test initial parameter estimation for light response."""
    
    def test_initial_guess_typical_data(self):
        """Test initial guess with typical light response data."""
        # Generate synthetic data
        I = np.array([0, 20, 50, 100, 200, 400, 800, 1200, 1600, 2000])
        A_true = non_rectangular_hyperbola(I, phi=0.05, Amax=30, theta=0.7, Rd=2)
        
        # Add small noise
        np.random.seed(42)
        A = A_true + np.random.normal(0, 0.5, size=len(A_true))
        
        # Create ExtendedDataFrame
        data = pd.DataFrame({'Qin': I, 'A': A})
        exdf = ExtendedDataFrame(data)
        
        # Get initial guess
        guess = initial_guess_light_response(exdf)
        
        # Check parameter ranges
        assert 0 < guess['phi'] < 0.2
        assert 0 < guess['Amax'] < 100
        assert 0 <= guess['theta'] <= 1
        assert 0 <= guess['Rd'] < 10
        
        # Should be reasonably close to true values
        assert guess['phi'] == pytest.approx(0.05, rel=0.5)
        assert guess['Amax'] == pytest.approx(32, rel=0.3)  # Amax + Rd
        assert guess['Rd'] == pytest.approx(2, rel=0.5)
    
    def test_initial_guess_minimal_data(self):
        """Test initial guess with minimal data points."""
        # Minimal data
        I = np.array([0, 500, 1500])
        A = np.array([-1.5, 15, 25])
        
        data = pd.DataFrame({'Qin': I, 'A': A})
        exdf = ExtendedDataFrame(data)
        
        # Should still work
        guess = initial_guess_light_response(exdf)
        
        assert guess['phi'] > 0
        assert guess['Amax'] > 0
        assert guess['Rd'] >= 0


class TestLightResponseFitting:
    """Test light response curve fitting."""
    
    def test_fit_non_rectangular_hyperbola(self):
        """Test fitting non-rectangular hyperbola model."""
        # Generate synthetic data with known parameters
        I = np.array([0, 20, 50, 100, 200, 400, 600, 800, 1000, 1200, 1500, 2000])
        true_params = {'phi': 0.05, 'Amax': 30, 'theta': 0.7, 'Rd': 2}
        A_true = non_rectangular_hyperbola(I, **true_params)
        
        # Add noise
        np.random.seed(42)
        A = A_true + np.random.normal(0, 0.5, size=len(A_true))
        
        # Create ExtendedDataFrame
        data = pd.DataFrame({'Qin': I, 'A': A})
        exdf = ExtendedDataFrame(data)
        
        # Fit model
        result = fit_light_response(exdf, model_type='non_rectangular_hyperbola')
        
        # Check structure
        assert 'parameters' in result
        assert 'statistics' in result
        assert 'predicted' in result
        assert 'residuals' in result
        assert 'convergence' in result
        
        # Check fitted parameters are close to true values
        fitted = result['parameters']
        assert fitted['phi'] == pytest.approx(true_params['phi'], rel=0.2)
        assert fitted['Amax'] == pytest.approx(true_params['Amax'], rel=0.1)
        assert fitted['theta'] == pytest.approx(true_params['theta'], rel=0.3)
        assert fitted['Rd'] == pytest.approx(true_params['Rd'], rel=0.3)
        
        # Check fit quality
        assert result['statistics']['r_squared'] > 0.95
        assert result['statistics']['rmse'] < 1.0
        
        # Check convergence
        assert result['convergence']['success']
    
    def test_fit_with_fixed_parameters(self):
        """Test fitting with fixed parameters."""
        # Generate data
        I = np.linspace(0, 2000, 15)
        A = non_rectangular_hyperbola(I, phi=0.06, Amax=28, theta=0.8, Rd=1.5)
        
        data = pd.DataFrame({'Qin': I, 'A': A})
        exdf = ExtendedDataFrame(data)
        
        # Fix theta
        result = fit_light_response(
            exdf,
            model_type='non_rectangular_hyperbola',
            fixed_parameters={'theta': 0.8}
        )
        
        # Check theta is fixed
        assert result['parameters']['theta'] == 0.8
        
        # Other parameters should still be fitted well
        assert result['parameters']['phi'] == pytest.approx(0.06, rel=0.1)
        assert result['parameters']['Amax'] == pytest.approx(28, rel=0.1)
    
    def test_fit_rectangular_hyperbola(self):
        """Test fitting rectangular hyperbola model."""
        # Generate data with rectangular hyperbola (theta = 0)
        I = np.linspace(0, 2000, 20)
        A = rectangular_hyperbola(I, phi=0.05, Amax=25, Rd=1.8)
        
        # Add small noise
        np.random.seed(42)
        A += np.random.normal(0, 0.3, size=len(A))
        
        data = pd.DataFrame({'Qin': I, 'A': A})
        exdf = ExtendedDataFrame(data)
        
        # Fit model
        result = fit_light_response(exdf, model_type='rectangular_hyperbola')
        
        # Check parameters
        assert 'theta' not in result['parameters']  # No theta for rectangular
        assert result['parameters']['phi'] == pytest.approx(0.05, rel=0.1)
        assert result['parameters']['Amax'] == pytest.approx(25, rel=0.1)
        assert result['parameters']['Rd'] == pytest.approx(1.8, rel=0.2)
    
    def test_light_compensation_point(self):
        """Test calculation of light compensation point."""
        # Generate data where we know LCP
        I = np.linspace(0, 2000, 30)
        phi = 0.05
        Amax = 30
        Rd = 2
        
        # LCP is where A = 0, so phi * I * Amax / (phi * I + Amax) = Rd
        # Solving: I = Rd * Amax / (phi * (Amax - Rd))
        expected_lcp = Rd * Amax / (phi * (Amax - Rd))
        
        A = rectangular_hyperbola(I, phi, Amax, Rd)
        
        data = pd.DataFrame({'Qin': I, 'A': A})
        exdf = ExtendedDataFrame(data)
        
        result = fit_light_response(exdf, model_type='rectangular_hyperbola')
        
        # Check LCP calculation
        lcp = result['statistics']['light_compensation_point']
        assert lcp is not None
        assert lcp == pytest.approx(expected_lcp, rel=0.1)
    
    def test_compare_models(self):
        """Test model comparison functionality."""
        # Generate data that clearly favors non-rectangular hyperbola
        I = np.linspace(0, 2000, 25)
        A = non_rectangular_hyperbola(I, phi=0.05, Amax=30, theta=0.9, Rd=2)
        
        # Add noise
        np.random.seed(42)
        A += np.random.normal(0, 0.5, size=len(A))
        
        data = pd.DataFrame({'Qin': I, 'A': A})
        exdf = ExtendedDataFrame(data)
        
        # Compare models
        comparison = compare_light_models(exdf)
        
        # Check structure
        assert 'comparison_summary' in comparison
        assert 'non_rectangular_hyperbola' in comparison
        assert 'rectangular_hyperbola' in comparison
        assert 'exponential' in comparison
        
        # Check best model selection
        summary = comparison['comparison_summary']
        assert 'best_model' in summary
        assert 'aic_values' in summary
        assert 'delta_aic' in summary
        
        # Best model should have delta_aic = 0
        best_model = summary['best_model']
        assert summary['delta_aic'][best_model] == 0
    
    def test_missing_data_handling(self):
        """Test handling of missing data."""
        # Data with NaN values
        I = np.array([0, 50, np.nan, 200, 400, 800, np.nan, 1500])
        A = np.array([-2, 5, 10, np.nan, 18, 22, 25, np.nan])
        
        data = pd.DataFrame({'Qin': I, 'A': A})
        exdf = ExtendedDataFrame(data)
        
        # Should still fit with valid points
        result = fit_light_response(exdf)
        
        assert result['convergence']['success']
        assert result['statistics']['n_points'] == 4  # Only valid points
        
        # Residuals should have NaN in same places
        assert np.isnan(result['residuals'][2])
        assert np.isnan(result['residuals'][3])
        assert np.isnan(result['residuals'][6])
        assert np.isnan(result['residuals'][7])
    
    def test_insufficient_data_error(self):
        """Test error with insufficient data."""
        # Too few points
        data = pd.DataFrame({'Qin': [0, 500], 'A': [-1, 10]})
        exdf = ExtendedDataFrame(data)
        
        with pytest.raises(ValueError, match="Insufficient valid data points"):
            fit_light_response(exdf)
    
    def test_custom_bounds(self):
        """Test fitting with custom parameter bounds."""
        # Generate typical data
        I = np.linspace(0, 2000, 15)
        A = non_rectangular_hyperbola(I, phi=0.08, Amax=35, theta=0.6, Rd=3)
        
        data = pd.DataFrame({'Qin': I, 'A': A})
        exdf = ExtendedDataFrame(data)
        
        # Set restrictive bounds
        bounds = {
            'phi': (0.07, 0.09),
            'Amax': (30, 40),
            'theta': (0.5, 0.7),
            'Rd': (2.5, 3.5)
        }
        
        result = fit_light_response(
            exdf,
            model_type='non_rectangular_hyperbola',
            bounds=bounds
        )
        
        # Check parameters are within bounds
        params = result['parameters']
        for param, (lower, upper) in bounds.items():
            assert lower <= params[param] <= upper