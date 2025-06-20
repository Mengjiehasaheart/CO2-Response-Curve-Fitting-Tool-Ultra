"""
Tests for temperature response curve fitting module.
"""

import pytest
import numpy as np
import pandas as pd
from aci_py.core.data_structures import ExtendedDataFrame
from aci_py.analysis.temperature_response import (
    gaussian_peak_model,
    quadratic_temperature_response,
    modified_arrhenius_deactivation,
    thermal_performance_curve,
    initial_guess_temperature_response,
    fit_temperature_response,
    fit_arrhenius_with_photogea_params
)


class TestTemperatureResponseModels:
    """Test individual temperature response model functions."""
    
    def test_gaussian_peak_model(self):
        """Test Gaussian peak temperature response."""
        # Test single value
        T = 25
        amplitude = 30
        T_opt = 28
        width = 5
        baseline = 2
        
        value = gaussian_peak_model(T, amplitude, T_opt, width, baseline)
        
        # Should be positive
        assert value > baseline
        
        # Test array
        T_array = np.linspace(10, 40, 30)
        values = gaussian_peak_model(T_array, amplitude, T_opt, width, baseline)
        
        # Check shape
        assert values.shape == T_array.shape
        
        # Peak should be at T_opt
        max_idx = np.argmax(values)
        assert T_array[max_idx] == pytest.approx(T_opt, abs=1)
        
        # Peak value should be amplitude + baseline
        assert values[max_idx] == pytest.approx(amplitude + baseline, rel=0.01)
        
        # Should be symmetric around T_opt
        # Find points equidistant from T_opt
        dist = 5
        idx_below = np.argmin(np.abs(T_array - (T_opt - dist)))
        idx_above = np.argmin(np.abs(T_array - (T_opt + dist)))
        
        assert values[idx_below] == pytest.approx(values[idx_above], rel=0.1)
    
    def test_quadratic_temperature_response(self):
        """Test quadratic temperature response."""
        T = np.array([15, 20, 25, 30, 35])
        a = 5
        b = 2
        c = -0.05
        
        values = quadratic_temperature_response(T, a, b, c)
        
        # Check calculation
        expected = a + b * T + c * T**2
        np.testing.assert_allclose(values, expected)
        
        # With negative c, should have a maximum
        T_opt_calc = -b / (2 * c)
        assert T_opt_calc > 0  # Should be positive temperature
    
    def test_modified_arrhenius_deactivation(self):
        """Test modified Arrhenius model with deactivation."""
        T = np.linspace(10, 40, 30)
        amplitude = 50
        Ea = 65000  # J/mol
        Ed = 200000  # J/mol
        Hd = 650  # J/mol/K
        
        values = modified_arrhenius_deactivation(T, amplitude, Ea, Ed, Hd)
        
        # Should all be positive
        assert np.all(values > 0)
        
        # Should have a peak (not monotonic)
        max_idx = np.argmax(values)
        assert 0 < max_idx < len(T) - 1  # Peak not at edges
        
        # Calculate T_opt analytically
        R = 8.314
        T_opt_K = Ed / (Hd / R - np.log(Ed / Ea))
        T_opt = T_opt_K - 273.15
        
        # Peak should be near calculated T_opt
        assert T[max_idx] == pytest.approx(T_opt, abs=2)
    
    def test_thermal_performance_curve(self):
        """Test thermal performance curve model."""
        T = np.linspace(5, 45, 40)
        T_opt = 28
        T_min = 10
        T_max = 40
        amplitude = 30
        skewness = 0.7
        
        values = thermal_performance_curve(T, T_opt, T_min, T_max, amplitude, skewness)
        
        # Check boundaries
        below_min = T < T_min
        above_max = T > T_max
        assert np.all(values[below_min] == 0)
        assert np.all(values[above_max] == 0)
        
        # Check peak
        valid_mask = (T >= T_min) & (T <= T_max)
        valid_values = values[valid_mask]
        valid_T = T[valid_mask]
        
        max_idx = np.argmax(valid_values)
        assert valid_T[max_idx] == pytest.approx(T_opt, abs=1)
        assert valid_values[max_idx] == pytest.approx(amplitude, rel=0.01)
        
        # Test skewness effect
        # With skewness != 0.5, curve should be asymmetric
        if skewness != 0.5:
            # Find values at equal distances from T_opt
            dist = 5
            T_below = T_opt - dist
            T_above = T_opt + dist
            
            if T_min < T_below < T_opt < T_above < T_max:
                idx_below = np.argmin(np.abs(T - T_below))
                idx_above = np.argmin(np.abs(T - T_above))
                
                # Values should be different due to skewness
                assert values[idx_below] != pytest.approx(values[idx_above], rel=0.1)


class TestInitialGuessTemperature:
    """Test initial parameter estimation for temperature response."""
    
    def test_initial_guess_gaussian(self):
        """Test initial guess for Gaussian model."""
        # Generate synthetic data
        T = np.linspace(15, 35, 10)
        true_params = {'amplitude': 25, 'T_opt': 27, 'width': 6, 'baseline': 5}
        values = gaussian_peak_model(T, **true_params)
        
        # Add small noise
        np.random.seed(42)
        values += np.random.normal(0, 0.5, size=len(values))
        
        # Create ExtendedDataFrame
        data = pd.DataFrame({'Tleaf': T, 'Param': values})
        exdf = ExtendedDataFrame(data)
        
        # Get initial guess
        guess = initial_guess_temperature_response(
            exdf, 'Tleaf', 'Param', 'gaussian_peak'
        )
        
        # Check reasonable values
        assert guess['T_opt'] == pytest.approx(true_params['T_opt'], abs=2)
        assert guess['amplitude'] == pytest.approx(true_params['amplitude'], rel=0.2)
        assert guess['width'] > 0
        assert guess['baseline'] >= 0
    
    def test_initial_guess_thermal_performance(self):
        """Test initial guess for thermal performance curve."""
        # Generate data
        T = np.linspace(10, 40, 15)
        values = thermal_performance_curve(T, T_opt=28, T_min=12, T_max=38, 
                                         amplitude=30, skewness=0.6)
        
        data = pd.DataFrame({'Tleaf': T, 'Param': values})
        exdf = ExtendedDataFrame(data)
        
        guess = initial_guess_temperature_response(
            exdf, 'Tleaf', 'Param', 'thermal_performance'
        )
        
        # Check structure
        assert 'T_opt' in guess
        assert 'T_min' in guess
        assert 'T_max' in guess
        assert 'amplitude' in guess
        assert 'skewness' in guess
        
        # Check reasonable ranges
        assert 10 <= guess['T_opt'] <= 40
        assert guess['T_min'] < guess['T_opt']
        assert guess['T_opt'] < guess['T_max']
        assert guess['amplitude'] > 0


class TestTemperatureResponseFitting:
    """Test temperature response curve fitting."""
    
    def test_fit_gaussian_peak(self):
        """Test fitting Gaussian peak model."""
        # Generate synthetic data
        T = np.linspace(10, 40, 12)
        true_params = {'amplitude': 30, 'T_opt': 26, 'width': 7, 'baseline': 3}
        values_true = gaussian_peak_model(T, **true_params)
        
        # Add noise
        np.random.seed(42)
        values = values_true + np.random.normal(0, 1, size=len(values_true))
        
        # Create ExtendedDataFrame
        data = pd.DataFrame({'Tleaf': T, 'Vcmax': values})
        exdf = ExtendedDataFrame(data)
        
        # Fit model
        result = fit_temperature_response(
            exdf, 'Tleaf', 'Vcmax', model_type='gaussian_peak'
        )
        
        # Check structure
        assert 'parameters' in result
        assert 'statistics' in result
        assert 'T_opt' in result
        assert 'performance_range' in result
        
        # Check fitted parameters
        fitted = result['parameters']
        assert fitted['T_opt'] == pytest.approx(true_params['T_opt'], abs=1)
        assert fitted['amplitude'] == pytest.approx(true_params['amplitude'], rel=0.2)
        
        # Check fit quality
        assert result['statistics']['r_squared'] > 0.9
        
        # Check T_opt calculation
        assert result['T_opt'] == pytest.approx(true_params['T_opt'], abs=1)
    
    def test_fit_modified_arrhenius(self):
        """Test fitting modified Arrhenius model."""
        # Generate data
        T = np.linspace(15, 35, 10)
        values = modified_arrhenius_deactivation(
            T, amplitude=60, Ea=65000, Ed=200000, Hd=650
        )
        
        # Add small noise
        np.random.seed(42)
        values += np.random.normal(0, 1, size=len(values))
        
        data = pd.DataFrame({'Tleaf': T, 'Vcmax': values})
        exdf = ExtendedDataFrame(data)
        
        # Fit model
        result = fit_temperature_response(
            exdf, 'Tleaf', 'Vcmax', model_type='modified_arrhenius'
        )
        
        # Check convergence
        assert result['convergence']['success']
        
        # Check parameters are reasonable
        params = result['parameters']
        assert 10000 < params['Ea'] < 100000  # Activation energy
        assert 100000 < params['Ed'] < 300000  # Deactivation energy
        assert 100 < params['Hd'] < 1000  # Entropy term
        
        # Check T_opt is reasonable
        assert 20 < result['T_opt'] < 35
    
    def test_fit_thermal_performance(self):
        """Test fitting thermal performance curve."""
        # Generate data
        T = np.linspace(8, 42, 15)
        true_params = {
            'T_opt': 27, 'T_min': 10, 'T_max': 40, 
            'amplitude': 35, 'skewness': 0.6
        }
        values = thermal_performance_curve(T, **true_params)
        
        # Add noise
        np.random.seed(42)
        values += np.random.normal(0, 0.5, size=len(values))
        
        data = pd.DataFrame({'Tleaf': T, 'A': values})
        exdf = ExtendedDataFrame(data)
        
        # Fit model
        result = fit_temperature_response(
            exdf, 'Tleaf', 'A', model_type='thermal_performance'
        )
        
        # Check fitted parameters
        fitted = result['parameters']
        assert fitted['T_opt'] == pytest.approx(true_params['T_opt'], abs=1)
        assert fitted['T_min'] == pytest.approx(true_params['T_min'], abs=2)
        assert fitted['T_max'] == pytest.approx(true_params['T_max'], abs=2)
        
        # Check performance range calculation
        T_90_min, T_90_max = result['performance_range']
        assert true_params['T_min'] < T_90_min < true_params['T_opt']
        assert true_params['T_opt'] < T_90_max < true_params['T_max']
    
    def test_fit_with_weights(self):
        """Test weighted fitting."""
        # Generate data with variable noise
        T = np.linspace(15, 35, 10)
        values_true = gaussian_peak_model(T, amplitude=30, T_opt=25, width=5, baseline=5)
        
        # Add heteroscedastic noise
        np.random.seed(42)
        noise_std = 0.1 + 0.05 * np.abs(T - 25)  # More noise away from optimum
        values = values_true + np.random.normal(0, noise_std)
        
        # Weights inversely proportional to variance
        weights = 1 / noise_std**2
        
        data = pd.DataFrame({'Tleaf': T, 'Param': values})
        exdf = ExtendedDataFrame(data)
        
        # Fit with weights
        result = fit_temperature_response(
            exdf, 'Tleaf', 'Param', 
            model_type='gaussian_peak',
            weights=weights
        )
        
        # Should converge successfully
        assert result['convergence']['success']
        
        # T_opt should be well estimated despite variable noise
        assert result['T_opt'] == pytest.approx(25, abs=1)
    
    def test_q10_calculation(self):
        """Test Q10 temperature coefficient calculation."""
        # Generate data with known Q10
        T = np.array([10, 15, 20, 25, 30, 35])
        # Simple exponential with Q10 = 2
        values = 20 * 2**((T - 15) / 10)
        
        data = pd.DataFrame({'Tleaf': T, 'Param': values})
        exdf = ExtendedDataFrame(data)
        
        # Fit quadratic (will approximate exponential locally)
        result = fit_temperature_response(
            exdf, 'Tleaf', 'Param', model_type='quadratic'
        )
        
        # Check Q10 calculation
        if result.get('Q10') is not None:
            # Should be approximately 2
            assert result['Q10'] == pytest.approx(2, rel=0.3)
    
    def test_arrhenius_photogea_fitting(self):
        """Test PhotoGEA-compatible Arrhenius fitting."""
        # Generate data with known Arrhenius response
        T = np.array([15, 18, 21, 24, 27, 30, 33])
        Vcmax_25 = 55
        Ha = 65330
        
        # Calculate using Arrhenius function
        R = 8.314  # J/mol/K
        T_K = T + 273.15
        T_ref_K = 25 + 273.15
        values_true = Vcmax_25 * np.exp((Ha / R) * (1/T_ref_K - 1/T_K))
        
        # Add small noise
        np.random.seed(42)
        values = values_true + np.random.normal(0, 1, size=len(values_true))
        
        data = pd.DataFrame({'Tleaf': T, 'Vcmax': values})
        exdf = ExtendedDataFrame(data)
        
        # Fit with Arrhenius
        result = fit_arrhenius_with_photogea_params(
            exdf, 'Tleaf', 'Vcmax', temperature_params='bernacchi'
        )
        
        # Check success
        assert result['success']
        
        # Check fitted parameters
        assert result['parameters']['Vcmax_at_25'] == pytest.approx(Vcmax_25, rel=0.05)
        assert result['parameters']['Ha_Vcmax'] == pytest.approx(Ha, rel=0.1)
        
        # Check errors are provided
        assert 'parameter_errors' in result
        assert 'Vcmax_at_25' in result['parameter_errors']
        assert 'Ha_Vcmax' in result['parameter_errors']
    
    def test_missing_data_handling(self):
        """Test handling of missing temperature data."""
        # Data with NaN values
        T = np.array([15, 20, np.nan, 25, 30, np.nan, 35])
        values = np.array([10, 18, 22, np.nan, 28, 20, np.nan])
        
        data = pd.DataFrame({'Tleaf': T, 'Param': values})
        exdf = ExtendedDataFrame(data)
        
        # Should still fit with valid points
        result = fit_temperature_response(
            exdf, 'Tleaf', 'Param', model_type='gaussian_peak'
        )
        
        assert result['convergence']['success']
        assert result['statistics']['n_points'] == 3  # Only valid points
    
    def test_insufficient_data_error(self):
        """Test error with insufficient data."""
        # Too few points
        data = pd.DataFrame({'Tleaf': [20, 25], 'Param': [10, 15]})
        exdf = ExtendedDataFrame(data)
        
        with pytest.raises(ValueError, match="Insufficient valid data points"):
            fit_temperature_response(exdf, 'Tleaf', 'Param')
    
    def test_model_specific_bounds(self):
        """Test model-specific parameter bounds."""
        # Generate simple data
        T = np.linspace(15, 35, 10)
        values = gaussian_peak_model(T, amplitude=30, T_opt=25, width=5, baseline=0)
        
        data = pd.DataFrame({'Tleaf': T, 'Param': values})
        exdf = ExtendedDataFrame(data)
        
        # Test with custom bounds
        bounds = {
            'T_opt': (20, 30),
            'amplitude': (25, 35),
            'width': (3, 7),
            'baseline': (-2, 2)
        }
        
        result = fit_temperature_response(
            exdf, 'Tleaf', 'Param',
            model_type='gaussian_peak',
            bounds=bounds
        )
        
        # Check parameters are within bounds
        params = result['parameters']
        for param, (lower, upper) in bounds.items():
            assert lower <= params[param] <= upper