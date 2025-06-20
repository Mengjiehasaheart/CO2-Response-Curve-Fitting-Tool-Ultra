"""
Tests for confidence interval calculations.
"""

import numpy as np
import pytest
from aci_py.analysis.optimization import (
    calculate_confidence_intervals_profile,
    calculate_confidence_intervals_bootstrap,
    negative_log_likelihood,
    create_error_function
)


class TestConfidenceIntervals:
    """Test confidence interval calculation methods."""
    
    def test_profile_likelihood_simple(self):
        """Test profile likelihood CI with simple quadratic function."""
        # Simple quadratic: f(x) = (x - 2)^2
        def error_func(params):
            return (params[0] - 2)**2
        
        best_params = np.array([2.0])
        param_names = ['x']
        bounds = [(0, 4)]
        
        ci_result = calculate_confidence_intervals_profile(
            error_func,
            best_params,
            param_names,
            bounds,
            confidence_level=0.95,
            n_points=50
        )
        
        # For quadratic, 95% CI should be approximately ±1.96
        assert 'x' in ci_result
        lower, upper = ci_result['x']
        assert lower < 2.0 < upper
        # Should be roughly symmetric for quadratic
        assert abs((2.0 - lower) - (upper - 2.0)) < 0.5
    
    def test_profile_likelihood_multi_param(self):
        """Test profile likelihood CI with multiple parameters."""
        # f(x, y) = (x - 1)^2 + (y - 2)^2
        def error_func(params):
            return (params[0] - 1)**2 + (params[1] - 2)**2
        
        best_params = np.array([1.0, 2.0])
        param_names = ['x', 'y']
        bounds = [(-5, 5), (-5, 5)]
        
        ci_result = calculate_confidence_intervals_profile(
            error_func,
            best_params,
            param_names,
            bounds,
            confidence_level=0.95,
            n_points=30
        )
        
        assert 'x' in ci_result
        assert 'y' in ci_result
        
        # Check that CIs contain true values
        assert ci_result['x'][0] < 1.0 < ci_result['x'][1]
        assert ci_result['y'][0] < 2.0 < ci_result['y'][1]
    
    def test_bootstrap_confidence_intervals(self):
        """Test bootstrap CI calculation."""
        # Generate synthetic data
        np.random.seed(42)
        n_obs = 50
        true_slope = 2.0
        true_intercept = 1.0
        
        x = np.linspace(0, 10, n_obs)
        y_true = true_slope * x + true_intercept
        y_obs = y_true + np.random.normal(0, 0.5, n_obs)
        
        # Model function
        def model_func(params, data):
            return params['slope'] * data['x'] + params['intercept']
        
        # Data dictionary
        data = {'x': x, 'A': y_obs}
        
        # Best parameters (from a hypothetical fit)
        best_params = {'slope': 2.1, 'intercept': 0.9}
        param_names = ['slope', 'intercept']
        bounds = [(0, 5), (-5, 5)]
        
        ci_result = calculate_confidence_intervals_bootstrap(
            model_func,
            data,
            best_params,
            param_names,
            bounds,
            n_bootstrap=50,  # Reduced for faster test
            confidence_level=0.95,
            seed=42
        )
        
        # Check that CIs are reasonable
        assert 'slope' in ci_result
        assert 'intercept' in ci_result
        
        # Check that CIs are finite and reasonable
        assert not np.isnan(ci_result['slope'][0])
        assert not np.isnan(ci_result['slope'][1]) 
        assert not np.isnan(ci_result['intercept'][0])
        assert not np.isnan(ci_result['intercept'][1])
        
        # CIs should have reasonable width (not too narrow or wide)
        slope_width = ci_result['slope'][1] - ci_result['slope'][0]
        intercept_width = ci_result['intercept'][1] - ci_result['intercept'][0]
        assert 0.1 < slope_width < 2.0  # Reasonable for this problem
        assert 0.1 < intercept_width < 2.0
    
    def test_confidence_intervals_at_bounds(self):
        """Test CI calculation when parameter is at bounds."""
        # Function with minimum at boundary
        def error_func(params):
            return params[0]**2  # Minimum at x=0
        
        best_params = np.array([0.1])  # Near lower bound
        param_names = ['x']
        bounds = [(0, 10)]
        
        ci_result = calculate_confidence_intervals_profile(
            error_func,
            best_params,
            param_names,
            bounds,
            confidence_level=0.95,
            n_points=20
        )
        
        # Lower bound should be at parameter bound
        assert ci_result['x'][0] == bounds[0][0]
        assert ci_result['x'][1] > best_params[0]
    
    def test_different_confidence_levels(self):
        """Test CI calculation with different confidence levels."""
        def error_func(params):
            return (params[0] - 3)**2
        
        best_params = np.array([3.0])
        param_names = ['x']
        bounds = [(0, 6)]
        
        # Calculate CIs at different levels
        ci_68 = calculate_confidence_intervals_profile(
            error_func, best_params, param_names, bounds,
            confidence_level=0.68, n_points=30
        )
        
        ci_95 = calculate_confidence_intervals_profile(
            error_func, best_params, param_names, bounds,
            confidence_level=0.95, n_points=30
        )
        
        # 95% CI should be wider than 68% CI
        width_68 = ci_68['x'][1] - ci_68['x'][0]
        width_95 = ci_95['x'][1] - ci_95['x'][0]
        assert width_95 > width_68