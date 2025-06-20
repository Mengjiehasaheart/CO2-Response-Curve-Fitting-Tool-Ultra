"""
Tests for temperature response functions.

Validates temperature response calculations against known values and
ensures consistency with PhotoGEA R package.
"""

import numpy as np
import pytest
from aci_py.core.temperature import (
    arrhenius_response,
    johnson_eyring_williams_response,
    gaussian_response,
    polynomial_response,
    calculate_temperature_response,
    apply_temperature_response,
    TemperatureParameter,
    C3_TEMPERATURE_PARAM_BERNACCHI,
    C3_TEMPERATURE_PARAM_SHARKEY,
    C3_TEMPERATURE_PARAM_FLAT,
    IDEAL_GAS_CONSTANT,
    F_CONST
)


class TestArrheniusResponse:
    """Test Arrhenius temperature response function."""
    
    def test_basic_calculation(self):
        """Test basic Arrhenius calculation."""
        # At 25°C, should return exp(scaling)
        result = arrhenius_response(scaling=1.0, activation_energy=0, temperature_c=25.0)
        assert np.isclose(result, np.exp(1.0))
        
        # Test with actual activation energy
        result = arrhenius_response(scaling=0, activation_energy=50, temperature_c=25.0)
        assert 0 < result < 1
    
    def test_temperature_array(self):
        """Test with array of temperatures."""
        temps = np.array([15, 20, 25, 30, 35])
        results = arrhenius_response(scaling=0, activation_energy=50, temperature_c=temps)
        
        # Should increase with temperature
        assert np.all(np.diff(results) > 0)
        assert results.shape == temps.shape
    
    def test_vcmax_temperature_response(self):
        """Test Vcmax temperature response using Bernacchi parameters."""
        # Get Vcmax normalization at different temperatures
        temps = np.array([15, 20, 25, 30, 35])
        vcmax_param = C3_TEMPERATURE_PARAM_BERNACCHI['Vcmax_norm']
        
        results = arrhenius_response(vcmax_param.c, vcmax_param.Ea, temps)
        
        # At 25°C, normalization should be close to 1.0
        # Note: Vcmax uses c=26.35 (not c=Ea/f), so it's not exactly 1.0
        idx_25 = np.where(temps == 25)[0][0]
        assert np.isclose(results[idx_25], 0.9963, rtol=1e-3)
        
        # Should increase with temperature
        assert np.all(np.diff(results) > 0)


class TestJohnsonResponse:
    """Test Johnson-Eyring-Williams temperature response function."""
    
    def test_basic_calculation(self):
        """Test basic Johnson calculation."""
        result = johnson_eyring_williams_response(
            scaling=20.01,
            activation_enthalpy=49.6,
            deactivation_enthalpy=437.4,
            entropy=1.4,
            temperature_c=25.0
        )
        # At 25°C with Bernacchi gmc parameters, should be close to 1.0
        assert np.isclose(result, 1.0, rtol=0.01)
    
    def test_temperature_dependence(self):
        """Test temperature dependence shows optimum."""
        temps = np.linspace(0, 50, 51)
        gmc_param = C3_TEMPERATURE_PARAM_BERNACCHI['gmc_norm']
        
        results = johnson_eyring_williams_response(
            gmc_param.c, gmc_param.Ha, gmc_param.Hd, gmc_param.S, temps
        )
        
        # Should have a maximum (optimum temperature)
        max_idx = np.argmax(results)
        assert 20 < temps[max_idx] < 40  # Optimum should be reasonable
        
        # Should decrease at very high temperatures
        assert results[-1] < results[max_idx]


class TestGaussianResponse:
    """Test Gaussian temperature response function."""
    
    def test_basic_calculation(self):
        """Test basic Gaussian calculation."""
        # At optimum, should return optimum_rate
        result = gaussian_response(
            optimum_rate=100,
            t_opt=25,
            sigma=10,
            temperature_c=25
        )
        assert result == 100
        
        # Away from optimum, should be less
        result = gaussian_response(
            optimum_rate=100,
            t_opt=25,
            sigma=10,
            temperature_c=35
        )
        assert result < 100
    
    def test_symmetry(self):
        """Test Gaussian response is symmetric."""
        optimum_rate = 100
        t_opt = 25
        sigma = 10
        
        # Same distance above and below optimum should give same result
        result_below = gaussian_response(optimum_rate, t_opt, sigma, 15)
        result_above = gaussian_response(optimum_rate, t_opt, sigma, 35)
        
        assert np.isclose(result_below, result_above)


class TestPolynomialResponse:
    """Test polynomial temperature response function."""
    
    def test_constant_polynomial(self):
        """Test constant (0th order) polynomial."""
        result = polynomial_response(coefficients=42.0, temperature_c=25.0)
        assert result == 42.0
        
        # Should be same for any temperature
        temps = np.array([10, 20, 30])
        results = polynomial_response(coefficients=42.0, temperature_c=temps)
        assert np.all(results == 42.0)
    
    def test_linear_polynomial(self):
        """Test linear polynomial."""
        # y = 2 + 3*x
        coeffs = [2.0, 3.0]
        result = polynomial_response(coeffs, 10.0)
        assert result == 2.0 + 3.0 * 10.0
    
    def test_quadratic_polynomial(self):
        """Test quadratic polynomial."""
        # y = 1 + 2*x + 3*x^2
        coeffs = [1.0, 2.0, 3.0]
        x = 5.0
        expected = 1.0 + 2.0 * x + 3.0 * x**2
        result = polynomial_response(coeffs, x)
        assert np.isclose(result, expected)


class TestCalculateTemperatureResponse:
    """Test unified temperature response calculation."""
    
    def test_arrhenius_type(self):
        """Test calculation with Arrhenius type."""
        param = TemperatureParameter(
            type='arrhenius',
            c=26.35,
            Ea=65.33,
            units='normalized'
        )
        result = calculate_temperature_response(param, 25.0)
        assert np.isclose(result, 1.0, rtol=0.01)
    
    def test_johnson_type(self):
        """Test calculation with Johnson type."""
        param = TemperatureParameter(
            type='johnson',
            c=20.01,
            Ha=49.6,
            Hd=437.4,
            S=1.4,
            units='normalized'
        )
        result = calculate_temperature_response(param, 25.0)
        assert np.isclose(result, 1.0, rtol=0.01)
    
    def test_gaussian_type(self):
        """Test calculation with Gaussian type."""
        param = TemperatureParameter(
            type='gaussian',
            optimum_rate=100,
            t_opt=30,
            sigma=10,
            units='test units'
        )
        result = calculate_temperature_response(param, 30.0)
        assert result == 100
    
    def test_polynomial_type(self):
        """Test calculation with polynomial type."""
        param = TemperatureParameter(
            type='polynomial',
            coef=[10, 2],
            units='test units'
        )
        result = calculate_temperature_response(param, 5.0)
        assert result == 10 + 2 * 5
    
    def test_invalid_type(self):
        """Test error with invalid type."""
        param = TemperatureParameter(
            type='invalid',
            units='test'
        )
        with pytest.raises(ValueError, match="Unknown temperature response type"):
            calculate_temperature_response(param, 25.0)
    
    def test_missing_parameters(self):
        """Test error with missing required parameters."""
        # Missing Ea for Arrhenius
        param = TemperatureParameter(
            type='arrhenius',
            c=1.0,
            units='test'
        )
        with pytest.raises(ValueError, match="Arrhenius parameters require"):
            calculate_temperature_response(param, 25.0)


class TestApplyTemperatureResponse:
    """Test applying temperature responses to multiple parameters."""
    
    def test_normalized_parameters(self):
        """Test application of normalized temperature responses."""
        base_params = {
            'Vcmax': 100.0,
            'J': 200.0,
            'RL': 2.0
        }
        
        temp_params = {
            'Vcmax_norm': TemperatureParameter(
                type='arrhenius', c=26.35, Ea=65.33, units='normalized'
            ),
            'J_norm': TemperatureParameter(
                type='arrhenius', c=17.57, Ea=43.5, units='normalized'
            ),
            'RL_norm': TemperatureParameter(
                type='arrhenius', c=18.72, Ea=46.39, units='normalized'
            )
        }
        
        # At 25°C, parameters should be close to base values
        # Note: Parameters don't normalize to exactly 1.0 due to their c values
        adjusted = apply_temperature_response(base_params, temp_params, 25.0)
        
        # Expected values based on actual normalization factors at 25°C
        assert np.isclose(adjusted['Vcmax'], 100.0 * 0.9963, rtol=1e-3)
        assert np.isclose(adjusted['J'], 200.0 * 1.0226, rtol=1e-3)
        assert np.isclose(adjusted['RL'], 2.0 * 1.0066, rtol=1e-3)
        
        # At higher temperature, values should increase
        adjusted_35 = apply_temperature_response(base_params, temp_params, 35.0)
        
        assert adjusted_35['Vcmax'] > adjusted['Vcmax']
        assert adjusted_35['J'] > adjusted['J']
        assert adjusted_35['RL'] > adjusted['RL']
    
    def test_absolute_parameters(self):
        """Test application of absolute temperature responses."""
        base_params = {}  # No base values needed for absolute params
        
        temp_params = {
            'Gamma_star_at_25': TemperatureParameter(
                type='polynomial', coef=42.93205, units='micromol mol^(-1)'
            ),
            'Kc_at_25': TemperatureParameter(
                type='polynomial', coef=406.8494, units='micromol mol^(-1)'
            )
        }
        
        adjusted = apply_temperature_response(base_params, temp_params, 25.0)
        
        assert 'Gamma_star' in adjusted
        assert 'Kc' in adjusted
        assert adjusted['Gamma_star'] == 42.93205
        assert adjusted['Kc'] == 406.8494
    
    def test_mixed_parameters(self):
        """Test mix of normalized and absolute parameters."""
        base_params = {'Vcmax': 100.0}
        
        temp_params = {
            'Vcmax_norm': TemperatureParameter(
                type='arrhenius', c=26.35, Ea=65.33, units='normalized'
            ),
            'Gamma_star_at_25': TemperatureParameter(
                type='polynomial', coef=42.93205, units='micromol mol^(-1)'
            )
        }
        
        adjusted = apply_temperature_response(base_params, temp_params, 30.0)
        
        assert 'Vcmax' in adjusted
        assert 'Gamma_star' in adjusted
        assert adjusted['Vcmax'] > 100.0  # Should increase with temperature
        assert adjusted['Gamma_star'] == 42.93205  # Constant polynomial


class TestParameterSets:
    """Test predefined parameter sets."""
    
    def test_bernacchi_parameters(self):
        """Test Bernacchi parameter set structure."""
        assert 'Vcmax_norm' in C3_TEMPERATURE_PARAM_BERNACCHI
        assert 'Gamma_star_at_25' in C3_TEMPERATURE_PARAM_BERNACCHI
        
        vcmax_param = C3_TEMPERATURE_PARAM_BERNACCHI['Vcmax_norm']
        assert vcmax_param.type == 'arrhenius'
        assert vcmax_param.Ea == 65.33
        
        gamma_param = C3_TEMPERATURE_PARAM_BERNACCHI['Gamma_star_at_25']
        assert gamma_param.type == 'polynomial'
        assert gamma_param.coef == 42.93205
    
    def test_sharkey_parameters(self):
        """Test Sharkey parameter set structure."""
        assert 'Vcmax_norm' in C3_TEMPERATURE_PARAM_SHARKEY
        
        # Should have different values than Bernacchi
        vcmax_bernacchi = C3_TEMPERATURE_PARAM_BERNACCHI['Gamma_star_norm']
        vcmax_sharkey = C3_TEMPERATURE_PARAM_SHARKEY['Gamma_star_norm']
        
        assert vcmax_bernacchi.Ea != vcmax_sharkey.Ea
    
    def test_flat_parameters(self):
        """Test flat (no temperature response) parameter set."""
        # All normalized parameters should have Ea = 0
        for key, param in C3_TEMPERATURE_PARAM_FLAT.items():
            if key.endswith('_norm'):
                assert param.Ea == 0
                assert param.c == 0
    
    def test_parameter_consistency(self):
        """Test consistency across parameter sets."""
        # All sets should have the same parameter names
        bernacchi_keys = set(C3_TEMPERATURE_PARAM_BERNACCHI.keys())
        sharkey_keys = set(C3_TEMPERATURE_PARAM_SHARKEY.keys())
        flat_keys = set(C3_TEMPERATURE_PARAM_FLAT.keys())
        
        # Sharkey has all Bernacchi parameters except Vomax_norm
        assert 'Vomax_norm' in bernacchi_keys
        assert 'Vomax_norm' not in sharkey_keys
        bernacchi_keys.remove('Vomax_norm')
        
        assert bernacchi_keys == sharkey_keys == flat_keys


if __name__ == '__main__':
    pytest.main([__file__, '-v'])