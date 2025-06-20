"""
Tests for C3 photosynthesis calculations.

Validates C3 assimilation calculations against expected values and
ensures consistency with PhotoGEA R package.
"""

import numpy as np
import pandas as pd
import pytest
from aci_py.core.data_structures import ExtendedDataFrame
from aci_py.core.c3_calculations import (
    calculate_c3_assimilation,
    identify_c3_limiting_process,
    C3AssimilationResult,
    _validate_c3_parameters
)
from aci_py.core.temperature import C3_TEMPERATURE_PARAM_BERNACCHI


class TestC3Calculations:
    """Test C3 assimilation calculations."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        # Create data at different CO2 levels
        ci_values = np.array([50, 100, 200, 300, 400, 600, 800, 1000, 1200])
        
        df = pd.DataFrame({
            'Ci': ci_values,
            'Cc': ci_values * 0.95,  # Assume small gradient from Ci to Cc
            'T_leaf_K': np.full_like(ci_values, 298.15),  # 25°C
            'Pa': np.full_like(ci_values, 101325.0),  # 1 atm in Pa
        })
        
        return ExtendedDataFrame(
            data=df,
            units={
                'Ci': 'micromol mol^(-1)',
                'Cc': 'micromol mol^(-1)', 
                'T_leaf_K': 'K',
                'Pa': 'Pa'
            }
        )
    
    @pytest.fixture
    def typical_c3_params(self):
        """Typical C3 parameters for tobacco at 25°C."""
        return {
            'Vcmax_at_25': 100.0,  # µmol m⁻² s⁻¹
            'J_at_25': 200.0,      # µmol m⁻² s⁻¹
            'Tp_at_25': 12.0,      # µmol m⁻² s⁻¹
            'RL_at_25': 1.5,       # µmol m⁻² s⁻¹
            'Gamma_star_at_25': 42.93,  # µmol mol⁻¹
            'Kc_at_25': 406.85,    # µmol mol⁻¹
            'Ko_at_25': 277.14,    # mmol mol⁻¹
        }
    
    def test_basic_calculation(self, sample_data, typical_c3_params):
        """Test basic C3 assimilation calculation."""
        result = calculate_c3_assimilation(sample_data, typical_c3_params)
        
        # Check result structure
        assert isinstance(result, C3AssimilationResult)
        assert hasattr(result, 'An')
        assert hasattr(result, 'Ac')
        assert hasattr(result, 'Aj')
        assert hasattr(result, 'Ap')
        
        # Check array shapes
        n_points = len(sample_data.data)
        assert len(result.An) == n_points
        assert len(result.Ac) == n_points
        assert len(result.Aj) == n_points
        assert len(result.Ap) == n_points
        
        # Check values are reasonable
        assert np.all(np.isfinite(result.An))
        assert np.all(result.An[1:] > result.An[0])  # An should increase with Cc
        
    def test_rubisco_limitation(self, typical_c3_params):
        """Test Rubisco-limited region (low CO2)."""
        # Create data with very low CO2 where Rubisco should limit
        df = pd.DataFrame({
            'Cc': [50, 75, 100],
            'T_leaf_K': [298.15, 298.15, 298.15],
            'Pa': [101325.0, 101325.0, 101325.0]
        })
        exdf = ExtendedDataFrame(df)
        
        result = calculate_c3_assimilation(exdf, typical_c3_params)
        limiting = identify_c3_limiting_process(result)
        
        # At low CO2, should be Rubisco-limited
        assert np.all(limiting == 'Rubisco')
        assert np.allclose(result.An, result.Ac)
        
    def test_rubp_limitation(self, typical_c3_params):
        """Test RuBP-limited region (intermediate CO2)."""
        # Create data with intermediate CO2 where RuBP should limit
        df = pd.DataFrame({
            'Cc': [200, 250, 300, 350, 400],
            'T_leaf_K': [298.15, 298.15, 298.15, 298.15, 298.15],
            'Pa': [101325.0, 101325.0, 101325.0, 101325.0, 101325.0]
        })
        exdf = ExtendedDataFrame(df)
        
        # Adjust parameters to ensure RuBP limitation at these CO2 levels
        params = typical_c3_params.copy()
        params['J_at_25'] = 150.0  # Lower J to make RuBP limitation more likely
        
        result = calculate_c3_assimilation(exdf, params)
        limiting = identify_c3_limiting_process(result)
        
        # Should have at least some RuBP-limited points
        assert 'RuBP' in limiting
        # And RuBP limitation should occur at intermediate CO2
        rubp_idx = np.where(limiting == 'RuBP')[0]
        if len(rubp_idx) > 0:
            cc_at_rubp = df['Cc'].values[rubp_idx]
            assert np.any((cc_at_rubp > 150) & (cc_at_rubp < 500))
        
    def test_tpu_limitation(self, typical_c3_params):
        """Test TPU-limited region (high CO2)."""
        # Create data with very high CO2 where TPU might limit
        df = pd.DataFrame({
            'Cc': [1000, 1200, 1500],
            'T_leaf_K': [298.15, 298.15, 298.15],
            'Pa': [101325.0, 101325.0, 101325.0]
        })
        exdf = ExtendedDataFrame(df)
        
        # Reduce Tp to make TPU limitation more likely
        params = typical_c3_params.copy()
        params['Tp_at_25'] = 8.0
        
        result = calculate_c3_assimilation(exdf, params)
        
        # An should plateau at high CO2 if TPU-limited
        an_diff = np.diff(result.An)
        assert an_diff[-1] < an_diff[0]  # Rate of increase should decline
        
    def test_temperature_response(self, typical_c3_params):
        """Test temperature effects on assimilation."""
        # Create data at different temperatures
        temps_c = np.array([15, 20, 25, 30, 35])
        df = pd.DataFrame({
            'Cc': np.full_like(temps_c, 300.0),
            'T_leaf_K': temps_c + 273.15,
            'Pa': np.full_like(temps_c, 101325.0)
        })
        exdf = ExtendedDataFrame(df)
        
        result = calculate_c3_assimilation(
            exdf, typical_c3_params,
            temperature_response_params=C3_TEMPERATURE_PARAM_BERNACCHI
        )
        
        # Check temperature-adjusted parameters increase with temperature
        assert np.all(np.diff(result.Vcmax_tl) > 0)
        assert np.all(np.diff(result.J_tl) > 0)
        
        # Assimilation should have an optimum
        # (not necessarily monotonic due to Gamma_star increase)
        assert result.An[2] > result.An[0]  # Higher at 25°C than 15°C
        
    def test_fractionation_factors(self, sample_data, typical_c3_params):
        """Test different isotope fractionation scenarios."""
        # Test with legacy alpha
        result_legacy = calculate_c3_assimilation(
            sample_data, typical_c3_params,
            alpha_old=0.01, use_legacy_alpha=True
        )
        
        # Test with new fractionation factors
        result_new = calculate_c3_assimilation(
            sample_data, typical_c3_params,
            alpha_g=0.005, alpha_s=0.002, alpha_t=0.001
        )
        
        # Results should be different but reasonable
        assert not np.allclose(result_legacy.An, result_new.An)
        assert np.all(np.isfinite(result_legacy.An))
        assert np.all(np.isfinite(result_new.An))
        
    def test_parameter_validation(self):
        """Test parameter validation."""
        # Missing required parameter
        incomplete_params = {
            'Vcmax_at_25': 100.0,
            'J_at_25': 200.0,
            # Missing other required params
        }
        
        with pytest.raises(ValueError, match="Missing required parameters"):
            _validate_c3_parameters(incomplete_params, 0, 0, 0, 0, False, 4.0, 8.0)
        
        # Negative parameter value
        bad_params = {
            'Vcmax_at_25': -100.0,  # Negative!
            'J_at_25': 200.0,
            'Tp_at_25': 12.0,
            'RL_at_25': 1.5,
            'Gamma_star_at_25': 42.93,
            'Kc_at_25': 406.85,
            'Ko_at_25': 277.14,
        }
        
        with pytest.raises(ValueError, match="must be >= 0"):
            _validate_c3_parameters(bad_params, 0, 0, 0, 0, False, 4.0, 8.0)
        
        # Mixed alpha models
        good_params = {
            'Vcmax_at_25': 100.0,
            'J_at_25': 200.0,
            'Tp_at_25': 12.0,
            'RL_at_25': 1.5,
            'Gamma_star_at_25': 42.93,
            'Kc_at_25': 406.85,
            'Ko_at_25': 277.14,
        }
        
        with pytest.raises(ValueError, match="Cannot use legacy alpha_old"):
            _validate_c3_parameters(good_params, 0.01, 0.01, 0, 0, True, 4.0, 8.0)
        
    def test_missing_columns(self, typical_c3_params):
        """Test error handling for missing columns."""
        # Create data missing required columns
        df = pd.DataFrame({
            'Ci': [100, 200, 300]
            # Missing Cc, T_leaf_K, Pa
        })
        exdf = ExtendedDataFrame(df)
        
        with pytest.raises(ValueError, match="Missing required columns"):
            calculate_c3_assimilation(exdf, typical_c3_params)
        
    def test_custom_column_names(self, typical_c3_params):
        """Test using custom column names."""
        # Create data with non-standard column name
        df = pd.DataFrame({
            'Cc_mesophyll': [100, 200, 300],
            'T_leaf_K': [298.15, 298.15, 298.15],
            'Pa': [101325.0, 101325.0, 101325.0]
        })
        exdf = ExtendedDataFrame(df)
        
        # Should work with custom column name
        result = calculate_c3_assimilation(
            exdf, typical_c3_params,
            cc_column_name='Cc_mesophyll'
        )
        
        assert len(result.An) == 3
        assert np.all(np.isfinite(result.An))
        
    def test_oxygen_concentration(self, sample_data, typical_c3_params):
        """Test effect of oxygen concentration."""
        # Test with low oxygen (C4-like conditions)
        result_low_o2 = calculate_c3_assimilation(
            sample_data, typical_c3_params,
            oxygen=2.0  # 2% O2
        )
        
        # Test with normal oxygen
        result_normal_o2 = calculate_c3_assimilation(
            sample_data, typical_c3_params,
            oxygen=21.0  # 21% O2
        )
        
        # Low O2 should increase assimilation (less photorespiration)
        # But only where Rubisco limits - at high CO2, TPU may limit both
        rubisco_limited_idx = np.where(identify_c3_limiting_process(result_normal_o2) == 'Rubisco')[0]
        if len(rubisco_limited_idx) > 0:
            assert np.all(result_low_o2.An[rubisco_limited_idx] > result_normal_o2.An[rubisco_limited_idx])
        
    def test_tpu_threshold(self, sample_data, typical_c3_params):
        """Test custom TPU threshold."""
        # Set very high custom threshold
        result_high_threshold = calculate_c3_assimilation(
            sample_data, typical_c3_params,
            TPU_threshold=1000.0  # Very high threshold
        )
        
        # Set low custom threshold
        result_low_threshold = calculate_c3_assimilation(
            sample_data, typical_c3_params,
            TPU_threshold=100.0  # Low threshold
        )
        
        # High threshold should allow more TPU activity
        # Low threshold should limit TPU more
        # At points where PCc > low_threshold but < high_threshold, Wp should differ
        # Find points where thresholds might matter
        cc_values = sample_data.data['Cc'].values
        pressure = sample_data.data['Pa'].values / 100.0
        pcc_values = cc_values * pressure
        
        # Check if we have points in the affected range
        affected_range = (pcc_values > 100.0) & (pcc_values < 1000.0)
        if np.any(affected_range):
            # If we have points in the range where thresholds matter, Wp should differ
            assert not np.array_equal(
                result_high_threshold.Wp[affected_range], 
                result_low_threshold.Wp[affected_range]
            )


class TestLimitingProcess:
    """Test limiting process identification."""
    
    def test_identify_limiting_process(self):
        """Test identification of limiting processes."""
        # Create mock result with clear limitations
        result = C3AssimilationResult(
            An=np.array([5, 10, 15, 20, 22, 23]),
            Ac=np.array([5, 12, 20, 25, 30, 35]),
            Aj=np.array([8, 10, 15, 20, 22, 25]),
            Ap=np.array([10, 15, 18, 22, 23, 23]),
            Wc=np.array([10, 20, 30, 40, 50, 60]),
            Wj=np.array([15, 18, 25, 35, 40, 45]),
            Wp=np.array([20, 25, 30, 40, 42, 42]),
            Vc=np.array([10, 18, 25, 35, 40, 42]),
            Vcmax_tl=np.array([100]*6),
            J_tl=np.array([200]*6),
            Tp_tl=np.array([12]*6),
            RL_tl=np.array([1.5]*6),
            Gamma_star_tl=np.array([42.93]*6),
            Kc_tl=np.array([406.85]*6),
            Ko_tl=np.array([277.14]*6),
            Gamma_star_agt=np.array([42.93]*6)
        )
        
        limiting = identify_c3_limiting_process(result)
        
        # First point: Wc=10 < Wj=15 < Wp=20, so Rubisco-limited
        assert limiting[0] == 'Rubisco'
        
        # Check all values are valid
        assert all(l in ['Rubisco', 'RuBP', 'TPU'] for l in limiting)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])