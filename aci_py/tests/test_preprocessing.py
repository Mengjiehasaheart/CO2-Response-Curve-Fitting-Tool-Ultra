"""
Tests for data preprocessing and quality control.
"""

import pytest
import numpy as np
import pandas as pd
from ..core.data_structures import ExtendedDataFrame
from ..core.preprocessing import (
    detect_outliers_iqr,
    detect_outliers_zscore,
    detect_outliers_mad,
    check_environmental_stability,
    identify_aci_outliers,
    remove_outliers,
    check_aci_data_quality,
    preprocess_aci_data,
    flag_points_for_removal
)


class TestOutlierDetection:
    """Test outlier detection methods."""
    
    def test_detect_outliers_iqr(self):
        """Test IQR-based outlier detection."""
        # Create data with clear outliers
        normal_data = np.random.normal(10, 2, 100)
        outliers = np.array([50, -20])
        data = np.concatenate([normal_data, outliers])
        
        # Detect outliers
        mask = detect_outliers_iqr(data, factor=1.5)
        
        # Check that outliers are detected
        assert mask[-2]  # 50 should be outlier
        assert mask[-1]  # -20 should be outlier
        
        # Most normal data should not be outliers
        assert np.sum(mask[:-2]) < 10  # Less than 10% false positives
    
    def test_detect_outliers_zscore(self):
        """Test z-score based outlier detection."""
        # Create data with outliers
        data = np.array([1, 2, 3, 4, 5, 100, 2, 3, 4, -50])
        
        mask = detect_outliers_zscore(data, threshold=2.0)
        
        assert mask[5]  # 100 is outlier
        assert mask[9]  # -50 is outlier
        assert not mask[0]  # 1 is not outlier
    
    def test_detect_outliers_mad(self):
        """Test MAD-based outlier detection."""
        # MAD is more robust to outliers than std
        data = np.array([1, 2, 3, 4, 5, 100, 2, 3, 4])
        
        mask_mad = detect_outliers_mad(data, threshold=2.5)
        mask_zscore = detect_outliers_zscore(data, threshold=2.5)
        
        # MAD should detect the outlier
        assert mask_mad[5]
        
        # MAD should be more conservative than z-score
        assert np.sum(mask_mad) <= np.sum(mask_zscore)


class TestEnvironmentalChecks:
    """Test environmental stability checks."""
    
    @pytest.fixture
    def stable_data(self):
        """Create environmentally stable data."""
        n = 10
        data = pd.DataFrame({
            'Tleaf': np.random.normal(25, 0.1, n),  # Stable temperature
            'RHcham': np.random.normal(60, 0.5, n),  # Stable RH
            'Qin': np.random.normal(1500, 5, n),    # Stable PAR
            'CO2_r': np.array([400]*3 + [600]*3 + [800]*4)  # Step changes
        })
        return ExtendedDataFrame(data)
    
    @pytest.fixture
    def unstable_data(self):
        """Create environmentally unstable data."""
        n = 10
        data = pd.DataFrame({
            'Tleaf': np.linspace(20, 30, n),      # Large temperature drift
            'RHcham': np.linspace(40, 70, n),     # Large RH drift
            'Qin': np.linspace(1000, 1600, n),    # Large PAR drift
            'CO2_r': np.random.normal(400, 20, n)  # Noisy CO2
        })
        return ExtendedDataFrame(data)
    
    def test_stable_environment(self, stable_data):
        """Test detection of stable conditions."""
        results = check_environmental_stability(stable_data)
        
        assert results['Tleaf_stable']
        assert results['RH_stable']
        assert results['PAR_stable']
        assert results['Tleaf_range'] < 1.0
    
    def test_unstable_environment(self, unstable_data):
        """Test detection of unstable conditions."""
        results = check_environmental_stability(
            unstable_data,
            temp_tolerance=2.0,
            rh_tolerance=5.0
        )
        
        assert not results['Tleaf_stable']
        assert not results['RH_stable']
        assert not results['PAR_stable']
        assert results['Tleaf_range'] > 5.0


class TestACIOutliers:
    """Test ACI-specific outlier detection."""
    
    @pytest.fixture
    def aci_data(self):
        """Create synthetic ACI data with outliers."""
        # Normal ACI curve
        ci = np.array([50, 100, 150, 200, 300, 400, 600, 800, 1000])
        a = np.array([5, 10, 15, 18, 22, 25, 28, 30, 31])
        
        # Add outliers
        ci = np.append(ci, [500, 700])
        a = np.append(a, [-10, 50])  # Negative and too high
        
        data = pd.DataFrame({'Ci': ci, 'A': a})
        return ExtendedDataFrame(data)
    
    def test_identify_aci_outliers(self, aci_data):
        """Test ACI outlier identification."""
        mask = identify_aci_outliers(
            aci_data,
            method='combined',
            check_negative_a=True
        )
        
        # Should detect the outliers we added
        assert mask[-2]  # Negative A
        assert mask[-1]  # Too high A
        
        # Normal points should not be outliers
        assert np.sum(mask[:-2]) == 0
    
    def test_extreme_ci_detection(self):
        """Test detection of extreme Ci values."""
        data = pd.DataFrame({
            'Ci': [-50, 10, 100, 500, 1000, 3000],
            'A': [5, 8, 15, 25, 30, 32]
        })
        exdf = ExtendedDataFrame(data)
        
        mask = identify_aci_outliers(
            exdf,
            check_extreme_ci=True,
            ci_min=0,
            ci_max=2000
        )
        
        assert mask[0]  # Negative Ci
        assert mask[5]  # Ci > 2000
        assert not mask[2]  # Normal Ci


class TestDataQuality:
    """Test data quality checks."""
    
    def test_check_aci_data_quality_good(self):
        """Test quality check on good data."""
        data = pd.DataFrame({
            'Ci': np.linspace(50, 1000, 15),
            'A': np.linspace(5, 35, 15)
        })
        exdf = ExtendedDataFrame(data)
        
        results = check_aci_data_quality(exdf)
        
        assert results['quality_ok']
        assert results['sufficient_points']
        assert results['sufficient_ci_range']
        assert results['has_low_ci']
        assert results['has_high_ci']
        assert len(results['quality_issues']) == 0
    
    def test_check_aci_data_quality_bad(self):
        """Test quality check on poor data."""
        # Too few points, narrow range
        data = pd.DataFrame({
            'Ci': [400, 450, 500],
            'A': [20, 22, 23]
        })
        exdf = ExtendedDataFrame(data)
        
        results = check_aci_data_quality(
            exdf,
            min_points=5,
            require_low_ci=True,
            require_high_ci=True
        )
        
        assert not results['quality_ok']
        assert not results['sufficient_points']
        assert not results['has_low_ci']
        assert len(results['quality_issues']) > 0


class TestPreprocessing:
    """Test complete preprocessing pipeline."""
    
    @pytest.fixture
    def raw_aci_data(self):
        """Create raw ACI data with issues."""
        # Base curve
        ci_base = np.array([50, 100, 200, 300, 400, 600, 800, 1000])
        a_base = np.array([5, 10, 18, 22, 25, 28, 30, 31])
        
        # Add outlier
        ci = np.append(ci_base, 500)
        a = np.append(a_base, -5)  # Negative outlier
        
        # Add environmental data with drift
        n = len(ci)
        data = pd.DataFrame({
            'Ci': ci,
            'A': a,
            'Tleaf': np.linspace(24.5, 25.5, n),  # 1°C drift
            'RHcham': np.linspace(58, 62, n),     # 4% drift
            'Qin': np.full(n, 1500)               # Stable
        })
        
        return ExtendedDataFrame(data)
    
    def test_preprocess_aci_data_complete(self, raw_aci_data):
        """Test complete preprocessing pipeline."""
        processed, report = preprocess_aci_data(
            raw_aci_data,
            remove_outliers_flag=True,
            check_environment=True,
            check_quality=True,
            verbose=False
        )
        
        # Check that outlier was removed
        assert report['original_n_points'] == 9
        assert report['final_n_points'] == 8
        assert report['outliers_removed'] == 1
        
        # Check environmental stability was assessed
        assert 'environmental_stability' in report
        assert report['environmental_stability']['Tleaf_stable']
        
        # Check quality was assessed
        assert 'quality_check' in report
        assert report['quality_check']['quality_ok']
    
    def test_preprocess_preserve_minimum_points(self):
        """Test that preprocessing preserves minimum points."""
        # Create data where most points would be outliers
        data = pd.DataFrame({
            'Ci': [100, 200, 300, 400, 500],
            'A': [-10, 50, -5, 60, -15]  # Mostly outliers
        })
        exdf = ExtendedDataFrame(data)
        
        processed, report = preprocess_aci_data(
            exdf,
            remove_outliers_flag=True,
            min_points=4,
            verbose=False
        )
        
        # Should keep at least min_points
        assert len(processed.data) >= 4


class TestFlagging:
    """Test point flagging functionality."""
    
    def test_flag_points_for_removal(self):
        """Test adding flags to data."""
        data = pd.DataFrame({
            'Ci': np.arange(10),
            'A': np.arange(10)
        })
        exdf = ExtendedDataFrame(data)
        
        # Create some flags
        flags = {
            'outlier': np.array([True] + [False]*8 + [True]),
            'unstable': np.array([False]*5 + [True]*5)
        }
        
        result = flag_points_for_removal(exdf, flags)
        
        # Check flags were added
        assert 'flag_outlier' in result.data.columns
        assert 'flag_unstable' in result.data.columns
        assert 'flag_any' in result.data.columns
        
        # Check combined flag
        assert result.data['flag_any'].iloc[0]  # Has outlier flag
        assert result.data['flag_any'].iloc[9]  # Has both flags
        assert not result.data['flag_any'].iloc[2]  # No flags
        
        # Check that 5 points have unstable flag
        assert np.sum(result.data['flag_unstable']) == 5