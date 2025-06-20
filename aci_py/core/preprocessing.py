"""
Data preprocessing and quality control for photosynthesis measurements.

This module provides functions for:
- Outlier detection and removal
- Data quality checks
- Environmental stability validation
- Curve preprocessing for fitting
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from scipy import stats
import warnings

from .data_structures import ExtendedDataFrame


def detect_outliers_iqr(
    data: np.ndarray,
    factor: float = 1.5
) -> np.ndarray:
    """
    Detect outliers using the Interquartile Range (IQR) method.
    
    Parameters
    ----------
    data : np.ndarray
        Data values to check
    factor : float, optional
        IQR multiplier for outlier bounds. Default is 1.5.
        
    Returns
    -------
    np.ndarray
        Boolean mask where True indicates an outlier
    """
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    
    return (data < lower_bound) | (data > upper_bound)


def detect_outliers_zscore(
    data: np.ndarray,
    threshold: float = 3.0
) -> np.ndarray:
    """
    Detect outliers using z-score method.
    
    Parameters
    ----------
    data : np.ndarray
        Data values to check
    threshold : float, optional
        Z-score threshold. Default is 3.0.
        
    Returns
    -------
    np.ndarray
        Boolean mask where True indicates an outlier
    """
    z_scores = np.abs(stats.zscore(data, nan_policy='omit'))
    return z_scores > threshold


def detect_outliers_mad(
    data: np.ndarray,
    threshold: float = 3.5
) -> np.ndarray:
    """
    Detect outliers using Median Absolute Deviation (MAD).
    
    More robust than z-score for non-normal distributions.
    
    Parameters
    ----------
    data : np.ndarray
        Data values to check
    threshold : float, optional
        MAD threshold. Default is 3.5.
        
    Returns
    -------
    np.ndarray
        Boolean mask where True indicates an outlier
    """
    median = np.median(data)
    mad = np.median(np.abs(data - median))
    
    # Modified z-score using MAD
    if mad == 0:
        # If MAD is 0, use a different approach
        return np.abs(data - median) > threshold * np.std(data)
    
    modified_z_scores = 0.6745 * (data - median) / mad
    return np.abs(modified_z_scores) > threshold


def check_environmental_stability(
    exdf: ExtendedDataFrame,
    temp_tolerance: float = 1.0,
    rh_tolerance: float = 5.0,
    par_tolerance: float = 50.0,
    co2_ref_tolerance: float = 10.0
) -> Dict[str, np.ndarray]:
    """
    Check environmental stability during measurements.
    
    Parameters
    ----------
    exdf : ExtendedDataFrame
        Measurement data
    temp_tolerance : float, optional
        Maximum allowed temperature variation (°C)
    rh_tolerance : float, optional
        Maximum allowed RH variation (%)
    par_tolerance : float, optional
        Maximum allowed PAR variation (µmol/m²/s)
    co2_ref_tolerance : float, optional
        Maximum allowed reference CO2 variation (µmol/mol)
        
    Returns
    -------
    dict
        Dictionary with stability check results for each parameter
    """
    results = {}
    
    # Check leaf temperature stability
    if 'Tleaf' in exdf.data.columns:
        tleaf = exdf.data['Tleaf'].values
        tleaf_range = np.max(tleaf) - np.min(tleaf)
        results['Tleaf_stable'] = tleaf_range <= temp_tolerance
        results['Tleaf_range'] = tleaf_range
    
    # Check RH stability
    if 'RHcham' in exdf.data.columns:
        rh = exdf.data['RHcham'].values
        rh_range = np.max(rh) - np.min(rh)
        results['RH_stable'] = rh_range <= rh_tolerance
        results['RH_range'] = rh_range
    
    # Check PAR stability
    if 'Qin' in exdf.data.columns:
        par = exdf.data['Qin'].values
        par_range = np.max(par) - np.min(par)
        results['PAR_stable'] = par_range <= par_tolerance
        results['PAR_range'] = par_range
    elif 'PARi' in exdf.data.columns:
        par = exdf.data['PARi'].values
        par_range = np.max(par) - np.min(par)
        results['PAR_stable'] = par_range <= par_tolerance
        results['PAR_range'] = par_range
    
    # Check reference CO2 stability (for quality control)
    if 'CO2_r' in exdf.data.columns:
        co2_ref = exdf.data['CO2_r'].values
        # Check variation around each setpoint
        unique_targets = np.unique(np.round(co2_ref, -1))  # Round to nearest 10
        max_deviation = 0
        for target in unique_targets:
            mask = np.abs(co2_ref - target) < 50  # Points near this target
            if np.sum(mask) > 0:
                deviation = np.max(np.abs(co2_ref[mask] - target))
                max_deviation = max(max_deviation, deviation)
        results['CO2_ref_stable'] = max_deviation <= co2_ref_tolerance
        results['CO2_ref_max_deviation'] = max_deviation
    
    return results


def identify_aci_outliers(
    exdf: ExtendedDataFrame,
    a_column: str = 'A',
    ci_column: str = 'Ci',
    method: str = 'combined',
    iqr_factor: float = 2.0,
    zscore_threshold: float = 3.0,
    mad_threshold: float = 3.5,
    check_negative_a: bool = True,
    check_extreme_ci: bool = True,
    ci_min: float = 0.0,
    ci_max: float = 2000.0
) -> np.ndarray:
    """
    Identify outliers in ACI curve data using multiple methods.
    
    Parameters
    ----------
    exdf : ExtendedDataFrame
        ACI curve data
    a_column : str, optional
        Column name for assimilation rate
    ci_column : str, optional
        Column name for intercellular CO2
    method : str, optional
        Outlier detection method: 'iqr', 'zscore', 'mad', or 'combined'
    iqr_factor : float, optional
        IQR multiplier for outlier detection
    zscore_threshold : float, optional
        Z-score threshold
    mad_threshold : float, optional
        MAD threshold
    check_negative_a : bool, optional
        Whether to flag negative assimilation as outliers
    check_extreme_ci : bool, optional
        Whether to flag extreme Ci values as outliers
    ci_min : float, optional
        Minimum reasonable Ci value
    ci_max : float, optional
        Maximum reasonable Ci value
        
    Returns
    -------
    np.ndarray
        Boolean mask where True indicates an outlier
    """
    n_points = len(exdf.data)
    outlier_mask = np.zeros(n_points, dtype=bool)
    
    # Get data
    a_values = exdf.data[a_column].values
    ci_values = exdf.data[ci_column].values
    
    # Apply statistical outlier detection
    if method in ['iqr', 'combined']:
        outlier_mask |= detect_outliers_iqr(a_values, iqr_factor)
    
    if method in ['zscore', 'combined']:
        outlier_mask |= detect_outliers_zscore(a_values, zscore_threshold)
    
    if method in ['mad', 'combined']:
        outlier_mask |= detect_outliers_mad(a_values, mad_threshold)
    
    # Domain-specific checks
    if check_negative_a:
        # Flag strongly negative A values (small negatives OK for respiration)
        outlier_mask |= a_values < -5.0
    
    if check_extreme_ci:
        outlier_mask |= (ci_values < ci_min) | (ci_values > ci_max)
    
    # Check for non-monotonic behavior in specific regions
    # (This is more sophisticated and specific to ACI curves)
    if len(ci_values) > 5:
        # Sort by Ci to check progression
        sort_idx = np.argsort(ci_values)
        ci_sorted = ci_values[sort_idx]
        a_sorted = a_values[sort_idx]
        
        # In the initial rise (low Ci), A should generally increase
        low_ci_mask = ci_sorted < 200
        if np.sum(low_ci_mask) > 3:
            # Check for strong decreases in the low Ci region
            a_diff = np.diff(a_sorted[low_ci_mask])
            strong_decrease = a_diff < -5.0  # Large drop in A
            if np.any(strong_decrease):
                decrease_idx = np.where(strong_decrease)[0] + 1
                outlier_mask[sort_idx[np.where(low_ci_mask)[0][decrease_idx]]] = True
    
    return outlier_mask


def remove_outliers(
    exdf: ExtendedDataFrame,
    outlier_mask: np.ndarray,
    min_points: int = 5
) -> ExtendedDataFrame:
    """
    Remove outliers from data, ensuring minimum points remain.
    
    Parameters
    ----------
    exdf : ExtendedDataFrame
        Original data
    outlier_mask : np.ndarray
        Boolean mask indicating outliers
    min_points : int, optional
        Minimum number of points to retain
        
    Returns
    -------
    ExtendedDataFrame
        Data with outliers removed
    """
    n_outliers = np.sum(outlier_mask)
    n_remaining = len(outlier_mask) - n_outliers
    
    if n_remaining < min_points:
        warnings.warn(
            f"Removing {n_outliers} outliers would leave only {n_remaining} points. "
            f"Keeping all data (minimum required: {min_points})."
        )
        return exdf
    
    # Create clean data
    clean_mask = ~outlier_mask
    clean_data = exdf.data[clean_mask].copy()
    
    # Create new ExtendedDataFrame with same units and categories
    result = ExtendedDataFrame(
        clean_data,
        units=exdf.units.copy(),
        categories=exdf.categories.copy()
    )
    
    return result


def check_aci_data_quality(
    exdf: ExtendedDataFrame,
    min_points: int = 5,
    min_ci_range: float = 100.0,
    require_low_ci: bool = True,
    low_ci_threshold: float = 100.0,
    require_high_ci: bool = True,
    high_ci_threshold: float = 500.0
) -> Dict[str, Union[bool, float, int]]:
    """
    Comprehensive quality checks for ACI curve data.
    
    Parameters
    ----------
    exdf : ExtendedDataFrame
        ACI curve data
    min_points : int, optional
        Minimum number of points required
    min_ci_range : float, optional
        Minimum range of Ci values required
    require_low_ci : bool, optional
        Whether to require low Ci measurements
    low_ci_threshold : float, optional
        Threshold for low Ci
    require_high_ci : bool, optional
        Whether to require high Ci measurements
    high_ci_threshold : float, optional
        Threshold for high Ci
        
    Returns
    -------
    dict
        Quality check results
    """
    results = {}
    
    # Check number of points
    n_points = len(exdf.data)
    results['n_points'] = n_points
    results['sufficient_points'] = n_points >= min_points
    
    # Check Ci range
    ci_values = exdf.data['Ci'].values if 'Ci' in exdf.data.columns else np.array([])
    if len(ci_values) > 0:
        ci_range = np.max(ci_values) - np.min(ci_values)
        results['ci_range'] = ci_range
        results['sufficient_ci_range'] = ci_range >= min_ci_range
        
        # Check for low Ci points
        if require_low_ci:
            has_low_ci = np.any(ci_values < low_ci_threshold)
            results['has_low_ci'] = has_low_ci
            results['min_ci'] = np.min(ci_values)
        
        # Check for high Ci points
        if require_high_ci:
            has_high_ci = np.any(ci_values > high_ci_threshold)
            results['has_high_ci'] = has_high_ci
            results['max_ci'] = np.max(ci_values)
    
    # Check data distribution
    if 'A' in exdf.data.columns:
        a_values = exdf.data['A'].values
        results['mean_a'] = np.mean(a_values)
        results['std_a'] = np.std(a_values)
        results['min_a'] = np.min(a_values)
        results['max_a'] = np.max(a_values)
    
    # Overall quality assessment
    quality_issues = []
    if not results.get('sufficient_points', False):
        quality_issues.append(f"Too few points ({n_points} < {min_points})")
    if not results.get('sufficient_ci_range', False):
        quality_issues.append(f"Insufficient Ci range ({results.get('ci_range', 0):.1f})")
    if require_low_ci and not results.get('has_low_ci', False):
        quality_issues.append("Missing low Ci measurements")
    if require_high_ci and not results.get('has_high_ci', False):
        quality_issues.append("Missing high Ci measurements")
    
    results['quality_ok'] = len(quality_issues) == 0
    results['quality_issues'] = quality_issues
    
    return results


def preprocess_aci_data(
    exdf: ExtendedDataFrame,
    remove_outliers_flag: bool = True,
    outlier_method: str = 'combined',
    check_environment: bool = True,
    check_quality: bool = True,
    min_points: int = 5,
    verbose: bool = True
) -> Tuple[ExtendedDataFrame, Dict[str, any]]:
    """
    Complete preprocessing pipeline for ACI data.
    
    Parameters
    ----------
    exdf : ExtendedDataFrame
        Raw ACI curve data
    remove_outliers_flag : bool, optional
        Whether to remove outliers
    outlier_method : str, optional
        Method for outlier detection
    check_environment : bool, optional
        Whether to check environmental stability
    check_quality : bool, optional
        Whether to perform quality checks
    min_points : int, optional
        Minimum points to retain
    verbose : bool, optional
        Whether to print preprocessing summary
        
    Returns
    -------
    ExtendedDataFrame
        Preprocessed data
    dict
        Preprocessing report with all checks and actions
    """
    report = {
        'original_n_points': len(exdf.data),
        'preprocessing_steps': []
    }
    
    result = exdf.copy()
    
    # 1. Environmental stability check
    if check_environment:
        env_results = check_environmental_stability(result)
        report['environmental_stability'] = env_results
        
        if verbose and not all(v for k, v in env_results.items() if k.endswith('_stable')):
            print("Warning: Environmental conditions may not be stable")
            for key, value in env_results.items():
                if key.endswith('_range') or key.endswith('_deviation'):
                    print(f"  {key}: {value:.2f}")
    
    # 2. Outlier detection and removal
    if remove_outliers_flag:
        outlier_mask = identify_aci_outliers(result, method=outlier_method)
        n_outliers = np.sum(outlier_mask)
        
        if n_outliers > 0:
            result_clean = remove_outliers(result, outlier_mask, min_points)
            n_removed = len(result.data) - len(result_clean.data)
            
            report['outliers_detected'] = int(n_outliers)
            report['outliers_removed'] = int(n_removed)
            report['preprocessing_steps'].append(f"Removed {n_removed} outliers")
            
            result = result_clean
            
            if verbose:
                print(f"Removed {n_removed} outliers ({n_outliers} detected)")
    
    # 3. Data quality check
    if check_quality:
        quality_results = check_aci_data_quality(result, min_points=min_points)
        report['quality_check'] = quality_results
        
        if verbose and not quality_results['quality_ok']:
            print("Data quality issues detected:")
            for issue in quality_results['quality_issues']:
                print(f"  - {issue}")
    
    # 4. Final summary
    report['final_n_points'] = len(result.data)
    report['points_removed'] = report['original_n_points'] - report['final_n_points']
    
    if verbose:
        print(f"\nPreprocessing complete:")
        print(f"  Original points: {report['original_n_points']}")
        print(f"  Final points: {report['final_n_points']}")
        print(f"  Points removed: {report['points_removed']}")
    
    return result, report


def flag_points_for_removal(
    exdf: ExtendedDataFrame,
    flags: Dict[str, np.ndarray]
) -> ExtendedDataFrame:
    """
    Add flags to data for manual review rather than automatic removal.
    
    Parameters
    ----------
    exdf : ExtendedDataFrame
        Data to flag
    flags : dict
        Dictionary of flag names and boolean arrays
        
    Returns
    -------
    ExtendedDataFrame
        Data with flag columns added
    """
    result = exdf.copy()
    
    # Add individual flags
    for flag_name, flag_values in flags.items():
        column_name = f'flag_{flag_name}'
        result.data[column_name] = flag_values
        result.units[column_name] = 'dimensionless'
        result.categories[column_name] = 'preprocessing_flags'
    
    # Add combined flag
    combined_flag = np.zeros(len(result.data), dtype=bool)
    for flag_values in flags.values():
        combined_flag |= flag_values
    
    result.data['flag_any'] = combined_flag
    result.units['flag_any'] = 'dimensionless'
    result.categories['flag_any'] = 'preprocessing_flags'
    
    return result