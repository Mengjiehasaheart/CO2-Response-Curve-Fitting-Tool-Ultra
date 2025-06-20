

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
from scipy import stats
from ..core.data_structures import ExtendedDataFrame


def estimate_c3_initial_parameters(
    exdf: ExtendedDataFrame,
    a_column: str = 'A',
    ci_column: str = 'Ci',
    temperature_response_params: Optional[Dict] = None,
    average_temperature: Optional[float] = None
) -> Dict[str, float]:
    """
    Estimate initial C3 model parameters from A-Ci curve data.
    
    Based on PhotoGEA's initial_guess_c3_aci.R implementation.
    
    Args:
        exdf: Extended dataframe with A-Ci curve data
        a_column: Column name for net assimilation
        ci_column: Column name for intercellular CO2
        temperature_response_params: Temperature parameters for adjustments
        average_temperature: Average leaf temperature (°C)
    
    Returns:
        Dictionary of initial parameter estimates
    """
    # Extract data
    A = exdf.data[a_column].values
    Ci = exdf.data[ci_column].values
    
    # Get temperature if available
    if average_temperature is None and 'Tleaf' in exdf.data.columns:
        average_temperature = exdf.data['Tleaf'].mean()
    if average_temperature is None:
        average_temperature = 25.0  # Default to 25°C
    
    # Sort by Ci for analysis
    sort_idx = np.argsort(Ci)
    Ci_sorted = Ci[sort_idx]
    A_sorted = A[sort_idx]
    
    # 1. Estimate Vcmax from Rubisco-limited region (low Ci)
    vcmax_guess = estimate_vcmax_from_initial_slope(Ci_sorted, A_sorted)
    
    # 2. Estimate J from RuBP-limited region (intermediate Ci)
    j_guess = estimate_j_from_plateau(Ci_sorted, A_sorted, vcmax_guess)
    
    # 3. Estimate Tp from TPU-limited region (high Ci)
    tp_guess = estimate_tp_from_high_ci(Ci_sorted, A_sorted)
    
    # 4. Estimate Rd from low-light intercept or minimum A
    rd_guess = estimate_rd(A_sorted)
    
    # 5. Default gm (mesophyll conductance) - often fixed
    gm_guess = 3.0  # mol m⁻² s⁻¹ bar⁻¹
    
    # Apply reasonable bounds - use _at_25 suffix for parameters
    initial_params = {
        'Vcmax_at_25': max(10.0, min(vcmax_guess, 300.0)),
        'J_at_25': max(20.0, min(j_guess, 500.0)),
        'Tp_at_25': max(5.0, min(tp_guess, 50.0)),
        'RL_at_25': max(0.0, min(rd_guess, 10.0)),  # RL instead of Rd
        'gmc': gm_guess  # gmc instead of gm
    }
    
    return initial_params


def estimate_vcmax_from_initial_slope(
    ci: np.ndarray,
    a: np.ndarray,
    ci_threshold: float = 300.0
) -> float:
    """
    Estimate Vcmax from the initial slope of the A-Ci curve.
    
    In the Rubisco-limited region (low Ci), the relationship is approximately linear.
    
    Args:
        ci: Sorted intercellular CO2 concentrations
        a: Sorted assimilation rates
        ci_threshold: Maximum Ci for Rubisco-limited region
    
    Returns:
        Estimated Vcmax value
    """
    # Select points in Rubisco-limited region
    mask = ci < ci_threshold
    if np.sum(mask) < 3:
        # Not enough points, use all data
        mask = np.ones_like(ci, dtype=bool)
    
    ci_low = ci[mask]
    a_low = a[mask]
    
    # Fit linear regression to estimate slope
    if len(ci_low) >= 2:
        slope, intercept, _, _, _ = stats.linregress(ci_low, a_low)
        
        # Vcmax is approximately related to the slope
        # In the Rubisco-limited region: A ≈ Vcmax * (Ci - Γ*) / (Ci + Kc*(1 + O/Ko))
        # At low Ci with typical kinetic constants, slope ≈ Vcmax / (Kc + typical_Ci)
        # With Kc ≈ 270 µmol/mol and typical low Ci ≈ 100, scaling factor ≈ 370/slope
        # But this is very approximate, so we use a more empirical approach
        
        # Use the maximum A value as additional information
        max_a = np.max(a)
        
        # Estimate Vcmax from both slope and max A
        vcmax_from_slope = slope * 25.0  # Increased empirical scaling
        vcmax_from_max = max_a * 4.0     # Vcmax typically 3-5x max A
        
        # Use weighted average favoring the slope estimate
        vcmax_guess = 0.7 * vcmax_from_slope + 0.3 * vcmax_from_max
        
        # Ensure positive and reasonable
        vcmax_guess = max(20.0, min(vcmax_guess, 300.0))
    else:
        # Fallback estimate based on maximum A value
        max_a = np.max(a)
        vcmax_guess = max(50.0, min(max_a * 4.0, 200.0))
    
    return vcmax_guess


def estimate_j_from_plateau(
    ci: np.ndarray,
    a: np.ndarray,
    vcmax_est: float,
    ci_range: Tuple[float, float] = (300.0, 700.0)
) -> float:
    """
    Estimate J from the RuBP-limited plateau region.
    
    Args:
        ci: Sorted intercellular CO2 concentrations
        a: Sorted assimilation rates
        vcmax_est: Estimated Vcmax value
        ci_range: Ci range for RuBP-limited region
    
    Returns:
        Estimated J value
    """
    # Select points in RuBP-limited region
    mask = (ci >= ci_range[0]) & (ci <= ci_range[1])
    if np.sum(mask) < 3:
        # Adjust range if not enough points
        mask = (ci >= 200.0) & (ci <= 800.0)
    
    if np.sum(mask) >= 2:
        a_plateau = a[mask]
        
        # In RuBP-limited region, A ≈ J/4 - Rd (simplified)
        # So J ≈ 4 * (A + Rd)
        a_mean = np.mean(a_plateau)
        j_guess = 4.0 * (a_mean + 1.0)  # Assume Rd ≈ 1.0
        
        # J should be greater than Vcmax typically
        j_guess = max(j_guess, vcmax_est * 1.5)
    else:
        # Fallback: J is typically 2-2.5x Vcmax
        j_guess = vcmax_est * 2.0
    
    return j_guess


def estimate_tp_from_high_ci(
    ci: np.ndarray,
    a: np.ndarray,
    ci_threshold: float = 700.0
) -> float:
    """
    Estimate Tp from the TPU-limited region at high Ci.
    
    Args:
        ci: Sorted intercellular CO2 concentrations
        a: Sorted assimilation rates
        ci_threshold: Minimum Ci for TPU-limited region
    
    Returns:
        Estimated Tp value
    """
    # Select points in potential TPU-limited region
    mask = ci > ci_threshold
    
    if np.sum(mask) >= 3:
        ci_high = ci[mask]
        a_high = a[mask]
        
        # Check if A decreases with increasing Ci (TPU limitation signature)
        correlation = np.corrcoef(ci_high, a_high)[0, 1]
        
        if correlation < -0.3:  # Negative correlation suggests TPU limitation
            # Tp ≈ A_max / 3 (rough approximation)
            tp_guess = np.max(a_high) / 3.0
        else:
            # No clear TPU limitation
            tp_guess = 15.0  # Default moderate value
    else:
        # Not enough high Ci points
        tp_guess = 15.0  # Default
    
    return tp_guess


def estimate_rd(a: np.ndarray) -> float:
    """
    Estimate dark respiration (Rd) from assimilation data.
    
    Args:
        a: Assimilation rates
    
    Returns:
        Estimated Rd value
    """
    # Method 1: Use minimum A value (if negative)
    a_min = np.min(a)
    if a_min < 0:
        rd_guess = abs(a_min)
    else:
        # Method 2: Estimate as small fraction of mean A
        rd_guess = 0.05 * np.mean(a[a > 0])
    
    # Ensure reasonable range
    rd_guess = max(0.5, min(rd_guess, 5.0))
    
    return rd_guess


def estimate_c3_parameter_bounds(
    initial_params: Dict[str, float],
    fixed_params: Optional[Dict[str, float]] = None
) -> Dict[str, Tuple[float, float]]:
    """
    Generate parameter bounds based on initial estimates.
    
    Args:
        initial_params: Initial parameter estimates
        fixed_params: Parameters to fix (not optimize)
    
    Returns:
        Dictionary of parameter bounds
    """
    fixed_params = fixed_params or {}
    
    bounds = {}
    
    # Vcmax bounds: 0.2x to 5x initial estimate
    if 'Vcmax_at_25' not in fixed_params:
        vcmax_init = initial_params.get('Vcmax_at_25', 100.0)
        bounds['Vcmax_at_25'] = (
            max(0.0, 0.2 * vcmax_init),
            min(500.0, 5.0 * vcmax_init)
        )
    
    # J bounds: 0.5x to 3x initial estimate
    if 'J_at_25' not in fixed_params:
        j_init = initial_params.get('J_at_25', 200.0)
        bounds['J_at_25'] = (
            max(0.0, 0.5 * j_init),
            min(600.0, 3.0 * j_init)
        )
    
    # Tp bounds: 0.3x to 5x initial estimate
    if 'Tp_at_25' not in fixed_params:
        tp_init = initial_params.get('Tp_at_25', 15.0)
        bounds['Tp_at_25'] = (
            max(0.0, 0.3 * tp_init),
            min(100.0, 5.0 * tp_init)
        )
    
    # RL bounds: 0 to 3x initial estimate
    if 'RL_at_25' not in fixed_params:
        rd_init = initial_params.get('RL_at_25', 1.0)
        bounds['RL_at_25'] = (
            0.0,
            min(10.0, 3.0 * rd_init)
        )
    
    # gmc bounds: wide range if not fixed
    if 'gmc' not in fixed_params:
        bounds['gmc'] = (0.1, 20.0)
    
    return bounds


def identify_limiting_regions(
    exdf: ExtendedDataFrame,
    ci_column: str = 'Ci'
) -> Dict[str, np.ndarray]:
    """
    Identify approximate regions where different processes limit photosynthesis.
    
    Args:
        exdf: Extended dataframe with A-Ci curve data
        ci_column: Column name for intercellular CO2
    
    Returns:
        Dictionary with boolean masks for each limiting region
    """
    ci = exdf.data[ci_column].values
    
    # Define approximate Ci ranges for each limitation
    regions = {
        'rubisco_limited': ci < 300.0,
        'rubp_limited': (ci >= 300.0) & (ci <= 700.0),
        'tpu_limited': ci > 700.0
    }
    
    return regions


def validate_initial_guess(
    params: Dict[str, float],
    data_stats: Optional[Dict[str, float]] = None
) -> Tuple[bool, List[str]]:
    """
    Validate initial parameter guesses.
    
    Args:
        params: Initial parameter estimates
        data_stats: Optional statistics from the data
    
    Returns:
        Tuple of (is_valid, list_of_warnings)
    """
    warnings = []
    
    # Check parameter relationships
    if 'Vcmax_at_25' in params and 'J_at_25' in params:
        j_vcmax_ratio = params['J_at_25'] / params['Vcmax_at_25']
        if j_vcmax_ratio < 1.0:
            warnings.append(f"J/Vcmax ratio is low: {j_vcmax_ratio:.2f}")
        elif j_vcmax_ratio > 4.0:
            warnings.append(f"J/Vcmax ratio is high: {j_vcmax_ratio:.2f}")
    
    # Check individual parameters
    if params.get('Vcmax_at_25', 0) < 10:
        warnings.append("Vcmax seems too low")
    if params.get('J_at_25', 0) < 20:
        warnings.append("J seems too low")
    if params.get('RL_at_25', 0) > 5:
        warnings.append("RL seems too high")
    
    is_valid = len(warnings) == 0
    return is_valid, warnings