"""
C3 photosynthesis calculations using the Farquhar-von Caemmerer-Berry model.

This module implements the FvCB model for C3 photosynthesis, including:
- Rubisco-limited carboxylation (Wc)
- RuBP-regeneration-limited carboxylation (Wj)
- TPU-limited carboxylation (Wp)
- Net CO2 assimilation calculation

Based on PhotoGEA R package implementations.
"""

import numpy as np
from typing import Dict, Union, Optional, Tuple
from dataclasses import dataclass
from .data_structures import ExtendedDataFrame
from .temperature import apply_temperature_response, C3_TEMPERATURE_PARAM_BERNACCHI


@dataclass
class C3AssimilationResult:
    """Container for C3 assimilation calculation results."""
    # Net assimilation rates
    An: np.ndarray  # Net CO2 assimilation rate (µmol m⁻² s⁻¹)
    Ac: np.ndarray  # Rubisco-limited net assimilation
    Aj: np.ndarray  # RuBP-limited net assimilation
    Ap: np.ndarray  # TPU-limited net assimilation
    
    # Gross carboxylation rates
    Vc: np.ndarray  # Overall carboxylation rate
    Wc: np.ndarray  # Rubisco-limited carboxylation
    Wj: np.ndarray  # RuBP-limited carboxylation
    Wp: np.ndarray  # TPU-limited carboxylation
    
    # Temperature-adjusted parameters
    Vcmax_tl: np.ndarray  # Temperature-adjusted Vcmax
    J_tl: np.ndarray      # Temperature-adjusted J
    Tp_tl: np.ndarray     # Temperature-adjusted Tp
    RL_tl: np.ndarray     # Temperature-adjusted RL
    Gamma_star_tl: np.ndarray  # Temperature-adjusted Gamma_star
    Kc_tl: np.ndarray     # Temperature-adjusted Kc
    Ko_tl: np.ndarray     # Temperature-adjusted Ko
    
    # Effective CO2 compensation point
    Gamma_star_agt: np.ndarray  # Effective Gamma_star with fractionation


def calculate_c3_assimilation(
    exdf: ExtendedDataFrame,
    parameters: Dict[str, float],
    cc_column_name: str = 'Cc',
    temperature_response_params: Optional[Dict] = None,
    alpha_g: float = 0.0,
    alpha_old: float = 0.0,
    alpha_s: float = 0.0,
    alpha_t: float = 0.0,
    Wj_coef_C: float = 4.0,
    Wj_coef_Gamma_star: float = 8.0,
    TPU_threshold: Optional[float] = None,
    oxygen: float = 21.0,  # O2 percentage
    use_legacy_alpha: bool = False,
    perform_checks: bool = True
) -> C3AssimilationResult:
    """
    Calculate C3 assimilation using the Farquhar-von Caemmerer-Berry model.
    
    The FvCB model calculates photosynthesis as limited by:
    1. Rubisco activity (Wc)
    2. RuBP regeneration/electron transport (Wj)
    3. Triose phosphate utilization (Wp)
    
    Args:
        exdf: ExtendedDataFrame with gas exchange data
        parameters: Dictionary with model parameters:
            - Vcmax_at_25: Maximum carboxylation rate at 25°C (µmol m⁻² s⁻¹)
            - J_at_25: Maximum electron transport rate at 25°C (µmol m⁻² s⁻¹)
            - Tp_at_25: Triose phosphate utilization rate at 25°C (µmol m⁻² s⁻¹)
            - RL_at_25: Day respiration rate at 25°C (µmol m⁻² s⁻¹)
            - Gamma_star_at_25: CO2 compensation point at 25°C (µmol mol⁻¹)
            - Kc_at_25: Michaelis constant for CO2 at 25°C (µmol mol⁻¹)
            - Ko_at_25: Michaelis constant for O2 at 25°C (mmol mol⁻¹)
        cc_column_name: Column name for mesophyll CO2 concentration
        temperature_response_params: Temperature response parameters (default: Bernacchi)
        alpha_g: Fractionation factor for gaseous diffusion
        alpha_old: Legacy fractionation factor (mutually exclusive with others)
        alpha_s: Fractionation factor for CO2 dissolution
        alpha_t: Fractionation factor during transport
        Wj_coef_C: Coefficient for Wj calculation (default 4.0)
        Wj_coef_Gamma_star: Coefficient for Wj calculation (default 8.0)
        TPU_threshold: Custom TPU threshold (uses biochemical threshold if None)
        oxygen: O2 concentration (percent)
        use_legacy_alpha: Use alpha_old instead of new fractionation factors
        perform_checks: Whether to perform input validation
        
    Returns:
        C3AssimilationResult with calculated values
        
    Raises:
        ValueError: If required columns are missing or parameters are invalid
    """
    # Default temperature response if not provided
    if temperature_response_params is None:
        temperature_response_params = C3_TEMPERATURE_PARAM_BERNACCHI
    
    # Check for required columns
    required_columns = [cc_column_name, 'T_leaf_K', 'Pa']
    if perform_checks:
        missing = [col for col in required_columns if col not in exdf.data.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
    
    # Check parameter validity
    if perform_checks:
        _validate_c3_parameters(parameters, alpha_g, alpha_old, alpha_s, alpha_t, 
                               use_legacy_alpha, Wj_coef_C, Wj_coef_Gamma_star)
    
    # Extract data arrays
    Cc = exdf.data[cc_column_name].values  # µmol/mol
    T_leaf_C = exdf.data['T_leaf_K'].values - 273.15  # Convert to °C
    pressure = exdf.data['Pa'].values / 100.0  # Convert Pa to bar
    
    # Calculate partial pressures
    PCc = Cc * pressure  # µbar
    POc = oxygen * pressure * 1e4  # µbar (oxygen partial pressure)
    
    # Apply temperature corrections to parameters
    # Convert parameter names from '_at_25' format to base names
    base_params = {
        'Vcmax': parameters['Vcmax_at_25'],
        'J': parameters['J_at_25'],
        'Tp': parameters['Tp_at_25'],
        'RL': parameters['RL_at_25'],
        'Gamma_star': parameters['Gamma_star_at_25'],
        'Kc': parameters['Kc_at_25'],
        'Ko': parameters['Ko_at_25']
    }
    
    temp_adjusted = apply_temperature_response(
        base_params, temperature_response_params, T_leaf_C
    )
    
    # Extract temperature-adjusted values
    # Handle case where temp_adjusted values might be scalars or arrays
    def get_array_value(key, default):
        val = temp_adjusted.get(key, default)
        if isinstance(val, (int, float)):
            return np.full_like(T_leaf_C, val, dtype=float)
        return val
    
    Vcmax_tl = get_array_value('Vcmax', parameters['Vcmax_at_25'])
    J_tl = get_array_value('J', parameters['J_at_25'])
    Tp_tl = get_array_value('Tp', parameters['Tp_at_25'])
    RL_tl = get_array_value('RL', parameters['RL_at_25'])
    Gamma_star_tl = get_array_value('Gamma_star', parameters['Gamma_star_at_25'])
    
    # Convert Kc and Ko to appropriate units with pressure
    Kc_base = get_array_value('Kc', parameters['Kc_at_25'])
    Ko_base = get_array_value('Ko', parameters['Ko_at_25'])
    Kc_tl = Kc_base * pressure  # µbar
    Ko_tl = Ko_base * pressure * 1000  # µbar
    
    # Calculate effective CO2 compensation point with fractionation
    if use_legacy_alpha:
        Gamma_star_agt = (1 - alpha_old) * Gamma_star_tl * pressure
    else:
        Gamma_star_agt = (1 - alpha_g + 2 * alpha_t) * Gamma_star_tl * pressure  # µbar
    
    # Calculate Rubisco-limited carboxylation rate (µmol m⁻² s⁻¹)
    Wc = PCc * Vcmax_tl / (PCc + Kc_tl * (1.0 + POc / Ko_tl))
    
    # Calculate RuBP-regeneration-limited carboxylation rate
    if use_legacy_alpha:
        Wj_denominator = PCc * Wj_coef_C + Gamma_star_agt * Wj_coef_Gamma_star
    else:
        Wj_denominator = (PCc * Wj_coef_C + 
                         Gamma_star_agt * (Wj_coef_Gamma_star + 16 * alpha_g - 
                                          8 * alpha_t + 8 * alpha_s))
    Wj = PCc * J_tl / Wj_denominator
    
    # Calculate TPU-limited carboxylation rate
    if use_legacy_alpha:
        Wp_denominator = PCc - Gamma_star_agt * (1 + 3 * alpha_old)
    else:
        Wp_denominator = (PCc - Gamma_star_agt * (1 + 3 * alpha_g + 
                                                  6 * alpha_t + 4 * alpha_s))
    
    # Initialize Wp with infinity
    Wp = np.full_like(PCc, np.inf, dtype=float)
    
    # Apply TPU threshold
    if TPU_threshold is None:
        # Use biochemical threshold
        if use_legacy_alpha:
            threshold = Gamma_star_agt * (1 + 3 * alpha_old)
        else:
            threshold = Gamma_star_agt * (1 + 3 * alpha_g + 6 * alpha_t + 4 * alpha_s)
    else:
        # Use custom threshold
        threshold = TPU_threshold
    
    # Calculate Wp only where PCc > threshold
    valid_idx = PCc > threshold
    if np.any(valid_idx):
        Wp[valid_idx] = PCc[valid_idx] * 3 * Tp_tl[valid_idx] / Wp_denominator[valid_idx]
    
    # Overall carboxylation rate (minimum of three limitations)
    Vc = np.minimum(np.minimum(Wc, Wj), Wp)
    
    # Calculate net CO2 assimilation rates
    photo_resp_factor = 1.0 - Gamma_star_agt / PCc  # Photorespiration factor
    
    # Prevent division by zero
    photo_resp_factor = np.where(PCc > 0, photo_resp_factor, 0)
    
    # Net assimilation for each limitation
    Ac = photo_resp_factor * Wc - RL_tl
    Aj = photo_resp_factor * Wj - RL_tl
    Ap = photo_resp_factor * Wp - RL_tl
    
    # Overall net assimilation
    An = photo_resp_factor * Vc - RL_tl
    
    return C3AssimilationResult(
        An=An, Ac=Ac, Aj=Aj, Ap=Ap,
        Vc=Vc, Wc=Wc, Wj=Wj, Wp=Wp,
        Vcmax_tl=Vcmax_tl, J_tl=J_tl, Tp_tl=Tp_tl, RL_tl=RL_tl,
        Gamma_star_tl=Gamma_star_tl, Kc_tl=Kc_tl, Ko_tl=Ko_tl,
        Gamma_star_agt=Gamma_star_agt
    )


def _validate_c3_parameters(
    parameters: Dict[str, float],
    alpha_g: float, 
    alpha_old: float,
    alpha_s: float,
    alpha_t: float,
    use_legacy_alpha: bool,
    Wj_coef_C: float,
    Wj_coef_Gamma_star: float
) -> None:
    """
    Validate C3 model parameters.
    
    Raises:
        ValueError: If parameters are invalid or inconsistent
    """
    # Check required parameters
    required_params = [
        'Vcmax_at_25', 'J_at_25', 'Tp_at_25', 'RL_at_25',
        'Gamma_star_at_25', 'Kc_at_25', 'Ko_at_25'
    ]
    missing = [p for p in required_params if p not in parameters]
    if missing:
        raise ValueError(f"Missing required parameters: {missing}")
    
    # Check for mixing of alpha models
    if use_legacy_alpha and (alpha_g > 0 or alpha_s > 0 or alpha_t > 0):
        raise ValueError(
            "Cannot use legacy alpha_old with new fractionation factors "
            "(alpha_g, alpha_s, alpha_t)"
        )
    
    if not use_legacy_alpha and alpha_old > 0:
        raise ValueError(
            "alpha_old > 0 but use_legacy_alpha is False. "
            "Set use_legacy_alpha=True to use alpha_old."
        )
    
    # Check Wj coefficients when using new fractionation
    if (alpha_g > 0 or alpha_s > 0 or alpha_t > 0):
        if abs(Wj_coef_C - 4.0) > 1e-10 or abs(Wj_coef_Gamma_star - 8.0) > 1e-10:
            raise ValueError(
                "Wj_coef_C must be 4.0 and Wj_coef_Gamma_star must be 8.0 "
                "when using new fractionation factors"
            )
    
    # Check parameter values
    for param_name, value in parameters.items():
        if value < 0:
            raise ValueError(f"{param_name} must be >= 0, got {value}")
    
    # Check fractionation factors
    for name, value in [('alpha_g', alpha_g), ('alpha_old', alpha_old), 
                       ('alpha_s', alpha_s), ('alpha_t', alpha_t)]:
        if value < 0 or value > 1:
            raise ValueError(f"{name} must be between 0 and 1, got {value}")
    
    # Check combined fractionation constraint
    if alpha_g + 2 * alpha_t + 4 * alpha_s / 3 > 1:
        raise ValueError(
            "alpha_g + 2 * alpha_t + 4 * alpha_s / 3 must be <= 1"
        )


def identify_c3_limiting_process(result: C3AssimilationResult) -> np.ndarray:
    """
    Identify which process limits photosynthesis at each point.
    
    Args:
        result: C3AssimilationResult from calculate_c3_assimilation
        
    Returns:
        Array of strings indicating limiting process for each point:
        - 'Rubisco': Rubisco-limited
        - 'RuBP': RuBP-regeneration-limited  
        - 'TPU': TPU-limited
    """
    n_points = len(result.An)
    limiting_process = np.empty(n_points, dtype=object)
    
    # Compare carboxylation rates to identify limitation
    for i in range(n_points):
        wc = result.Wc[i]
        wj = result.Wj[i]
        wp = result.Wp[i]
        
        if wc <= wj and wc <= wp:
            limiting_process[i] = 'Rubisco'
        elif wj <= wc and wj <= wp:
            limiting_process[i] = 'RuBP'
        else:
            limiting_process[i] = 'TPU'
    
    return limiting_process