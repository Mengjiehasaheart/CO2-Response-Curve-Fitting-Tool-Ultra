"""
C4 photosynthesis calculations based on the von Caemmerer (2000) model.

This module implements the C4 photosynthesis model as described in:
- von Caemmerer, S. (2000). Biochemical models of leaf photosynthesis.
- von Caemmerer, S. (2021). Updating the steady-state model of C4 photosynthesis.

The model calculates net CO2 assimilation rate (An) based on the minimum of 
enzyme-limited and light-limited rates, accounting for bundle sheath CO2 
concentration and the unique C4 carbon concentrating mechanism.
"""

import numpy as np
from typing import Dict, Optional, Union, Tuple
from .data_structures import ExtendedDataFrame
from .temperature import apply_temperature_response


def quadratic_root_minus(a: float, b: float, c: float) -> float:
    """
    Find the smaller root of ax² + bx + c = 0 using the quadratic formula.
    
    Returns the root using x = (-b - sqrt(b² - 4ac)) / (2a).
    If no real roots exist, returns NaN.
    """
    if abs(a) < 1e-15:
        # Linear equation: bx + c = 0
        return -c / b if abs(b) > 1e-15 else np.nan
    
    discriminant = b**2 - 4*a*c
    if discriminant < 0:
        return np.nan
    
    return (-b - np.sqrt(discriminant)) / (2*a)


def get_array_value(
    value: Union[float, np.ndarray], 
    n_points: int
) -> np.ndarray:
    """
    Convert a scalar or array to an array of the specified length.
    
    Parameters
    ----------
    value : float or array-like
        The value to convert
    n_points : int
        The desired array length
        
    Returns
    -------
    np.ndarray
        Array of length n_points
    """
    if np.isscalar(value):
        return np.full(n_points, value)
    else:
        return np.asarray(value)


def calculate_c4_assimilation(
    exdf: ExtendedDataFrame,
    alpha_psii: Union[float, np.ndarray] = 0.0,  # Fraction of PSII in bundle sheath
    gbs: Union[float, np.ndarray] = 0.003,       # Bundle sheath conductance (mol/m²/s/bar)
    J_at_25: Union[float, np.ndarray] = 1000,    # Max electron transport rate at 25°C
    RL_at_25: Union[float, np.ndarray] = 1.0,    # Day respiration at 25°C
    Rm_frac: Union[float, np.ndarray] = 0.5,     # Fraction of respiration in mesophyll
    Vcmax_at_25: Union[float, np.ndarray] = 100, # Max carboxylation rate at 25°C
    Vpmax_at_25: Union[float, np.ndarray] = 150, # Max PEP carboxylation rate at 25°C
    Vpr: Union[float, np.ndarray] = 80,          # PEP regeneration rate
    x_etr: float = 0.4,                          # Fraction of electron transport for C4 cycle
    temperature_response_params: Optional[Dict] = None,
    return_extended: bool = True,
    check_inputs: bool = True
) -> Union[np.ndarray, ExtendedDataFrame]:
    """
    Calculate C4 photosynthesis using the von Caemmerer model.
    
    The model calculates net assimilation as the minimum of enzyme-limited (Ac)
    and light-limited (Aj) rates. The enzyme-limited rate depends on the minimum
    of Rubisco carboxylation (Ar), PEP carboxylation (Ap), and PEP regeneration (Apr).
    
    Parameters
    ----------
    exdf : ExtendedDataFrame
        Extended data frame containing required columns:
        - PCm: Mesophyll CO2 partial pressure (µbar)
        - ao: Fraction of O2 evolved in bundle sheath
        - gamma_star: CO2 compensation point (dimensionless) 
        - Kc: Michaelis constant for CO2 (µbar)
        - Ko: Michaelis constant for O2 (mbar)
        - Kp: Michaelis constant for PEP carboxylation (µbar)
        - oxygen: O2 percentage
        - total_pressure: Atmospheric pressure (bar)
        - Temperature response normalized parameters (if not using defaults)
        
    alpha_psii : float or array-like, optional
        Fraction of PSII activity in bundle sheath cells (0-1).
        Default is 0 (all PSII in mesophyll).
        
    gbs : float or array-like, optional
        Bundle sheath conductance to CO2 (mol/m²/s/bar).
        Default is 0.003.
        
    J_at_25 : float or array-like, optional
        Maximum electron transport rate at 25°C (µmol/m²/s).
        
    RL_at_25 : float or array-like, optional
        Day respiration rate at 25°C (µmol/m²/s).
        
    Rm_frac : float or array-like, optional
        Fraction of respiration occurring in mesophyll cells (0-1).
        Default is 0.5.
        
    Vcmax_at_25 : float or array-like, optional
        Maximum Rubisco carboxylation rate at 25°C (µmol/m²/s).
        
    Vpmax_at_25 : float or array-like, optional
        Maximum PEP carboxylation rate at 25°C (µmol/m²/s).
        
    Vpr : float or array-like, optional
        PEP regeneration rate (µmol/m²/s).
        
    x_etr : float, optional
        Fraction of electron transport used for C4 cycle.
        Default is 0.4.
        
    temperature_response_params : dict, optional
        Temperature response parameters. If None, uses defaults.
        
    return_extended : bool, optional
        If True, returns ExtendedDataFrame with all calculated values.
        If False, returns only the net assimilation array.
        
    check_inputs : bool, optional
        If True, validates input parameters and data.
        
    Returns
    -------
    np.ndarray or ExtendedDataFrame
        If return_extended is False: Array of net assimilation rates
        If return_extended is True: ExtendedDataFrame with all calculations
        
    References
    ----------
    von Caemmerer, S. (2000). Biochemical models of leaf photosynthesis. 
    Techniques in Plant Sciences No. 2. CSIRO Publishing, Collingwood.
    """
    
    # Extract required columns from exdf
    n_points = len(exdf.data)
    
    # Get mesophyll CO2 partial pressure
    if 'PCm' not in exdf.data.columns:
        raise ValueError("PCm (mesophyll CO2 partial pressure) must be in the data")
    Cm = exdf.data['PCm'].values  # microbar
    
    # Get other required parameters
    ao = exdf.data['ao'].values if 'ao' in exdf.data.columns else 0.21
    gamma_star = exdf.data['gamma_star'].values
    Kc = exdf.data['Kc'].values  # microbar
    Ko = exdf.data['Ko'].values * 1000  # Convert from mbar to microbar
    Kp = exdf.data['Kp'].values  # microbar
    oxygen = exdf.data['oxygen'].values
    pressure = exdf.data['total_pressure'].values  # bar
    
    # Calculate oxygen partial pressure
    POm = oxygen * pressure * 1e4  # microbar
    
    # Apply temperature responses if normalized columns are present
    if 'Vcmax_norm' in exdf.data.columns:
        Vcmax_tl = get_array_value(Vcmax_at_25, n_points) * exdf.data['Vcmax_norm'].values
    else:
        Vcmax_tl = get_array_value(Vcmax_at_25, n_points)
        
    if 'Vpmax_norm' in exdf.data.columns:
        Vpmax_tl = get_array_value(Vpmax_at_25, n_points) * exdf.data['Vpmax_norm'].values
    else:
        Vpmax_tl = get_array_value(Vpmax_at_25, n_points)
        
    if 'RL_norm' in exdf.data.columns:
        RL_tl = get_array_value(RL_at_25, n_points) * exdf.data['RL_norm'].values
    else:
        RL_tl = get_array_value(RL_at_25, n_points)
        
    if 'J_norm' in exdf.data.columns:
        J_tl = get_array_value(J_at_25, n_points) * exdf.data['J_norm'].values
    else:
        J_tl = get_array_value(J_at_25, n_points)
    
    # Calculate mesophyll respiration rate
    Rm_frac_arr = get_array_value(Rm_frac, n_points)
    RLm_tl = Rm_frac_arr * RL_tl
    
    # Get array values for other parameters
    alpha_psii_arr = get_array_value(alpha_psii, n_points)
    gbs_arr = get_array_value(gbs, n_points)
    Vpr_arr = get_array_value(Vpr, n_points)
    
    # Input validation if requested
    if check_inputs:
        # Check parameter bounds
        if np.any(alpha_psii_arr < 0) or np.any(alpha_psii_arr > 1):
            raise ValueError("alpha_psii must be between 0 and 1")
        if np.any(gbs_arr < 0):
            raise ValueError("gbs must be >= 0")
        if np.any(Rm_frac_arr < 0) or np.any(Rm_frac_arr > 1):
            raise ValueError("Rm_frac must be between 0 and 1")
        if np.any(x_etr < 0) or np.any(x_etr > 1):
            raise ValueError("x_etr must be between 0 and 1")
            
        # Check for negative rates
        if np.any(Vcmax_tl < 0):
            raise ValueError("Vcmax must be >= 0")
        if np.any(Vpmax_tl < 0):
            raise ValueError("Vpmax must be >= 0")
        if np.any(J_tl < 0):
            raise ValueError("J must be >= 0")
        if np.any(RL_tl < 0):
            raise ValueError("RL must be >= 0")
        if np.any(Vpr_arr < 0):
            raise ValueError("Vpr must be >= 0")
    
    # Calculate PEP carboxylation rates (Equations 4.17 and 4.19)
    Vpc = Cm * Vpmax_tl / (Cm + Kp)  # CO2-limited PEP carboxylation
    Vp = np.minimum(Vpc, Vpr_arr)    # Actual PEP carboxylation (limited by PEP regeneration)
    
    # Calculate individual process-limited assimilation rates
    Apr = Vpr_arr - RLm_tl + gbs_arr * Cm              # PEP-regeneration limited
    Apc = Vpc - RLm_tl + gbs_arr * Cm                  # CO2-limited PEP carboxylation
    Ap = Vp - RLm_tl + gbs_arr * Cm                    # PEP carboxylation limited (Eq 4.25)
    Ar = Vcmax_tl - RL_tl                              # Rubisco limited (Eq 4.25)
    Ajm = x_etr * J_tl / 2 - RLm_tl + gbs_arr * Cm    # Mesophyll electron transport (Eq 4.45)
    Ajbs = (1 - x_etr) * J_tl / 3 - RL_tl             # Bundle sheath electron transport (Eq 4.45)
    
    # Calculate terms for enzyme-limited quadratic (used in Equations 4.21-4.24)
    f1 = alpha_psii_arr / ao                          # dimensionless
    f2 = gbs_arr * Kc * (1.0 + POm / Ko)            # µmol/m²/s
    f3 = gamma_star * Vcmax_tl                       # µmol/m²/s
    f4 = Kc / Ko                                      # dimensionless
    f5 = 7 * gamma_star / 3                          # dimensionless
    f6 = (1 - x_etr) * J_tl / 3 + 7 * RL_tl / 3     # µmol/m²/s
    
    # Enzyme-limited quadratic coefficients (Equations 4.22-4.24)
    ea = 1.0 - f1 * f4                                            # dimensionless
    eb = -(Ap + Ar + f2 + f1 * (f3 + RL_tl * f4))                # µmol/m²/s
    ec = Ar * Ap - (f3 * gbs_arr * POm + RL_tl * f2)            # (µmol/m²/s)²
    
    # Calculate enzyme-limited assimilation rate (Equation 4.21)
    Ac = np.zeros_like(Cm)
    for i in range(len(Cm)):
        Ac[i] = quadratic_root_minus(ea[i], eb[i], ec[i])
    
    # Light-limited quadratic coefficients (Equations 4.42-4.44)
    la = 1.0 - f1 * f5                                                      # dimensionless
    lb = -(Ajm + Ajbs + gbs_arr * POm * f5 + gamma_star * f1 * f6)        # µmol/m²/s
    lc = Ajm * Ajbs - gamma_star * gbs_arr * POm * f6                     # (µmol/m²/s)²
    
    # Calculate light-limited assimilation rate (Equation 4.41)
    Aj = np.zeros_like(Cm)
    for i in range(len(Cm)):
        Aj[i] = quadratic_root_minus(la[i], lb[i], lc[i])
    
    # Net assimilation is the minimum of enzyme and light limited rates (Equation 4.47)
    An = np.minimum(Ac, Aj)
    
    # Replace any NaN or negative values with 0
    An = np.where(np.isnan(An) | (An < 0), 0, An)
    
    if not return_extended:
        return An
    
    # Create extended output with all calculated values
    result_data = {
        'alpha_psii': alpha_psii_arr,
        'gbs': gbs_arr,
        'J_at_25': get_array_value(J_at_25, n_points),
        'J_tl': J_tl,
        'RL_at_25': get_array_value(RL_at_25, n_points),
        'RL_tl': RL_tl,
        'Rm_frac': Rm_frac_arr,
        'RLm_tl': RLm_tl,
        'Vcmax_at_25': get_array_value(Vcmax_at_25, n_points),
        'Vcmax_tl': Vcmax_tl,
        'Vpmax_at_25': get_array_value(Vpmax_at_25, n_points),
        'Vpmax_tl': Vpmax_tl,
        'Vpr': Vpr_arr,
        'Vpc': Vpc,
        'Vp': Vp,
        'Apc': Apc,
        'Apr': Apr,
        'Ap': Ap,
        'Ar': Ar,
        'Ajm': Ajm,
        'Ajbs': Ajbs,
        'Ac': Ac,
        'Aj': Aj,
        'An': An,
        'An_model': An  # For consistency with fitting functions
    }
    
    # Define units for all output variables
    units = {
        'alpha_psii': 'dimensionless',
        'gbs': 'mol m^(-2) s^(-1) bar^(-1)',
        'J_at_25': 'micromol m^(-2) s^(-1)',
        'J_tl': 'micromol m^(-2) s^(-1)',
        'RL_at_25': 'micromol m^(-2) s^(-1)',
        'RL_tl': 'micromol m^(-2) s^(-1)',
        'Rm_frac': 'dimensionless',
        'RLm_tl': 'micromol m^(-2) s^(-1)',
        'Vcmax_at_25': 'micromol m^(-2) s^(-1)',
        'Vcmax_tl': 'micromol m^(-2) s^(-1)',
        'Vpmax_at_25': 'micromol m^(-2) s^(-1)',
        'Vpmax_tl': 'micromol m^(-2) s^(-1)',
        'Vpr': 'micromol m^(-2) s^(-1)',
        'Vpc': 'micromol m^(-2) s^(-1)',
        'Vp': 'micromol m^(-2) s^(-1)',
        'Apc': 'micromol m^(-2) s^(-1)',
        'Apr': 'micromol m^(-2) s^(-1)',
        'Ap': 'micromol m^(-2) s^(-1)',
        'Ar': 'micromol m^(-2) s^(-1)',
        'Ajm': 'micromol m^(-2) s^(-1)',
        'Ajbs': 'micromol m^(-2) s^(-1)',
        'Ac': 'micromol m^(-2) s^(-1)',
        'Aj': 'micromol m^(-2) s^(-1)',
        'An': 'micromol m^(-2) s^(-1)',
        'An_model': 'micromol m^(-2) s^(-1)'
    }
    
    # Create categories for all variables
    categories = {col: 'calculate_c4_assimilation' for col in result_data.keys()}
    
    # Create result ExtendedDataFrame
    result = ExtendedDataFrame(result_data, units=units, categories=categories)
    
    # Merge with original data to preserve all columns
    for col in exdf.data.columns:
        if col not in result.data.columns:
            result.data[col] = exdf.data[col]
            if col in exdf.units:
                result.units[col] = exdf.units[col]
            if col in exdf.categories:
                result.categories[col] = exdf.categories[col]
    
    return result


def identify_c4_limiting_processes(
    exdf: ExtendedDataFrame,
    limiting_tolerance: float = 1e-10
) -> ExtendedDataFrame:
    """
    Identify which process is limiting C4 photosynthesis at each point.
    
    Parameters
    ----------
    exdf : ExtendedDataFrame
        Must contain columns: An, Ac, Aj, Ar, Ap, Apr from calculate_c4_assimilation
        
    limiting_tolerance : float, optional
        Tolerance for determining if two rates are equal.
        
    Returns
    -------
    ExtendedDataFrame
        Input data with added columns:
        - limiting_process: 'enzyme' or 'light'
        - enzyme_limited_process: 'rubisco', 'pep_carboxylation', or 'pep_regeneration'
    """
    result = exdf.copy()
    
    # Determine overall limiting process
    is_enzyme_limited = np.abs(exdf.data['An'] - exdf.data['Ac']) < limiting_tolerance
    result.data['limiting_process'] = np.where(is_enzyme_limited, 'enzyme', 'light')
    
    # For enzyme-limited points, determine specific limitation
    enzyme_process = np.full(len(exdf.data), '', dtype=object)
    
    # Check which enzyme process is limiting
    is_rubisco = np.abs(exdf.data['Ac'] - exdf.data['Ar']) < limiting_tolerance
    is_pep_carb = np.abs(exdf.data['Ac'] - exdf.data['Ap']) < limiting_tolerance
    
    # Ap can be limited by either CO2 (Apc) or PEP regeneration (Apr)
    is_co2_limited = np.abs(exdf.data['Ap'] - exdf.data['Apc']) < limiting_tolerance
    is_pep_regen = np.abs(exdf.data['Ap'] - exdf.data['Apr']) < limiting_tolerance
    
    enzyme_process[is_enzyme_limited & is_rubisco] = 'rubisco'
    enzyme_process[is_enzyme_limited & is_pep_carb & is_co2_limited] = 'pep_carboxylation_co2'
    enzyme_process[is_enzyme_limited & is_pep_carb & is_pep_regen] = 'pep_regeneration'
    
    result.data['enzyme_limited_process'] = enzyme_process
    
    # Add units and categories
    result.units['limiting_process'] = 'dimensionless'
    result.units['enzyme_limited_process'] = 'dimensionless'
    result.categories['limiting_process'] = 'identify_c4_limiting_processes'
    result.categories['enzyme_limited_process'] = 'identify_c4_limiting_processes'
    
    return result


def apply_gm_c4(
    exdf: ExtendedDataFrame,
    gmc_at_25: Union[float, np.ndarray] = 1.0,
    temperature_response_params: Optional[Dict] = None
) -> ExtendedDataFrame:
    """
    Apply mesophyll conductance to convert Ci to Cm for C4 photosynthesis.
    
    For C4 plants, this calculates the mesophyll CO2 partial pressure (PCm)
    from the intercellular CO2 partial pressure (PCi) using mesophyll conductance.
    
    Parameters
    ----------
    exdf : ExtendedDataFrame
        Must contain columns: PCi, A, total_pressure
        
    gmc_at_25 : float or array-like, optional
        Mesophyll conductance at 25°C (mol/m²/s/bar).
        Default is 1.0.
        
    temperature_response_params : dict, optional
        Temperature response parameters for gmc.
        
    Returns
    -------
    ExtendedDataFrame
        Input data with added columns:
        - gmc: Temperature-adjusted mesophyll conductance
        - PCm: Mesophyll CO2 partial pressure
    """
    result = exdf.copy()
    n_points = len(result.data)
    
    # Get array value for gmc
    gmc_25 = get_array_value(gmc_at_25, n_points)
    
    # Apply temperature response if available
    if 'gmc_norm' in result.data.columns:
        gmc = gmc_25 * result.data['gmc_norm'].values
    else:
        gmc = gmc_25
    
    # Calculate PCm from PCi
    # PCm = PCi - A / (gmc * Patm)
    PCi = result.data['PCi'].values
    A = result.data['A'].values
    Patm = result.data['total_pressure'].values
    
    PCm = PCi - A / (gmc * Patm)
    
    # Store results
    result.data['gmc'] = gmc
    result.data['PCm'] = PCm
    result.data['gmc_at_25'] = gmc_25
    
    # Add units
    result.units['gmc'] = 'mol m^(-2) s^(-1) bar^(-1)'
    result.units['PCm'] = 'microbar'
    result.units['gmc_at_25'] = 'mol m^(-2) s^(-1) bar^(-1)'
    
    # Add categories
    result.categories['gmc'] = 'apply_gm_c4'
    result.categories['PCm'] = 'apply_gm_c4'
    result.categories['gmc_at_25'] = 'apply_gm_c4'
    
    return result