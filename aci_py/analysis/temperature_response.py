
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Union, List
from scipy.optimize import differential_evolution, minimize, curve_fit
from lmfit import Model, Parameters

from ..core.data_structures import ExtendedDataFrame
from ..core.temperature import (
    arrhenius_response,
    johnson_eyring_williams_response,
    gaussian_response,
    polynomial_response
)


def quadratic_temperature_response(
    T: Union[float, np.ndarray],
    a: float,
    b: float,
    c: float
) -> Union[float, np.ndarray]:
    """
    Simple quadratic model for temperature response.
    
    Args:
        T: Temperature in °C
        a, b, c: Quadratic coefficients
    
    Returns:
        Parameter value at temperature T
    """
    return a + b * T + c * T**2


def gaussian_peak_model(
    T: Union[float, np.ndarray],
    amplitude: float,
    T_opt: float,
    width: float,
    baseline: float = 0
) -> Union[float, np.ndarray]:
    """
    Gaussian model for temperature response with a clear peak.
    
    Args:
        T: Temperature in °C
        amplitude: Peak height above baseline
        T_opt: Optimal temperature (peak position)
        width: Width of the response curve
        baseline: Baseline value
    
    Returns:
        Parameter value at temperature T
    """
    return baseline + amplitude * np.exp(-0.5 * ((T - T_opt) / width)**2)


def modified_arrhenius_deactivation(
    T: Union[float, np.ndarray],
    amplitude: float,
    Ea: float,
    Ed: float,
    Hd: float,
    T_opt: Optional[float] = None
) -> Union[float, np.ndarray]:
    """
    Modified Arrhenius model with high-temperature deactivation.
    
    This model combines Arrhenius activation with an entropy-driven
    deactivation at high temperatures, commonly used for Vcmax and Jmax.
    
    Args:
        T: Temperature in °C
        amplitude: Scaling factor
        Ea: Activation energy (J/mol)
        Ed: Deactivation energy (J/mol)
        Hd: Entropy term for deactivation (J/mol/K)
        T_opt: Reference optimal temperature (optional, calculated if None)
    
    Returns:
        Parameter value at temperature T
    """
    T_K = T + 273.15
    R = 8.314  # J/mol/K
    
    # If T_opt not provided, calculate it
    if T_opt is None:
        # Analytical solution for temperature optimum
        T_opt_K = Ed / (Hd / R - np.log(Ed / Ea))
        T_opt = T_opt_K - 273.15
    else:
        T_opt_K = T_opt + 273.15
    
    # Arrhenius activation term
    f_activation = np.exp(Ea * (T_K - 298.15) / (R * T_K * 298.15))
    
    # Deactivation term
    numerator = 1 + np.exp((Hd - Ed) / (R * 298.15))
    denominator = 1 + np.exp((Hd - Ed) / (R * T_K))
    f_deactivation = numerator / denominator
    
    # Combined response
    response = amplitude * f_activation * f_deactivation
    
    return response


def thermal_performance_curve(
    T: Union[float, np.ndarray],
    T_opt: float,
    T_min: float,
    T_max: float,
    amplitude: float,
    skewness: float = 0.5
) -> Union[float, np.ndarray]:
    """
    Thermal performance curve model with flexible shape.
    
    This model allows for asymmetric responses around the optimum temperature,
    which is common in biological systems.
    
    Args:
        T: Temperature in °C
        T_opt: Optimal temperature
        T_min: Minimum temperature (performance = 0)
        T_max: Maximum temperature (performance = 0)
        amplitude: Maximum performance at T_opt
        skewness: Asymmetry parameter (0.5 = symmetric)
    
    Returns:
        Performance value at temperature T
    """
    # Check if input is scalar
    is_scalar = np.isscalar(T)
    
    # Ensure T is array-like
    T = np.atleast_1d(T)
    
    # Initialize output
    performance = np.zeros_like(T, dtype=float)
    
    # Only calculate for temperatures within range
    valid_mask = (T > T_min) & (T < T_max)
    
    if np.any(valid_mask):
        T_valid = T[valid_mask]
        
        # Calculate performance for valid temperatures
        for i, t in enumerate(T_valid):
            if t <= T_opt:
                # Below optimum
                normalized = (t - T_min) / (T_opt - T_min)
                perf = amplitude * (normalized ** skewness)
            else:
                # Above optimum
                normalized = (T_max - t) / (T_max - T_opt)
                perf = amplitude * (normalized ** (1 / skewness))
            
            # Find where to put this value back
            idx = np.where((T == t) & valid_mask)[0][0]
            performance[idx] = perf
    
    return performance[0] if is_scalar else performance


def initial_guess_temperature_response(
    exdf: ExtendedDataFrame,
    T_column: str = 'Tleaf',
    param_column: str = 'A',
    model_type: str = 'gaussian_peak'
) -> Dict[str, float]:
    """
    Generate initial parameter guesses for temperature response fitting.
    
    Args:
        exdf: Extended DataFrame containing temperature response data
        T_column: Name of temperature column
        param_column: Name of parameter column (e.g., 'A', 'Vcmax')
        model_type: Type of model being fitted
    
    Returns:
        Dictionary of initial parameter guesses
    """
    # Extract data
    T = exdf.data[T_column].values
    param = exdf.data[param_column].values
    
    # Remove NaN values
    valid_mask = ~(np.isnan(T) | np.isnan(param))
    T_valid = T[valid_mask]
    param_valid = param[valid_mask]
    
    # Find optimal temperature (maximum parameter value)
    max_idx = np.argmax(param_valid)
    T_opt_guess = T_valid[max_idx]
    max_param = param_valid[max_idx]
    
    # Find temperature range
    T_min = np.min(T_valid)
    T_max = np.max(T_valid)
    
    # Model-specific guesses
    if model_type == 'gaussian_peak':
        # Estimate width from temperature range where param > 50% of max
        half_max_mask = param_valid > 0.5 * max_param
        if np.sum(half_max_mask) > 2:
            T_half_max = T_valid[half_max_mask]
            width_guess = (np.max(T_half_max) - np.min(T_half_max)) / 2.355  # FWHM to sigma
        else:
            width_guess = (T_max - T_min) / 4
        
        return {
            'amplitude': max_param,
            'T_opt': T_opt_guess,
            'width': width_guess,
            'baseline': 0
        }
    
    elif model_type == 'quadratic':
        # Fit quadratic to get initial guesses
        coeffs = np.polyfit(T_valid, param_valid, 2)
        return {
            'a': coeffs[2],
            'b': coeffs[1],
            'c': coeffs[0]
        }
    
    elif model_type == 'modified_arrhenius':
        # Typical values for photosynthetic parameters
        return {
            'amplitude': max_param / 2,  # Normalize to ~25°C
            'Ea': 50000,  # J/mol, typical activation energy
            'Ed': 200000,  # J/mol, typical deactivation energy
            'Hd': 650,  # J/mol/K, typical entropy term
            'T_opt': T_opt_guess
        }
    
    elif model_type == 'thermal_performance':
        # Find approximate temperature limits (10% of max)
        threshold = 0.1 * max_param
        above_threshold = param_valid > threshold
        if np.sum(above_threshold) > 2:
            T_above = T_valid[above_threshold]
            T_min_guess = np.min(T_above) - 5
            T_max_guess = np.max(T_above) + 5
        else:
            T_min_guess = T_min - 5
            T_max_guess = T_max + 5
        
        return {
            'T_opt': T_opt_guess,
            'T_min': T_min_guess,
            'T_max': T_max_guess,
            'amplitude': max_param,
            'skewness': 0.5
        }
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def fit_temperature_response(
    exdf: ExtendedDataFrame,
    T_column: str = 'Tleaf',
    param_column: str = 'A',
    model_type: str = 'gaussian_peak',
    initial_guess: Optional[Dict[str, float]] = None,
    bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    fixed_parameters: Optional[Dict[str, float]] = None,
    optimizer_params: Optional[Dict] = None,
    weights: Optional[np.ndarray] = None
) -> Dict:
    """
    Fit a temperature response curve to photosynthetic parameter data.
    
    Args:
        exdf: Extended DataFrame containing temperature response data
        T_column: Name of temperature column
        param_column: Name of parameter column (e.g., 'A', 'Vcmax', 'Jmax')
        model_type: Type of model ('gaussian_peak', 'quadratic', 'modified_arrhenius', 
                    'thermal_performance')
        initial_guess: Optional dictionary of initial parameter values
        bounds: Optional dictionary of parameter bounds
        fixed_parameters: Optional dictionary of parameters to fix
        optimizer_params: Optional parameters for the optimizer
        weights: Optional weights for weighted least squares
    
    Returns:
        Dictionary containing:
            - 'parameters': Fitted parameter values
            - 'statistics': R², RMSE, AIC, BIC
            - 'predicted': Model predictions
            - 'residuals': Fit residuals
            - 'model_type': Model used
            - 'T_opt': Optimal temperature
            - 'performance_range': Temperature range for 90% performance
            - 'convergence': Optimization convergence info
    """
    # Validate inputs
    if T_column not in exdf.data.columns:
        raise ValueError(f"Column '{T_column}' not found in data")
    if param_column not in exdf.data.columns:
        raise ValueError(f"Column '{param_column}' not found in data")
    
    # Extract data
    T = exdf.data[T_column].values
    param = exdf.data[param_column].values
    
    # Remove NaN values
    valid_mask = ~(np.isnan(T) | np.isnan(param))
    T_valid = T[valid_mask]
    param_valid = param[valid_mask]
    
    # Require at least 3 points for most models, but quadratic needs 4
    min_points = 4 if model_type == 'quadratic' else 3
    if len(T_valid) < min_points:
        raise ValueError(f"Insufficient valid data points for fitting (need at least {min_points})")
    
    # Get initial guesses
    if initial_guess is None:
        # Create subset with valid data
        valid_indices = np.where(valid_mask)[0]
        valid_exdf = exdf.subset_rows(valid_indices)
        initial_guess = initial_guess_temperature_response(
            valid_exdf,
            T_column,
            param_column,
            model_type
        )
    
    # Set up model
    model_funcs = {
        'gaussian_peak': gaussian_peak_model,
        'quadratic': quadratic_temperature_response,
        'modified_arrhenius': modified_arrhenius_deactivation,
        'thermal_performance': thermal_performance_curve
    }
    
    if model_type not in model_funcs:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model_func = model_funcs[model_type]
    param_names = list(initial_guess.keys())
    
    # Set default bounds
    default_bounds = {
        # Gaussian peak model
        'amplitude': (0, 1000),  # Reasonable upper limit for photosynthesis
        'T_opt': (0, 50),
        'width': (0.1, 50),
        'baseline': (-100, 100),  # Reasonable range for baseline
        # Quadratic model
        'a': (-1000, 1000),
        'b': (-100, 100),
        'c': (-10, 0),  # Negative for downward curve
        # Modified Arrhenius
        'Ea': (10000, 200000),
        'Ed': (50000, 500000),
        'Hd': (100, 2000),
        # Thermal performance
        'T_min': (-10, 30),
        'T_max': (20, 60),
        'skewness': (0.1, 10)
    }
    
    if bounds is None:
        bounds = {}
    
    # Merge with default bounds
    for param in param_names:
        if param not in bounds:
            bounds[param] = default_bounds.get(param, (-np.inf, np.inf))
    
    # Handle fixed parameters
    free_params = [p for p in param_names if p not in (fixed_parameters or {})]
    
    # Define objective function
    def objective(params_array):
        params_dict = {name: params_array[i] for i, name in enumerate(free_params)}
        
        # Add fixed parameters
        if fixed_parameters:
            params_dict.update(fixed_parameters)
        
        # Calculate predictions
        predictions = model_func(T_valid, **params_dict)
        
        # Calculate weighted residuals
        residuals = param_valid - predictions
        if weights is not None:
            residuals = residuals * np.sqrt(weights[valid_mask])
        
        # Return sum of squared residuals
        return np.sum(residuals**2)
    
    # Prepare bounds for optimizer
    bounds_list = [bounds[param] for param in free_params]
    initial_array = [initial_guess[param] for param in free_params]
    
    # Ensure initial values are within bounds
    for i, (lower, upper) in enumerate(bounds_list):
        initial_array[i] = np.clip(initial_array[i], lower, upper)
    
    # Run optimization
    if optimizer_params is None:
        optimizer_params = {}
    
    result = differential_evolution(
        objective,
        bounds_list,
        x0=initial_array,
        seed=42,
        maxiter=optimizer_params.get('maxiter', 1000),
        **{k: v for k, v in optimizer_params.items() if k != 'maxiter'}
    )
    
    # Extract fitted parameters
    fitted_params = {name: result.x[i] for i, name in enumerate(free_params)}
    if fixed_parameters:
        fitted_params.update(fixed_parameters)
    
    # Calculate predictions with fitted parameters
    predictions = model_func(T_valid, **fitted_params)
    all_predictions = model_func(T, **fitted_params)
    
    # Calculate statistics
    residuals = param_valid - predictions
    n = len(param_valid)
    k = len(free_params)
    
    # Weighted statistics if applicable
    if weights is not None:
        w = weights[valid_mask]
        ss_res = np.sum(w * residuals**2)
        ss_tot = np.sum(w * (param_valid - np.average(param_valid, weights=w))**2)
    else:
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((param_valid - np.mean(param_valid))**2)
    
    # R-squared
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # RMSE
    rmse = np.sqrt(ss_res / n)
    
    # AIC and BIC
    if ss_res > 0:
        log_likelihood = -0.5 * n * (np.log(2 * np.pi) + np.log(ss_res / n) + 1)
    else:
        log_likelihood = np.inf
    
    aic = 2 * k - 2 * log_likelihood
    bic = k * np.log(n) - 2 * log_likelihood
    
    # Find optimal temperature
    T_test = np.linspace(np.min(T_valid) - 5, np.max(T_valid) + 5, 1000)
    param_test = model_func(T_test, **fitted_params)
    T_opt_idx = np.argmax(param_test)
    T_opt = T_test[T_opt_idx]
    param_opt = param_test[T_opt_idx]
    
    # Find performance range (90% of optimum)
    threshold = 0.9 * param_opt
    above_threshold = param_test >= threshold
    if np.any(above_threshold):
        T_above = T_test[above_threshold]
        performance_range = (np.min(T_above), np.max(T_above))
    else:
        performance_range = (T_opt, T_opt)
    
    # Calculate Q10 (temperature coefficient) if applicable
    if model_type != 'thermal_performance':
        # Q10 between 15 and 25°C
        try:
            param_15 = model_func(15.0, **fitted_params)
            param_25 = model_func(25.0, **fitted_params)
            if param_15 > 0 and param_25 > 0:
                Q10 = (param_25 / param_15)
            else:
                Q10 = None
        except:
            Q10 = None
    else:
        Q10 = None
    
    # Initialize residuals array
    all_residuals = np.full_like(param, np.nan)
    all_residuals[valid_mask] = residuals
    
    # Prepare results
    results = {
        'parameters': fitted_params,
        'statistics': {
            'r_squared': r_squared,
            'rmse': rmse,
            'aic': aic,
            'bic': bic,
            'n_points': n,
            'n_parameters': k
        },
        'predicted': all_predictions,
        'residuals': all_residuals,
        'model_type': model_type,
        'T_opt': T_opt,
        'param_at_T_opt': param_opt,
        'performance_range': performance_range,
        'Q10': Q10,
        'convergence': {
            'success': result.success,
            'message': result.message,
            'n_iterations': result.nit,
            'n_evaluations': result.nfev,
            'final_cost': result.fun
        }
    }
    
    return results


def fit_arrhenius_with_photogea_params(
    exdf: ExtendedDataFrame,
    T_column: str = 'Tleaf',
    param_column: str = 'Vcmax',
    param_at_25: Optional[float] = None,
    Ha: Optional[float] = None,
    temperature_params: Optional[str] = 'bernacchi'
) -> Dict:
    """
    Fit temperature response using PhotoGEA-style Arrhenius parameters.
    
    This function specifically handles the temperature response fitting
    using the same parameterization as PhotoGEA for compatibility.
    
    Args:
        exdf: Extended DataFrame containing temperature response data
        T_column: Name of temperature column
        param_column: Name of parameter column
        param_at_25: Value at 25°C (if known)
        Ha: Activation energy (if known)
        temperature_params: Name of parameter set ('bernacchi', 'sharkey')
    
    Returns:
        Fitted parameters and statistics
    """
    # Import temperature parameter sets
    from ..core.temperature import (
        C3_TEMPERATURE_PARAM_BERNACCHI,
        C3_TEMPERATURE_PARAM_SHARKEY
    )
    
    # Get default Ha values if not provided
    if Ha is None:
        param_sets = {
            'bernacchi': C3_TEMPERATURE_PARAM_BERNACCHI,
            'sharkey': C3_TEMPERATURE_PARAM_SHARKEY
        }
        
        if temperature_params in param_sets:
            param_set = param_sets[temperature_params]
            
            # Map parameter names to Ha values
            # Check what's in the parameter set
            ha_map = {
                'Vcmax': 65330,  # Default values
                'Jmax': 43540,
                'Kc': 79430,
                'Ko': 36380,
                'gamma_star': 37830
            }
            
            # Update from parameter set if available
            for key, temp_param in param_set.items():
                if hasattr(temp_param, 'Ea') and temp_param.Ea is not None:
                    # Convert from kJ/mol to J/mol
                    ha_map[key] = temp_param.Ea * 1000
            
            Ha = ha_map.get(param_column, 50000)  # Default if not found
    
    # Extract data
    T = exdf.data[T_column].values
    param = exdf.data[param_column].values
    
    # Define simple Arrhenius function for fitting
    def arrhenius_func(T_C, param_25, Ha):
        # Simple Arrhenius: param(T) = param_25 * exp(Ha/R * (1/298.15 - 1/T_K))
        R = 8.314  # J/mol/K
        T_K = T_C + 273.15
        T_ref_K = 25 + 273.15
        return param_25 * np.exp((Ha / R) * (1/T_ref_K - 1/T_K))
    
    # Initial guess
    if param_at_25 is None:
        # Estimate from data near 25°C
        near_25_mask = np.abs(T - 25) < 2
        if np.any(near_25_mask):
            param_at_25 = np.mean(param[near_25_mask])
        else:
            param_at_25 = np.mean(param)
    
    # Fit using curve_fit
    try:
        popt, pcov = curve_fit(
            arrhenius_func,
            T,
            param,
            p0=[param_at_25, Ha],
            bounds=([0, 10000], [np.inf, 200000])
        )
        
        fitted_param_25, fitted_Ha = popt
        param_err_25, Ha_err = np.sqrt(np.diag(pcov))
        
        # Calculate predictions
        predictions = arrhenius_func(T, fitted_param_25, fitted_Ha)
        
        # Statistics
        residuals = param - predictions
        n = len(param)
        k = 2  # Two parameters
        
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((param - np.mean(param))**2)
        r_squared = 1 - (ss_res / ss_tot)
        rmse = np.sqrt(ss_res / n)
        
        # AIC and BIC
        log_likelihood = -0.5 * n * (np.log(2 * np.pi) + np.log(ss_res / n) + 1)
        aic = 2 * k - 2 * log_likelihood
        bic = k * np.log(n) - 2 * log_likelihood
        
        results = {
            'parameters': {
                f'{param_column}_at_25': fitted_param_25,
                f'Ha_{param_column}': fitted_Ha
            },
            'parameter_errors': {
                f'{param_column}_at_25': param_err_25,
                f'Ha_{param_column}': Ha_err
            },
            'statistics': {
                'r_squared': r_squared,
                'rmse': rmse,
                'aic': aic,
                'bic': bic
            },
            'predicted': predictions,
            'residuals': residuals,
            'success': True
        }
        
    except Exception as e:
        results = {
            'success': False,
            'error': str(e)
        }
    
    return results