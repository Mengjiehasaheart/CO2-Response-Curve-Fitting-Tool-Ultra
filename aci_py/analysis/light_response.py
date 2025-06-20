

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Union, List
from scipy.optimize import differential_evolution, minimize
from lmfit import Model, Parameters

from ..core.data_structures import ExtendedDataFrame


def non_rectangular_hyperbola(
    I: Union[float, np.ndarray],
    phi: float,
    Amax: float,
    theta: float,
    Rd: float
) -> Union[float, np.ndarray]:
    """
    Calculate photosynthesis rate using the non-rectangular hyperbola model.
    
    The non-rectangular hyperbola is the most widely used model for light response curves
    in photosynthesis research. It accounts for the gradual transition between light-limited
    and light-saturated photosynthesis.
    
    Args:
        I: Photosynthetically active radiation (PAR) in µmol m⁻² s⁻¹
        phi: Apparent quantum yield (initial slope) in mol CO₂ mol⁻¹ photons
        Amax: Maximum gross photosynthesis rate in µmol m⁻² s⁻¹
        theta: Curvature factor (0-1), where 0 = rectangular hyperbola
        Rd: Dark respiration rate in µmol m⁻² s⁻¹
    
    Returns:
        Net photosynthesis rate (A) in µmol m⁻² s⁻¹
    
    Notes:
        The model equation is:
        A = (1/(2*theta)) * (phi*I + Amax - sqrt((phi*I + Amax)² - 4*theta*phi*I*Amax)) - Rd
    """
    # Calculate the discriminant for the quadratic solution
    term1 = phi * I + Amax
    discriminant = term1**2 - 4 * theta * phi * I * Amax
    
    # Ensure discriminant is non-negative
    discriminant = np.maximum(discriminant, 0)
    
    # Calculate gross photosynthesis using quadratic solution
    if theta > 0:
        Ag = (term1 - np.sqrt(discriminant)) / (2 * theta)
    else:
        # When theta = 0, use rectangular hyperbola
        Ag = (phi * I * Amax) / (phi * I + Amax)
    
    # Calculate net photosynthesis
    A = Ag - Rd
    
    return A


def rectangular_hyperbola(
    I: Union[float, np.ndarray],
    phi: float,
    Amax: float,
    Rd: float
) -> Union[float, np.ndarray]:
    """
    Calculate photosynthesis rate using the rectangular hyperbola model.
    
    This is a simplified version of the non-rectangular hyperbola with theta = 0.
    It's the classic Michaelis-Menten type response.
    
    Args:
        I: Photosynthetically active radiation (PAR) in µmol m⁻² s⁻¹
        phi: Apparent quantum yield in mol CO₂ mol⁻¹ photons
        Amax: Maximum gross photosynthesis rate in µmol m⁻² s⁻¹
        Rd: Dark respiration rate in µmol m⁻² s⁻¹
    
    Returns:
        Net photosynthesis rate (A) in µmol m⁻² s⁻¹
    """
    Ag = (phi * I * Amax) / (phi * I + Amax)
    A = Ag - Rd
    return A


def exponential_model(
    I: Union[float, np.ndarray],
    phi: float,
    Amax: float,
    Rd: float
) -> Union[float, np.ndarray]:
    """
    Calculate photosynthesis rate using the exponential model.
    
    This model assumes no photoinhibition and provides a smooth transition
    to light saturation.
    
    Args:
        I: Photosynthetically active radiation (PAR) in µmol m⁻² s⁻¹
        phi: Apparent quantum yield in mol CO₂ mol⁻¹ photons
        Amax: Maximum gross photosynthesis rate in µmol m⁻² s⁻¹
        Rd: Dark respiration rate in µmol m⁻² s⁻¹
    
    Returns:
        Net photosynthesis rate (A) in µmol m⁻² s⁻¹
    """
    Ag = Amax * (1 - np.exp(-phi * I / Amax))
    A = Ag - Rd
    return A


def initial_guess_light_response(
    exdf: ExtendedDataFrame,
    I_column: str = 'Qin',
    A_column: str = 'A'
) -> Dict[str, float]:
    """
    Generate initial parameter guesses for light response curve fitting.
    
    Args:
        exdf: Extended DataFrame containing light response data
        I_column: Name of PAR/light intensity column
        A_column: Name of net assimilation column
    
    Returns:
        Dictionary of initial parameter guesses
    """
    # Extract data
    I = exdf.data[I_column].values
    A = exdf.data[A_column].values
    
    # Sort by light intensity
    sort_idx = np.argsort(I)
    I_sorted = I[sort_idx]
    A_sorted = A[sort_idx]
    
    # Estimate Rd from low light points (I < 50)
    low_light_mask = I_sorted < 50
    if np.sum(low_light_mask) >= 2:
        # Linear regression for low light
        coeffs = np.polyfit(I_sorted[low_light_mask], A_sorted[low_light_mask], 1)
        phi_guess = coeffs[0]  # Initial slope
        Rd_guess = -coeffs[1]  # Y-intercept (negative because Rd is positive)
    else:
        # Use first few points
        phi_guess = (A_sorted[1] - A_sorted[0]) / (I_sorted[1] - I_sorted[0])
        Rd_guess = -np.min(A_sorted)
    
    # Ensure positive values
    phi_guess = max(phi_guess, 0.001)
    Rd_guess = max(Rd_guess, 0)
    
    # Estimate Amax from high light points
    high_light_mask = I_sorted > 0.8 * np.max(I_sorted)
    if np.sum(high_light_mask) >= 3:
        Amax_net = np.mean(A_sorted[high_light_mask])
    else:
        Amax_net = np.max(A_sorted)
    
    # Gross Amax = net Amax + Rd
    Amax_guess = Amax_net + Rd_guess
    
    # Default theta for non-rectangular hyperbola
    theta_guess = 0.7  # Common value for many species
    
    return {
        'phi': phi_guess,
        'Amax': Amax_guess,
        'theta': theta_guess,
        'Rd': Rd_guess
    }


def fit_light_response(
    exdf: ExtendedDataFrame,
    I_column: str = 'Qin',
    A_column: str = 'A',
    model_type: str = 'non_rectangular_hyperbola',
    fit_theta: bool = True,
    initial_guess: Optional[Dict[str, float]] = None,
    bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    fixed_parameters: Optional[Dict[str, float]] = None,
    optimizer_params: Optional[Dict] = None
) -> Dict:
    """
    Fit a light response curve to photosynthesis data.
    
    Args:
        exdf: Extended DataFrame containing light response data
        I_column: Name of PAR/light intensity column
        A_column: Name of net assimilation column
        model_type: Type of model ('non_rectangular_hyperbola', 'rectangular_hyperbola', 'exponential')
        fit_theta: Whether to fit theta parameter (only for non_rectangular_hyperbola)
        initial_guess: Optional dictionary of initial parameter values
        bounds: Optional dictionary of parameter bounds
        fixed_parameters: Optional dictionary of parameters to fix
        optimizer_params: Optional parameters for the optimizer
    
    Returns:
        Dictionary containing:
            - 'parameters': Fitted parameter values
            - 'statistics': R², RMSE, AIC, BIC
            - 'predicted': Model predictions
            - 'residuals': Fit residuals
            - 'model_type': Model used
            - 'convergence': Optimization convergence info
    """
    # Validate inputs
    if I_column not in exdf.data.columns:
        raise ValueError(f"Column '{I_column}' not found in data")
    if A_column not in exdf.data.columns:
        raise ValueError(f"Column '{A_column}' not found in data")
    
    # Extract data
    I = exdf.data[I_column].values
    A = exdf.data[A_column].values
    
    # Remove NaN values
    valid_mask = ~(np.isnan(I) | np.isnan(A))
    I_valid = I[valid_mask]
    A_valid = A[valid_mask]
    
    if len(I_valid) < 4:
        raise ValueError("Insufficient valid data points for fitting")
    
    # Get initial guesses
    if initial_guess is None:
        # Create subset with valid data
        valid_indices = np.where(valid_mask)[0]
        valid_exdf = exdf.subset_rows(valid_indices)
        initial_guess = initial_guess_light_response(
            valid_exdf,
            I_column,
            A_column
        )
    
    # Set up model
    if model_type == 'non_rectangular_hyperbola':
        model_func = non_rectangular_hyperbola
        param_names = ['phi', 'Amax', 'theta', 'Rd'] if fit_theta else ['phi', 'Amax', 'Rd']
        if not fit_theta:
            fixed_parameters = fixed_parameters or {}
            fixed_parameters['theta'] = initial_guess.get('theta', 0.7)
    elif model_type == 'rectangular_hyperbola':
        model_func = rectangular_hyperbola
        param_names = ['phi', 'Amax', 'Rd']
    elif model_type == 'exponential':
        model_func = exponential_model
        param_names = ['phi', 'Amax', 'Rd']
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Set default bounds
    default_bounds = {
        'phi': (0, 0.2),      # Quantum yield typically < 0.1
        'Amax': (0, 100),     # Maximum assimilation
        'theta': (0, 1),      # Curvature factor
        'Rd': (0, 10)         # Dark respiration
    }
    
    if bounds is None:
        bounds = {}
    
    # Merge with default bounds
    for param in param_names:
        if param not in bounds:
            bounds[param] = default_bounds.get(param, (0, np.inf))
    
    # Define objective function
    def objective(params_array):
        params_dict = {name: params_array[i] for i, name in enumerate(param_names)}
        
        # Add fixed parameters
        if fixed_parameters:
            params_dict.update(fixed_parameters)
        
        # Calculate predictions
        if model_type == 'non_rectangular_hyperbola':
            predictions = model_func(I_valid, **params_dict)
        else:
            # For models without theta
            predictions = model_func(
                I_valid,
                params_dict['phi'],
                params_dict['Amax'],
                params_dict['Rd']
            )
        
        # Calculate residuals
        residuals = A_valid - predictions
        
        # Return sum of squared residuals
        return np.sum(residuals**2)
    
    # Prepare bounds for optimizer
    bounds_list = [bounds[param] for param in param_names]
    initial_array = [initial_guess[param] for param in param_names]
    
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
        popsize=optimizer_params.get('popsize', 15),
        **{k: v for k, v in optimizer_params.items() if k not in ['maxiter', 'popsize']}
    )
    
    # Extract fitted parameters
    fitted_params = {name: result.x[i] for i, name in enumerate(param_names)}
    if fixed_parameters:
        fitted_params.update(fixed_parameters)
    
    # Calculate predictions with fitted parameters
    if model_type == 'non_rectangular_hyperbola':
        predictions = model_func(I_valid, **fitted_params)
        all_predictions = model_func(I, **fitted_params)
    else:
        predictions = model_func(
            I_valid,
            fitted_params['phi'],
            fitted_params['Amax'],
            fitted_params['Rd']
        )
        all_predictions = model_func(
            I,
            fitted_params['phi'],
            fitted_params['Amax'],
            fitted_params['Rd']
        )
    
    # Calculate statistics
    residuals = A_valid - predictions
    n = len(A_valid)
    k = len(param_names)
    
    # R-squared
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((A_valid - np.mean(A_valid))**2)
    r_squared = 1 - (ss_res / ss_tot)
    
    # RMSE
    rmse = np.sqrt(ss_res / n)
    
    # AIC and BIC
    log_likelihood = -0.5 * n * (np.log(2 * np.pi) + np.log(ss_res / n) + 1)
    aic = 2 * k - 2 * log_likelihood
    bic = k * np.log(n) - 2 * log_likelihood
    
    # Calculate light compensation point (where A = 0)
    if fitted_params['Rd'] < fitted_params['Amax']:
        # Use numerical solution to find compensation point
        I_test = np.linspace(0, 200, 1000)
        if model_type == 'non_rectangular_hyperbola':
            A_test = model_func(I_test, **fitted_params)
        else:
            A_test = model_func(
                I_test,
                fitted_params['phi'],
                fitted_params['Amax'],
                fitted_params['Rd']
            )
        
        # Find where A crosses zero
        zero_crossings = np.where(np.diff(np.sign(A_test)))[0]
        if len(zero_crossings) > 0:
            lcp = I_test[zero_crossings[0]]
        else:
            lcp = None
    else:
        lcp = None
    
    # Calculate light saturation point (95% of Amax)
    saturation_A = 0.95 * (fitted_params['Amax'] - fitted_params['Rd'])
    I_test = np.linspace(0, 2000, 1000)
    if model_type == 'non_rectangular_hyperbola':
        A_test = model_func(I_test, **fitted_params)
    else:
        A_test = model_func(
            I_test,
            fitted_params['phi'],
            fitted_params['Amax'],
            fitted_params['Rd']
        )
    
    saturation_idx = np.where(A_test >= saturation_A)[0]
    if len(saturation_idx) > 0:
        lsp = I_test[saturation_idx[0]]
    else:
        lsp = None
    
    # Prepare results
    results = {
        'parameters': fitted_params,
        'statistics': {
            'r_squared': r_squared,
            'rmse': rmse,
            'aic': aic,
            'bic': bic,
            'n_points': n,
            'n_parameters': k,
            'light_compensation_point': lcp,
            'light_saturation_point': lsp
        },
        'predicted': all_predictions,
        'residuals': np.full_like(A, np.nan),
        'model_type': model_type,
        'convergence': {
            'success': result.success,
            'message': result.message,
            'n_iterations': result.nit,
            'n_evaluations': result.nfev,
            'final_cost': result.fun
        }
    }
    
    # Fill in residuals for valid points
    results['residuals'][valid_mask] = residuals
    
    return results


def compare_light_models(
    exdf: ExtendedDataFrame,
    I_column: str = 'Qin',
    A_column: str = 'A',
    models: Optional[List[str]] = None,
    **kwargs
) -> Dict[str, Dict]:
    """
    Compare multiple light response models on the same dataset.
    
    Args:
        exdf: Extended DataFrame containing light response data
        I_column: Name of PAR/light intensity column
        A_column: Name of net assimilation column
        models: List of model types to compare (default: all available)
        **kwargs: Additional arguments passed to fit_light_response
    
    Returns:
        Dictionary with model names as keys and fit results as values
    """
    if models is None:
        models = ['non_rectangular_hyperbola', 'rectangular_hyperbola', 'exponential']
    
    results = {}
    
    for model in models:
        try:
            fit_result = fit_light_response(
                exdf,
                I_column,
                A_column,
                model_type=model,
                **kwargs
            )
            results[model] = fit_result
        except Exception as e:
            print(f"Failed to fit {model}: {str(e)}")
            results[model] = {'error': str(e)}
    
    # Add model comparison summary
    valid_models = [m for m in results if 'error' not in results[m]]
    if valid_models:
        aic_values = {m: results[m]['statistics']['aic'] for m in valid_models}
        best_model = min(aic_values, key=aic_values.get)
        
        results['comparison_summary'] = {
            'best_model': best_model,
            'aic_values': aic_values,
            'delta_aic': {m: aic_values[m] - aic_values[best_model] for m in valid_models}
        }
    
    return results