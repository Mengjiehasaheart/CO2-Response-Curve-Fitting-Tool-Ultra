
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List, Union
from dataclasses import dataclass
import warnings

from ..core.data_structures import ExtendedDataFrame
from ..core.c3_calculations import calculate_c3_assimilation
from ..core.temperature import (
    apply_temperature_response,
    C3_TEMPERATURE_PARAM_BERNACCHI
)
from .optimization import (
    create_error_function,
    fit_with_differential_evolution,
    fit_with_nelder_mead,
    validate_optimization_result,
    calculate_aic,
    calculate_bic,
    calculate_confidence_intervals_profile,
    calculate_confidence_intervals_bootstrap
)
from .initial_guess import (
    estimate_c3_initial_parameters,
    estimate_c3_parameter_bounds,
    validate_initial_guess
)


@dataclass
class C3FitResult:
    """Container for C3 model fitting results."""
    # Fitted parameters
    parameters: Dict[str, float]
    
    # Optimization results
    success: bool
    error_value: float
    n_function_evals: int
    message: str
    
    # Model predictions
    fitted_A: np.ndarray
    residuals: np.ndarray
    
    # Limiting processes
    limiting_process: np.ndarray  # 'Wc', 'Wj', or 'Wp' for each point
    
    # Statistics
    rmse: float
    r_squared: float
    aic: float
    bic: float
    
    # Warnings
    warnings: List[str]
    
    # Additional info
    n_points: int = 0  # Number of data points
    
    # Additional results
    temperature_adjusted_params: Optional[Dict[str, float]] = None
    confidence_intervals: Optional[Dict[str, Tuple[float, float]]] = None
    convergence_info: Optional[Dict] = None


def fit_c3_aci(
    exdf: ExtendedDataFrame,
    # Column names
    a_column: str = 'A',
    ci_column: str = 'Ci',
    tleaf_column: str = 'Tleaf',
    patm_column: str = 'Pa',
    o2_column: Optional[str] = None,
    # Model options
    fixed_parameters: Optional[Dict[str, float]] = None,
    parameter_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    initial_guess: Optional[Dict[str, float]] = None,
    # Temperature response
    temperature_response_params: Optional[Dict] = None,
    use_gm: bool = True,
    # Optimization options
    optimizer: str = 'differential_evolution',
    error_metric: str = 'nll',
    sigma: float = 1.0,
    seed: Optional[int] = None,
    maxiter: int = 1000,
    # Output options
    calculate_confidence_intervals: bool = False,
    verbose: bool = False
) -> C3FitResult:
    """
    Fit C3 photosynthesis model to A-Ci curve data.
    
    Args:
        exdf: ExtendedDataFrame with A-Ci curve data
        a_column: Column name for net assimilation
        ci_column: Column name for intercellular CO2
        tleaf_column: Column name for leaf temperature
        patm_column: Column name for atmospheric pressure
        o2_column: Optional column name for O2 concentration
        fixed_parameters: Parameters to fix during optimization
        parameter_bounds: Custom parameter bounds
        initial_guess: Custom initial parameter values
        temperature_response_params: Temperature response parameters
        use_gm: Whether to include mesophyll conductance
        optimizer: 'differential_evolution' or 'nelder_mead'
        error_metric: 'nll' (negative log-likelihood) or 'rmse'
        sigma: Standard deviation for likelihood calculation
        seed: Random seed for reproducibility
        maxiter: Maximum iterations for optimization
        calculate_confidence_intervals: Whether to calculate CIs
        verbose: Print progress information
    
    Returns:
        C3FitResult object with fitting results
    """
    # Validate input data
    required_cols = [a_column, ci_column, tleaf_column, patm_column]
    if not exdf.check_required_variables(required_cols):
        raise ValueError(f"Missing required columns: {required_cols}")
    
    # Set default values
    fixed_parameters = fixed_parameters or {}
    temperature_response_params = temperature_response_params or C3_TEMPERATURE_PARAM_BERNACCHI
    
    # Prepare data dictionary for model
    data = {
        'Ci': exdf.data[ci_column].values,
        'Tleaf': exdf.data[tleaf_column].values,
        'Pa': exdf.data[patm_column].values,
        'A': exdf.data[a_column].values
    }
    
    # Add O2 if available
    if o2_column and o2_column in exdf.data.columns:
        data['O'] = exdf.data[o2_column].values
    else:
        # Use default O2 = 21%
        data['O'] = np.full_like(data['Ci'], 0.21)
    
    # Calculate average temperature for parameter normalization
    avg_temp = np.mean(data['Tleaf'])
    
    # Generate initial parameter estimates
    if initial_guess is None:
        if verbose:
            print("Estimating initial parameters...")
        initial_params = estimate_c3_initial_parameters(
            exdf, a_column, ci_column,
            temperature_response_params, avg_temp
        )
    else:
        initial_params = initial_guess.copy()
    
    # Add default kinetic parameters if not provided
    default_kinetic = {
        'Gamma_star_at_25': 36.94438,  # µmol/mol
        'Kc_at_25': 269.3391,  # µmol/mol
        'Ko_at_25': 163.7146   # mmol/mol
    }
    for param, value in default_kinetic.items():
        if param not in initial_params and param not in fixed_parameters:
            initial_params[param] = value
    
    # Update with any fixed parameters
    initial_params.update(fixed_parameters)
    
    # Determine parameters to optimize - use correct parameter names
    param_names = ['Vcmax_at_25', 'J_at_25', 'Tp_at_25', 'RL_at_25']
    if use_gm:
        param_names.append('gmc')
    
    # Remove fixed parameters from optimization
    optimize_params = [p for p in param_names if p not in fixed_parameters]
    
    if verbose:
        print(f"Optimizing parameters: {optimize_params}")
        print(f"Fixed parameters: {list(fixed_parameters.keys())}")
    
    # Generate parameter bounds
    if parameter_bounds is None:
        bounds_dict = estimate_c3_parameter_bounds(initial_params, fixed_parameters)
    else:
        bounds_dict = parameter_bounds.copy()
    
    # Create bounds list for optimizer
    bounds = [bounds_dict[p] for p in optimize_params]
    
    # Create model function for optimization (defined early for all-fixed case)
    def model_func(params: Dict, data: Dict) -> np.ndarray:
        """Calculate C3 assimilation for given parameters."""
        # Create ExtendedDataFrame for calculations
        model_exdf = ExtendedDataFrame({
            'Ci': data['Ci'],
            'Tleaf': data['Tleaf'],
            'Pa': data['Pa'],
            'O': data['O']
        })
        
        # Calculate gas properties
        model_exdf.calculate_gas_properties()
        
        # If using mesophyll conductance, we need to calculate Cc from Ci
        # For now, use Ci as Cc when gm is not being used
        if use_gm and 'gmc' in params:
            # This would require iterative solution - simplified for now
            model_exdf.set_variable('Cc', data['Ci'])
        else:
            model_exdf.set_variable('Cc', data['Ci'])
        
        # Add kinetic parameters if not present
        full_params = params.copy()
        for param, value in default_kinetic.items():
            if param not in full_params:
                full_params[param] = value
        
        # Calculate assimilation
        result = calculate_c3_assimilation(
            model_exdf,
            full_params,
            temperature_response_params=temperature_response_params,
            cc_column_name='Cc',
            oxygen=data['O'][0] * 100  # Convert fraction to percentage
        )
        
        return result.An
    
    # Check if we have parameters to optimize
    if not optimize_params:
        # All parameters are fixed - just evaluate the model
        fitted_A = model_func(fixed_parameters, data)
        residuals = data['A'] - fitted_A
        
        # Prepare for final result
        if use_gm and 'gmc' in fixed_parameters:
            exdf.set_variable('Cc', exdf.data[ci_column])
        else:
            exdf.set_variable('Cc', exdf.data[ci_column])
        
        # Add kinetic parameters if not in fixed_parameters
        full_fixed_params = fixed_parameters.copy()
        for param, value in default_kinetic.items():
            if param not in full_fixed_params:
                full_fixed_params[param] = value
        
        final_result = calculate_c3_assimilation(
            exdf,
            full_fixed_params,
            temperature_response_params=temperature_response_params,
            cc_column_name='Cc',
            oxygen=(data['O'][0] if o2_column else 0.21) * 100
        )
        
        # Determine limiting process
        limiting_process = []
        for i in range(len(fitted_A)):
            if np.abs(final_result.An[i] - final_result.Ac[i]) < 1e-6:
                limiting_process.append('Wc')
            elif np.abs(final_result.An[i] - final_result.Aj[i]) < 1e-6:
                limiting_process.append('Wj')
            else:
                limiting_process.append('Wp')
        limiting_process = np.array(limiting_process)
        
        # Calculate statistics
        rmse = np.sqrt(np.mean(residuals**2))
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((data['A'] - np.mean(data['A']))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return C3FitResult(
            parameters=fixed_parameters,
            success=True,
            error_value=0.0,
            n_function_evals=1,
            message='All parameters fixed',
            fitted_A=fitted_A,
            residuals=residuals,
            limiting_process=limiting_process,
            rmse=rmse,
            r_squared=r_squared,
            aic=np.nan,
            bic=np.nan,
            n_points=len(data['A']),
            warnings=['All parameters were fixed - no optimization performed'],
            temperature_adjusted_params={
                'Vcmax_tl': final_result.Vcmax_tl,
                'J_tl': final_result.J_tl,
                'Tp_tl': final_result.Tp_tl,
                'RL_tl': final_result.RL_tl,
                'Gamma_star_tl': final_result.Gamma_star_tl
            }
        )
    
    # Validate initial guess
    is_valid, warnings = validate_initial_guess(initial_params)
    if not is_valid and verbose:
        print(f"Initial guess warnings: {warnings}")
    
    # Create error function
    error_func = create_error_function(
        model_func,
        data,
        optimize_params,
        fixed_parameters,
        error_metric,
        sigma
    )
    
    # Run optimization
    if verbose:
        print(f"Running {optimizer} optimization...")
    
    if optimizer == 'differential_evolution':
        opt_result = fit_with_differential_evolution(
            error_func,
            bounds,
            seed=seed,
            maxiter=maxiter
        )
    elif optimizer == 'nelder_mead':
        initial_values = [initial_params[p] for p in optimize_params]
        opt_result = fit_with_nelder_mead(
            error_func,
            initial_values,
            bounds,
            maxiter=maxiter
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer}")
    
    # Extract fitted parameters
    fitted_params = fixed_parameters.copy()
    for i, name in enumerate(optimize_params):
        fitted_params[name] = opt_result['parameters'][i]
    
    # Validate optimization results
    is_valid, opt_warnings = validate_optimization_result(
        opt_result, bounds, optimize_params
    )
    
    # Calculate final model predictions
    fitted_A = model_func(fitted_params, data)
    residuals = data['A'] - fitted_A
    
    # Prepare data for final calculation
    if use_gm and 'gmc' in fitted_params:
        exdf.set_variable('Cc', exdf.data[ci_column])
    else:
        exdf.set_variable('Cc', exdf.data[ci_column])
    
    # Add kinetic parameters to fitted params
    full_fitted_params = fitted_params.copy()
    for param, value in default_kinetic.items():
        if param not in full_fitted_params:
            full_fitted_params[param] = value
    
    # Identify limiting processes
    final_result = calculate_c3_assimilation(
        exdf,
        full_fitted_params,
        temperature_response_params=temperature_response_params,
        cc_column_name='Cc',
        oxygen=(data['O'][0] if o2_column else 0.21) * 100
    )
    
    # Determine limiting process for each point
    limiting_process = []
    for i in range(len(fitted_A)):
        if np.abs(final_result.An[i] - final_result.Ac[i]) < 1e-6:
            limiting_process.append('Wc')
        elif np.abs(final_result.An[i] - final_result.Aj[i]) < 1e-6:
            limiting_process.append('Wj')
        else:
            limiting_process.append('Wp')
    limiting_process = np.array(limiting_process)
    
    # Calculate statistics
    rmse = np.sqrt(np.mean(residuals**2))
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((data['A'] - np.mean(data['A']))**2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # Calculate information criteria
    n_obs = len(data['A'])
    n_params = len(optimize_params)
    
    if error_metric == 'nll':
        nll = opt_result['error']
    else:
        # Convert RMSE to negative log-likelihood
        nll = negative_log_likelihood(data['A'], fitted_A, sigma)
    
    aic = calculate_aic(n_params, n_obs, nll)
    bic = calculate_bic(n_params, n_obs, nll)
    
    # Compile warnings
    all_warnings = warnings + opt_warnings
    if not opt_result['success']:
        all_warnings.append(f"Optimization may not have converged: {opt_result['message']}")
    
    # Create result object
    result = C3FitResult(
        parameters=fitted_params,
        success=opt_result['success'],
        error_value=opt_result['error'],
        n_function_evals=opt_result['nfev'],
        message=opt_result['message'],
        fitted_A=fitted_A,
        residuals=residuals,
        limiting_process=limiting_process,
        rmse=rmse,
        r_squared=r_squared,
        aic=aic,
        bic=bic,
        n_points=n_obs,
        warnings=all_warnings,
        temperature_adjusted_params={
            'Vcmax_tl': final_result.Vcmax_tl,
            'J_tl': final_result.J_tl,
            'Tp_tl': final_result.Tp_tl,
            'RL_tl': final_result.RL_tl,
            'Gamma_star_tl': final_result.Gamma_star_tl
        },
        convergence_info=opt_result.get('convergence', {})
    )
    
    # Calculate confidence intervals if requested
    if calculate_confidence_intervals and opt_result['success']:
        if verbose:
            print("Calculating confidence intervals...")
        
        # Use profile likelihood method by default
        ci_result = calculate_confidence_intervals_profile(
            error_func,
            opt_result['parameters'],
            optimize_params,
            bounds,
            confidence_level=0.95,
            n_points=20
        )
        
        result.confidence_intervals = ci_result
        
        if verbose:
            print("Confidence intervals (95%):")
            for param, (lower, upper) in ci_result.items():
                print(f"  {param}: [{lower:.2f}, {upper:.2f}]")
    
    if verbose:
        print(f"Optimization complete. Success: {result.success}")
        print(f"Fitted parameters: {result.parameters}")
        print(f"RMSE: {result.rmse:.3f}, R²: {result.r_squared:.3f}")
    
    return result


def negative_log_likelihood(
    observed: np.ndarray,
    predicted: np.ndarray,
    sigma: float
) -> float:
    """Calculate negative log-likelihood."""
    residuals = observed - predicted
    n = len(residuals)
    nll = 0.5 * n * np.log(2 * np.pi) + n * np.log(sigma) + \
          0.5 * np.sum(residuals**2) / (sigma**2)
    return nll


def summarize_c3_fit(result: C3FitResult) -> pd.DataFrame:
    """
    Create a summary DataFrame of fitting results.
    
    Args:
        result: C3FitResult object
    
    Returns:
        DataFrame with parameter estimates and statistics
    """
    summary_data = {
        'Parameter': [],
        'Value': [],
        'Unit': []
    }
    
    # Add fitted parameters
    param_units = {
        'Vcmax_at_25': 'µmol m⁻² s⁻¹',
        'J_at_25': 'µmol m⁻² s⁻¹',
        'Tp_at_25': 'µmol m⁻² s⁻¹',
        'RL_at_25': 'µmol m⁻² s⁻¹',
        'gmc': 'mol m⁻² s⁻¹ bar⁻¹'
    }
    
    for param, value in result.parameters.items():
        summary_data['Parameter'].append(param)
        summary_data['Value'].append(value)
        summary_data['Unit'].append(param_units.get(param, ''))
    
    # Add statistics
    stats = {
        'RMSE': (result.rmse, 'µmol m⁻² s⁻¹'),
        'R²': (result.r_squared, ''),
        'AIC': (result.aic, ''),
        'BIC': (result.bic, '')
    }
    
    for stat, (value, unit) in stats.items():
        summary_data['Parameter'].append(stat)
        summary_data['Value'].append(value)
        summary_data['Unit'].append(unit)
    
    return pd.DataFrame(summary_data)