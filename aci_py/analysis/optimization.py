

import numpy as np
from scipy.optimize import differential_evolution, minimize
from typing import Dict, Tuple, Callable, Optional, List, Any
from dataclasses import dataclass
import warnings
from ..core.data_structures import ExtendedDataFrame


@dataclass
class FittingResult:
    """Container for model fitting results."""
    parameters: Dict[str, float]  # Fitted parameter values
    result: Any  # Full optimization result object
    exdf: ExtendedDataFrame  # Data with model predictions
    rmse: float  # Root mean square error
    r_squared: float  # Coefficient of determination
    confidence_intervals: Optional[Dict[str, Tuple[float, float]]] = None
    covariance: Optional[np.ndarray] = None
    n_points: Optional[int] = None
    n_parameters: Optional[int] = None
    fixed_parameters: Optional[Dict[str, float]] = None
    parameter_names: Optional[List[str]] = None


def negative_log_likelihood(
    observed: np.ndarray,
    predicted: np.ndarray,
    sigma: float = 1.0
) -> float:
    """
    Calculate negative log-likelihood for normally distributed residuals.
    
    Args:
        observed: Observed values
        predicted: Model predictions
        sigma: Standard deviation of residuals (default 1.0)
    
    Returns:
        Negative log-likelihood value
    """
    residuals = observed - predicted
    n = len(residuals)
    
    # Calculate negative log-likelihood
    # -ln(L) = n/2 * ln(2π) + n * ln(σ) + 1/(2σ²) * Σ(residuals²)
    nll = 0.5 * n * np.log(2 * np.pi) + n * np.log(sigma) + \
          0.5 * np.sum(residuals**2) / (sigma**2)
    
    return nll


def rmse(observed: np.ndarray, predicted: np.ndarray) -> float:
    """
    Calculate root mean squared error.
    
    Args:
        observed: Observed values
        predicted: Model predictions
    
    Returns:
        RMSE value
    """
    return np.sqrt(np.mean((observed - predicted)**2))


def create_error_function(
    model_func: Callable,
    data: Dict[str, np.ndarray],
    param_names: List[str],
    fixed_params: Optional[Dict[str, float]] = None,
    error_metric: str = 'nll',
    sigma: float = 1.0
) -> Callable:
    """
    Create an error function for optimization.
    
    Args:
        model_func: Function that calculates model predictions
        data: Dictionary with required data arrays
        param_names: Names of parameters to optimize
        fixed_params: Fixed parameter values (not optimized)
        error_metric: 'nll' for negative log-likelihood, 'rmse' for RMSE
        sigma: Standard deviation for likelihood calculation
    
    Returns:
        Error function suitable for optimization
    """
    fixed_params = fixed_params or {}
    
    def error_function(param_values: np.ndarray) -> float:
        # Combine optimized and fixed parameters
        all_params = fixed_params.copy()
        for i, name in enumerate(param_names):
            all_params[name] = param_values[i]
        
        # Calculate model predictions
        try:
            predictions = model_func(all_params, data)
            
            # Calculate error
            if error_metric == 'nll':
                return negative_log_likelihood(
                    data['A'], predictions, sigma
                )
            elif error_metric == 'rmse':
                return rmse(data['A'], predictions)
            else:
                raise ValueError(f"Unknown error metric: {error_metric}")
                
        except Exception as e:
            # Return large error value for invalid parameters
            warnings.warn(f"Model calculation failed: {e}")
            return 1e10
    
    return error_function


def fit_with_differential_evolution(
    error_func: Callable,
    bounds: List[Tuple[float, float]],
    seed: Optional[int] = None,
    maxiter: int = 1000,
    popsize: int = 15,
    atol: float = 0.01,
    tol: float = 0.01,
    mutation: Tuple[float, float] = (0.5, 1),
    recombination: float = 0.7,
    polish: bool = True,
    workers: int = 1
) -> Dict:
    """
    Fit model using differential evolution global optimization.
    
    Args:
        error_func: Error function to minimize
        bounds: Parameter bounds [(min, max), ...]
        seed: Random seed for reproducibility
        maxiter: Maximum iterations
        popsize: Population size multiplier
        atol: Absolute tolerance for convergence
        tol: Relative tolerance for convergence
        mutation: Mutation constant range
        recombination: Crossover probability
        polish: Whether to polish with L-BFGS-B
        workers: Number of parallel workers
    
    Returns:
        Dictionary with optimization results
    """
    result = differential_evolution(
        error_func,
        bounds,
        seed=seed,
        maxiter=maxiter,
        popsize=popsize,
        atol=atol,
        tol=tol,
        mutation=mutation,
        recombination=recombination,
        polish=polish,
        workers=workers,
        updating='deferred'
    )
    
    return {
        'success': result.success,
        'parameters': result.x,
        'error': result.fun,
        'nfev': result.nfev,
        'message': result.message,
        'convergence': {
            'iterations': result.nit,
            'population_energies': result.population_energies
        }
    }


def fit_with_nelder_mead(
    error_func: Callable,
    initial_guess: np.ndarray,
    bounds: Optional[List[Tuple[float, float]]] = None,
    maxiter: int = 1000,
    xatol: float = 1e-4,
    fatol: float = 1e-4
) -> Dict:
    """
    Fit model using Nelder-Mead simplex algorithm.
    
    Args:
        error_func: Error function to minimize
        initial_guess: Starting parameter values
        bounds: Parameter bounds (optional)
        maxiter: Maximum iterations
        xatol: Absolute tolerance on parameters
        fatol: Absolute tolerance on function value
    
    Returns:
        Dictionary with optimization results
    """
    # Convert bounds to constraints if provided
    method = 'Nelder-Mead'
    options = {
        'maxiter': maxiter,
        'xatol': xatol,
        'fatol': fatol,
        'disp': False
    }
    
    if bounds is not None:
        # Use L-BFGS-B if bounds are provided
        method = 'L-BFGS-B'
        options = {'maxiter': maxiter, 'disp': False}
    
    result = minimize(
        error_func,
        initial_guess,
        method=method,
        bounds=bounds,
        options=options
    )
    
    return {
        'success': result.success,
        'parameters': result.x,
        'error': result.fun,
        'nfev': result.nfev,
        'message': result.message,
        'convergence': {
            'iterations': result.nit,
            'final_simplex': result.get('final_simplex_vertices', None)
        }
    }


def parameter_penalty(
    value: float,
    lower: float,
    upper: float,
    penalty_factor: float = 1000.0
) -> float:
    """
    Calculate penalty for parameter values outside bounds.
    
    Used for soft constraints in unbounded optimization.
    
    Args:
        value: Parameter value
        lower: Lower bound
        upper: Upper bound
        penalty_factor: Penalty multiplier
    
    Returns:
        Penalty value (0 if within bounds)
    """
    if value < lower:
        return penalty_factor * (lower - value)**2
    elif value > upper:
        return penalty_factor * (value - upper)**2
    else:
        return 0.0


def validate_optimization_result(
    result: Dict,
    bounds: List[Tuple[float, float]],
    param_names: List[str]
) -> Tuple[bool, List[str]]:
    """
    Validate optimization results.
    
    Args:
        result: Optimization result dictionary
        bounds: Parameter bounds
        param_names: Parameter names
    
    Returns:
        Tuple of (is_valid, list_of_warnings)
    """
    warnings_list = []
    
    # Check convergence
    if not result['success']:
        warnings_list.append(f"Optimization did not converge: {result['message']}")
    
    # Check parameter bounds
    params = result['parameters']
    for i, (value, (lower, upper), name) in enumerate(zip(params, bounds, param_names)):
        if value <= lower * 1.001:  # Within 0.1% of lower bound
            warnings_list.append(f"{name} at lower bound: {value:.3f}")
        elif value >= upper * 0.999:  # Within 0.1% of upper bound
            warnings_list.append(f"{name} at upper bound: {value:.3f}")
    
    # Check for unreasonable error values
    if result['error'] > 1e6:
        warnings_list.append(f"Very high error value: {result['error']:.2e}")
    
    is_valid = len(warnings_list) == 0
    return is_valid, warnings_list


def calculate_aic(n_params: int, n_obs: int, nll: float) -> float:
    """
    Calculate Akaike Information Criterion.
    
    Args:
        n_params: Number of parameters
        n_obs: Number of observations
        nll: Negative log-likelihood
    
    Returns:
        AIC value
    """
    aic = 2 * n_params + 2 * nll
    
    # Small sample correction
    if n_obs < 40 * n_params:
        aic_c = aic + 2 * n_params * (n_params + 1) / (n_obs - n_params - 1)
        return aic_c
    
    return aic


def calculate_bic(n_params: int, n_obs: int, nll: float) -> float:
    """
    Calculate Bayesian Information Criterion.
    
    Args:
        n_params: Number of parameters
        n_obs: Number of observations
        nll: Negative log-likelihood
    
    Returns:
        BIC value
    """
    return np.log(n_obs) * n_params + 2 * nll


def calculate_confidence_intervals_profile(
    error_func: Callable,
    best_params: np.ndarray,
    param_names: List[str],
    bounds: List[Tuple[float, float]],
    confidence_level: float = 0.95,
    n_points: int = 20,
    chi2_threshold: Optional[float] = None
) -> Dict[str, Tuple[float, float]]:
    """
    Calculate confidence intervals using profile likelihood method.
    
    This method fixes each parameter at various values and re-optimizes
    the other parameters, finding where the likelihood drops by chi-squared
    critical value.
    
    Args:
        error_func: Error function (negative log-likelihood)
        best_params: Best-fit parameter values
        param_names: Names of parameters
        bounds: Parameter bounds
        confidence_level: Confidence level (default 0.95)
        n_points: Number of points to sample for each parameter
        chi2_threshold: Chi-squared threshold (auto-calculated if None)
    
    Returns:
        Dictionary mapping parameter names to (lower, upper) confidence bounds
    """
    from scipy.stats import chi2
    
    # Calculate chi-squared threshold for given confidence level
    if chi2_threshold is None:
        chi2_threshold = chi2.ppf(confidence_level, df=1) / 2
    
    # Best negative log-likelihood
    best_nll = error_func(best_params)
    threshold_nll = best_nll + chi2_threshold
    
    confidence_intervals = {}
    
    for i, param_name in enumerate(param_names):
        # Create arrays for other parameters
        other_indices = [j for j in range(len(best_params)) if j != i]
        other_bounds = [bounds[j] for j in other_indices]
        
        # Function to optimize other parameters with current parameter fixed
        def profile_error(fixed_value):
            def error_with_fixed(other_params):
                full_params = np.zeros(len(best_params))
                full_params[i] = fixed_value
                for j, idx in enumerate(other_indices):
                    full_params[idx] = other_params[j]
                return error_func(full_params)
            
            # Optimize other parameters
            if len(other_indices) > 0:
                initial_other = best_params[other_indices]
                result = minimize(
                    error_with_fixed,
                    initial_other,
                    method='L-BFGS-B',
                    bounds=other_bounds
                )
                return result.fun
            else:
                return error_with_fixed([])
        
        # Search for lower bound
        param_range = np.linspace(bounds[i][0], best_params[i], n_points)
        lower_bound = bounds[i][0]
        
        for test_value in reversed(param_range[:-1]):
            profile_nll = profile_error(test_value)
            if profile_nll > threshold_nll:
                # Interpolate to find more precise bound
                if test_value < best_params[i] - 1e-6:
                    lower_bound = test_value
                break
        
        # Search for upper bound
        param_range = np.linspace(best_params[i], bounds[i][1], n_points)
        upper_bound = bounds[i][1]
        
        for test_value in param_range[1:]:
            profile_nll = profile_error(test_value)
            if profile_nll > threshold_nll:
                # Interpolate to find more precise bound
                if test_value > best_params[i] + 1e-6:
                    upper_bound = test_value
                break
        
        confidence_intervals[param_name] = (lower_bound, upper_bound)
    
    return confidence_intervals


def calculate_confidence_intervals_bootstrap(
    model_func: Callable,
    data: Dict[str, np.ndarray],
    best_params: Dict[str, float],
    param_names: List[str],
    bounds: List[Tuple[float, float]],
    n_bootstrap: int = 100,
    confidence_level: float = 0.95,
    seed: Optional[int] = None
) -> Dict[str, Tuple[float, float]]:
    """
    Calculate confidence intervals using bootstrap resampling.
    
    Args:
        model_func: Function that calculates model predictions
        data: Dictionary with data arrays
        best_params: Best-fit parameters as dictionary
        param_names: Names of parameters to fit
        bounds: Parameter bounds
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level
        seed: Random seed
    
    Returns:
        Dictionary mapping parameter names to (lower, upper) confidence bounds
    """
    if seed is not None:
        np.random.seed(seed)
    
    n_obs = len(data['A'])
    bootstrap_params = {name: [] for name in param_names}
    
    # Calculate residuals from best fit
    best_predictions = model_func(best_params, data)
    residuals = data['A'] - best_predictions
    
    for i in range(n_bootstrap):
        # Resample residuals with replacement
        resampled_indices = np.random.choice(n_obs, n_obs, replace=True)
        resampled_residuals = residuals[resampled_indices]
        
        # Create new "observations" by adding resampled residuals to predictions
        bootstrap_data = data.copy()
        bootstrap_data['A'] = best_predictions + resampled_residuals
        
        # Fit model to bootstrap sample
        error_func = create_error_function(
            model_func, bootstrap_data, param_names,
            fixed_params={k: v for k, v in best_params.items() if k not in param_names}
        )
        
        # Use best params as initial guess
        initial_guess = [best_params[name] for name in param_names]
        
        try:
            result = minimize(
                error_func,
                initial_guess,
                method='L-BFGS-B',
                bounds=bounds
            )
            
            if result.success:
                for j, name in enumerate(param_names):
                    bootstrap_params[name].append(result.x[j])
        except:
            # Skip failed optimizations
            continue
    
    # Calculate percentile confidence intervals
    alpha = 1 - confidence_level
    confidence_intervals = {}
    
    for name in param_names:
        if len(bootstrap_params[name]) > 10:  # Need enough successful fits
            lower = np.percentile(bootstrap_params[name], 100 * alpha / 2)
            upper = np.percentile(bootstrap_params[name], 100 * (1 - alpha / 2))
            confidence_intervals[name] = (lower, upper)
        else:
            # Not enough data for bootstrap
            confidence_intervals[name] = (np.nan, np.nan)
    
    return confidence_intervals