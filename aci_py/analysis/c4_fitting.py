

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple, Any
from scipy.optimize import differential_evolution, minimize
import warnings

from ..core.data_structures import ExtendedDataFrame
from ..core.c4_calculations import calculate_c4_assimilation, apply_gm_c4
from ..core.models import C4Model
from .optimization import FittingResult


def initial_guess_c4_aci(
    exdf: ExtendedDataFrame,
    alpha_psii: float = 0.0,
    gbs: float = 0.003,
    gmc_at_25: float = 1.0,
    Rm_frac: float = 0.5,
    pcm_threshold_rlm: float = 40.0,
    x_etr: float = 0.4
) -> Dict[str, float]:
    """
    Generate initial parameter guesses for C4 ACI fitting.
    
    This function analyzes the ACI curve to estimate reasonable starting values
    for the optimization. It uses different regions of the curve to estimate
    different parameters.
    
    Parameters
    ----------
    exdf : ExtendedDataFrame
        ACI curve data with columns: A, Ci, PCi, PCm (if available)
        
    alpha_psii : float, optional
        Fixed value for fraction of PSII in bundle sheath.
        
    gbs : float, optional
        Fixed value for bundle sheath conductance.
        
    gmc_at_25 : float, optional
        Fixed value for mesophyll conductance at 25°C.
        
    Rm_frac : float, optional
        Fixed value for fraction of respiration in mesophyll.
        
    pcm_threshold_rlm : float, optional
        PCm threshold for estimating RLm from low CO2 points.
        
    x_etr : float, optional
        Fraction of electron transport for C4 cycle.
        
    Returns
    -------
    dict
        Initial parameter guesses:
        - RL_at_25: Day respiration
        - Vcmax_at_25: Maximum carboxylation rate
        - Vpmax_at_25: Maximum PEP carboxylation rate
        - Vpr: PEP regeneration rate
        - J_at_25: Maximum electron transport rate
    """
    # Apply mesophyll conductance if PCm is not present
    if 'PCm' not in exdf.data.columns:
        exdf = apply_gm_c4(exdf, gmc_at_25)
    
    # Get PCm values
    PCm = exdf.data['PCm'].values
    A = exdf.data['A'].values
    
    # Get temperature normalization factors if available
    rl_norm = exdf.data['RL_norm'].values if 'RL_norm' in exdf.data else np.ones(len(A))
    vcmax_norm = exdf.data['Vcmax_norm'].values if 'Vcmax_norm' in exdf.data else np.ones(len(A))
    vpmax_norm = exdf.data['Vpmax_norm'].values if 'Vpmax_norm' in exdf.data else np.ones(len(A))
    j_norm = exdf.data['J_norm'].values if 'J_norm' in exdf.data else np.ones(len(A))
    
    # Get Kp values if available
    Kp = exdf.data['Kp'].values if 'Kp' in exdf.data else 80.0  # Default value
    
    # 1. Estimate RLm from low PCm points
    low_pcm_mask = PCm <= pcm_threshold_rlm
    if np.sum(low_pcm_mask) >= 2:
        # Linear fit of A vs PCm at low PCm
        # A ≈ -RLm + slope * PCm
        from scipy import stats
        slope, intercept, _, _, _ = stats.linregress(
            PCm[low_pcm_mask], 
            A[low_pcm_mask]
        )
        RLm_estimate = max(-intercept / np.mean(rl_norm[low_pcm_mask]), 0.1)
    else:
        # Default estimate
        RLm_estimate = 0.5
    
    # Calculate RL from RLm
    RL_estimate = RLm_estimate / Rm_frac
    
    # 2. Estimate Vpmax from PEP carboxylation limited region
    # When PEP carboxylation is limiting: A ≈ Vpc - RLm + gbs * PCm
    # Vpc = PCm * Vpmax / (PCm + Kp)
    # Solving for Vpmax: Vpmax = (A + RLm - gbs * PCm) * (PCm + Kp) / PCm
    vpmax_estimates = np.zeros(len(A))
    for i in range(len(A)):
        if PCm[i] > 0:
            vpmax_estimates[i] = (A[i] + RLm_estimate * rl_norm[i] - gbs * PCm[i]) * \
                               (PCm[i] + Kp[i]) / PCm[i] / vpmax_norm[i]
    
    # Use high percentile as estimate (PEP carboxylation limited at intermediate PCm)
    valid_vpmax = vpmax_estimates[vpmax_estimates > 0]
    if len(valid_vpmax) > 0:
        Vpmax_estimate = np.percentile(valid_vpmax, 90)
    else:
        Vpmax_estimate = 150.0
    
    # 3. Estimate Vcmax from Rubisco limited region
    # When Rubisco is limiting: A ≈ Vcmax - RL
    vcmax_estimates = (A + RL_estimate * rl_norm) / vcmax_norm
    
    # Use high percentile as estimate (Rubisco limited at high PCm)
    valid_vcmax = vcmax_estimates[vcmax_estimates > 0]
    if len(valid_vcmax) > 0:
        Vcmax_estimate = np.percentile(valid_vcmax, 90)
    else:
        Vcmax_estimate = 100.0
    
    # 4. Estimate Vpr from PEP regeneration limited region
    # When PEP regeneration is limiting: A ≈ Vpr - RLm + gbs * PCm
    vpr_estimates = A + RLm_estimate * rl_norm - gbs * PCm
    
    # Use high percentile as estimate
    valid_vpr = vpr_estimates[vpr_estimates > 0]
    if len(valid_vpr) > 0:
        Vpr_estimate = np.percentile(valid_vpr, 80)
    else:
        Vpr_estimate = min(Vpmax_estimate * 0.8, 80.0)
    
    # 5. Estimate J from light limited region
    # When bundle sheath electron transport is limiting:
    # A ≈ (1 - x_etr) * J / 3 - RL
    j_estimates = 3 * (A + RL_estimate * rl_norm) / ((1 - x_etr) * j_norm)
    
    # Use high percentile as estimate
    valid_j = j_estimates[j_estimates > 0]
    if len(valid_j) > 0:
        J_estimate = np.percentile(valid_j, 90)
    else:
        J_estimate = 400.0
    
    # Ensure reasonable bounds
    RL_estimate = np.clip(RL_estimate, 0.1, 10.0)
    Vcmax_estimate = np.clip(Vcmax_estimate, 10.0, 500.0)
    Vpmax_estimate = np.clip(Vpmax_estimate, 10.0, 500.0)
    Vpr_estimate = np.clip(Vpr_estimate, 10.0, min(Vpmax_estimate, 200.0))
    J_estimate = np.clip(J_estimate, 50.0, 1000.0)
    
    return {
        'RL_at_25': float(RL_estimate),
        'Vcmax_at_25': float(Vcmax_estimate),
        'Vpmax_at_25': float(Vpmax_estimate),
        'Vpr': float(Vpr_estimate),
        'J_at_25': float(J_estimate)
    }


def fit_c4_aci(
    exdf: ExtendedDataFrame,
    fixed_parameters: Optional[Dict[str, float]] = None,
    bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    initial_guess: Optional[Dict[str, float]] = None,
    optimizer: str = 'differential_evolution',
    de_options: Optional[Dict[str, Any]] = None,
    nm_options: Optional[Dict[str, Any]] = None,
    calculate_confidence: bool = True,
    confidence_level: float = 0.95,
    sd_A: str = 'RMSE'
) -> FittingResult:
    """
    Fit C4 photosynthesis parameters to ACI curve data.
    
    Parameters
    ----------
    exdf : ExtendedDataFrame
        ACI curve data with required columns:
        - A: Net assimilation rate
        - Ci or PCi: Intercellular or internal CO2
        - Tleaf: Leaf temperature
        - Kinetic parameters (Kc, Ko, Kp, gamma_star, ao)
        
    fixed_parameters : dict, optional
        Parameters to fix during fitting. Common fixed parameters:
        - alpha_psii: 0.0 (all PSII in mesophyll)
        - gbs: 0.003 (bundle sheath conductance)
        - gmc_at_25: 1.0 (mesophyll conductance)
        - Rm_frac: 0.5 (respiration fraction)
        - x_etr: 0.4 (electron transport fraction)
        
    bounds : dict, optional
        Custom bounds for parameters. Default bounds:
        - RL_at_25: (0, 10)
        - Vcmax_at_25: (1, 1000)
        - Vpmax_at_25: (1, 1000)
        - Vpr: (1, 1000)
        - J_at_25: (1, 1000)
        - alpha_psii: (0, 1)
        - gbs: (0, 10)
        - gmc_at_25: (0, 10)
        - Rm_frac: (0, 1)
        
    initial_guess : dict, optional
        Initial parameter values. If None, uses initial_guess_c4_aci.
        
    optimizer : str, optional
        Optimization method: 'differential_evolution' or 'nelder-mead'.
        
    de_options : dict, optional
        Options for differential evolution optimizer.
        
    nm_options : dict, optional
        Options for Nelder-Mead optimizer.
        
    calculate_confidence : bool, optional
        Whether to calculate confidence intervals.
        
    confidence_level : float, optional
        Confidence level for intervals (default 0.95).
        
    sd_A : str, optional
        Method for estimating standard deviation: 'RMSE' only currently.
        
    Returns
    -------
    FittingResult
        Object containing:
        - parameters: Fitted parameter values
        - result: Full optimization result
        - exdf: ExtendedDataFrame with model predictions
        - rmse: Root mean square error
        - confidence_intervals: Parameter confidence intervals (if calculated)
        - covariance: Parameter covariance matrix (if calculated)
    """
    # Set default fixed parameters for C4
    default_fixed = {
        'alpha_psii': 0.0,
        'gbs': 0.003,
        'gmc_at_25': 1.0,
        'Rm_frac': 0.5,
        'x_etr': 0.4
    }
    
    if fixed_parameters is None:
        fixed_parameters = default_fixed
    else:
        # Merge with defaults
        fixed_parameters = {**default_fixed, **fixed_parameters}
    
    # Define default bounds
    default_bounds = {
        'RL_at_25': (0, 10),
        'Vcmax_at_25': (1, 1000), 
        'Vpmax_at_25': (1, 1000),
        'Vpr': (1, 1000),
        'J_at_25': (1, 1000),
        'alpha_psii': (0, 1),
        'gbs': (0, 10),
        'gmc_at_25': (0, 10),
        'Rm_frac': (0, 1)
    }
    
    if bounds is not None:
        default_bounds.update(bounds)
    bounds = default_bounds
    
    # Apply mesophyll conductance if needed
    if 'PCm' not in exdf.data.columns:
        gmc_val = fixed_parameters.get('gmc_at_25', 1.0)
        exdf = apply_gm_c4(exdf, gmc_val)
    
    # Get initial guess
    if initial_guess is None:
        initial_guess = initial_guess_c4_aci(
            exdf,
            alpha_psii=fixed_parameters.get('alpha_psii', 0.0),
            gbs=fixed_parameters.get('gbs', 0.003),
            gmc_at_25=fixed_parameters.get('gmc_at_25', 1.0),
            Rm_frac=fixed_parameters.get('Rm_frac', 0.5),
            x_etr=fixed_parameters.get('x_etr', 0.4)
        )
    
    # Determine parameters to fit
    all_params = ['RL_at_25', 'Vcmax_at_25', 'Vpmax_at_25', 'Vpr', 'J_at_25',
                  'alpha_psii', 'gbs', 'gmc_at_25', 'Rm_frac']
    params_to_fit = [p for p in all_params if p not in fixed_parameters]
    
    # Get measured assimilation
    A_measured = exdf.data['A'].values
    
    # Define objective function
    def objective(params_array):
        # Build parameter dictionary
        param_dict = fixed_parameters.copy()
        for i, param_name in enumerate(params_to_fit):
            param_dict[param_name] = params_array[i]
        
        # Calculate model predictions
        result = calculate_c4_assimilation(
            exdf,
            alpha_psii=param_dict.get('alpha_psii', 0.0),
            gbs=param_dict.get('gbs', 0.003),
            J_at_25=param_dict['J_at_25'],
            RL_at_25=param_dict['RL_at_25'],
            Rm_frac=param_dict.get('Rm_frac', 0.5),
            Vcmax_at_25=param_dict['Vcmax_at_25'],
            Vpmax_at_25=param_dict['Vpmax_at_25'],
            Vpr=param_dict['Vpr'],
            x_etr=fixed_parameters.get('x_etr', 0.4),
            return_extended=False,
            check_inputs=False
        )
        
        # Calculate residuals
        residuals = A_measured - result
        
        # Return sum of squares
        return np.sum(residuals**2)
    
    # Set up bounds for optimization
    param_bounds = [bounds[p] for p in params_to_fit]
    
    # Get initial values for parameters to fit
    x0 = [initial_guess.get(p, (bounds[p][0] + bounds[p][1])/2) for p in params_to_fit]
    
    # Optimize
    if optimizer == 'differential_evolution':
        de_opts = {
            'strategy': 'best1bin',
            'maxiter': 1000,
            'popsize': 15,
            'tol': 1e-6,
            'mutation': (0.5, 1.5),
            'recombination': 0.7,
            'seed': 42,
            'polish': True,
            'disp': False
        }
        if de_options is not None:
            de_opts.update(de_options)
        
        result = differential_evolution(objective, param_bounds, **de_opts)
        
    elif optimizer == 'nelder-mead':
        nm_opts = {
            'maxiter': 5000,
            'xatol': 1e-8,
            'fatol': 1e-8,
            'disp': False
        }
        if nm_options is not None:
            nm_opts.update(nm_options)
        
        result = minimize(objective, x0, method='Nelder-Mead',
                         bounds=param_bounds, options=nm_opts)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer}")
    
    # Build final parameter dictionary
    fitted_params = fixed_parameters.copy()
    for i, param_name in enumerate(params_to_fit):
        fitted_params[param_name] = result.x[i]
    
    # Calculate final model predictions with extended output
    final_result = calculate_c4_assimilation(
        exdf,
        alpha_psii=fitted_params.get('alpha_psii', 0.0),
        gbs=fitted_params.get('gbs', 0.003),
        J_at_25=fitted_params['J_at_25'],
        RL_at_25=fitted_params['RL_at_25'],
        Rm_frac=fitted_params.get('Rm_frac', 0.5),
        Vcmax_at_25=fitted_params['Vcmax_at_25'],
        Vpmax_at_25=fitted_params['Vpmax_at_25'],
        Vpr=fitted_params['Vpr'],
        x_etr=fixed_parameters.get('x_etr', 0.4),
        return_extended=True,
        check_inputs=False
    )
    
    # Calculate RMSE
    A_model = final_result.data['An'].values
    residuals = A_measured - A_model
    rmse = np.sqrt(np.mean(residuals**2))
    
    # Calculate R-squared
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((A_measured - np.mean(A_measured))**2)
    r_squared = 1 - (ss_res / ss_tot)
    
    # Store residuals
    final_result.data['residuals'] = residuals
    final_result.units['residuals'] = 'micromol m^(-2) s^(-1)'
    final_result.categories['residuals'] = 'fit_c4_aci'
    
    # Calculate confidence intervals if requested
    confidence_intervals = None
    covariance = None
    
    if calculate_confidence and len(params_to_fit) > 0:
        try:
            # Estimate covariance matrix using finite differences
            from scipy.optimize import approx_fprime
            
            def residual_func(params):
                param_dict = fixed_parameters.copy()
                for i, pname in enumerate(params_to_fit):
                    param_dict[pname] = params[i]
                
                model = calculate_c4_assimilation(
                    exdf,
                    alpha_psii=param_dict.get('alpha_psii', 0.0),
                    gbs=param_dict.get('gbs', 0.003),
                    J_at_25=param_dict['J_at_25'],
                    RL_at_25=param_dict['RL_at_25'],
                    Rm_frac=param_dict.get('Rm_frac', 0.5),
                    Vcmax_at_25=param_dict['Vcmax_at_25'],
                    Vpmax_at_25=param_dict['Vpmax_at_25'],
                    Vpr=param_dict['Vpr'],
                    x_etr=fixed_parameters.get('x_etr', 0.4),
                    return_extended=False,
                    check_inputs=False
                )
                return A_measured - model
            
            # Calculate Jacobian
            jacobian = np.zeros((len(A_measured), len(params_to_fit)))
            eps = np.sqrt(np.finfo(float).eps)
            
            for i in range(len(A_measured)):
                def single_residual(params):
                    return residual_func(params)[i]
                
                jacobian[i, :] = approx_fprime(result.x, single_residual, eps)
            
            # Calculate covariance matrix
            try:
                hessian = jacobian.T @ jacobian
                covariance = np.linalg.inv(hessian) * rmse**2
                
                # Calculate confidence intervals
                from scipy import stats
                t_value = stats.t.ppf((1 + confidence_level) / 2, len(A_measured) - len(params_to_fit))
                
                confidence_intervals = {}
                for i, param in enumerate(params_to_fit):
                    stderr = np.sqrt(covariance[i, i])
                    margin = t_value * stderr
                    confidence_intervals[param] = (
                        fitted_params[param] - margin,
                        fitted_params[param] + margin
                    )
                    
            except np.linalg.LinAlgError:
                warnings.warn("Could not calculate confidence intervals: singular matrix")
                
        except Exception as e:
            warnings.warn(f"Could not calculate confidence intervals: {str(e)}")
    
    # Create fitting result
    fitting_result = FittingResult(
        parameters=fitted_params,
        result=result,
        exdf=final_result,
        rmse=rmse,
        r_squared=r_squared,
        confidence_intervals=confidence_intervals,
        covariance=covariance,
        n_points=len(A_measured),
        n_parameters=len(params_to_fit),
        fixed_parameters=fixed_parameters,
        parameter_names=params_to_fit
    )
    
    return fitting_result