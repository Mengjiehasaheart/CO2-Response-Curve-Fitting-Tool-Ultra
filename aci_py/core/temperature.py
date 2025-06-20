"""
Temperature response functions for photosynthesis parameters.

This module implements various temperature response functions used to adjust
photosynthesis parameters based on leaf temperature. Includes Arrhenius,
Johnson-Eyring-Williams, Gaussian, and polynomial response functions.

Based on PhotoGEA R package implementations.
"""

import numpy as np
from typing import Dict, Union, List, Optional
from dataclasses import dataclass

# Constants for temperature calculations
IDEAL_GAS_CONSTANT = 8.3145e-3  # kJ / mol / K
ABSOLUTE_ZERO = -273.15  # degrees C
T_REF_K = 25.0 - ABSOLUTE_ZERO  # Reference temperature (25°C) in Kelvin
F_CONST = IDEAL_GAS_CONSTANT * T_REF_K  # R * T_ref
C_PA_TO_PPM = np.log(1e6 / 101325)


@dataclass
class TemperatureParameter:
    """Container for temperature response parameter data."""
    type: str  # 'arrhenius', 'gaussian', 'johnson', or 'polynomial'
    units: str
    # Arrhenius parameters
    c: Optional[float] = None
    Ea: Optional[float] = None  # Activation energy (kJ/mol)
    # Johnson parameters  
    Ha: Optional[float] = None  # Activation enthalpy (kJ/mol)
    Hd: Optional[float] = None  # Deactivation enthalpy (kJ/mol)
    S: Optional[float] = None   # Entropy (kJ/K/mol)
    # Gaussian parameters
    optimum_rate: Optional[float] = None
    t_opt: Optional[float] = None  # Optimum temperature (°C)
    sigma: Optional[float] = None  # Standard deviation (°C)
    # Polynomial parameters
    coef: Optional[Union[float, List[float]]] = None


def arrhenius_response(
    scaling: float,
    activation_energy: float,
    temperature_c: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """
    Calculate Arrhenius temperature response.
    
    The Arrhenius equation describes the temperature dependence of reaction
    rates:
    
    response = exp(scaling - Ea / (R * T))
    
    Args:
        scaling: Dimensionless scaling factor
        activation_energy: Activation energy (kJ/mol)
        temperature_c: Temperature in degrees Celsius
        
    Returns:
        Temperature response factor
    """
    temperature_k = temperature_c - ABSOLUTE_ZERO
    return np.exp(scaling - activation_energy / (IDEAL_GAS_CONSTANT * temperature_k))


def johnson_eyring_williams_response(
    scaling: float,
    activation_enthalpy: float,
    deactivation_enthalpy: float,
    entropy: float,
    temperature_c: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """
    Calculate Johnson-Eyring-Williams temperature response.
    
    This function describes temperature response with both activation and
    deactivation at high temperatures:
    
    response = arrhenius(c, Ha, T) / (1 + arrhenius(S/R, Hd, T))
    
    Args:
        scaling: Dimensionless scaling factor
        activation_enthalpy: Activation enthalpy (kJ/mol)
        deactivation_enthalpy: Deactivation enthalpy (kJ/mol)
        entropy: Entropy term (kJ/K/mol)
        temperature_c: Temperature in degrees Celsius
        
    Returns:
        Temperature response factor
    """
    top = arrhenius_response(scaling, activation_enthalpy, temperature_c)
    bot = 1.0 + arrhenius_response(
        entropy / IDEAL_GAS_CONSTANT, 
        deactivation_enthalpy, 
        temperature_c
    )
    return top / bot


def gaussian_response(
    optimum_rate: float,
    t_opt: float,
    sigma: float,
    temperature_c: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """
    Calculate Gaussian temperature response.
    
    response = optimum_rate * exp(-(T - T_opt)^2 / sigma^2)
    
    Args:
        optimum_rate: Maximum rate at optimum temperature
        t_opt: Optimum temperature (°C)
        sigma: Standard deviation of response curve (°C)
        temperature_c: Temperature in degrees Celsius
        
    Returns:
        Temperature response in same units as optimum_rate
    """
    return optimum_rate * np.exp(-(temperature_c - t_opt)**2 / sigma**2)


def polynomial_response(
    coefficients: Union[float, List[float]],
    temperature_c: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """
    Calculate polynomial temperature response.
    
    response = sum(coef[i] * T^i for i in range(len(coef)))
    
    Args:
        coefficients: Polynomial coefficients (constant term first)
        temperature_c: Temperature in degrees Celsius
        
    Returns:
        Polynomial value
    """
    if isinstance(coefficients, (int, float)):
        coefficients = [coefficients]
    
    result = np.zeros_like(temperature_c, dtype=float)
    for i, coef in enumerate(coefficients):
        result += coef * temperature_c**i
    
    return result


def calculate_temperature_response(
    parameter: TemperatureParameter,
    temperature_c: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """
    Calculate temperature response for a single parameter.
    
    Args:
        parameter: Temperature parameter definition
        temperature_c: Temperature in degrees Celsius
        
    Returns:
        Temperature response value
        
    Raises:
        ValueError: If parameter type is unknown or required fields are missing
    """
    param_type = parameter.type.lower()
    
    if param_type == 'arrhenius':
        if parameter.c is None or parameter.Ea is None:
            raise ValueError("Arrhenius parameters require 'c' and 'Ea' values")
        return arrhenius_response(parameter.c, parameter.Ea, temperature_c)
        
    elif param_type == 'johnson':
        if any(x is None for x in [parameter.c, parameter.Ha, parameter.Hd, parameter.S]):
            raise ValueError("Johnson parameters require 'c', 'Ha', 'Hd', and 'S' values")
        return johnson_eyring_williams_response(
            parameter.c, parameter.Ha, parameter.Hd, parameter.S, temperature_c
        )
        
    elif param_type == 'gaussian':
        if any(x is None for x in [parameter.optimum_rate, parameter.t_opt, parameter.sigma]):
            raise ValueError("Gaussian parameters require 'optimum_rate', 't_opt', and 'sigma' values")
        return gaussian_response(
            parameter.optimum_rate, parameter.t_opt, parameter.sigma, temperature_c
        )
        
    elif param_type == 'polynomial':
        if parameter.coef is None:
            raise ValueError("Polynomial parameters require 'coef' value(s)")
        return polynomial_response(parameter.coef, temperature_c)
        
    else:
        raise ValueError(
            f"Unknown temperature response type: '{param_type}'. "
            "Supported types are: arrhenius, johnson, gaussian, polynomial"
        )


def apply_temperature_response(
    parameters: Dict[str, float],
    temperature_params: Dict[str, TemperatureParameter],
    temperature_c: Union[float, np.ndarray]
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Apply temperature responses to multiple parameters.
    
    Args:
        parameters: Base parameter values at 25°C
        temperature_params: Temperature response definitions for each parameter
        temperature_c: Leaf temperature in degrees Celsius
        
    Returns:
        Dictionary of temperature-adjusted parameter values
    """
    adjusted = {}
    
    for param_name, param_def in temperature_params.items():
        temp_response = calculate_temperature_response(param_def, temperature_c)
        
        # Handle normalized vs absolute parameters
        if param_name.endswith('_norm'):
            # This is a normalized response, multiply by base value
            base_param = param_name.replace('_norm', '')
            if base_param in parameters:
                adjusted[base_param] = parameters[base_param] * temp_response
            else:
                # Store the normalization factor itself
                adjusted[param_name] = temp_response
        elif param_name.endswith('_at_25'):
            # This is an absolute value at 25°C, use directly
            base_param = param_name.replace('_at_25', '')
            adjusted[base_param] = temp_response
        else:
            # Direct temperature-dependent parameter
            adjusted[param_name] = temp_response
    
    return adjusted


# Predefined parameter sets
C3_TEMPERATURE_PARAM_BERNACCHI = {
    'Gamma_star_norm': TemperatureParameter(
        type='arrhenius', c=37.83/F_CONST, Ea=37.83,
        units='normalized to Gamma_star at 25 degrees C'
    ),
    'J_norm': TemperatureParameter(
        type='arrhenius', c=17.57, Ea=43.5,
        units='normalized to J at 25 degrees C'
    ),
    'Kc_norm': TemperatureParameter(
        type='arrhenius', c=79.43/F_CONST, Ea=79.43,
        units='normalized to Kc at 25 degrees C'
    ),
    'Ko_norm': TemperatureParameter(
        type='arrhenius', c=36.38/F_CONST, Ea=36.38,
        units='normalized to Ko at 25 degrees C'
    ),
    'RL_norm': TemperatureParameter(
        type='arrhenius', c=18.72, Ea=46.39,
        units='normalized to RL at 25 degrees C'
    ),
    'Vcmax_norm': TemperatureParameter(
        type='arrhenius', c=26.35, Ea=65.33,
        units='normalized to Vcmax at 25 degrees C'
    ),
    'Vomax_norm': TemperatureParameter(
        type='arrhenius', c=22.98, Ea=60.11,
        units='normalized to Vomax at 25 degrees C'
    ),
    'gmc_norm': TemperatureParameter(
        type='johnson', c=20.01, Ha=49.6, Hd=437.4, S=1.4,
        units='normalized to gmc at 25 degrees C'
    ),
    'Tp_norm': TemperatureParameter(
        type='johnson', c=21.46, Ha=53.1, Hd=201.8, S=0.65,
        units='normalized to Tp at 25 degrees C'
    ),
    'Gamma_star_at_25': TemperatureParameter(
        type='polynomial', coef=42.93205,
        units='micromol mol^(-1)'
    ),
    'Kc_at_25': TemperatureParameter(
        type='polynomial', coef=406.8494,
        units='micromol mol^(-1)'
    ),
    'Ko_at_25': TemperatureParameter(
        type='polynomial', coef=277.1446,
        units='mmol mol^(-1)'
    ),
}

C3_TEMPERATURE_PARAM_SHARKEY = {
    'Gamma_star_norm': TemperatureParameter(
        type='arrhenius', c=24.46/F_CONST, Ea=24.46,
        units='normalized to Gamma_star at 25 degrees C'
    ),
    'J_norm': TemperatureParameter(
        type='arrhenius', c=17.71, Ea=43.9,
        units='normalized to J at 25 degrees C'
    ),
    'Kc_norm': TemperatureParameter(
        type='arrhenius', c=80.99/F_CONST, Ea=80.99,
        units='normalized to Kc at 25 degrees C'
    ),
    'Ko_norm': TemperatureParameter(
        type='arrhenius', c=23.72/F_CONST, Ea=23.72,
        units='normalized to Ko at 25 degrees C'
    ),
    'RL_norm': TemperatureParameter(
        type='arrhenius', c=18.7145, Ea=46.39,
        units='normalized to RL at 25 degrees C'
    ),
    'Vcmax_norm': TemperatureParameter(
        type='arrhenius', c=26.355, Ea=65.33,
        units='normalized to Vcmax at 25 degrees C'
    ),
    'gmc_norm': TemperatureParameter(
        type='johnson', c=20.01, Ha=49.6, Hd=437.4, S=1.4,
        units='normalized to gmc at 25 degrees C'
    ),
    'Tp_norm': TemperatureParameter(
        type='johnson', c=21.46, Ha=53.1, Hd=201.8, S=0.65,
        units='normalized to Tp at 25 degrees C'
    ),
    'Gamma_star_at_25': TemperatureParameter(
        type='polynomial', coef=36.94438,
        units='micromol mol^(-1)'
    ),
    'Kc_at_25': TemperatureParameter(
        type='polynomial', coef=269.3391,
        units='micromol mol^(-1)'
    ),
    'Ko_at_25': TemperatureParameter(
        type='polynomial', coef=163.7146,
        units='mmol mol^(-1)'
    ),
}

C3_TEMPERATURE_PARAM_FLAT = {
    'Gamma_star_norm': TemperatureParameter(
        type='arrhenius', c=0, Ea=0,
        units='normalized to Gamma_star at 25 degrees C'
    ),
    'gmc_norm': TemperatureParameter(
        type='arrhenius', c=0, Ea=0,
        units='normalized to gmc at 25 degrees C'
    ),
    'J_norm': TemperatureParameter(
        type='arrhenius', c=0, Ea=0,
        units='normalized to J at 25 degrees C'
    ),
    'Kc_norm': TemperatureParameter(
        type='arrhenius', c=0, Ea=0,
        units='normalized to Kc at 25 degrees C'
    ),
    'Ko_norm': TemperatureParameter(
        type='arrhenius', c=0, Ea=0,
        units='normalized to Ko at 25 degrees C'
    ),
    'RL_norm': TemperatureParameter(
        type='arrhenius', c=0, Ea=0,
        units='normalized to RL at 25 degrees C'
    ),
    'Tp_norm': TemperatureParameter(
        type='arrhenius', c=0, Ea=0,
        units='normalized to Tp at 25 degrees C'
    ),
    'Vcmax_norm': TemperatureParameter(
        type='arrhenius', c=0, Ea=0,
        units='normalized to Vcmax at 25 degrees C'
    ),
    'Gamma_star_at_25': TemperatureParameter(
        type='polynomial', coef=36.94438,
        units='micromol mol^(-1)'
    ),
    'Kc_at_25': TemperatureParameter(
        type='polynomial', coef=269.3391,
        units='micromol mol^(-1)'
    ),
    'Ko_at_25': TemperatureParameter(
        type='polynomial', coef=163.7146,
        units='mmol mol^(-1)'
    ),
}