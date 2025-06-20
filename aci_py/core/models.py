"""
Base classes and interfaces for photosynthesis models.

This module provides abstract base classes for implementing different
photosynthesis models (C3, C4, CAM, etc.).
"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional, Union, List
import numpy as np
from aci_py.core.data_structures import ExtendedDataFrame


class PhotosynthesisModel(ABC):
    """
    Abstract base class for photosynthesis models.
    
    All photosynthesis models should inherit from this class and implement
    the required abstract methods.
    """
    
    def __init__(self, name: str):
        """
        Initialize the model.
        
        Args:
            name: Model name (e.g., "C3", "C4")
        """
        self.name = name
        self._parameter_names: List[str] = []
        self._parameter_bounds: Dict[str, Tuple[float, float]] = {}
        self._parameter_units: Dict[str, str] = {}
        
    @abstractmethod
    def calculate_assimilation(
        self, 
        parameters: Dict[str, float], 
        exdf: ExtendedDataFrame,
        return_diagnostics: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, Dict]]:
        """
        Calculate assimilation rate given parameters and environmental data.
        
        Args:
            parameters: Dictionary of model parameters
            exdf: ExtendedDataFrame containing environmental data
            return_diagnostics: If True, return additional diagnostic information
            
        Returns:
            If return_diagnostics=False: Array of calculated assimilation rates
            If return_diagnostics=True: Tuple of (assimilation rates, diagnostics dict)
        """
        pass
    
    @abstractmethod
    def get_parameter_bounds(self) -> Dict[str, Tuple[float, float]]:
        """
        Get parameter bounds for optimization.
        
        Returns:
            Dictionary mapping parameter names to (lower, upper) bounds
        """
        pass
    
    @abstractmethod
    def get_default_parameters(self, species: str = "tobacco") -> Dict[str, float]:
        """
        Get default parameter values for a given species.
        
        Args:
            species: Species name (e.g., "tobacco", "soybean")
            
        Returns:
            Dictionary of default parameter values
        """
        pass
    
    @abstractmethod
    def initial_guess(
        self, 
        exdf: ExtendedDataFrame,
        fixed_parameters: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        Generate initial parameter guesses from data.
        
        Args:
            exdf: ExtendedDataFrame containing measurement data
            fixed_parameters: Parameters to fix (not estimate)
            
        Returns:
            Dictionary of initial parameter values
        """
        pass
    
    @abstractmethod
    def validate_parameters(self, parameters: Dict[str, float]) -> Tuple[bool, str]:
        """
        Validate parameter values for reasonableness.
        
        Args:
            parameters: Dictionary of parameter values to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        pass
    
    def get_required_columns(self) -> List[str]:
        """
        Get list of required data columns for this model.
        
        Returns:
            List of required column names
        """
        # Base requirements common to all models
        return ["A", "Ci", "Tleaf", "Pa"]
    
    def get_parameter_units(self) -> Dict[str, str]:
        """
        Get units for all parameters.
        
        Returns:
            Dictionary mapping parameter names to unit strings
        """
        return self._parameter_units.copy()


class C3Model(PhotosynthesisModel):
    """
    C3 photosynthesis model (Farquhar-von Caemmerer-Berry).
    
    This is a placeholder - full implementation will follow.
    """
    
    def __init__(self):
        super().__init__("C3")
        
        # Define C3 parameter names and bounds
        self._parameter_names = ["Vcmax", "J", "Tp", "Rd", "gm"]
        self._parameter_bounds = {
            "Vcmax": (0.0, 1000.0),  # µmol m⁻² s⁻¹
            "J": (0.0, 1000.0),       # µmol m⁻² s⁻¹
            "Tp": (0.0, 100.0),       # µmol m⁻² s⁻¹
            "Rd": (0.0, 50.0),        # µmol m⁻² s⁻¹
            "gm": (0.0, 10.0),        # mol m⁻² s⁻¹ bar⁻¹
        }
        self._parameter_units = {
            "Vcmax": "µmol m⁻² s⁻¹",
            "J": "µmol m⁻² s⁻¹",
            "Tp": "µmol m⁻² s⁻¹",
            "Rd": "µmol m⁻² s⁻¹",
            "gm": "mol m⁻² s⁻¹ bar⁻¹",
        }
    
    def calculate_assimilation(
        self, 
        parameters: Dict[str, float], 
        exdf: ExtendedDataFrame,
        return_diagnostics: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, Dict]]:
        """Calculate C3 assimilation using FvCB model."""
        from aci_py.core.c3_calculations import calculate_c3_assimilation, identify_c3_limiting_process
        
        # Calculate assimilation
        result = calculate_c3_assimilation(exdf, parameters)
        
        if return_diagnostics:
            # Return assimilation and diagnostic info
            diagnostics = {
                'Ac': result.Ac,
                'Aj': result.Aj,
                'Ap': result.Ap,
                'Vcmax_tl': result.Vcmax_tl,
                'J_tl': result.J_tl,
                'Tp_tl': result.Tp_tl,
                'RL_tl': result.RL_tl,
                'limiting_process': identify_c3_limiting_process(result)
            }
            return result.An, diagnostics
        else:
            return result.An
    
    def get_parameter_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Get C3 parameter bounds."""
        return self._parameter_bounds.copy()
    
    def get_default_parameters(self, species: str = "tobacco") -> Dict[str, float]:
        """Get default C3 parameters for species."""
        # Default tobacco parameters at 25°C
        if species == "tobacco":
            return {
                "Vcmax": 100.0,
                "J": 200.0,
                "Tp": 12.0,
                "Rd": 1.5,
                "gm": 0.5,
            }
        else:
            # Generic defaults
            return {
                "Vcmax": 80.0,
                "J": 160.0,
                "Tp": 10.0,
                "Rd": 1.0,
                "gm": 0.4,
            }
    
    def initial_guess(
        self, 
        exdf: ExtendedDataFrame,
        fixed_parameters: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """Generate initial C3 parameter guesses."""
        from aci_py.analysis.initial_guess import estimate_c3_initial_parameters
        
        # Get initial estimates from data
        initial_params = estimate_c3_initial_parameters(exdf)
        
        # Override with any fixed parameters
        if fixed_parameters:
            initial_params.update(fixed_parameters)
            
        return initial_params
    
    def validate_parameters(self, parameters: Dict[str, float]) -> Tuple[bool, str]:
        """Validate C3 parameters."""
        # Check all required parameters are present
        for param in self._parameter_names:
            if param not in parameters:
                return False, f"Missing required parameter: {param}"
        
        # Check bounds
        for param, value in parameters.items():
            if param in self._parameter_bounds:
                lower, upper = self._parameter_bounds[param]
                if not lower <= value <= upper:
                    return False, f"{param} = {value} is outside bounds [{lower}, {upper}]"
        
        # Check parameter relationships
        if parameters["J"] < parameters["Vcmax"]:
            return False, "J should typically be greater than Vcmax"
        
        return True, ""


class C4Model(PhotosynthesisModel):
    """
    C4 photosynthesis model (von Caemmerer).
    
    This is a placeholder - full implementation will follow.
    """
    
    def __init__(self):
        super().__init__("C4")
        
        # Define C4 parameter names and bounds
        self._parameter_names = ["Vcmax", "Vpmax", "J", "Rd", "gbs", "Rm_frac"]
        self._parameter_bounds = {
            "Vcmax": (0.0, 200.0),    # µmol m⁻² s⁻¹
            "Vpmax": (0.0, 400.0),    # µmol m⁻² s⁻¹
            "J": (0.0, 1000.0),       # µmol m⁻² s⁻¹
            "Rd": (0.0, 10.0),        # µmol m⁻² s⁻¹
            "gbs": (0.0, 100.0),      # mmol m⁻² s⁻¹
            "Rm_frac": (0.0, 1.0),    # dimensionless
        }
        self._parameter_units = {
            "Vcmax": "µmol m⁻² s⁻¹",
            "Vpmax": "µmol m⁻² s⁻¹",
            "J": "µmol m⁻² s⁻¹",
            "Rd": "µmol m⁻² s⁻¹",
            "gbs": "mmol m⁻² s⁻¹",
            "Rm_frac": "dimensionless",
        }
    
    def calculate_assimilation(
        self, 
        parameters: Dict[str, float], 
        exdf: ExtendedDataFrame,
        return_diagnostics: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, Dict]]:
        """Calculate C4 assimilation using von Caemmerer model."""
        from aci_py.core.c4_calculations import calculate_c4_assimilation
        
        # Map parameters to expected names
        mapped_params = {
            'Vcmax_at_25': parameters.get('Vcmax', parameters.get('Vcmax_at_25', 50)),
            'Vpmax_at_25': parameters.get('Vpmax', parameters.get('Vpmax_at_25', 150)),
            'J_at_25': parameters.get('J', parameters.get('J_at_25', 400)),
            'RL_at_25': parameters.get('Rd', parameters.get('RL_at_25', 1.0)),
            'gbs': parameters.get('gbs', 0.003),
            'Rm_frac': parameters.get('Rm_frac', 0.5),
            'Vpr': parameters.get('Vpr', parameters.get('Vpmax', 150) * 0.8)  # Default to 80% of Vpmax
        }
        
        # Calculate assimilation
        result = calculate_c4_assimilation(
            exdf,
            **mapped_params,
            return_extended=return_diagnostics
        )
        
        if return_diagnostics:
            # For extended output, result is an ExtendedDataFrame
            An = result.data['An'].values
            diagnostics = {
                'Ac': result.data['Ac'].values,
                'Aj': result.data['Aj'].values,
                'Vpc': result.data['Vpc'].values if 'Vpc' in result.data else None,
                'Vp': result.data['Vp'].values if 'Vp' in result.data else None,
                'limiting_process': result.data['C4_limiting_process'].values if 'C4_limiting_process' in result.data else None
            }
            return An, diagnostics
        else:
            # For simple output, result is just the An array
            return result
    
    def get_parameter_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Get C4 parameter bounds."""
        return self._parameter_bounds.copy()
    
    def get_default_parameters(self, species: str = "maize") -> Dict[str, float]:
        """Get default C4 parameters for species."""
        # Default maize parameters at 25°C
        if species == "maize":
            return {
                "Vcmax": 50.0,
                "Vpmax": 120.0,
                "J": 400.0,
                "Rd": 2.0,
                "gbs": 3.0,
                "Rm_frac": 0.5,
            }
        else:
            # Generic defaults
            return {
                "Vcmax": 40.0,
                "Vpmax": 100.0,
                "J": 300.0,
                "Rd": 1.5,
                "gbs": 2.5,
                "Rm_frac": 0.5,
            }
    
    def initial_guess(
        self, 
        exdf: ExtendedDataFrame,
        fixed_parameters: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """Generate initial C4 parameter guesses."""
        from aci_py.analysis.c4_fitting import initial_guess_c4_aci
        
        # Get initial estimates from data
        initial_params = initial_guess_c4_aci(exdf)
        
        # Map to our parameter names
        mapped_params = {
            'Vcmax': initial_params.get('Vcmax_at_25', 50),
            'Vpmax': initial_params.get('Vpmax_at_25', 150),
            'J': initial_params.get('J_at_25', 400),
            'Rd': initial_params.get('RL_at_25', 1.0),
            'gbs': 3.0,  # Default value
            'Rm_frac': 0.5  # Default value
        }
        
        # Add Vpr if present
        if 'Vpr' in initial_params:
            mapped_params['Vpr'] = initial_params['Vpr']
        
        # Override with any fixed parameters
        if fixed_parameters:
            mapped_params.update(fixed_parameters)
            
        return mapped_params
    
    def validate_parameters(self, parameters: Dict[str, float]) -> Tuple[bool, str]:
        """Validate C4 parameters."""
        # Check all required parameters are present
        for param in self._parameter_names:
            if param not in parameters:
                return False, f"Missing required parameter: {param}"
        
        # Check bounds
        for param, value in parameters.items():
            if param in self._parameter_bounds:
                lower, upper = self._parameter_bounds[param]
                if not lower <= value <= upper:
                    return False, f"{param} = {value} is outside bounds [{lower}, {upper}]"
        
        # Check parameter relationships
        if parameters["Vpmax"] < parameters["Vcmax"]:
            return False, "Vpmax should typically be greater than Vcmax"
        
        return True, ""