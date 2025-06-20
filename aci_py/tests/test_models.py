"""
Unit tests for photosynthesis model classes.

Tests the models.py module including:
- C3Model calculate_assimilation implementation
- C4Model calculate_assimilation implementation
- Initial guess methods
- Parameter validation
"""

import pytest
import numpy as np
import pandas as pd

from aci_py.core.models import PhotosynthesisModel, C3Model, C4Model
from aci_py.core.data_structures import ExtendedDataFrame


class TestPhotoSynthesisModelBase:
    """Test the abstract base class."""
    
    def test_initialization(self):
        """Test base class initialization."""
        # Create a concrete implementation for testing
        class TestModel(PhotosynthesisModel):
            def calculate_assimilation(self, parameters, exdf, return_diagnostics=False):
                return np.zeros(10)
            
            def get_parameter_bounds(self):
                return {'param1': (0, 100)}
            
            def get_default_parameters(self, species="test"):
                return {'param1': 50}
            
            def initial_guess(self, exdf, fixed_parameters=None):
                return {'param1': 50}
            
            def validate_parameters(self, parameters):
                return True, ""
        
        model = TestModel("Test")
        assert model.name == "Test"
        assert model._parameter_names == []
        assert model._parameter_bounds == {}
        assert model._parameter_units == {}
    
    def test_get_required_columns(self):
        """Test default required columns."""
        class TestModel(PhotosynthesisModel):
            def calculate_assimilation(self, parameters, exdf, return_diagnostics=False):
                return np.zeros(10)
            
            def get_parameter_bounds(self):
                return {}
            
            def get_default_parameters(self, species="test"):
                return {}
            
            def initial_guess(self, exdf, fixed_parameters=None):
                return {}
            
            def validate_parameters(self, parameters):
                return True, ""
        
        model = TestModel("Test")
        required = model.get_required_columns()
        assert "A" in required
        assert "Ci" in required
        assert "Tleaf" in required
        assert "Pa" in required


class TestC3Model:
    """Test C3 photosynthesis model."""
    
    @staticmethod
    def create_test_data(n_points=13):
        """Create test ExtendedDataFrame with required columns."""
        df = pd.DataFrame({
            'A': np.random.normal(20, 2, n_points),
            'Ci': np.linspace(50, 1500, n_points),
            'Cc': np.linspace(45, 1450, n_points),  # Slightly lower than Ci
            'Tleaf': 25.0,
            'T_leaf_K': 298.15,
            'Pa': 101.325,
            'oxygen': 21.0
        })
        return ExtendedDataFrame(df)
    
    def test_initialization(self):
        """Test C3Model initialization."""
        model = C3Model()
        assert model.name == "C3"
        assert "Vcmax" in model._parameter_bounds
        assert "J" in model._parameter_bounds
        assert "Tp" in model._parameter_bounds
        assert "Rd" in model._parameter_bounds
        assert "gm" in model._parameter_bounds
    
    def test_get_parameter_bounds(self):
        """Test parameter bounds."""
        model = C3Model()
        bounds = model.get_parameter_bounds()
        
        assert bounds["Vcmax"] == (0.0, 1000.0)
        assert bounds["J"] == (0.0, 1000.0)
        assert bounds["Tp"] == (0.0, 100.0)
        assert bounds["Rd"] == (0.0, 50.0)
        assert bounds["gm"] == (0.0, 10.0)
    
    def test_get_default_parameters(self):
        """Test default parameters for different species."""
        model = C3Model()
        
        # Tobacco defaults
        tobacco_params = model.get_default_parameters("tobacco")
        assert tobacco_params["Vcmax"] == 100.0
        assert tobacco_params["J"] == 200.0
        assert tobacco_params["Tp"] == 12.0
        assert tobacco_params["Rd"] == 1.5
        assert tobacco_params["gm"] == 0.5
        
        # Generic defaults
        generic_params = model.get_default_parameters("other")
        assert generic_params["Vcmax"] == 80.0
        assert generic_params["J"] == 160.0
    
    def test_validate_parameters(self):
        """Test parameter validation."""
        model = C3Model()
        
        # Valid parameters
        valid_params = {
            "Vcmax": 100.0,
            "J": 200.0,
            "Tp": 12.0,
            "Rd": 1.5,
            "gm": 0.5
        }
        is_valid, msg = model.validate_parameters(valid_params)
        assert is_valid
        assert msg == ""
        
        # Missing parameter
        invalid_params = {
            "Vcmax": 100.0,
            "J": 200.0,
            # Missing Tp, Rd, gm
        }
        is_valid, msg = model.validate_parameters(invalid_params)
        assert not is_valid
        assert "Missing required parameter" in msg
        
        # Out of bounds
        invalid_params = valid_params.copy()
        invalid_params["Vcmax"] = -10.0
        is_valid, msg = model.validate_parameters(invalid_params)
        assert not is_valid
        assert "outside bounds" in msg
        
        # Invalid relationship (J < Vcmax)
        invalid_params = valid_params.copy()
        invalid_params["J"] = 50.0  # Less than Vcmax
        is_valid, msg = model.validate_parameters(invalid_params)
        assert not is_valid
        assert "J should typically be greater than Vcmax" in msg
    
    def test_calculate_assimilation(self):
        """Test C3 assimilation calculation."""
        model = C3Model()
        exdf = self.create_test_data()
        
        # Use proper parameter names for C3 calculations
        parameters = {
            'Vcmax_at_25': 100.0,
            'J_at_25': 200.0,
            'Tp_at_25': 12.0,
            'RL_at_25': 1.5,
            'gmc': 0.5,
            # Add required kinetic parameters
            'Gamma_star_at_25': 36.94438,
            'Kc_at_25': 269.3391,
            'Ko_at_25': 163.7146
        }
        
        # Calculate without diagnostics
        An = model.calculate_assimilation(parameters, exdf, return_diagnostics=False)
        assert isinstance(An, np.ndarray)
        assert len(An) == len(exdf.data)
        assert np.all(np.isfinite(An))  # No NaN or inf values
        
        # Calculate with diagnostics
        An, diagnostics = model.calculate_assimilation(parameters, exdf, return_diagnostics=True)
        assert isinstance(An, np.ndarray)
        assert isinstance(diagnostics, dict)
        assert 'Ac' in diagnostics
        assert 'Aj' in diagnostics
        assert 'Ap' in diagnostics
        assert 'limiting_process' in diagnostics
    
    def test_initial_guess(self):
        """Test initial parameter guess generation."""
        model = C3Model()
        exdf = self.create_test_data()
        
        # Without fixed parameters
        initial = model.initial_guess(exdf)
        assert isinstance(initial, dict)
        assert 'Vcmax_at_25' in initial
        assert 'J_at_25' in initial
        
        # With fixed parameters
        fixed = {'gmc': 1.0}
        initial = model.initial_guess(exdf, fixed_parameters=fixed)
        assert initial['gmc'] == 1.0


class TestC4Model:
    """Test C4 photosynthesis model."""
    
    @staticmethod
    def create_test_data(n_points=13):
        """Create test ExtendedDataFrame with C4-required columns."""
        df = pd.DataFrame({
            'A': np.random.normal(25, 2, n_points),
            'Ci': np.linspace(50, 1500, n_points),
            'PCm': np.linspace(35, 1050, n_points) * 0.101325,  # Mesophyll CO2 in µbar
            'PCi': np.linspace(50, 1500, n_points) * 0.101325,  # Intercellular CO2 in µbar
            'Tleaf': 25.0,
            'T_leaf_K': 298.15,
            'Pa': 101.325,
            'oxygen': 21.0,
            'ao': 0.21,  # Fraction of O2
            'gamma_star': 0.000193,  # C4 compensation point
            'Kc': 650.0,  # µbar
            'Ko': 450.0,  # mbar
            'Kp': 80.0,   # µbar
            'total_pressure': 1.01325  # bar
        })
        return ExtendedDataFrame(df)
    
    def test_initialization(self):
        """Test C4Model initialization."""
        model = C4Model()
        assert model.name == "C4"
        assert "Vcmax" in model._parameter_bounds
        assert "Vpmax" in model._parameter_bounds
        assert "J" in model._parameter_bounds
        assert "Rd" in model._parameter_bounds
        assert "gbs" in model._parameter_bounds
        assert "Rm_frac" in model._parameter_bounds
    
    def test_get_parameter_bounds(self):
        """Test C4 parameter bounds."""
        model = C4Model()
        bounds = model.get_parameter_bounds()
        
        assert bounds["Vcmax"] == (0.0, 200.0)
        assert bounds["Vpmax"] == (0.0, 400.0)
        assert bounds["J"] == (0.0, 1000.0)
        assert bounds["gbs"] == (0.0, 100.0)
        assert bounds["Rm_frac"] == (0.0, 1.0)
    
    def test_get_default_parameters(self):
        """Test C4 default parameters."""
        model = C4Model()
        
        # Maize defaults
        maize_params = model.get_default_parameters("maize")
        assert maize_params["Vcmax"] == 50.0
        assert maize_params["Vpmax"] == 120.0
        assert maize_params["J"] == 400.0
        
        # Generic defaults
        generic_params = model.get_default_parameters("other")
        assert generic_params["Vcmax"] == 40.0
    
    def test_validate_parameters(self):
        """Test C4 parameter validation."""
        model = C4Model()
        
        # Valid parameters
        valid_params = {
            "Vcmax": 50.0,
            "Vpmax": 120.0,
            "J": 400.0,
            "Rd": 2.0,
            "gbs": 3.0,
            "Rm_frac": 0.5
        }
        is_valid, msg = model.validate_parameters(valid_params)
        assert is_valid
        
        # Invalid relationship (Vpmax < Vcmax)
        invalid_params = valid_params.copy()
        invalid_params["Vpmax"] = 30.0  # Less than Vcmax
        is_valid, msg = model.validate_parameters(invalid_params)
        assert not is_valid
        assert "Vpmax should typically be greater than Vcmax" in msg
    
    def test_calculate_assimilation(self):
        """Test C4 assimilation calculation."""
        model = C4Model()
        exdf = self.create_test_data()
        
        parameters = {
            'Vcmax': 50.0,
            'Vpmax': 120.0,
            'J': 400.0,
            'Rd': 2.0,
            'gbs': 3.0,
            'Rm_frac': 0.5,
            'Vpr': 100.0
        }
        
        # Calculate without diagnostics
        An = model.calculate_assimilation(parameters, exdf, return_diagnostics=False)
        assert isinstance(An, np.ndarray)
        assert len(An) == len(exdf.data)
        
        # Calculate with diagnostics
        An, diagnostics = model.calculate_assimilation(parameters, exdf, return_diagnostics=True)
        assert isinstance(An, np.ndarray)
        assert isinstance(diagnostics, dict)
        assert 'Ac' in diagnostics
        assert 'Aj' in diagnostics
        # C4 model returns Vpc and Vp instead of Cbs
        assert 'Vpc' in diagnostics or 'Vp' in diagnostics
    
    def test_initial_guess(self):
        """Test C4 initial parameter guess."""
        model = C4Model()
        exdf = self.create_test_data()
        
        initial = model.initial_guess(exdf)
        assert isinstance(initial, dict)
        assert 'Vcmax' in initial
        assert 'Vpmax' in initial
        assert 'J' in initial
        assert 'Rd' in initial
        
        # With fixed parameters
        fixed = {'gbs': 5.0}
        initial = model.initial_guess(exdf, fixed_parameters=fixed)
        assert initial['gbs'] == 5.0
    
    def test_parameter_mapping(self):
        """Test parameter name mapping between models and calculations."""
        model = C4Model()
        exdf = self.create_test_data()
        
        # Test with alternative parameter names
        parameters = {
            'Vcmax_at_25': 50.0,  # Should be mapped to Vcmax
            'Vpmax_at_25': 120.0,
            'J_at_25': 400.0,
            'RL_at_25': 2.0,  # Should be mapped to Rd
            'gbs': 3.0,
            'Rm_frac': 0.5
        }
        
        # Should still work with mapped names
        An = model.calculate_assimilation(parameters, exdf, return_diagnostics=False)
        assert isinstance(An, np.ndarray)
        assert len(An) == len(exdf.data)


class TestModelIntegration:
    """Integration tests for model calculations."""
    
    def test_c3_c4_comparison(self):
        """Test that C3 and C4 models produce different results on same data."""
        # Create data suitable for both models
        df = pd.DataFrame({
            'A': np.random.normal(20, 2, 10),
            'Ci': np.linspace(50, 1500, 10),
            'Cc': np.linspace(45, 1450, 10),
            'PCm': np.linspace(35, 1050, 10) * 0.101325,
            'PCi': np.linspace(50, 1500, 10) * 0.101325,
            'Tleaf': 25.0,
            'T_leaf_K': 298.15,
            'Pa': 101.325,
            'oxygen': 21.0,
            'ao': 0.21,
            'gamma_star': 0.000193,
            'Kc': 650.0,
            'Ko': 450.0,
            'Kp': 80.0,
            'total_pressure': 1.01325
        })
        exdf = ExtendedDataFrame(df)
        
        # C3 calculation
        c3_model = C3Model()
        c3_params = {
            'Vcmax_at_25': 100.0,
            'J_at_25': 200.0,
            'Tp_at_25': 12.0,
            'RL_at_25': 1.5,
            'gmc': 0.5,
            'Gamma_star_at_25': 36.94438,
            'Kc_at_25': 269.3391,
            'Ko_at_25': 163.7146
        }
        c3_an = c3_model.calculate_assimilation(c3_params, exdf)
        
        # C4 calculation
        c4_model = C4Model()
        c4_params = {
            'Vcmax': 50.0,
            'Vpmax': 120.0,
            'J': 400.0,
            'Rd': 2.0,
            'gbs': 3.0,
            'Rm_frac': 0.5,
            'Vpr': 100.0
        }
        c4_an = c4_model.calculate_assimilation(c4_params, exdf)
        
        # Results should be different
        assert not np.allclose(c3_an, c4_an)
        
        # Both should be valid (no NaN)
        assert np.all(np.isfinite(c3_an))
        assert np.all(np.isfinite(c4_an))