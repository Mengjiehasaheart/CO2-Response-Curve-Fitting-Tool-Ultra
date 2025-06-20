"""
ACI_py: Professional photosynthesis A-Ci curve analysis tool.

A Python translation and optimization of the PhotoGEA R package's ACI fitting
functionality, designed to be user-friendly for photosynthesis researchers.
"""

__version__ = "0.2.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Import main components for easier access
from aci_py.core.data_structures import ExtendedDataFrame
from aci_py.core.models import C3Model, C4Model
from aci_py.core.c3_calculations import calculate_c3_assimilation, identify_c3_limiting_process
from aci_py.core.temperature import (
    C3_TEMPERATURE_PARAM_BERNACCHI,
    C3_TEMPERATURE_PARAM_SHARKEY,
    C3_TEMPERATURE_PARAM_FLAT,
)
from aci_py.io.licor import read_licor_file

__all__ = [
    # Core classes
    "ExtendedDataFrame",
    "C3Model", 
    "C4Model",
    # Main functions
    "calculate_c3_assimilation",
    "identify_c3_limiting_process",
    "read_licor_file",
    # Temperature parameter sets
    "C3_TEMPERATURE_PARAM_BERNACCHI",
    "C3_TEMPERATURE_PARAM_SHARKEY",
    "C3_TEMPERATURE_PARAM_FLAT",
    # Version
    "__version__",
]