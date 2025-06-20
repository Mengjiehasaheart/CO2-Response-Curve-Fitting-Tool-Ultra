"""
Core modules for ACI_py.

This package contains the fundamental data structures, models, and algorithms
for photosynthesis analysis.
"""

from aci_py.core.data_structures import ExtendedDataFrame, identify_common_columns
from aci_py.core.models import PhotosynthesisModel, C3Model, C4Model
from aci_py.core.temperature import (
    arrhenius_response,
    johnson_eyring_williams_response,
    gaussian_response,
    polynomial_response,
    calculate_temperature_response,
    apply_temperature_response,
    TemperatureParameter,
    C3_TEMPERATURE_PARAM_BERNACCHI,
    C3_TEMPERATURE_PARAM_SHARKEY,
    C3_TEMPERATURE_PARAM_FLAT,
)
from aci_py.core.c3_calculations import (
    calculate_c3_assimilation,
    identify_c3_limiting_process,
    C3AssimilationResult,
)
from aci_py.core.c4_calculations import (
    calculate_c4_assimilation,
    identify_c4_limiting_processes,
    apply_gm_c4,
)
from aci_py.core.preprocessing import (
    detect_outliers_iqr,
    detect_outliers_zscore,
    detect_outliers_mad,
    check_environmental_stability,
    identify_aci_outliers,
    remove_outliers,
    check_aci_data_quality,
    preprocess_aci_data,
    flag_points_for_removal,
)

__all__ = [
    # Data structures
    "ExtendedDataFrame",
    "identify_common_columns",
    # Models
    "PhotosynthesisModel",
    "C3Model",
    "C4Model",
    # Temperature response
    "arrhenius_response",
    "johnson_eyring_williams_response",
    "gaussian_response",
    "polynomial_response",
    "calculate_temperature_response",
    "apply_temperature_response",
    "TemperatureParameter",
    "C3_TEMPERATURE_PARAM_BERNACCHI",
    "C3_TEMPERATURE_PARAM_SHARKEY",
    "C3_TEMPERATURE_PARAM_FLAT",
    # C3 calculations
    "calculate_c3_assimilation",
    "identify_c3_limiting_process",
    "C3AssimilationResult",
    # C4 calculations
    "calculate_c4_assimilation",
    "identify_c4_limiting_processes",
    "apply_gm_c4",
    # Preprocessing
    "detect_outliers_iqr",
    "detect_outliers_zscore",
    "detect_outliers_mad",
    "check_environmental_stability",
    "identify_aci_outliers",
    "remove_outliers",
    "check_aci_data_quality",
    "preprocess_aci_data",
    "flag_points_for_removal",
]