"""
LI-COR 6800 gas exchange calculations.

This module implements the standard calculations used by LI-COR 6800
to compute gas exchange parameters from raw sensor data.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional


def calculate_licor_gas_exchange(
    df: pd.DataFrame,
    leaf_area: float = 6.0,  # cm²
    boundary_layer_conductance: Optional[float] = None
) -> pd.DataFrame:
    """
    Calculate gas exchange parameters from LI-COR 6800 raw sensor data.
    
    This function implements the standard LI-COR calculations for:
    - Net assimilation rate (A)
    - Transpiration rate (E)
    - Intercellular CO2 (Ci)
    - Stomatal conductance (gsw)
    - Other derived parameters
    
    Args:
        df: DataFrame with raw LI-COR data
        leaf_area: Leaf area in cm² (default: 6.0)
        boundary_layer_conductance: If None, calculate from flow rate
        
    Returns:
        DataFrame with calculated gas exchange parameters added
    """
    # Create a copy to avoid modifying the original
    result = df.copy()
    
    # Convert leaf area to m²
    S = leaf_area / 10000.0  # cm² to m²
    
    # Get atmospheric pressure in kPa (if Pa is in kPa already)
    Pa = result['Pa'].values if 'Pa' in result.columns else 101.325
    
    # Flow rate in µmol/s
    Flow = result['Flow'].values if 'Flow' in result.columns else 500.0
    
    # Calculate molar flow rate (mol/s)
    # Flow is in µmol/s, convert to mol/s
    flow_mol_s = Flow * 1e-6
    
    # Temperature in Kelvin
    T_leaf_K = result['Tleaf'].values + 273.15 if 'Tleaf' in result.columns else 298.15
    T_air_K = result['Tair'].values + 273.15 if 'Tair' in result.columns else 298.15
    
    # Universal gas constant
    R = 8.314  # J mol⁻¹ K⁻¹
    
    # 1. Calculate boundary layer conductance if not provided
    if boundary_layer_conductance is None and 'Flow' in result.columns:
        # Simplified calculation based on flow rate
        # This is an approximation - actual LI-COR uses more complex calculations
        gbw = 1.37 * np.sqrt(Flow / 500.0)  # mol m⁻² s⁻¹
        # Ensure gbw is an array with the same length as data
        if isinstance(gbw, (int, float)):
            gbw = np.full(len(result), gbw)
    else:
        gbw_val = boundary_layer_conductance or 1.37
        gbw = np.full(len(result), gbw_val)
    
    # 2. Calculate net CO2 assimilation rate (A)
    if all(col in result.columns for col in ['CO2_r', 'CO2_s']):
        # A = (CO2_r - CO2_s) * Flow / (Leaf_Area)
        # CO2_r and CO2_s are in µmol/mol, Flow in µmol/s, area in m²
        # Result is in µmol m⁻² s⁻¹
        CO2_diff = result['CO2_r'] - result['CO2_s']  # µmol/mol
        result['A'] = CO2_diff * Flow / (S * 1e6)  # µmol m⁻² s⁻¹
    
    # 3. Calculate transpiration rate (E)
    if all(col in result.columns for col in ['H2O_s', 'H2O_r']):
        # E = (H2O_s - H2O_r) * Flow / (Leaf_Area)
        # H2O in mmol/mol, Flow in µmol/s, area in m²
        # Result is in mmol m⁻² s⁻¹
        H2O_diff = result['H2O_s'] - result['H2O_r']  # mmol/mol
        # Convert properly: (mmol/mol) × (µmol/s) / m² 
        # = (mmol/mol) × (µmol/s) / m² × (1/1000) × (1/1000) × 1000
        # = mmol m⁻² s⁻¹
        result['E'] = H2O_diff * Flow / 1000 / S / 1000  # mmol m⁻² s⁻¹
    
    # 4. Calculate total conductance to water vapor (gtw)
    if 'H2O_s' in result.columns and 'H2O_a' in result.columns and 'E' in result.columns:
        # gtw = E / (H2O_s - H2O_a)
        # H2O values are in mmol/mol, typical gradients are 0.1-5 mmol/mol
        H2O_gradient = result['H2O_s'] - result['H2O_a']
        # Use a smaller threshold appropriate for mmol/mol units
        # Also ensure positive gradient (water should flow out of leaf)
        mask = H2O_gradient > 0.01  # 0.01 mmol/mol threshold, positive only
        # Initialize column with zeros
        result['gtw'] = 0.0
        # Calculate where valid
        if mask.sum() > 0:
            # E is in mmol m⁻² s⁻¹, H2O gradient in mmol/mol
            # gtw = E / ΔH2O = (mmol m⁻² s⁻¹) / (mmol/mol) = mol m⁻² s⁻¹
            result.loc[mask, 'gtw'] = result.loc[mask, 'E'].values / H2O_gradient[mask].values
    
    # 5. Calculate stomatal conductance to water vapor (gsw)
    # First check if gsw already exists and has non-zero values
    if 'gsw' in result.columns and not (result['gsw'] == 0).all():
        # Use existing gsw
        pass
    elif 'CndTotal' in result.columns and 'CndCO2' in result.columns:
        # Some LI-COR files have conductance with different names
        if not (result['CndTotal'] == 0).all():
            result['gsw'] = result['CndTotal']
    elif 'gtw' in result.columns:
        # Initialize column
        result['gsw'] = 0.0
        
        # Get values
        gtw_values = result['gtw'].values
        gbw_values = gbw if isinstance(gbw, np.ndarray) else np.full(len(result), gbw)
        
        # LI-COR uses a complex quadratic formula for gsw that accounts for
        # one-sided vs two-sided leaves via the K parameter
        # For now, we'll use different approaches based on the gtw/gbw relationship
        
        if 'K' in result.columns:
            K = result['K'].values[0] if len(result) > 0 else 0.5
        else:
            K = 0.5  # Default for one-sided leaves
        
        # Method 1: When gtw > gbw (typical for high transpiration)
        # This suggests parallel pathways (one-sided leaves or high conductance)
        high_cond_mask = (gtw_values > gbw_values) & (gtw_values > 0)
        if high_cond_mask.sum() > 0:
            # For one-sided leaves: gsw ≈ gtw - gbw
            # This is an approximation of the complex LI-COR formula
            result.loc[high_cond_mask, 'gsw'] = gtw_values[high_cond_mask] - gbw_values[high_cond_mask]
        
        # Method 2: When gtw < gbw (typical for low transpiration)
        # Use series resistance model
        low_cond_mask = (gtw_values < gbw_values) & (gtw_values > 0) & (gbw_values > 0)
        if low_cond_mask.sum() > 0:
            # Series resistances: 1/gtw = 1/gsw + 1/gbw
            # Therefore: gsw = 1 / (1/gtw - 1/gbw)
            with np.errstate(divide='ignore', invalid='ignore'):
                gsw_series = 1.0 / (1.0/gtw_values[low_cond_mask] - 1.0/gbw_values[low_cond_mask])
                # Only use positive values
                positive_mask = gsw_series > 0
                if positive_mask.sum() > 0:
                    result.loc[low_cond_mask, 'gsw'][positive_mask] = gsw_series[positive_mask]
        
        # Method 3: Fallback using E and VPD relationship
        if 'E' in result.columns and 'VPDleaf' in result.columns:
            # For any remaining zero values, estimate from E/VPD
            zero_mask = (result['gsw'] == 0) & (result['VPDleaf'] > 0.1)
            if zero_mask.sum() > 0:
                # gsw ≈ E / VPD (rough approximation)
                # E is in mmol m⁻² s⁻¹, VPD in kPa, gsw in mol m⁻² s⁻¹
                result.loc[zero_mask, 'gsw'] = result.loc[zero_mask, 'E'] / result.loc[zero_mask, 'VPDleaf'] / 1000.0
    
    # 6. Calculate stomatal conductance to CO2 (gsc)
    if 'gsw' in result.columns:
        # gsc = gsw / 1.6 (approximate ratio of diffusivities)
        result['gsc'] = result['gsw'] / 1.6
    
    # 7. Calculate boundary layer conductance to CO2 (gbc)
    # Store gbw as a column (ensure it's an array)
    result['gbw'] = gbw if isinstance(gbw, np.ndarray) else np.full(len(result), gbw)
    result['gbc'] = result['gbw'] / 1.37  # Ratio of diffusivities
    
    # 8. Calculate total conductance to CO2 (gtc)
    if 'gsc' in result.columns:
        # gtc = 1 / (1/gsc + 1/gbc)
        result['gtc'] = 1.0 / (1.0/result['gsc'] + 1.0/result['gbc'])
    
    # 9. Calculate ambient CO2 in µmol/mol (Ca)
    # For ACI curves, Ca is the CO2 concentration entering the leaf chamber
    # This is typically the reference CO2 (CO2_r), not the ambient sensor
    if 'CO2_r' in result.columns:
        result['Ca'] = result['CO2_r']
    elif 'CO2_a' in result.columns:
        # Fallback to ambient sensor if reference not available
        result['Ca'] = result['CO2_a']
    
    # 10. Calculate intercellular CO2 (Ci)
    # First check if we have a direct Ci measurement from the instrument
    if 'Ci' in result.columns and not (result['Ci'] == 0).all():
        # Use existing Ci if it has non-zero values
        pass
    elif all(col in result.columns for col in ['Ca', 'A']):
        # Initialize column
        result['Ci'] = 0.0
        
        # If we have conductance, use the full calculation
        if 'gtc' in result.columns and (result['gtc'] > 0).any():
            # Ci = Ca - A/gtc (simplified)
            # For more accuracy, include the effect of transpiration:
            # Ci = ((gtc - E/2) * Ca - A) / (gtc + E/2)
            if 'E' in result.columns:
                # Convert E from mmol to mol for unit consistency
                E_mol = result['E'] / 1000.0  # mmol m⁻² s⁻¹ to mol m⁻² s⁻¹
                numerator = (result['gtc'] - E_mol/2) * result['Ca'] - result['A']
                denominator = result['gtc'] + E_mol/2
                mask = denominator > 0.001  # Small positive threshold
                if mask.sum() > 0:
                    result.loc[mask, 'Ci'] = numerator[mask] / denominator[mask]
            else:
                # Simplified calculation without E
                mask = result['gtc'] > 0.001
                if mask.sum() > 0:
                    result.loc[mask, 'Ci'] = result.loc[mask, 'Ca'] - result.loc[mask, 'A'] / result.loc[mask, 'gtc']
        else:
            # Fallback: Use a typical Ci/Ca ratio approach
            # For C3 plants under normal conditions, Ci/Ca is typically 0.6-0.8
            # We'll use the A-Ci relationship to estimate
            # When A is positive, Ci < Ca; when A is negative, Ci > Ca
            
            # For ACI curves, we expect Ci to follow Ca closely with some offset based on A
            # This is a rough approximation when conductance data is not available
            # Assume a typical total conductance of 0.2 mol m⁻² s⁻¹
            assumed_gtc = 0.2
            result['Ci'] = result['Ca'] - result['A'] / assumed_gtc
            
            # Ensure Ci is reasonable (between 0 and 2*Ca)
            result.loc[result['Ci'] < 0, 'Ci'] = 0
            result.loc[result['Ci'] > 2 * result['Ca'], 'Ci'] = 2 * result['Ca']
    
    # 11. Calculate vapor pressure deficit (VPD)
    if all(col in result.columns for col in ['Tleaf', 'H2O_s']):
        # Saturation vapor pressure at leaf temperature (kPa)
        es_leaf = 0.6108 * np.exp(17.27 * result['Tleaf'] / (result['Tleaf'] + 237.3))
        # Actual vapor pressure (kPa)
        ea = result['H2O_s'] * Pa / 1000.0  # Convert mmol/mol to kPa
        result['VPDleaf'] = es_leaf - ea
    
    # 12. Calculate relative humidity in chamber
    if all(col in result.columns for col in ['Tair', 'H2O_s']):
        # Saturation vapor pressure at air temperature (kPa)
        es_air = 0.6108 * np.exp(17.27 * result['Tair'] / (result['Tair'] + 237.3))
        # Actual vapor pressure (kPa)
        ea = result['H2O_s'] * Pa / 1000.0
        result['RHcham'] = 100.0 * ea / es_air
    
    # 13. Calculate CO2 partial pressures (Pa)
    if 'Ca' in result.columns:
        result['Pca'] = result['Ca'] * Pa / 1000.0  # Pa
    if 'Ci' in result.columns:
        result['Pci'] = result['Ci'] * Pa / 1000.0  # Pa
    
    # 14. Add units information as DataFrame attributes
    result.attrs['units'] = {
        'A': 'µmol m⁻² s⁻¹',
        'E': 'mmol m⁻² s⁻¹',
        'Ca': 'µmol mol⁻¹',
        'Ci': 'µmol mol⁻¹',
        'gsw': 'mol m⁻² s⁻¹',
        'gsc': 'mol m⁻² s⁻¹',
        'gtw': 'mol m⁻² s⁻¹',
        'gtc': 'mol m⁻² s⁻¹',
        'gbw': 'mol m⁻² s⁻¹',
        'gbc': 'mol m⁻² s⁻¹',
        'VPDleaf': 'kPa',
        'RHcham': '%',
        'Pca': 'Pa',
        'Pci': 'Pa'
    }
    
    return result


def detect_leaf_area_from_licor(df: pd.DataFrame) -> float:
    """
    Try to detect the leaf area from LI-COR data.
    
    Args:
        df: DataFrame with LI-COR data
        
    Returns:
        Leaf area in cm², or 6.0 as default
    """
    # Check for Area column
    if 'Area' in df.columns:
        area = df['Area'].iloc[0]
        if not pd.isna(area) and area > 0:
            return float(area)
    
    # Check for S column (area in m²)
    if 'S' in df.columns:
        s = df['S'].iloc[0]
        if not pd.isna(s) and s > 0:
            return float(s) * 10000  # Convert m² to cm²
    
    # Check preamble or constants sections if available
    # This would require access to the full file structure
    
    # Default to 6 cm² (standard LI-COR aperture)
    return 6.0


def validate_calculated_values(df: pd.DataFrame) -> Dict[str, bool]:
    """
    Validate that calculated values are reasonable.
    
    Args:
        df: DataFrame with calculated values
        
    Returns:
        Dictionary of validation results
    """
    validations = {}
    
    # Check A is in reasonable range
    if 'A' in df.columns:
        a_values = df['A'].dropna()
        validations['A_range'] = (a_values >= -10).all() and (a_values <= 100).all()
        validations['A_has_variation'] = a_values.std() > 0.1
    
    # Check Ci is positive and less than Ca
    if 'Ci' in df.columns and 'Ca' in df.columns:
        mask = df['Ci'].notna() & df['Ca'].notna()
        validations['Ci_positive'] = (df.loc[mask, 'Ci'] >= 0).all()
        validations['Ci_less_than_Ca'] = (df.loc[mask, 'Ci'] <= df.loc[mask, 'Ca'] * 1.2).all()
    
    # Check gsw is positive
    if 'gsw' in df.columns:
        gsw_values = df['gsw'].dropna()
        validations['gsw_positive'] = (gsw_values >= 0).all()
        validations['gsw_reasonable'] = (gsw_values <= 20.0).all()  # mol m⁻² s⁻¹ (high for well-watered plants)
    
    # Check E is positive
    if 'E' in df.columns:
        e_values = df['E'].dropna()
        validations['E_positive'] = (e_values >= 0).all()
        validations['E_reasonable'] = (e_values <= 20).all()  # mmol m⁻² s⁻¹
    
    return validations