"""
LI-COR file reading utilities.

This module provides functions to read data files from LI-COR gas exchange
systems, particularly the LI-6800.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple, Union
import re
from pathlib import Path
import hashlib
import tempfile
import os
import warnings
from aci_py.core.data_structures import ExtendedDataFrame


# Common LI-COR column name mappings to standardized names
LICOR_COLUMN_MAPPING = {
    "A": "A",                    # Net assimilation
    "Ca": "Ca",                  # Ambient CO2
    "Ci": "Ci",                  # Intercellular CO2
    "Pci": "Pci",                # Intercellular CO2 pressure
    "Pca": "Pca",                # Ambient CO2 pressure
    "E": "E",                    # Transpiration
    "gsw": "gsw",                # Stomatal conductance to water
    "gbw": "gbw",                # Boundary layer conductance to water
    "gtw": "gtw",                # Total conductance to water
    "Tleaf": "Tleaf",            # Leaf temperature
    "Tair": "Tair",              # Air temperature
    "TleafCnd": "TleafCnd",      # Leaf temperature (conductance)
    "Pa": "Pa",                  # Atmospheric pressure
    "RHcham": "RHcham",          # Chamber relative humidity
    "VPDleaf": "VPDleaf",        # Leaf vapor pressure deficit
    "Q": "Q",                    # PAR
    "Qin": "Qin",                # Incident PAR
    "Flow": "Flow",              # Flow rate
}

# Common units for LI-COR measurements
LICOR_UNITS = {
    "A": "µmol m⁻² s⁻¹",
    "Ca": "µmol mol⁻¹",
    "Ci": "µmol mol⁻¹",
    "Pci": "Pa",
    "Pca": "Pa",
    "E": "mmol m⁻² s⁻¹",
    "gsw": "mol m⁻² s⁻¹",
    "gbw": "mol m⁻² s⁻¹",
    "gtw": "mol m⁻² s⁻¹",
    "Tleaf": "°C",
    "Tair": "°C",
    "TleafCnd": "°C",
    "Pa": "kPa",
    "RHcham": "%",
    "VPDleaf": "kPa",
    "Q": "µmol m⁻² s⁻¹",
    "Qin": "µmol m⁻² s⁻¹",
    "Flow": "µmol s⁻¹",
}


def read_licor_6800_csv(
    filepath: Union[str, Path],
    group_by: Optional[List[str]] = None,
    standardize_columns: bool = True
) -> Union[ExtendedDataFrame, Dict[str, ExtendedDataFrame]]:
    """
    Read a LI-COR 6800 CSV file.
    
    Args:
        filepath: Path to the CSV file
        group_by: Optional list of columns to group data by (e.g., ["Genotype", "Plant_NR"])
        standardize_columns: Whether to map column names to standard names
        
    Returns:
        If group_by is None: Single ExtendedDataFrame
        If group_by is specified: Dictionary of group names to ExtendedDataFrames
    """
    filepath = Path(filepath)
    
    # Check if this is a cached file with units in comments
    units_from_cache = None
    with open(filepath, 'r') as f:
        first_line = f.readline()
        if first_line.startswith('# Units:'):
            try:
                import ast
                units_str = first_line[8:].strip()
                units_from_cache = ast.literal_eval(units_str)
            except:
                pass
    
    # Read the CSV file
    # LI-COR files often have metadata rows at the top, try to detect data start
    if units_from_cache:
        # Skip the comment line for cached files
        df = pd.read_csv(filepath, comment='#')
    else:
        df = pd.read_csv(filepath)
    
    # Clean column names (remove special characters, trailing spaces)
    df.columns = [col.strip() for col in df.columns]
    
    # Drop any unnamed columns
    unnamed_mask = df.columns.str.contains('^Unnamed')
    df = df.loc[:, ~unnamed_mask]
    
    # Convert numeric columns
    # But preserve string columns like Genotype
    numeric_columns = []
    string_columns = ['Genotype', 'Type_Measurement', 'date', 'hhmmss']  # Known string columns
    
    for col in df.columns:
        if col not in string_columns:
            try:
                # Try to convert to numeric, but only if it makes sense
                temp = pd.to_numeric(df[col], errors='coerce')
                # Check if we lost too much data (more than 50% became NaN)
                if temp.notna().sum() > len(df) * 0.5:
                    df[col] = temp
                    numeric_columns.append(col)
            except:
                pass
    
    # Create units dictionary
    if units_from_cache:
        # Use cached units
        units = units_from_cache
    else:
        # Create from standard mapping
        units = {}
        for col in df.columns:
            if col in LICOR_UNITS:
                units[col] = LICOR_UNITS[col]
            else:
                units[col] = "unknown"
    
    # Create categories dictionary (all from LI-COR)
    categories = {col: "LI-COR 6800" for col in df.columns}
    
    # Create ExtendedDataFrame
    exdf = ExtendedDataFrame(data=df, units=units, categories=categories)
    
    # Add calculated gas properties if core columns exist
    if "Pa" in df.columns and "Tleaf" in df.columns:
        exdf.calculate_gas_properties(pressure_col="Pa", temperature_col="Tleaf")
    
    # Group if requested
    if group_by:
        # Ensure grouping columns are properly handled
        # Note: Genotype is already string, Plant_NR can be kept as numeric
        
        grouped = {}
        for name, group_df in df.groupby(group_by):
            # Convert single value to tuple for consistent handling
            if not isinstance(name, tuple):
                name = (name,)
            
            # Create key string - ensure proper string conversion
            # Handle numpy types by converting to native Python types
            key_parts = []
            for n in name:
                if hasattr(n, 'item'):  # numpy scalar
                    key_parts.append(str(n.item()))
                else:
                    key_parts.append(str(n))
            key = "_".join(key_parts)
            
            # Create ExtendedDataFrame for this group
            group_exdf = ExtendedDataFrame(
                data=group_df.reset_index(drop=True),
                units=units,
                categories=categories
            )
            
            # Add calculated properties
            if "Pa" in group_df.columns and "Tleaf" in group_df.columns:
                group_exdf.calculate_gas_properties()
            
            grouped[key] = group_exdf
        
        return grouped
    else:
        return exdf


def read_licor_6800_excel(
    filepath: Union[str, Path],
    sheet_name: Union[str, int] = "Measurements",
    column_name: str = 'obs',
    group_by: Optional[List[str]] = None,
    standardize_columns: bool = True,
    check_for_formulas: bool = True
) -> Union[ExtendedDataFrame, Dict[str, ExtendedDataFrame]]:
    """
    Read a LI-COR 6800 Excel file with proper handling of its complex structure.
    
    LI-COR 6800 Excel files have a specific structure:
    - Multiple sheets (typically "Measurements" and "Remarks")
    - Header information before the main data table
    - Categories row, column names row, units row, then data
    - May contain Excel formulas that need to be calculated
    
    Args:
        filepath: Path to the Excel file
        sheet_name: Sheet name to read (default: "Measurements")
        column_name: Column name to search for to identify data start (default: 'obs')
        group_by: Optional list of columns to group data by
        standardize_columns: Whether to map column names to standard names
        check_for_formulas: Whether to check for uncalculated formulas
        
    Returns:
        If group_by is None: Single ExtendedDataFrame
        If group_by is specified: Dictionary of group names to ExtendedDataFrames
        
    Raises:
        ValueError: If sheet not found or data structure is invalid
        
    Note:
        If the file contains uncalculated formulas (all values are 0), you need to:
        1. Open the file in Excel
        2. Press F9 or go to Formulas > Calculate Now
        3. Save and close the file
    """
    filepath = Path(filepath)
    
    try:
        import openpyxl
    except ImportError:
        raise ImportError("openpyxl is required to read Excel files. Install with: pip install openpyxl")
    
    # Check if file exists
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    # Try to read with openpyxl first to check structure
    try:
        wb = openpyxl.load_workbook(filepath, read_only=True, data_only=True)
        sheet_names = wb.sheetnames
        wb.close()
    except Exception as e:
        raise ValueError(f"Error reading Excel file: {e}")
    
    # Verify sheet exists
    if sheet_name not in sheet_names:
        if "Measurements" in sheet_names:
            sheet_name = "Measurements"
        else:
            raise ValueError(f"Sheet '{sheet_name}' not found. Available sheets: {sheet_names}")
    
    # Read the entire sheet without headers to find structure
    df_raw = pd.read_excel(filepath, sheet_name=sheet_name, header=None)
    
    # Find the row with column names by searching for the column_name
    data_row = None
    for idx, row in df_raw.iterrows():
        if column_name in row.values or any(column_name == str(v) for v in row.values if pd.notna(v)):
            data_row = idx
            break
    
    if data_row is None:
        raise ValueError(f"Could not find column '{column_name}' in the Excel file. "
                        f"This may not be a valid LI-COR 6800 file.")
    
    # Extract structure based on PhotoGEA approach
    # Row structure: ..., categories_row, names_row, units_row, data...
    categories_row = data_row - 1 if data_row > 0 else None
    names_row = data_row
    units_row = data_row + 1
    first_data_row = data_row + 2
    
    # Get column names
    column_names = df_raw.iloc[names_row].values
    # Clean column names - remove NaN and convert to string
    column_names = [str(col).strip() if pd.notna(col) else f'Column_{i}' 
                    for i, col in enumerate(column_names)]
    
    # Get units (if available)
    units_dict = {}
    if units_row < len(df_raw):
        units_values = df_raw.iloc[units_row].values
        for i, (col, unit) in enumerate(zip(column_names, units_values)):
            if pd.notna(unit):
                units_dict[col] = str(unit).strip()
            else:
                units_dict[col] = LICOR_UNITS.get(col, "unknown")
    
    # Get categories (if available)
    categories_dict = {}
    if categories_row is not None and categories_row >= 0:
        categories_values = df_raw.iloc[categories_row].values
        for i, (col, cat) in enumerate(zip(column_names, categories_values)):
            if pd.notna(cat):
                categories_dict[col] = str(cat).strip()
            else:
                categories_dict[col] = "LI-COR 6800"
    else:
        categories_dict = {col: "LI-COR 6800" for col in column_names}
    
    # Extract main data
    if first_data_row < len(df_raw):
        data_df = df_raw.iloc[first_data_row:].copy()
        data_df.columns = column_names
        
        # Remove any rows that are all NaN
        data_df = data_df.dropna(how='all')
        
        # Reset index
        data_df = data_df.reset_index(drop=True)
    else:
        raise ValueError("No data found after header rows")
    
    # Clean up column names - remove duplicates by making unique
    if len(column_names) != len(set(column_names)):
        # Make column names unique
        seen = {}
        unique_names = []
        for name in column_names:
            if name in seen:
                seen[name] += 1
                unique_names.append(f"{name}_{seen[name]}")
            else:
                seen[name] = 0
                unique_names.append(name)
        data_df.columns = unique_names
        
        # Update units and categories dicts
        new_units = {}
        new_categories = {}
        for old_name, new_name in zip(column_names, unique_names):
            new_units[new_name] = units_dict.get(old_name, "unknown")
            new_categories[new_name] = categories_dict.get(old_name, "LI-COR 6800")
        units_dict = new_units
        categories_dict = new_categories
    
    # Convert numeric columns
    string_columns = ['Genotype', 'Type_Measurement', 'date', 'hhmmss', 'user_remark']
    for col in data_df.columns:
        if col not in string_columns:
            try:
                # Try to convert to numeric
                temp = pd.to_numeric(data_df[col], errors='coerce')
                # Only replace if we didn't lose too much data
                if temp.notna().sum() > 0:  # At least some valid values
                    data_df[col] = temp
            except:
                pass
    
    # Check for potential formula issues and fix them
    if check_for_formulas:
        # Check key columns that should not be all zeros
        check_columns = ['A', 'gsw', 'Ci', 'E']
        problem_columns = []
        for col in check_columns:
            if col in data_df.columns:
                if (data_df[col] == 0).all() or data_df[col].isna().all():
                    problem_columns.append(col)
        
        if problem_columns:
            # Try to calculate the values from raw data
            print(f"Detected uncalculated formulas in columns: {', '.join(problem_columns)}")
            print("Attempting to calculate gas exchange parameters from raw sensor data...")
            
            # Check if we have the necessary raw data columns
            required_raw_cols = ['CO2_r', 'CO2_s', 'H2O_s', 'H2O_r', 'Flow']
            missing_raw = [col for col in required_raw_cols if col not in data_df.columns]
            
            if not missing_raw:
                # We have the raw data, calculate the values
                from aci_py.io.licor_calculations import calculate_licor_gas_exchange, detect_leaf_area_from_licor
                
                # Detect leaf area
                leaf_area = detect_leaf_area_from_licor(data_df)
                
                # Calculate gas exchange parameters
                data_df = calculate_licor_gas_exchange(data_df, leaf_area=leaf_area)
                
                # Update units dictionary with calculated values
                if hasattr(data_df, 'attrs') and 'units' in data_df.attrs:
                    units_dict.update(data_df.attrs['units'])
                
                print(f"Successfully calculated gas exchange parameters using leaf area = {leaf_area} cm²")
            else:
                # Can't calculate, show original error
                raise ValueError(
                    f"The following columns contain uncalculated formulas (all zeros): {', '.join(problem_columns)}.\n"
                    f"Missing raw data columns needed for calculation: {', '.join(missing_raw)}.\n\n"
                    f"This Excel file was exported from LI-COR 6800 but Excel hasn't calculated the formulas.\n"
                    f"You have two options:\n\n"
                    f"Option 1 - Fix in Excel:\n"
                    f"  1. Open the file in Microsoft Excel\n"
                    f"  2. Press F9 or go to Formulas > Calculate Now\n"
                    f"  3. Save and close the file\n"
                    f"  4. Try importing again\n\n"
                    f"Option 2 - Use CSV export:\n"
                    f"  1. On the LI-COR 6800, export data as CSV instead of Excel\n"
                    f"  2. CSV files contain calculated values and import more reliably"
                )
    
    # Create ExtendedDataFrame
    exdf = ExtendedDataFrame(data=data_df, units=units_dict, categories=categories_dict)
    
    # Add calculated gas properties if possible
    if "Pa" in data_df.columns and "Tleaf" in data_df.columns:
        try:
            exdf.calculate_gas_properties(pressure_col="Pa", temperature_col="Tleaf")
        except:
            pass  # Don't fail if calculation doesn't work
    
    # Group if requested
    if group_by:
        grouped = {}
        for name, group_df in data_df.groupby(group_by):
            if not isinstance(name, tuple):
                name = (name,)
            
            # Create key string
            key_parts = []
            for n in name:
                if hasattr(n, 'item'):  # numpy scalar
                    key_parts.append(str(n.item()))
                else:
                    key_parts.append(str(n))
            key = "_".join(key_parts)
            
            # Create ExtendedDataFrame for this group
            group_exdf = ExtendedDataFrame(
                data=group_df.reset_index(drop=True),
                units=units_dict,
                categories=categories_dict
            )
            
            # Add calculated properties
            if "Pa" in group_df.columns and "Tleaf" in group_df.columns:
                try:
                    group_exdf.calculate_gas_properties(pressure_col="Pa", temperature_col="Tleaf")
                except:
                    pass
            
            grouped[key] = group_exdf
        
        return grouped
    else:
        return exdf


def detect_licor_format(filepath: Union[str, Path]) -> str:
    """
    Detect the format of a LI-COR file.
    
    Args:
        filepath: Path to the file
        
    Returns:
        File format string: "csv", "excel", or "unknown"
    """
    filepath = Path(filepath)
    
    # Check extension
    ext = filepath.suffix.lower()
    
    if ext in [".csv", ".txt", ".dat"]:
        return "csv"
    elif ext in [".xlsx", ".xls"]:
        return "excel"
    else:
        # Try to read first few bytes
        try:
            with open(filepath, 'rb') as f:
                header = f.read(8)
                
            # Check for Excel magic numbers
            if header[:4] == b'\x50\x4b\x03\x04':  # XLSX
                return "excel"
            elif header[:8] == b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1':  # XLS
                return "excel"
            else:
                # Assume CSV/text
                return "csv"
        except:
            return "unknown"


def _get_file_hash(filepath: str) -> str:
    """Get a hash of the file for caching purposes."""
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def _get_cache_path(original_path: str) -> str:
    """Get the cache file path for a given Excel file."""
    file_hash = _get_file_hash(original_path)
    cache_dir = Path(tempfile.gettempdir()) / "aci_py_cache"
    cache_dir.mkdir(exist_ok=True)
    return str(cache_dir / f"{file_hash}.csv")

def _should_use_cache(original_path: str, cache_path: str) -> bool:
    """Check if cache is valid and should be used."""
    if not os.path.exists(cache_path):
        return False
    # Cache is valid if it's newer than the original file
    return os.path.getmtime(cache_path) > os.path.getmtime(original_path)

def _save_to_cache(df: pd.DataFrame, units: dict, cache_path: str):
    """Save processed data to CSV cache with units in comments."""
    with open(cache_path, 'w') as f:
        # Write units as comment
        f.write("# Units: " + str(units) + "\n")
        # Write data
        df.to_csv(f, index=False)

def read_licor_file(
    filepath: Union[str, Path],
    group_by: Optional[List[str]] = None,
    use_cache: bool = True,
    **kwargs
) -> Union[ExtendedDataFrame, Dict[str, ExtendedDataFrame]]:
    """
    Read a LI-COR file, automatically detecting the format.
    
    For Excel files, this function implements smart caching to handle formula calculations:
    - Excel files exported directly from LI-COR may contain uncalculated formulas
    - The function will attempt to calculate these values from raw sensor data
    - Successfully calculated data is cached as CSV for faster subsequent reads
    
    Args:
        filepath: Path to the file
        group_by: Optional list of columns to group data by
        use_cache: Whether to use cached CSV files for Excel imports (default: True)
        **kwargs: Additional arguments passed to specific readers
        
    Returns:
        ExtendedDataFrame or dictionary of ExtendedDataFrames
        
    Raises:
        ValueError: If file format cannot be determined or read
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    # Detect format
    format_type = detect_licor_format(filepath)
    
    if format_type == "csv":
        return read_licor_6800_csv(filepath, group_by=group_by, **kwargs)
    elif format_type == "excel":
        # Check for cached CSV version if enabled
        if use_cache:
            cache_path = _get_cache_path(str(filepath))
            if _should_use_cache(str(filepath), cache_path):
                try:
                    result = read_licor_6800_csv(Path(cache_path), group_by=group_by, **kwargs)
                    print(f"Using cached data from: {cache_path}")
                    return result
                except Exception:
                    # If cache read fails, fall back to Excel reading
                    pass
        
        # Read from Excel
        result = read_licor_6800_excel(filepath, group_by=group_by, **kwargs)
        
        # If single result and has calculated data, save to cache
        if use_cache and isinstance(result, ExtendedDataFrame):
            if 'A' in result.data.columns and not result.data['A'].isna().all():
                try:
                    cache_path = _get_cache_path(str(filepath))
                    _save_to_cache(result.data, result.units, cache_path)
                    print(f"Cached processed data to: {cache_path}")
                except Exception as e:
                    warnings.warn(f"Could not save cache: {e}")
        
        return result
    else:
        raise ValueError(f"Unable to determine file format for: {filepath}")


def validate_aci_data(exdf: ExtendedDataFrame) -> Tuple[bool, List[str]]:
    """
    Validate that an ExtendedDataFrame contains required columns for ACI analysis.
    
    Args:
        exdf: ExtendedDataFrame to validate
        
    Returns:
        Tuple of (is_valid, list_of_missing_columns)
    """
    required_columns = ["A", "Ci", "Ca", "Tleaf", "Pa"]
    missing = []
    
    for col in required_columns:
        if col not in exdf.data.columns:
            missing.append(col)
    
    return len(missing) == 0, missing