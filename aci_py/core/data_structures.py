"""
Data structures for ACI_py, including the ExtendedDataFrame class.

This module provides enhanced data structures that track units and metadata
similar to PhotoGEA's exdf objects.
"""

from typing import Dict, List, Optional, Union, Any
import pandas as pd
import numpy as np
from copy import deepcopy


class ExtendedDataFrame:
    """
    Enhanced DataFrame with units and metadata tracking.
    
    Similar to PhotoGEA's exdf structure, this class wraps a pandas DataFrame
    with additional metadata about units and data categories/sources.
    
    Attributes:
        data: The main pandas DataFrame containing the data
        units: Dictionary mapping column names to their units
        categories: Dictionary mapping column names to their categories/sources
    """
    
    def __init__(
        self, 
        data: Union[pd.DataFrame, Dict, List], 
        units: Optional[Dict[str, str]] = None,
        categories: Optional[Dict[str, str]] = None
    ):
        """
        Initialize an ExtendedDataFrame.
        
        Args:
            data: Data to store (DataFrame, dict, or list)
            units: Dictionary of column names to unit strings
            categories: Dictionary of column names to category strings
        """
        if isinstance(data, pd.DataFrame):
            self.data = data.copy()
        else:
            self.data = pd.DataFrame(data)
            
        self.units = units or {}
        self.categories = categories or {}
        
        # Ensure all columns have entries in units and categories
        for col in self.data.columns:
            if col not in self.units:
                self.units[col] = "dimensionless"
            if col not in self.categories:
                self.categories[col] = "unknown"
    
    def check_required_variables(
        self, 
        required: List[str], 
        raise_error: bool = True
    ) -> bool:
        """
        Check if required columns exist in the data.
        
        Args:
            required: List of required column names
            raise_error: If True, raise ValueError if columns are missing
            
        Returns:
            True if all required columns exist, False otherwise
            
        Raises:
            ValueError: If raise_error=True and columns are missing
        """
        missing = [col for col in required if col not in self.data.columns]
        
        if missing:
            msg = f"Missing required columns: {', '.join(missing)}"
            if raise_error:
                raise ValueError(msg)
            else:
                print(f"Warning: {msg}")
                return False
        return True
    
    def get_column_units(self, column: str) -> str:
        """Get units for a specific column."""
        return self.units.get(column, "dimensionless")
    
    def get_column_category(self, column: str) -> str:
        """Get category for a specific column."""
        return self.categories.get(column, "unknown")
    
    def set_variable(
        self, 
        name: str, 
        values: Union[np.ndarray, pd.Series, List, float],
        units: str = "dimensionless",
        category: str = "calculated"
    ) -> None:
        """
        Add or update a variable in the ExtendedDataFrame.
        
        Args:
            name: Column name
            values: Values to set
            units: Units for the variable
            category: Category/source for the variable
        """
        self.data[name] = values
        self.units[name] = units
        self.categories[name] = category
    
    def calculate_gas_properties(
        self,
        pressure_col: str = "Pa",
        temperature_col: str = "Tleaf"
    ) -> None:
        """
        Calculate additional gas exchange properties.
        
        Adds calculated columns for partial pressures and other derived values
        commonly needed for photosynthesis calculations.
        
        Args:
            pressure_col: Column name for atmospheric pressure (kPa)
            temperature_col: Column name for leaf temperature (°C)
        """
        # Check required columns
        self.check_required_variables([pressure_col, temperature_col])
        
        # Convert temperature to Kelvin
        T_K = self.data[temperature_col] + 273.15
        self.set_variable("T_leaf_K", T_K, "K", "calculated")
        
        # Calculate partial pressures if CO2/O2 data exists
        if "Ca" in self.data.columns:
            # CO2 partial pressure in Pa
            PCa = self.data["Ca"] * self.data[pressure_col] * 0.1  # µbar
            self.set_variable("PCa", PCa, "µbar", "calculated")
            
        if "Ci" in self.data.columns:
            # Intercellular CO2 partial pressure
            PCi = self.data["Ci"] * self.data[pressure_col] * 0.1  # µbar
            self.set_variable("PCi", PCi, "µbar", "calculated")
    
    def copy(self) -> 'ExtendedDataFrame':
        """Create a deep copy of the ExtendedDataFrame."""
        return ExtendedDataFrame(
            data=self.data.copy(),
            units=deepcopy(self.units),
            categories=deepcopy(self.categories)
        )
    
    def subset_rows(self, indices: Union[pd.Index, np.ndarray, List]) -> 'ExtendedDataFrame':
        """
        Create a subset of the ExtendedDataFrame with specified rows.
        
        Args:
            indices: Row indices or boolean mask
            
        Returns:
            New ExtendedDataFrame with subset of rows
        """
        return ExtendedDataFrame(
            data=self.data.loc[indices].copy(),
            units=deepcopy(self.units),
            categories=deepcopy(self.categories)
        )
    
    def subset_columns(self, columns: List[str]) -> 'ExtendedDataFrame':
        """
        Create a subset of the ExtendedDataFrame with specified columns.
        
        Args:
            columns: List of column names to keep
            
        Returns:
            New ExtendedDataFrame with subset of columns
        """
        subset_units = {col: self.units[col] for col in columns if col in self.units}
        subset_categories = {col: self.categories[col] for col in columns if col in self.categories}
        
        return ExtendedDataFrame(
            data=self.data[columns].copy(),
            units=subset_units,
            categories=subset_categories
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for serialization."""
        return {
            "data": self.data.to_dict(),
            "units": self.units,
            "categories": self.categories
        }
    
    @classmethod
    def from_dict(cls, data_dict: Dict[str, Any]) -> 'ExtendedDataFrame':
        """Create ExtendedDataFrame from dictionary."""
        return cls(
            data=pd.DataFrame(data_dict["data"]),
            units=data_dict.get("units", {}),
            categories=data_dict.get("categories", {})
        )
    
    def __repr__(self) -> str:
        """String representation of ExtendedDataFrame."""
        n_rows, n_cols = self.data.shape
        cols_with_units = [
            f"{col} [{self.units.get(col, '?')}]" 
            for col in self.data.columns[:5]
        ]
        if n_cols > 5:
            cols_with_units.append("...")
            
        return (
            f"ExtendedDataFrame with {n_rows} rows and {n_cols} columns:\n"
            f"Columns: {', '.join(cols_with_units)}\n"
            f"Categories: {len(set(self.categories.values()))} unique"
        )
    
    def __len__(self) -> int:
        """Return number of rows."""
        return len(self.data)
    
    def __getitem__(self, key: str) -> pd.Series:
        """Allow direct column access like a DataFrame."""
        return self.data[key]
    
    def __setitem__(self, key: str, value: Any) -> None:
        """Allow direct column setting with default units/category."""
        self.set_variable(key, value)


def identify_common_columns(
    exdf_list: List[ExtendedDataFrame], 
    require_all: bool = True
) -> List[str]:
    """
    Identify columns that are common across multiple ExtendedDataFrames.
    
    Args:
        exdf_list: List of ExtendedDataFrames to compare
        require_all: If True, return only columns present in ALL DataFrames
                    If False, return columns present in ANY DataFrame
    
    Returns:
        List of common column names
    """
    if not exdf_list:
        return []
    
    if require_all:
        # Find intersection of all column sets
        common = set(exdf_list[0].data.columns)
        for exdf in exdf_list[1:]:
            common = common.intersection(set(exdf.data.columns))
        return sorted(list(common))
    else:
        # Find union of all column sets
        all_cols = set()
        for exdf in exdf_list:
            all_cols = all_cols.union(set(exdf.data.columns))
        return sorted(list(all_cols))