
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Callable, Any, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import warnings
from pathlib import Path

from ..core.data_structures import ExtendedDataFrame
from ..io.licor import read_licor_file
from .c3_fitting import fit_c3_aci
from .c4_fitting import fit_c4_aci
from .optimization import FittingResult


class BatchResult:
    """Container for batch processing results."""
    
    def __init__(self):
        self.results: Dict[str, FittingResult] = {}
        self.summary_df: Optional[pd.DataFrame] = None
        self.failed_curves: List[str] = []
        self.warnings: Dict[str, List[str]] = {}
    
    def add_result(self, curve_id: str, result: FittingResult):
        """
        Add a fitting result for a curve.
        
        Args:
            curve_id: Unique identifier for the curve
            result: FittingResult object from fit_c3_aci or fit_c4_aci
        """
        self.results[curve_id] = result
    
    def add_failure(self, curve_id: str, error_msg: str):
        """
        Record a failed curve with error message.
        
        Args:
            curve_id: Unique identifier for the curve
            error_msg: Description of the failure
            
        Note:
            Failed curves are tracked separately and included in summary
            with success=False flag.
        """
        self.failed_curves.append(curve_id)
        if curve_id not in self.warnings:
            self.warnings[curve_id] = []
        self.warnings[curve_id].append(f"Fitting failed: {error_msg}")
    
    def add_warning(self, curve_id: str, warning_msg: str):
        """
        Add a warning for a curve without marking it as failed.
        
        Args:
            curve_id: Unique identifier for the curve
            warning_msg: Warning message (e.g., "Low R-squared", "Convergence issues")
            
        Note:
            Warnings don't prevent a curve from being marked as successful,
            but are tracked for quality control purposes.
        """
        if curve_id not in self.warnings:
            self.warnings[curve_id] = []
        self.warnings[curve_id].append(warning_msg)
    
    def generate_summary(self):
        """Generate summary DataFrame from all results."""
        summary_data = []
        
        for curve_id, result in self.results.items():
            row = {'curve_id': curve_id}
            
            # Add parameter values
            for param, value in result.parameters.items():
                row[param] = value
            
            # Add confidence intervals if available
            if result.confidence_intervals:
                for param, (lower, upper) in result.confidence_intervals.items():
                    row[f'{param}_CI_lower'] = lower
                    row[f'{param}_CI_upper'] = upper
            
            # Add fit statistics
            row['rmse'] = result.rmse
            row['r_squared'] = result.r_squared
            row['n_points'] = result.n_points
            # Check if optimization was successful
            # Result objects from fit_c3_aci and fit_c4_aci have 'success' attribute
            row['success'] = getattr(result, 'success', True)
            
            summary_data.append(row)
        
        # Add failed curves
        for curve_id in self.failed_curves:
            row = {'curve_id': curve_id, 'success': False}
            summary_data.append(row)
        
        self.summary_df = pd.DataFrame(summary_data)
        return self.summary_df


def process_single_curve(
    curve_data: Union[ExtendedDataFrame, pd.DataFrame],
    curve_id: str,
    fit_function: Callable,
    fit_kwargs: Dict[str, Any]
) -> Tuple[str, Union[FittingResult, Exception]]:
    """
    Process a single ACI curve.
    
    This function is designed to be used with parallel processing.
    
    Args:
        curve_data: Data for a single curve
        curve_id: Identifier for the curve
        fit_function: Function to use for fitting (fit_c3_aci or fit_c4_aci)
        fit_kwargs: Keyword arguments for the fitting function
    
    Returns:
        Tuple of (curve_id, result or exception)
        
    Raises:
        No exceptions are raised directly; all exceptions are caught and returned
        in the result tuple to support parallel processing error handling
    """
    try:
        # Convert to ExtendedDataFrame if needed
        if isinstance(curve_data, pd.DataFrame):
            exdf = ExtendedDataFrame(curve_data)
        else:
            exdf = curve_data
        
        # Fit the curve
        result = fit_function(exdf, **fit_kwargs)
        return curve_id, result
        
    except Exception as e:
        return curve_id, e


def batch_fit_aci(
    data: Union[str, pd.DataFrame, ExtendedDataFrame, Dict[str, Union[pd.DataFrame, ExtendedDataFrame]]],
    model_type: str = 'C3',
    groupby: Optional[List[str]] = None,
    n_jobs: int = 1,
    progress_bar: bool = True,
    **fit_kwargs
) -> BatchResult:
    """
    Fit multiple ACI curves in batch.
    
    This function can process:
    1. A file path containing multiple curves
    2. A DataFrame with multiple curves (use groupby to separate)
    3. A dictionary of DataFrames/ExtendedDataFrames
    
    Args:
        data: Input data in various formats
        model_type: 'C3' or 'C4' photosynthesis model
        groupby: Column names to group by (for DataFrame input)
        n_jobs: Number of parallel jobs (-1 for all CPUs)
        progress_bar: Show progress bar
        **fit_kwargs: Additional arguments passed to fit function
    
    Returns:
        BatchResult object containing all fitting results
    
    Examples:
        # From file with grouping
        results = batch_fit_aci('data.csv', groupby=['Genotype', 'Plant'])
        
        # From dictionary of curves
        curves = {'curve1': df1, 'curve2': df2}
        results = batch_fit_aci(curves, n_jobs=4)
        
        # With specific fitting options
        results = batch_fit_aci(data, fixed_parameters={'gmc': 0.5})
    """
    # Select fitting function
    if model_type.upper() == 'C3':
        fit_function = fit_c3_aci
    elif model_type.upper() == 'C4':
        fit_function = fit_c4_aci
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Prepare data dictionary
    curves_dict = {}
    
    if isinstance(data, str):
        # Load from file
        if groupby:
            curves_dict = read_licor_file(data, groupby=groupby)
        else:
            # Single curve in file
            curves_dict = {'curve_1': read_licor_file(data)}
            
    elif isinstance(data, (pd.DataFrame, ExtendedDataFrame)):
        # DataFrame input
        if groupby:
            # Group the data
            df = data.data if isinstance(data, ExtendedDataFrame) else data
            grouped = df.groupby(groupby)
            
            for name, group in grouped:
                # Create curve ID from group names
                if isinstance(name, tuple):
                    curve_id = "_".join(str(n) for n in name)
                else:
                    curve_id = str(name)
                curves_dict[curve_id] = group.reset_index(drop=True)
        else:
            # Single curve
            curves_dict = {'curve_1': data}
            
    elif isinstance(data, dict):
        # Already a dictionary
        curves_dict = data
        
    else:
        raise ValueError("Data must be a file path, DataFrame, or dictionary")
    
    # Initialize results
    batch_result = BatchResult()
    
    # Determine number of jobs
    if n_jobs == -1:
        import multiprocessing
        n_jobs = multiprocessing.cpu_count()
    
    # Process curves
    if n_jobs == 1:
        # Sequential processing
        iterator = curves_dict.items()
        if progress_bar:
            iterator = tqdm(iterator, desc="Fitting curves", total=len(curves_dict))
        
        for curve_id, curve_data in iterator:
            _, result = process_single_curve(curve_data, curve_id, fit_function, fit_kwargs)
            
            if isinstance(result, Exception):
                batch_result.add_failure(curve_id, str(result))
            else:
                batch_result.add_result(curve_id, result)
                if hasattr(result, 'success') and not result.success:
                    batch_result.add_warning(curve_id, "Optimization did not converge")
    else:
        # Parallel processing
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            # Submit all tasks
            futures = {
                executor.submit(process_single_curve, curve_data, curve_id, fit_function, fit_kwargs): curve_id
                for curve_id, curve_data in curves_dict.items()
            }
            
            # Process completed tasks
            iterator = as_completed(futures)
            if progress_bar:
                iterator = tqdm(iterator, desc="Fitting curves", total=len(futures))
            
            for future in iterator:
                curve_id = futures[future]
                try:
                    _, result = future.result()
                    
                    if isinstance(result, Exception):
                        batch_result.add_failure(curve_id, str(result))
                    else:
                        batch_result.add_result(curve_id, result)
                        if hasattr(result, 'success') and not result.success:
                            batch_result.add_warning(curve_id, "Optimization did not converge")
                            
                except Exception as e:
                    batch_result.add_failure(curve_id, str(e))
    
    # Generate summary
    batch_result.generate_summary()
    
    # Print summary statistics
    n_total = len(curves_dict)
    n_success = len(batch_result.results)
    n_failed = len(batch_result.failed_curves)
    
    print(f"\nBatch fitting complete:")
    print(f"  Total curves: {n_total}")
    print(f"  Successful: {n_success}")
    print(f"  Failed: {n_failed}")
    
    if batch_result.warnings:
        print(f"  Curves with warnings: {len(batch_result.warnings)}")
    
    return batch_result


def compare_models(
    data: Union[pd.DataFrame, ExtendedDataFrame],
    models: List[str] = ['C3', 'C4'],
    **fit_kwargs
) -> Dict[str, FittingResult]:
    """
    Compare different photosynthesis models on the same data.
    
    Args:
        data: ACI curve data
        models: List of model types to compare
        **fit_kwargs: Additional arguments for fitting
    
    Returns:
        Dictionary mapping model type to fitting results
    """
    results = {}
    
    for model in models:
        if model.upper() == 'C3':
            results[model] = fit_c3_aci(data, **fit_kwargs)
        elif model.upper() == 'C4':
            results[model] = fit_c4_aci(data, **fit_kwargs)
        else:
            warnings.warn(f"Unknown model type: {model}")
    
    return results


def analyze_parameter_variability(
    batch_result: BatchResult,
    parameters: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Analyze parameter variability across batch results.
    
    Args:
        batch_result: Results from batch_fit_aci
        parameters: List of parameters to analyze (None for all)
    
    Returns:
        DataFrame with parameter statistics including mean, std, CV%, min, max, and n
        
    Example:
        >>> results = batch_fit_aci('data.csv', groupby=['Genotype'])
        >>> stats = analyze_parameter_variability(results, ['Vcmax_at_25', 'J_at_25'])
        >>> print(stats)
        #      parameter   mean    std     cv    min    max  n_curves
        # 0  Vcmax_at_25   95.3   8.2   8.6%  82.1  108.5        12
        # 1     J_at_25  185.7  15.3   8.2% 162.3  210.1        12
    """
    if batch_result.summary_df is None:
        batch_result.generate_summary()
    
    df = batch_result.summary_df[batch_result.summary_df['success'] == True].copy()
    
    if parameters is None:
        # Get all parameter columns (exclude metadata and CI columns)
        exclude_cols = ['curve_id', 'rmse', 'r_squared', 'n_points', 'success']
        parameters = [col for col in df.columns 
                     if col not in exclude_cols and not col.endswith('_CI_lower') 
                     and not col.endswith('_CI_upper')]
    
    stats_data = []
    for param in parameters:
        if param in df.columns:
            param_data = df[param].dropna()
            
            stats = {
                'parameter': param,
                'mean': param_data.mean(),
                'std': param_data.std(),
                'cv': param_data.std() / param_data.mean() * 100 if param_data.mean() != 0 else np.nan,
                'min': param_data.min(),
                'max': param_data.max(),
                'n_curves': len(param_data)
            }
            stats_data.append(stats)
    
    return pd.DataFrame(stats_data)