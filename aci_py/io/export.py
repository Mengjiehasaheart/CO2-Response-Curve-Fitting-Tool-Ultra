"""
Export functionality for saving analysis results.

This module provides functions to export fitting results, plots, and reports
in various formats for publication and further analysis.
"""

import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from ..core.data_structures import ExtendedDataFrame
from ..analysis.optimization import FittingResult
from ..analysis.batch import BatchResult


def export_fitting_result(
    result: FittingResult,
    output_dir: Union[str, Path],
    base_name: str = "aci_fit",
    formats: List[str] = ['csv', 'json'],
    include_plots: bool = True,
    plot_dpi: int = 300
) -> Dict[str, Path]:
    """
    Export a single fitting result to multiple formats.
    
    Args:
        result: FittingResult object to export
        output_dir: Directory to save files
        base_name: Base name for output files
        formats: List of formats ('csv', 'json', 'excel')
        include_plots: Whether to export plots
        plot_dpi: DPI for plot exports
    
    Returns:
        Dictionary mapping format to output file path
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_files = {}
    
    # Export parameters and statistics
    if 'json' in formats:
        json_path = output_dir / f"{base_name}_results.json"
        export_data = {
            'parameters': result.parameters,
            'confidence_intervals': result.confidence_intervals,
            'statistics': {
                'rmse': result.rmse,
                'r_squared': result.r_squared,
                'n_points': result.n_points,
                'success': result.success
            },
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'model_type': result.model_type if hasattr(result, 'model_type') else 'Unknown'
            }
        }
        
        with open(json_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        output_files['json'] = json_path
    
    # Export fitted values as CSV
    if 'csv' in formats and result.fitted_values is not None:
        csv_path = output_dir / f"{base_name}_fitted_values.csv"
        df = pd.DataFrame(result.fitted_values)
        df.to_csv(csv_path, index=False)
        output_files['csv'] = csv_path
    
    # Export to Excel with multiple sheets
    if 'excel' in formats:
        excel_path = output_dir / f"{base_name}_results.xlsx"
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # Parameters sheet
            params_df = pd.DataFrame([result.parameters])
            params_df.to_excel(writer, sheet_name='Parameters', index=False)
            
            # Confidence intervals sheet
            if result.confidence_intervals:
                ci_data = []
                for param, (lower, upper) in result.confidence_intervals.items():
                    ci_data.append({
                        'Parameter': param,
                        'Lower_95CI': lower,
                        'Upper_95CI': upper,
                        'Estimate': result.parameters.get(param, np.nan)
                    })
                ci_df = pd.DataFrame(ci_data)
                ci_df.to_excel(writer, sheet_name='Confidence_Intervals', index=False)
            
            # Fitted values sheet
            if result.fitted_values is not None:
                fitted_df = pd.DataFrame(result.fitted_values)
                fitted_df.to_excel(writer, sheet_name='Fitted_Values', index=False)
            
            # Statistics sheet
            stats_df = pd.DataFrame([{
                'RMSE': result.rmse,
                'R_squared': result.r_squared,
                'N_points': result.n_points,
                'Success': result.success
            }])
            stats_df.to_excel(writer, sheet_name='Statistics', index=False)
        
        output_files['excel'] = excel_path
    
    # Export plots
    if include_plots and hasattr(result, 'figure'):
        if result.figure is not None:
            # PNG format
            png_path = output_dir / f"{base_name}_plot.png"
            result.figure.savefig(png_path, dpi=plot_dpi, bbox_inches='tight')
            output_files['plot_png'] = png_path
            
            # PDF format
            pdf_path = output_dir / f"{base_name}_plot.pdf"
            result.figure.savefig(pdf_path, bbox_inches='tight')
            output_files['plot_pdf'] = pdf_path
    
    return output_files


def export_batch_results(
    batch_result: BatchResult,
    output_dir: Union[str, Path],
    base_name: str = "batch_results",
    include_individual: bool = False,
    formats: List[str] = ['csv', 'excel']
) -> Dict[str, Path]:
    """
    Export batch fitting results.
    
    Args:
        batch_result: BatchResult object to export
        output_dir: Directory to save files
        base_name: Base name for output files
        include_individual: Whether to export individual curve results
        formats: List of formats for summary ('csv', 'excel')
    
    Returns:
        Dictionary mapping format to output file path
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_files = {}
    
    # Ensure summary is generated
    if batch_result.summary_df is None:
        batch_result.generate_summary()
    
    # Export summary
    if 'csv' in formats:
        csv_path = output_dir / f"{base_name}_summary.csv"
        batch_result.summary_df.to_csv(csv_path, index=False)
        output_files['summary_csv'] = csv_path
    
    if 'excel' in formats:
        excel_path = output_dir / f"{base_name}_summary.xlsx"
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # Summary sheet
            batch_result.summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Failed curves sheet
            if batch_result.failed_curves:
                failed_df = pd.DataFrame({
                    'curve_id': batch_result.failed_curves,
                    'status': 'Failed'
                })
                if batch_result.warnings:
                    warnings_list = []
                    for curve_id in batch_result.failed_curves:
                        if curve_id in batch_result.warnings:
                            warnings_list.append('; '.join(batch_result.warnings[curve_id]))
                        else:
                            warnings_list.append('')
                    failed_df['warnings'] = warnings_list
                
                failed_df.to_excel(writer, sheet_name='Failed_Curves', index=False)
            
            # Parameter statistics sheet
            param_cols = [col for col in batch_result.summary_df.columns 
                         if col not in ['curve_id', 'rmse', 'r_squared', 'n_points', 'success']
                         and not col.endswith('_CI_lower') and not col.endswith('_CI_upper')]
            
            if param_cols:
                stats_data = []
                successful_df = batch_result.summary_df[batch_result.summary_df['success'] == True]
                
                for param in param_cols:
                    if param in successful_df.columns:
                        param_data = successful_df[param].dropna()
                        if len(param_data) > 0:
                            stats_data.append({
                                'Parameter': param,
                                'Mean': param_data.mean(),
                                'Std': param_data.std(),
                                'CV%': param_data.std() / param_data.mean() * 100 if param_data.mean() != 0 else np.nan,
                                'Min': param_data.min(),
                                'Max': param_data.max(),
                                'N': len(param_data)
                            })
                
                if stats_data:
                    stats_df = pd.DataFrame(stats_data)
                    stats_df.to_excel(writer, sheet_name='Parameter_Statistics', index=False)
        
        output_files['summary_excel'] = excel_path
    
    # Export individual results
    if include_individual:
        individual_dir = output_dir / 'individual_curves'
        individual_dir.mkdir(exist_ok=True)
        
        for curve_id, result in batch_result.results.items():
            curve_files = export_fitting_result(
                result,
                individual_dir,
                base_name=f"{curve_id}_fit",
                formats=['csv', 'json'],
                include_plots=True
            )
            output_files[f'individual_{curve_id}'] = curve_files
    
    return output_files


def create_analysis_report(
    results: Union[FittingResult, BatchResult, Dict[str, FittingResult]],
    output_path: Union[str, Path],
    title: str = "ACI Analysis Report",
    include_methods: bool = True,
    include_plots: bool = True,
    original_data: Optional[ExtendedDataFrame] = None
) -> Path:
    """
    Create a comprehensive PDF report of analysis results.
    
    Args:
        results: Analysis results (single, batch, or model comparison)
        output_path: Path for output PDF file
        title: Report title
        include_methods: Include methods description
        include_plots: Include plots in report
        original_data: Original measurement data (for plotting)
    
    Returns:
        Path to created PDF file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with PdfPages(output_path) as pdf:
        # Title page
        fig = plt.figure(figsize=(8.5, 11))
        fig.text(0.5, 0.7, title, ha='center', size=24, weight='bold')
        fig.text(0.5, 0.6, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", 
                ha='center', size=12)
        
        # Add summary information
        summary_text = []
        if isinstance(results, FittingResult):
            summary_text.append("Single Curve Analysis")
            summary_text.append(f"Model: {results.model_type if hasattr(results, 'model_type') else 'Unknown'}")
            summary_text.append(f"R² = {results.r_squared:.3f}")
            summary_text.append(f"RMSE = {results.rmse:.3f}")
        elif isinstance(results, BatchResult):
            n_success = len(results.results)
            n_total = n_success + len(results.failed_curves)
            summary_text.append("Batch Analysis")
            summary_text.append(f"Total curves: {n_total}")
            summary_text.append(f"Successful: {n_success}")
            summary_text.append(f"Failed: {len(results.failed_curves)}")
        elif isinstance(results, dict):
            summary_text.append("Model Comparison")
            summary_text.append(f"Models compared: {', '.join(results.keys())}")
        
        fig.text(0.5, 0.4, '\n'.join(summary_text), ha='center', size=14, 
                va='center', multialignment='center')
        
        plt.axis('off')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Methods page (if requested)
        if include_methods:
            fig = plt.figure(figsize=(8.5, 11))
            fig.text(0.1, 0.9, "Methods", size=18, weight='bold')
            
            methods_text = """
This analysis uses the Farquhar-von Caemmerer-Berry (FvCB) model for C3 photosynthesis
or the von Caemmerer model for C4 photosynthesis to fit A-Ci response curves.

Key parameters estimated:
• Vcmax: Maximum carboxylation rate
• J: Maximum electron transport rate  
• Tp: Triose phosphate utilization rate (C3)
• Vpmax: Maximum PEP carboxylation rate (C4)
• RL: Day respiration rate

The fitting uses differential evolution for global optimization followed by
local refinement. Confidence intervals are calculated using profile likelihood
or bootstrap methods.
            """
            
            fig.text(0.1, 0.1, methods_text, size=11, va='bottom', 
                    wrap=True, multialignment='left')
            plt.axis('off')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
        
        # Results pages
        if isinstance(results, FittingResult):
            # Single result - parameter table
            _add_parameter_table_page(pdf, results)
            
            # Add plot if requested
            if include_plots and original_data is not None:
                # Generate A-Ci plot for the report
                from ..analysis.plotting import plot_c3_fit, setup_plot_style
                setup_plot_style()
                fig = plot_c3_fit(
                    original_data,
                    results,
                    title=None,  # No title per request
                    show_parameters=True,
                    show_residuals=True
                )
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
                
        elif isinstance(results, BatchResult):
            # Batch results - summary statistics
            _add_batch_summary_page(pdf, results)
            
            # Parameter distributions
            if results.summary_df is not None and len(results.results) > 1:
                _add_parameter_distribution_page(pdf, results)
                
        elif isinstance(results, dict):
            # Model comparison
            _add_model_comparison_page(pdf, results)
        
        # Metadata
        d = pdf.infodict()
        d['Title'] = title
        d['Author'] = 'ACI_py Analysis Package'
        d['Subject'] = 'Photosynthesis Parameter Analysis'
        d['Keywords'] = 'Photosynthesis, A-Ci curves, FvCB model'
        d['CreationDate'] = datetime.now()
    
    return output_path


def _add_parameter_table_page(pdf: PdfPages, result: FittingResult):
    """Add parameter table page to PDF."""
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare parameter data
    param_data = []
    for param, value in result.parameters.items():
        row = [param, f"{value:.3f}"]
        
        # Add confidence interval if available
        if result.confidence_intervals and param in result.confidence_intervals:
            lower, upper = result.confidence_intervals[param]
            row.append(f"[{lower:.3f}, {upper:.3f}]")
        else:
            row.append("-")
        
        param_data.append(row)
    
    # Create table
    col_labels = ['Parameter', 'Value', '95% CI']
    table = ax.table(cellText=param_data, colLabels=col_labels,
                    cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)
    
    ax.text(0.5, 0.9, "Fitted Parameters", ha='center', 
           transform=ax.transAxes, size=16, weight='bold')
    
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()


def _add_batch_summary_page(pdf: PdfPages, batch_result: BatchResult):
    """Add batch summary statistics page to PDF."""
    fig = plt.figure(figsize=(8.5, 11))
    
    # Calculate summary statistics
    if batch_result.summary_df is None:
        batch_result.generate_summary()
    
    successful_df = batch_result.summary_df[batch_result.summary_df['success'] == True]
    
    text_lines = [
        "Batch Analysis Summary",
        "",
        f"Total curves analyzed: {len(batch_result.results) + len(batch_result.failed_curves)}",
        f"Successfully fitted: {len(batch_result.results)}",
        f"Failed: {len(batch_result.failed_curves)}",
        "",
        "Parameter Summary (mean ± std):"
    ]
    
    # Get parameter columns
    param_cols = [col for col in successful_df.columns 
                 if col not in ['curve_id', 'rmse', 'r_squared', 'n_points', 'success']
                 and not col.endswith('_CI_lower') and not col.endswith('_CI_upper')]
    
    for param in param_cols:
        if param in successful_df.columns:
            values = successful_df[param].dropna()
            if len(values) > 0:
                mean_val = values.mean()
                std_val = values.std()
                text_lines.append(f"  {param}: {mean_val:.3f} ± {std_val:.3f}")
    
    fig.text(0.1, 0.9, '\n'.join(text_lines), va='top', size=12,
            transform=fig.transFigure)
    
    plt.axis('off')
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()


def _add_parameter_distribution_page(pdf: PdfPages, batch_result: BatchResult):
    """Add parameter distribution plots page to PDF."""
    successful_df = batch_result.summary_df[batch_result.summary_df['success'] == True]
    
    # Get main parameters (not CI bounds)
    param_cols = [col for col in successful_df.columns 
                 if col not in ['curve_id', 'rmse', 'r_squared', 'n_points', 'success']
                 and not col.endswith('_CI_lower') and not col.endswith('_CI_upper')]
    
    # Limit to first 6 parameters for space
    param_cols = param_cols[:6]
    
    if param_cols:
        n_params = len(param_cols)
        fig, axes = plt.subplots(2, 3, figsize=(8.5, 11))
        axes = axes.flatten()
        
        for i, param in enumerate(param_cols):
            if i < len(axes):
                ax = axes[i]
                values = successful_df[param].dropna()
                
                if len(values) > 1:
                    ax.hist(values, bins='auto', alpha=0.7, edgecolor='black')
                    ax.axvline(values.mean(), color='red', linestyle='--', 
                             label=f'Mean: {values.mean():.2f}')
                    ax.set_xlabel(param)
                    ax.set_ylabel('Count')
                    ax.legend()
                else:
                    ax.text(0.5, 0.5, f"{param}\nOnly one value", 
                           ha='center', va='center', transform=ax.transAxes)
                    ax.set_xticks([])
                    ax.set_yticks([])
        
        # Hide unused subplots
        for i in range(n_params, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle('Parameter Distributions', fontsize=16, fontweight='bold')
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()


def _add_model_comparison_page(pdf: PdfPages, results: Dict[str, FittingResult]):
    """Add model comparison page to PDF."""
    fig, ax = plt.subplots(figsize=(8.5, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare comparison data
    comparison_data = []
    for model_name, result in results.items():
        row = [
            model_name,
            f"{result.rmse:.3f}",
            f"{result.r_squared:.3f}",
            f"{result.n_points}",
            "Yes" if result.success else "No"
        ]
        comparison_data.append(row)
    
    # Create table
    col_labels = ['Model', 'RMSE', 'R²', 'N Points', 'Converged']
    table = ax.table(cellText=comparison_data, colLabels=col_labels,
                    cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)
    
    ax.text(0.5, 0.9, "Model Comparison", ha='center', 
           transform=ax.transAxes, size=16, weight='bold')
    
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()


def save_for_photogea_compatibility(
    result: FittingResult,
    output_path: Union[str, Path],
    original_data: Optional[ExtendedDataFrame] = None
) -> Path:
    """
    Save results in a format compatible with PhotoGEA R package.
    
    Args:
        result: Fitting result to save
        output_path: Output file path
        original_data: Original measurement data
    
    Returns:
        Path to saved file
    """
    output_path = Path(output_path)
    
    # Create PhotoGEA-compatible structure
    output_data = {
        'parameters': {},
        'fits': {},
        'diagnostics': {}
    }
    
    # Map parameter names to PhotoGEA conventions
    param_mapping = {
        'Vcmax_at_25': 'Vcmax',
        'J_at_25': 'J',
        'Tp_at_25': 'Tp',
        'RL_at_25': 'Rd',
        'gmc': 'gm'
    }
    
    for our_name, photogea_name in param_mapping.items():
        if our_name in result.parameters:
            output_data['parameters'][photogea_name] = result.parameters[our_name]
    
    # Add fitted values if available
    if result.fitted_values is not None:
        output_data['fits'] = result.fitted_values
    
    # Add diagnostics
    output_data['diagnostics'] = {
        'rmse': result.rmse,
        'r_squared': result.r_squared,
        'n': result.n_points,
        'convergence': result.success
    }
    
    # Save as CSV with special structure
    if output_path.suffix == '.csv':
        # Parameters on top rows
        with open(output_path, 'w') as f:
            f.write("# PhotoGEA-compatible output\n")
            f.write("# Parameters:\n")
            for param, value in output_data['parameters'].items():
                f.write(f"# {param},{value}\n")
            f.write("# Diagnostics:\n")
            for diag, value in output_data['diagnostics'].items():
                f.write(f"# {diag},{value}\n")
            f.write("\n")
            
            # Fitted values as main data
            if result.fitted_values is not None:
                df = pd.DataFrame(result.fitted_values)
                df.to_csv(f, index=False)
    
    return output_path