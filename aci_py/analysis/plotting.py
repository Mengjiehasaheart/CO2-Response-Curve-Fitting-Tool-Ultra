import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional, List, Tuple, Union
import pandas as pd

from ..core.data_structures import ExtendedDataFrame
from .c3_fitting import C3FitResult


def setup_plot_style():
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        try:
            plt.style.use('seaborn-whitegrid')
        except:
            pass
    
    sns.set_palette("husl")
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.titleweight'] = 'bold'
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['axes.linewidth'] = 1.5
    plt.rcParams['axes.edgecolor'] = 'black'


def plot_aci_curve(
    exdf: ExtendedDataFrame,
    ci_column: str = 'Ci',
    a_column: str = 'A',
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    fig_size: Tuple[float, float] = (8, 6),
    color: str = 'black',
    marker: str = 'o',
    markersize: float = 8,
    alpha: float = 0.8,
    ax: Optional[plt.Axes] = None
) -> plt.Axes:

    if ax is None:
        fig, ax = plt.subplots(figsize=fig_size)
    
    # Extract data
    ci = exdf.data[ci_column].values
    a = exdf.data[a_column].values
    
    # Plot points
    ax.scatter(ci, a, color=color, marker=marker, s=markersize**2, 
               alpha=alpha, label='Observed', zorder=3)
    
    # Set labels with italic formatting for parameters
    if xlabel is None:
        xlabel = f"$\\mathit{{{ci_column}}}$"  # Italic parameter
        if ci_column in exdf.units:
            xlabel += f" ({exdf.units[ci_column]})"
    if ylabel is None:
        ylabel = f"$\\mathit{{{a_column}}}$"  # Italic parameter
        if a_column in exdf.units:
            ylabel += f" ({exdf.units[a_column]})"
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    # Remove title setting - no titles per request
    # if title:
    #     ax.set_title(title)
    
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_c3_fit(
    exdf: ExtendedDataFrame,
    fit_result: C3FitResult,
    ci_column: str = 'Ci',
    a_column: str = 'A',
    title: Optional[str] = None,
    show_limiting_processes: bool = True,
    show_confidence_intervals: bool = True,
    show_parameters: bool = True,
    show_residuals: bool = True,
    fig_size: Tuple[float, float] = (10, 10),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create comprehensive plot of C3 A-Ci curve fit.
    
    Args:
        exdf: ExtendedDataFrame with original data
        fit_result: C3FitResult object
        ci_column: Column name for Ci values
        a_column: Column name for A values
        title: Plot title
        show_limiting_processes: Color-code by limiting process
        show_confidence_intervals: Show parameter confidence intervals
        show_parameters: Show fitted parameter values
        show_residuals: Include residual subplot
        fig_size: Figure size
        save_path: Path to save figure (optional)
    
    Returns:
        Matplotlib figure object
    """
    setup_plot_style()
    
    # Create figure with subplots
    if show_residuals:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=fig_size,
                                       gridspec_kw={'height_ratios': [3, 1]})
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(fig_size[0], fig_size[1] * 0.6))
    
    # Extract data
    ci = exdf.data[ci_column].values
    a_obs = exdf.data[a_column].values
    a_fit = fit_result.fitted_A
    
    # Sort by Ci for smooth lines
    sort_idx = np.argsort(ci)
    ci_sorted = ci[sort_idx]
    a_obs_sorted = a_obs[sort_idx]
    a_fit_sorted = a_fit[sort_idx]
    
    # Plot observed data
    ax1.scatter(ci, a_obs, color='black', s=64, alpha=0.7,
                label='Observed', zorder=3, edgecolors='black', linewidth=0.5)
    
    if show_limiting_processes and hasattr(fit_result, 'limiting_process'):
        # Plot fitted curve colored by limiting process
        limiting_sorted = fit_result.limiting_process[sort_idx]
        
        colors = {'Wc': '#1f77b4', 'Wj': '#2ca02c', 'Wp': '#d62728'}
        labels = {'Wc': 'Rubisco-limited', 'Wj': 'RuBP-limited', 'Wp': 'TPU-limited'}
        
        # Plot each limiting region
        for process in ['Wc', 'Wj', 'Wp']:
            mask = limiting_sorted == process
            if np.any(mask):
                # Find continuous segments
                diff = np.diff(np.concatenate(([False], mask, [False])).astype(int))
                starts = np.where(diff == 1)[0]
                ends = np.where(diff == -1)[0]
                
                for start, end in zip(starts, ends):
                    label = labels[process] if start == starts[0] else None
                    ax1.plot(ci_sorted[start:end], a_fit_sorted[start:end],
                            color=colors[process], linewidth=3, label=label)
    else:
        # Simple fitted line
        ax1.plot(ci_sorted, a_fit_sorted, 'r-', linewidth=2, label='Fitted')
    
    # Labels and formatting with italic parameters
    xlabel = f"$\\mathit{{{ci_column}}}$"  # Italic Ci
    if ci_column in exdf.units:
        xlabel += f" ({exdf.units[ci_column]})"
    ylabel = f"$\\mathit{{{a_column}}}$"  # Italic A
    if a_column in exdf.units:
        ylabel += f" ({exdf.units[a_column]})"
    
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    
    # No title per request
    # if title:
    #     ax1.set_title(title)
    
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    
    # Add parameter text box
    if show_parameters:
        param_text = _format_parameters_text(fit_result, show_confidence_intervals)
        ax1.text(0.05, 0.95, param_text, transform=ax1.transAxes,
                 verticalalignment='top', horizontalalignment='left',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                          edgecolor='gray', alpha=0.9))
    
    # Residual plot
    if show_residuals:
        residuals_sorted = fit_result.residuals[sort_idx]
        ax2.scatter(ci_sorted, residuals_sorted, color='black', s=36, alpha=0.7)
        ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax2.set_xlabel(xlabel)
        ax2.set_ylabel('Residuals')
        ax2.grid(True, alpha=0.3)
        
        # Add RMSE text
        ax2.text(0.95, 0.95, f"RMSE = {fit_result.rmse:.2f}",
                 transform=ax2.transAxes,
                 verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_parameter_distributions(
    results: List[C3FitResult],
    parameters: Optional[List[str]] = None,
    group_labels: Optional[List[str]] = None,
    fig_size: Tuple[float, float] = (12, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot distributions of fitted parameters across multiple curves.
    
    Args:
        results: List of C3FitResult objects
        parameters: Parameters to plot (default: all)
        group_labels: Labels for each result
        fig_size: Figure size
        save_path: Path to save figure
    
    Returns:
        Matplotlib figure object
    """
    setup_plot_style()
    
    # Determine parameters to plot
    if parameters is None:
        parameters = ['Vcmax_at_25', 'J_at_25', 'Tp_at_25', 'RL_at_25']
        parameters = [p for p in parameters if p in results[0].parameters]
    
    n_params = len(parameters)
    n_cols = min(2, n_params)
    n_rows = (n_params + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=fig_size)
    if n_params == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, param in enumerate(parameters):
        ax = axes[idx]
        
        # Extract parameter values
        values = [r.parameters[param] for r in results if param in r.parameters]
        
        # Create DataFrame for easier plotting
        if group_labels:
            df = pd.DataFrame({
                'Value': values,
                'Group': group_labels[:len(values)]
            })
            
            # Box plot by group
            df.boxplot(column='Value', by='Group', ax=ax)
            ax.set_title(param)
            ax.set_xlabel('Group')
        else:
            # Histogram
            ax.hist(values, bins=20, edgecolor='black', alpha=0.7)
            ax.set_xlabel(param)
            ax.set_ylabel('Count')
            
            # Add statistics
            mean_val = np.mean(values)
            std_val = np.std(values)
            ax.axvline(mean_val, color='red', linestyle='--', 
                      label=f'Mean = {mean_val:.1f}')
            ax.text(0.95, 0.95, f'SD = {std_val:.1f}',
                   transform=ax.transAxes,
                   verticalalignment='top', horizontalalignment='right')
        
        ax.grid(True, alpha=0.3)
    
    # Hide extra subplots
    for idx in range(n_params, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_limiting_process_analysis(
    exdf: ExtendedDataFrame,
    fit_result: C3FitResult,
    ci_column: str = 'Ci',
    fig_size: Tuple[float, float] = (10, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create detailed plot showing limiting processes and component rates.
    
    Args:
        exdf: ExtendedDataFrame with data
        fit_result: C3FitResult object
        ci_column: Column name for Ci
        fig_size: Figure size
        save_path: Path to save figure
    
    Returns:
        Matplotlib figure object
    """
    setup_plot_style()
    
    fig, ax = plt.subplots(figsize=fig_size)
    
    # Get Ci values
    ci = exdf.data[ci_column].values
    sort_idx = np.argsort(ci)
    ci_sorted = ci[sort_idx]
    
    # Calculate component rates if available
    if hasattr(fit_result, 'temperature_adjusted_params'):
        # These would need to be calculated - placeholder for now
        a_obs = exdf.data['A'].values[sort_idx]
        a_fit = fit_result.fitted_A[sort_idx]
        
        ax.scatter(ci, exdf.data['A'].values, color='black', s=64, 
                  alpha=0.7, label='Observed', zorder=5)
        ax.plot(ci_sorted, a_fit, 'k-', linewidth=2, label='Net assimilation')
        
        # Add limiting regions as background colors
        if hasattr(fit_result, 'limiting_process'):
            limiting_sorted = fit_result.limiting_process[sort_idx]
            
            colors = {'Wc': '#1f77b4', 'Wj': '#2ca02c', 'Wp': '#d62728'}
            
            for process in ['Wc', 'Wj', 'Wp']:
                mask = limiting_sorted == process
                if np.any(mask):
                    # Find continuous segments
                    diff = np.diff(np.concatenate(([False], mask, [False])).astype(int))
                    starts = np.where(diff == 1)[0]
                    ends = np.where(diff == -1)[0]
                    
                    for start, end in zip(starts, ends):
                        ax.axvspan(ci_sorted[start], ci_sorted[end-1],
                                  alpha=0.2, color=colors[process])
    
    ax.set_xlabel(f"$\\mathit{{{ci_column}}}$ ({exdf.units.get(ci_column, '')})")
    ax.set_ylabel(f"$\\mathit{{A}}$ ({exdf.units.get('A', '')})")
    # No title per request
    # ax.set_title("Limiting Process Analysis")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def _format_parameters_text(
    fit_result: C3FitResult,
    show_confidence_intervals: bool = False
) -> str:
    """Format parameter values for display."""
    lines = []
    
    # Main parameters with italic formatting
    param_formats = {
        'Vcmax_at_25': ('$V_{cmax}$', '.1f'),
        'J_at_25': ('$J_{max}$', '.1f'),
        'Tp_at_25': ('$T_p$', '.1f'),
        'RL_at_25': ('$R_L$', '.2f'),
        'gmc': ('$g_{mc}$', '.2f')
    }
    
    for param, (display_name, fmt) in param_formats.items():
        if param in fit_result.parameters:
            value = fit_result.parameters[param]
            line = f"{display_name} = {value:{fmt}}"
            
            # Add confidence interval if available
            if show_confidence_intervals and fit_result.confidence_intervals:
                if param in fit_result.confidence_intervals:
                    ci_lower, ci_upper = fit_result.confidence_intervals[param]
                    line += f" [{ci_lower:{fmt}}, {ci_upper:{fmt}}]"
            
            lines.append(line)
    
    # Add statistics
    lines.append("")
    lines.append(f"R² = {fit_result.r_squared:.3f}")
    lines.append(f"RMSE = {fit_result.rmse:.2f}")
    
    if not np.isnan(fit_result.aic):
        lines.append(f"AIC = {fit_result.aic:.1f}")
    
    return '\n'.join(lines)