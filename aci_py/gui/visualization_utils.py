"""
Visualization utilities for ACI_py GUI

Enhanced plotting functions with confidence intervals and interactive features.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional, Tuple, List
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_aci_with_confidence_intervals(
        data: pd.DataFrame,
        fit_result,
        confidence_level: float = 0.95,
        show_residuals: bool = True,
        show_limiting_processes: bool = True
) -> plt.Figure:
    """
    Create comprehensive A-Ci plot with optional confidence intervals.

    Parameters
    ----------
    data : pd.DataFrame
        Original measurement data
    fit_result : FittingResult
        Result object from fitting
    confidence_level : float
        Confidence level for intervals (default 0.95)
    show_residuals : bool
        Whether to show residual subplot
    show_limiting_processes : bool
        Whether to color-code limiting processes

    Returns
    -------
    plt.Figure
        Matplotlib figure with enhanced visualization
    """
    # Create figure with subplots if showing residuals
    if show_residuals:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8),
                                       gridspec_kw={'height_ratios': [3, 1]})
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))

    # Main A-Ci plot
    ax1.scatter(data['Ci'], data['A'],
                s=100, alpha=0.7, edgecolors='darkblue',
                linewidth=2, label='Measured', zorder=5)

    # Plot fitted curve with confidence band if available
    if isinstance(fit_result, dict) and 'data' in fit_result and 'fitted_values' in fit_result:
        # Wrapper format from GUI
        ci_data = fit_result['data']['Ci'].values
        a_fit = fit_result['fitted_values']['A_fit'].values

        # Sort by Ci for proper line plotting
        sort_idx = np.argsort(ci_data)
        ci_sorted = ci_data[sort_idx]
        a_fit_sorted = a_fit[sort_idx]

        # Main fitted line
        ax1.plot(ci_sorted, a_fit_sorted, 'r-', linewidth=2,
                 label='Fitted', zorder=4)

        # Add confidence band if available
        if 'confidence_bands' in fit_result and fit_result['confidence_bands']:
            lower = fit_result['confidence_bands']['lower'][sort_idx]
            upper = fit_result['confidence_bands']['upper'][sort_idx]

            ax1.fill_between(ci_sorted, lower, upper,
                             alpha=0.2, color='red',
                             label=f'{int(confidence_level * 100)}% CI')

        # Color-code limiting processes if available
        if show_limiting_processes and 'limiting_process' in fit_result['fitted_values'].columns:
            limiting = fit_result['fitted_values']['limiting_process'].values[sort_idx]

            # Plot each limiting region with different color
            colors = {'Rubisco': '#e74c3c', 'RuBP': '#3498db', 'TPU': '#2ecc71'}

            for process, color in colors.items():
                mask = limiting == process
                if mask.any():
                    # Find continuous regions
                    regions = find_continuous_regions(mask)

                    for start, end in regions:
                        ax1.plot(ci_sorted[start:end + 1], a_fit_sorted[start:end + 1],
                                 color=color, linewidth=4, alpha=0.8,
                                 label=f'{process}-limited' if start == regions[0][0] else '')

    # Add parameter annotations
    if isinstance(fit_result, dict) and 'parameters' in fit_result:
        params = fit_result['parameters']
        param_text = []

        # Add main parameters with italic formatting
        if 'Vcmax_at_25' in params:
            param_text.append(f'$V_{{cmax}}$ = {params["Vcmax_at_25"]:.1f}')
        if 'J_at_25' in params:
            param_text.append(f'$J_{{max}}$ = {params["J_at_25"]:.1f}')
        if 'Tp_at_25' in params:
            param_text.append(f'$T_p$ = {params["Tp_at_25"]:.1f}')
        if 'RL_at_25' in params:
            param_text.append(f'$R_L$ = {params["RL_at_25"]:.2f}')

        # Add confidence intervals only if available (batch processing)
        # Skip for single file analysis
        # if 'confidence_intervals' in fit_result and fit_result['confidence_intervals']:
        #     ci_text = []
        #     for param, bounds in fit_result['confidence_intervals'].items():
        #         if bounds['lower'] is not None and bounds['upper'] is not None:
        #             param_name = param.replace('_at_25', '')
        #             ci_text.append(f'{param_name}: [{bounds["lower"]:.1f}, {bounds["upper"]:.1f}]')
        #
        #     if ci_text:
        #         param_text.append('\n95% CI:')
        #         param_text.extend(ci_text)

        # Add text box
        if param_text:
            text = '\n'.join(param_text)
            ax1.text(0.05, 0.95, text, transform=ax1.transAxes,
                     verticalalignment='top', horizontalalignment='left',
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                               edgecolor='gray', alpha=0.9),
                     fontsize=12, family='monospace')

    # Formatting with bold labels and italic parameters
    ax1.set_xlabel('$\\mathit{C_i}$ (µmol mol⁻¹)', fontsize=14, weight='bold')
    ax1.set_ylabel('$\\mathit{A}$ (µmol m⁻² s⁻¹)', fontsize=14, weight='bold')
    # No title per request
    # ax1.set_title('A-Ci Response Curve with Confidence Intervals', fontsize=14, pad=10)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='lower right', frameon=True, shadow=True, fontsize=12)

    # Add panel border
    for spine in ax1.spines.values():
        spine.set_linewidth(1.5)
        spine.set_edgecolor('black')

    # Residual plot
    if show_residuals and isinstance(fit_result, dict) and 'fitted_values' in fit_result:
        residuals = data['A'].values - fit_result['fitted_values']['A_fit'].values

        ax2.scatter(data['Ci'], residuals, s=50, alpha=0.7, color='darkblue')
        ax2.axhline(y=0, color='red', linestyle='--', linewidth=1)

        # Add confidence band for residuals
        if 'statistics' in fit_result and 'rmse' in fit_result['statistics']:
            rmse = fit_result['statistics']['rmse']
            ax2.axhline(y=rmse, color='gray', linestyle=':', alpha=0.5)
            ax2.axhline(y=-rmse, color='gray', linestyle=':', alpha=0.5)
            ax2.fill_between(data['Ci'], -rmse, rmse, alpha=0.1, color='gray')

        ax2.set_xlabel('$\\mathit{C_i}$ (µmol mol⁻¹)', fontsize=14, weight='bold')
        ax2.set_ylabel('Residuals', fontsize=14, weight='bold')
        ax2.grid(True, alpha=0.3)

        # Add panel border
        for spine in ax2.spines.values():
            spine.set_linewidth(1.5)
            spine.set_edgecolor('black')

    plt.tight_layout()
    return fig


def create_interactive_aci_plot(
        data: pd.DataFrame,
        fit_result,
        show_confidence: bool = True
) -> go.Figure:
    """
    Create interactive Plotly plot with hover information.

    Parameters
    ----------
    data : pd.DataFrame
        Original measurement data
    fit_result : FittingResult
        Result object from fitting
    show_confidence : bool
        Whether to show confidence intervals

    Returns
    -------
    go.Figure
        Plotly figure object
    """
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.75, 0.25],
        subplot_titles=('', ''),  # No titles per request
        vertical_spacing=0.1
    )

    # Measured data
    fig.add_trace(
        go.Scatter(
            x=data['Ci'],
            y=data['A'],
            mode='markers',
            name='Measured',
            marker=dict(
                size=10,
                color='darkblue',
                line=dict(width=2, color='darkblue')
            ),
            hovertemplate='Ci: %{x:.1f}<br>A: %{y:.2f}<extra></extra>'
        ),
        row=1, col=1
    )

    # Fitted curve
    if isinstance(fit_result, dict) and 'data' in fit_result and 'fitted_values' in fit_result:
        ci_data = fit_result['data']['Ci'].values
        a_fit = fit_result['fitted_values']['A_fit'].values

        # Sort for proper line
        sort_idx = np.argsort(ci_data)
        ci_sorted = ci_data[sort_idx]
        a_fit_sorted = a_fit[sort_idx]

        fig.add_trace(
            go.Scatter(
                x=ci_sorted,
                y=a_fit_sorted,
                mode='lines',
                name='Fitted',
                line=dict(color='red', width=2),
                hovertemplate='Ci: %{x:.1f}<br>A (fit): %{y:.2f}<extra></extra>'
            ),
            row=1, col=1
        )

        # Confidence bands
        if show_confidence and 'confidence_bands' in fit_result and fit_result['confidence_bands']:
            lower = fit_result['confidence_bands']['lower'][sort_idx]
            upper = fit_result['confidence_bands']['upper'][sort_idx]

            # Don't show confidence bands for single file analysis
            # These will be enabled when batch processing is implemented
            pass
            # # Upper bound
            # fig.add_trace(
            #     go.Scatter(
            #         x=ci_sorted,
            #         y=upper,
            #         mode='lines',
            #         name='95% CI Upper',
            #         line=dict(width=0),
            #         showlegend=False,
            #         hoverinfo='skip'
            #     ),
            #     row=1, col=1
            # )
            #
            # # Lower bound with fill
            # fig.add_trace(
            #     go.Scatter(
            #         x=ci_sorted,
            #         y=lower,
            #         mode='lines',
            #         name='95% CI',
            #         line=dict(width=0),
            #         fill='tonexty',
            #         fillcolor='rgba(255,0,0,0.2)',
            #         hoverinfo='skip'
            #     ),
            #     row=1, col=1
            # )

        # Residuals
        residuals = data['A'].values - fit_result['fitted_values']['A_fit'].values

        fig.add_trace(
            go.Scatter(
                x=data['Ci'],
                y=residuals,
                mode='markers',
                name='Residuals',
                marker=dict(size=6, color='darkblue'),
                showlegend=False,
                hovertemplate='Ci: %{x:.1f}<br>Residual: %{y:.3f}<extra></extra>'
            ),
            row=2, col=1
        )

        # Zero line for residuals
        fig.add_hline(y=0, line_dash="dash", line_color="red",
                      row=2, col=1)

    # Update layout with italic parameters and bold text
    fig.update_xaxes(title_text="<b><i>C<sub>i</sub></i> (µmol mol⁻¹)</b>", title_font_size=14, row=2, col=1)
    fig.update_xaxes(title_text="", row=1, col=1)
    fig.update_yaxes(title_text="<b><i>A</i> (µmol m⁻² s⁻¹)</b>", title_font_size=14, row=1, col=1)
    fig.update_yaxes(title_text="<b>Residual</b>", title_font_size=14, row=2, col=1)

    # Update tick font sizes
    fig.update_xaxes(tickfont_size=12)
    fig.update_yaxes(tickfont_size=12)

    fig.update_layout(
        height=700,
        hovermode='x unified',
        # No title per request
        # title_text="Interactive A-Ci Analysis",
        # title_font_size=16,
        showlegend=True,
        legend=dict(
            x=0.02,  # Move legend to left to avoid overlap
            y=0.48,  # Position in middle of top plot
            xanchor='left',
            yanchor='top',
            bgcolor='rgba(255, 255, 255, 0.9)',
            bordercolor='black',
            borderwidth=1
        ),
        # Add border to plots
        xaxis=dict(showline=True, linewidth=2, linecolor='black', mirror=True),
        yaxis=dict(showline=True, linewidth=2, linecolor='black', mirror=True),
        xaxis2=dict(showline=True, linewidth=2, linecolor='black', mirror=True),
        yaxis2=dict(showline=True, linewidth=2, linecolor='black', mirror=True)
    )

    return fig


def plot_parameter_confidence_intervals(
        fit_result,
        parameters: Optional[List[str]] = None
) -> plt.Figure:
    """
    Create bar plot of parameters with optional confidence intervals.

    Parameters
    ----------
    fit_result : FittingResult
        Result object with parameters and confidence intervals
    parameters : List[str], optional
        List of parameters to plot (default: all)

    Returns
    -------
    plt.Figure
        Matplotlib figure with parameter confidence intervals
    """
    if not (isinstance(fit_result, dict) and 'parameters' in fit_result and 'confidence_intervals' in fit_result):
        # Check if it's wrapped in result
        if isinstance(fit_result, dict) and 'result' in fit_result:
            result_obj = fit_result['result']
            if hasattr(result_obj, 'parameters') and hasattr(result_obj, 'confidence_intervals'):
                # Use the result object directly
                fit_result = {
                    'parameters': result_obj.parameters,
                    'confidence_intervals': result_obj.confidence_intervals
                }
            else:
                raise ValueError("Fit result must have parameters and confidence intervals")
        else:
            raise ValueError("Fit result must have parameters and confidence intervals")

    # Get parameters to plot
    if parameters is None:
        parameters = [p for p in fit_result['parameters'].keys()
                      if p in fit_result['confidence_intervals']]

    # Prepare data
    param_names = []
    values = []
    lower_errors = []
    upper_errors = []

    for param in parameters:
        if param in fit_result['confidence_intervals']:
            ci = fit_result['confidence_intervals'][param]
            if isinstance(ci, dict) and 'lower' in ci and 'upper' in ci:
                if ci['lower'] is not None and ci['upper'] is not None:
                    value = fit_result['parameters'][param]
                    param_names.append(param.replace('_at_25', ''))
                    values.append(value)
                    lower_errors.append(value - ci['lower'])
                    upper_errors.append(ci['upper'] - value)

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(param_names))
    bars = ax.bar(x, values, yerr=[lower_errors, upper_errors],
                  capsize=10, color='skyblue', edgecolor='darkblue',
                  linewidth=2, error_kw={'linewidth': 2})

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, values)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{val:.2f}',
                ha='center', va='bottom', fontsize=10)

    # Formatting with bold labels
    ax.set_xlabel('Parameter', fontsize=14, weight='bold')
    ax.set_ylabel('Value', fontsize=14, weight='bold')
    # No title per request
    # ax.set_title('Fitted Parameters', fontsize=14, pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(param_names, fontsize=11)
    ax.grid(True, axis='y', alpha=0.3)

    # Add panel border
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_edgecolor('black')

    # Add note only if confidence intervals are shown
    if len(lower_errors) > 0 and any(e > 0 for e in lower_errors):
        ax.text(0.02, 0.98, 'Error bars show 95% confidence intervals',
                transform=ax.transAxes, fontsize=12,
                verticalalignment='top', style='italic', alpha=0.7)

    plt.tight_layout()
    return fig


def find_continuous_regions(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Find continuous True regions in a boolean mask."""
    regions = []
    start = None

    for i, val in enumerate(mask):
        if val and start is None:
            start = i
        elif not val and start is not None:
            regions.append((start, i - 1))
            start = None

    if start is not None:
        regions.append((start, len(mask) - 1))

    return regions


def create_diagnostic_plots(fit_result) -> plt.Figure:
    """
    Create multi-panel diagnostic plots for fitting assessment.

    Parameters
    ----------
    fit_result : FittingResult
        Result object from fitting

    Returns
    -------
    plt.Figure
        Multi-panel diagnostic figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Fitted vs Measured
    ax = axes[0, 0]
    if isinstance(fit_result, dict) and 'data' in fit_result and 'fitted_values' in fit_result:
        measured = fit_result['data']['A'].values
        fitted = fit_result['fitted_values']['A_fit'].values

        ax.scatter(measured, fitted, alpha=0.7, s=50)

        # Add 1:1 line
        lims = [min(measured.min(), fitted.min()),
                max(measured.max(), fitted.max())]
        ax.plot(lims, lims, 'r--', alpha=0.7, label='1:1 line')

        # Add R² if available
        if 'statistics' in fit_result and 'r_squared' in fit_result['statistics']:
            ax.text(0.05, 0.95, f'R² = {fit_result["statistics"]["r_squared"]:.3f}',
                    transform=ax.transAxes, verticalalignment='top', fontsize=12,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax.set_xlabel('Measured $\\mathit{A}$', fontsize=14, weight='bold')
        ax.set_ylabel('Fitted $\\mathit{A}$', fontsize=14, weight='bold')
        # No title per request
        # ax.set_title('Fitted vs Measured')
        ax.legend()

        # Add panel border
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
            spine.set_edgecolor('black')

    # 2. Residuals vs Fitted
    ax = axes[0, 1]
    if isinstance(fit_result, dict) and 'data' in fit_result and 'fitted_values' in fit_result:
        residuals = measured - fitted

        ax.scatter(fitted, residuals, alpha=0.7, s=50)
        ax.axhline(y=0, color='r', linestyle='--')

        ax.set_xlabel('Fitted $\\mathit{A}$', fontsize=14, weight='bold')
        ax.set_ylabel('Residuals', fontsize=14, weight='bold')
        # No title per request
        # ax.set_title('Residual Plot')

        # Add panel border
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
            spine.set_edgecolor('black')

    # 3. Q-Q plot
    ax = axes[1, 0]
    if isinstance(fit_result, dict) and 'data' in fit_result and 'fitted_values' in fit_result:
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=ax)
        # No title per request
        # ax.set_title('Normal Q-Q Plot')
        ax.set_xlabel('Theoretical Quantiles', fontsize=14, weight='bold')
        ax.set_ylabel('Sample Quantiles', fontsize=14, weight='bold')

        # Add panel border
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
            spine.set_edgecolor('black')

    # 4. Parameter correlations or histogram
    ax = axes[1, 1]
    if isinstance(fit_result, dict) and 'data' in fit_result and 'fitted_values' in fit_result:
        ax.hist(residuals, bins=10, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Residuals', fontsize=14, weight='bold')
        ax.set_ylabel('Frequency', fontsize=14, weight='bold')
        # No title per request
        # ax.set_title('Residual Distribution')

        # Add panel border
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
            spine.set_edgecolor('black')

        # Add normal curve
        mu, std = residuals.mean(), residuals.std()
        x = np.linspace(residuals.min(), residuals.max(), 100)
        ax.plot(x, len(residuals) * (residuals.max() - residuals.min()) / 10 *
                stats.norm.pdf(x, mu, std), 'r-', linewidth=2, label='Normal')
        ax.legend()

    # No overall title per request
    # plt.suptitle('Diagnostic Plots for Model Fit', fontsize=14)
    plt.tight_layout()
    return fig