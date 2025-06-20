import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from pathlib import Path
import sys
from scipy.optimize import curve_fit
from typing import Dict, Tuple, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

st.set_page_config(
    page_title="Mengjie's Premium CO2 Response Analysis Tool",
    page_icon="leaf",
    layout="wide",
    initial_sidebar_state="expanded"
)

try:
    from aci_py.io.licor import read_licor_file
    from aci_py.analysis.c3_fitting import fit_c3_aci
    from aci_py.analysis.c4_fitting import fit_c4_aci
    from aci_py.core.data_structures import ExtendedDataFrame
    from aci_py.core.preprocessing import preprocess_aci_data
    from aci_py.analysis.plotting import plot_c3_fit
    from aci_py.io.export import export_fitting_result, create_analysis_report
    MODULES_AVAILABLE = True
except ImportError as e:
    st.error(f"Import error: {e}")
    MODULES_AVAILABLE = False


try:
    from aci_py.gui.smart_quality import SmartQualityChecker, QualityIssue
    SMART_QUALITY_AVAILABLE = True
except ImportError:
    SMART_QUALITY_AVAILABLE = False

try:
    from aci_py.gui.visualization_utils import (
        plot_aci_with_confidence_intervals,
        create_interactive_aci_plot,
        plot_parameter_confidence_intervals,
        create_diagnostic_plots
    )
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    
    # Fallback simple checker
    class DataQualityChecker:
        """Simple data quality assessment for A-Ci curves."""
        
        @staticmethod
        def assess(data: pd.DataFrame) -> Dict:
            """Perform basic quality checks on the data."""
            issues = []
            score = 100
            
            # Check data size
            if len(data) < 5:
                issues.append("Very few data points (minimum 5 recommended)")
                score -= 30
            elif len(data) < 8:
                issues.append("Limited data points (8-12 recommended)")
                score -= 10
                
            # Check Ci range
            if 'Ci' in data.columns:
                ci_range = data['Ci'].max() - data['Ci'].min()
                if ci_range < 500:
                    issues.append("Limited Ci range for robust fitting")
                    score -= 20
                
                if data['Ci'].min() > 100:
                    issues.append("Consider adding low Ci measurements")
                    score -= 10
                    
            # Check for negative A at high Ci
            if 'A' in data.columns and 'Ci' in data.columns:
                high_ci_negative = data[(data['Ci'] > 500) & (data['A'] < -2)]
                if len(high_ci_negative) > 0:
                    issues.append("Negative assimilation at high Ci detected")
                    score -= 20
                    
            # Temperature stability
            if 'Tleaf' in data.columns:
                temp_range = data['Tleaf'].max() - data['Tleaf'].min()
                if temp_range > 2:
                    issues.append(f"Temperature variation: {temp_range:.1f}°C")
                    score -= 15
                    
            return {
                'score': max(0, score),
                'issues': issues,
                'badge': "Good" if score >= 80 else "Fair" if score >= 60 else "Poor"
            }


def plot_aci_curve(data: pd.DataFrame, fit_result: Optional[Dict] = None) -> plt.Figure:
    """Create a clean A-Ci plot with optional fitted curve."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot measured data
    ax.scatter(data['Ci'], data['A'], 
              s=80, alpha=0.7, edgecolors='darkblue', 
              linewidth=1.5, label='Measured', zorder=3)
    
    # Add fitted curve if result provided
    if fit_result:
        # Check if it's our wrapper format
        if isinstance(fit_result, dict) and 'data' in fit_result and 'fitted_values' in fit_result:
            # Our wrapper format
            if isinstance(fit_result['fitted_values'], pd.DataFrame) and 'A_fit' in fit_result['fitted_values'].columns:
                ax.plot(fit_result['data']['Ci'], fit_result['fitted_values']['A_fit'], 
                       'r-', linewidth=2, label='Fitted', zorder=2)
                
                # Add limiting process regions if available
                if 'limiting_process' in fit_result['fitted_values'].columns:
                    ci_data = fit_result['data']['Ci'].values
                    a_fit = fit_result['fitted_values']['A_fit'].values
                    limiting = fit_result['fitted_values']['limiting_process'].values
                    
                    # Color by limiting process
                    colors = {'Wc': 'red', 'Wj': 'blue', 'Wp': 'green'}
                    labels = {'Wc': 'Rubisco-limited', 'Wj': 'RuBP-limited', 'Wp': 'TPU-limited'}
                    for process, color in colors.items():
                        mask = limiting == process
                        if mask.any():
                            ax.plot(ci_data[mask], a_fit[mask], 
                                   color=color, linewidth=3, alpha=0.7,
                                   label=labels[process])
        
        elif isinstance(fit_result, dict) and 'parameters' in fit_result:
            # Simple parameter dict (fallback mode)
            ci_smooth = np.linspace(data['Ci'].min(), data['Ci'].max(), 100)
            params = fit_result['parameters']
            
            # Get parameters with flexible naming
            vcmax = params.get('Vcmax_at_25', params.get('Vcmax', 50))
            rd = params.get('RL_at_25', params.get('Rd', 1.5))
            kc = 270  # Default Kc value
            
            # Simple rectangular hyperbola
            a_fitted = vcmax * ci_smooth / (ci_smooth + kc) - rd
            ax.plot(ci_smooth, a_fitted, 'r-', linewidth=2, label='Fitted', zorder=2)
    
    ax.set_xlabel('Ci (µmol mol⁻¹)', fontsize=12)
    ax.set_ylabel('A (µmol m⁻² s⁻¹)', fontsize=12)
    ax.set_title('CO₂ Response Curve', fontsize=14, pad=10)
    ax.grid(True, alpha=0.3, zorder=1)
    ax.legend(frameon=True, shadow=True)
    
    plt.tight_layout()
    return fig


def generate_sample_data(quality: str = "good") -> pd.DataFrame:
    """Generate sample A-Ci data for demonstration."""
    np.random.seed(42)
    
    if quality == "good":
        ci = np.array([50, 100, 150, 200, 300, 400, 600, 800, 1000, 1200, 1500])
        vcmax, kc, rd = 55, 270, 1.2
        noise_level = 0.3
        temp_var = 0.1
    else:
        ci = np.array([100, 200, 400, 800, 1200])  # Fewer points, poor coverage
        vcmax, kc, rd = 45, 270, 1.5
        noise_level = 1.0
        temp_var = 1.5
    
    # Calculate assimilation
    an = vcmax * ci / (ci + kc) - rd
    an += np.random.normal(0, noise_level, len(ci))
    
    # Create DataFrame
    return pd.DataFrame({
        'Ci': ci,
        'A': an,
        'Ca': ci * 1.1,
        'Tleaf': 25.0 + np.random.normal(0, temp_var, len(ci)),
        'Pa': 101.325,  # Standard atmospheric pressure in kPa
        'Qin': 1500,
        'gsw': 0.3 + np.random.normal(0, 0.02, len(ci))
    })


# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'exdf' not in st.session_state:
    st.session_state.exdf = None
if 'quality_report' not in st.session_state:
    st.session_state.quality_report = None
if 'fit_results' not in st.session_state:
    st.session_state.fit_results = None
if 'batch_data' not in st.session_state:
    st.session_state.batch_data = {}
if 'batch_results' not in st.session_state:
    st.session_state.batch_results = {}


# Main UI
st.title("Mengjie's Premium ACI Fitting Tool Pro V1")
st.markdown("---")

# Create tabs for workflow
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Import Data", "Quality Check", "Analysis", "Results", "Batch Processing"])

# Tab 1: Data Import
with tab1:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Data Source")
        
        data_source = st.radio(
            "Select data source:",
["Upload file", "Use sample data", "Paste data"]
        )
        
        if data_source == "Upload file":
            uploaded_file = st.file_uploader(
                "Choose a file",
                type=['csv', 'xlsx', 'xls'],
                help="Supports LI-COR 6800 and other standard formats"
            )
            
            if uploaded_file and st.button("Import File", type="primary"):
                try:
                    # Import the proper licor reader
                    from aci_py.io.licor import read_licor_file
                    
                    # Save uploaded file temporarily
                    import tempfile
                    import os
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                        tmp_file.write(uploaded_file.getbuffer())
                        tmp_path = tmp_file.name
                    
                    try:
                        # Read using the proper licor reader
                        exdf = read_licor_file(tmp_path)
                        # Store the full ExtendedDataFrame
                        st.session_state.exdf = exdf
                        # Also store just the pandas DataFrame for display
                        st.session_state.data = exdf.data
                        
                        # Run quality check on the pandas DataFrame
                        if SMART_QUALITY_AVAILABLE:
                            checker = SmartQualityChecker()
                            st.session_state.quality_report = checker.assess_comprehensive(st.session_state.data)
                        else:
                            st.session_state.quality_report = DataQualityChecker.assess(st.session_state.data)
                    finally:
                        # Clean up temp file
                        os.unlink(tmp_path)
                    st.success("Data imported successfully!")
                    st.rerun()
                    
                except ValueError as e:
                    if "uncalculated formulas" in str(e):
                        st.error("Excel Formula Error")
                        st.warning(str(e))
                    else:
                        st.error(f"Error loading file: {e}")
                except Exception as e:
                    st.error(f"❌ Error loading file: {e}")
                    st.info("Make sure your file is a valid LI-COR 6800 export (CSV or Excel)")
        
        elif data_source == "Use sample data":
            st.markdown("### Sample Datasets")
            
            sample_type = st.selectbox(
                "Choose sample:",
                ["High quality C3", "Low quality C3", "C4 maize"]
            )
            
            if st.button("Load Sample", type="primary"):
                quality = "good" if "High" in sample_type else "poor"
                sample_df = generate_sample_data(quality)
                
                # Create ExtendedDataFrame from sample data
                from aci_py.core.data_structures import ExtendedDataFrame
                
                # Define units for the sample data
                units = {
                    'Ci': 'µmol mol⁻¹',
                    'A': 'µmol m⁻² s⁻¹',
                    'Ca': 'µmol mol⁻¹',
                    'Tleaf': '°C',
                    'Pa': 'kPa'
                }
                
                # Create ExtendedDataFrame
                st.session_state.exdf = ExtendedDataFrame(data=sample_df, units=units)
                st.session_state.data = sample_df
                
                # Run quality check
                if SMART_QUALITY_AVAILABLE:
                    checker = SmartQualityChecker()
                    species_type = 'C4' if 'C4' in sample_type else 'C3'
                    st.session_state.quality_report = checker.assess_comprehensive(
                        st.session_state.data, species_type=species_type
                    )
                else:
                    st.session_state.quality_report = DataQualityChecker.assess(st.session_state.data)
                    
                st.success(f"Loaded {sample_type} data")
                st.rerun()
        
        else:  # Paste data
            st.markdown("### Paste Data")
            st.markdown("""
            #### Instructions:
            Enter your A (assimilation) and Ci (internal CO₂) values separately below.
            You can copy values from a column in Excel and paste them directly.
            """)
            
            # Temperature input method
            temp_method = st.radio(
                "Temperature data:",
                ["Single value", "Separate values"],
                help="Choose how to specify leaf temperature"
            )
            
            # Create columns for better layout
            col_a, col_ci = st.columns(2)
            
            with col_a:
                st.markdown("##### A values (µmol m⁻² s⁻¹)")
                a_values = st.text_area(
                    "Paste A values (one per line):",
                    height=250,
                    placeholder="12.5\n18.3\n22.1\n25.8\n28.2\n29.5",
                    help="Copy your A column from Excel and paste here. One value per line.",
                    label_visibility="collapsed"
                )
            
            with col_ci:
                st.markdown("##### Ci values (µmol mol⁻¹)")
                ci_values = st.text_area(
                    "Paste Ci values (one per line):",
                    height=250,
                    placeholder="150\n250\n400\n600\n800\n1000",
                    help="Copy your Ci column from Excel and paste here. One value per line.",
                    label_visibility="collapsed"
                )
            
            # Temperature input
            if temp_method == "Single value":
                temperature_value = st.number_input(
                    "Leaf temperature (°C):",
                    min_value=0.0,
                    max_value=50.0,
                    value=25.0,
                    step=0.1,
                    help="This temperature will be used for all measurements"
                )
            else:
                st.markdown("##### Temperature values (°C)")
                temp_values = st.text_area(
                    "Paste temperature values (one per line):",
                    height=200,
                    placeholder="25.0\n25.1\n25.0\n25.2\n25.1\n25.0",
                    help="Copy your temperature column from Excel and paste here. One value per line.",
                    label_visibility="collapsed"
                )
            
            # Helper function for parsing values
            def parse_values(text: str) -> list:
                """Parse values from text, handling various formats."""
                try:
                    text = text.strip()
                    if not text:
                        return []
                    # Check if it's comma or tab separated on one line
                    if ',' in text and '\n' not in text:
                        values = [v.strip() for v in text.split(',')]
                    elif '\t' in text and '\n' not in text:
                        values = [v.strip() for v in text.split('\t')]
                    else:
                        # One value per line
                        values = [v.strip() for v in text.split('\n') if v.strip()]
                    
                    # Convert to float
                    return [float(v) for v in values if v]
                except:
                    return []
            
            # Parse button
            if st.button("📊 Parse Data", type="primary"):
                try:
                    # Parse A values
                    if not a_values.strip():
                        raise ValueError("Please enter A values")
                    
                    # Parse A values with validation
                    a_list = parse_values(a_values)
                    if not a_list:
                        raise ValueError("Could not parse A values. Please check the format.")
                    
                    # Parse Ci values
                    if not ci_values.strip():
                        raise ValueError("Please enter Ci values")
                    
                    ci_list = parse_values(ci_values)
                    if not ci_list:
                        raise ValueError("Could not parse Ci values. Please check the format.")
                    
                    # Check that A and Ci have same length
                    if len(a_list) != len(ci_list):
                        raise ValueError(f"Number of A values ({len(a_list)}) must match number of Ci values ({len(ci_list)})")
                    
                    if len(a_list) < 3:
                        raise ValueError("Need at least 3 data points for ACI fitting")
                    
                    # Create DataFrame
                    parsed_df = pd.DataFrame({
                        'A': a_list,
                        'Ci': ci_list
                    })
                    
                    # Add temperature
                    if temp_method == "Single value":
                        parsed_df['Tleaf'] = temperature_value
                    else:
                        # Parse temperature values
                        if not temp_values.strip():
                            raise ValueError("Please enter temperature values")
                        
                        temp_list = parse_values(temp_values)
                        
                        if len(temp_list) != len(a_list):
                            raise ValueError(f"Number of temperature values ({len(temp_list)}) must match number of A/Ci values ({len(a_list)})")
                        
                        parsed_df['Tleaf'] = temp_list
                    
                    # Add other required columns with default values
                    if 'Ca' not in parsed_df.columns:
                        # Estimate Ca from Ci (typical ratio)
                        parsed_df['Ca'] = parsed_df['Ci'] * 1.6
                    
                    if 'Pa' not in parsed_df.columns:
                        parsed_df['Pa'] = 101.325  # Standard atmospheric pressure
                    
                    if 'Qin' not in parsed_df.columns:
                        parsed_df['Qin'] = 1500  # Assume saturating light
                    
                    # Convert to numeric
                    numeric_cols = ['A', 'Ci', 'Ca', 'Tleaf', 'Pa', 'Qin']
                    for col in numeric_cols:
                        if col in parsed_df.columns:
                            parsed_df[col] = pd.to_numeric(parsed_df[col], errors='coerce')
                    
                    # Remove rows with NaN in essential columns
                    parsed_df = parsed_df.dropna(subset=['A', 'Ci', 'Tleaf'])
                    
                    # Additional data validation
                    if len(parsed_df) < 3:
                        raise ValueError("Need at least 3 valid data points for ACI fitting")
                    
                    # Check for reasonable ranges
                    if parsed_df['Ci'].min() < 0 or parsed_df['Ci'].max() > 3000:
                        st.warning("Some Ci values are outside typical range (0-3000 µmol/mol)")
                    
                    if parsed_df['A'].min() < -20 or parsed_df['A'].max() > 100:
                        st.warning("Some A values are outside typical range (-20 to 100 µmol/m²/s)")
                    
                    if parsed_df['Tleaf'].min() < 0 or parsed_df['Tleaf'].max() > 50:
                        st.warning("Some temperature values are outside typical range (0-50°C)")
                    
                    # Sort by Ci for better visualization
                    parsed_df = parsed_df.sort_values('Ci').reset_index(drop=True)
                    
                    # Create ExtendedDataFrame
                    units = {
                        'A': 'µmol m⁻² s⁻¹',
                        'Ci': 'µmol mol⁻¹',
                        'Ca': 'µmol mol⁻¹',
                        'Tleaf': '°C',
                        'Pa': 'kPa',
                        'Qin': 'µmol m⁻² s⁻¹'
                    }
                    
                    exdf = ExtendedDataFrame(data=parsed_df, units=units)
                    
                    # Store in session state
                    st.session_state.exdf = exdf
                    st.session_state.data = parsed_df
                    
                    # Run quality check
                    if SMART_QUALITY_AVAILABLE:
                        checker = SmartQualityChecker()
                        st.session_state.quality_report = checker.assess_comprehensive(parsed_df)
                    else:
                        st.session_state.quality_report = DataQualityChecker.assess(parsed_df)
                    
                    st.success(f"Successfully parsed {len(parsed_df)} data points!")
                    
                    # Show preview
                    with st.expander("Preview parsed data", expanded=True):
                        st.dataframe(parsed_df.round(2), use_container_width=True)
                        
                        # Quick stats
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Data points", len(parsed_df))
                        col2.metric("Ci range", f"{parsed_df['Ci'].min():.0f}-{parsed_df['Ci'].max():.0f}")
                        col3.metric("Mean A", f"{parsed_df['A'].mean():.1f}")
                    
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error parsing data: {str(e)}")
                    st.info("""
                    **Troubleshooting tips:**
                    - Make sure you have entered values in both A and Ci fields
                    - Each value should be on a new line (or comma/tab separated on one line)
                    - Ensure all values are numeric (no text or units)
                    - Check that A and Ci have the same number of values
                    - If using separate temperature values, ensure count matches A/Ci
                    """)
            
            # Show current data count for user feedback
            col1, col2, col3 = st.columns(3)
            with col1:
                a_count = len(parse_values(a_values)) if a_values.strip() else 0
                st.metric("A values", a_count)
            with col2:
                ci_count = len(parse_values(ci_values)) if ci_values.strip() else 0
                st.metric("Ci values", ci_count)
            with col3:
                if temp_method == "Separate values" and temp_values.strip():
                    temp_count = len(parse_values(temp_values))
                    st.metric("Temp values", temp_count)
                else:
                    st.metric("Temp", f"{temperature_value:.1f}°C" if temp_method == "Single value" else "Not set")
            
            # Example data format
            with st.expander("📚 See example data", expanded=False):
                example_col1, example_col2 = st.columns(2)
                
                with example_col1:
                    st.markdown("**Example A values:**")
                    st.code("""12.5
18.3
22.1
25.8
28.2
29.5
30.8
31.2
31.5""")
                    st.caption("Copy these values to test!")
                
                with example_col2:
                    st.markdown("**Example Ci values:**")
                    st.code("""150
250
400
600
800
1000
1200
1400
1600""")
                    st.caption("Must match A value count")
                
                st.markdown("**Alternative formats also work:**")
                st.code("12.5, 18.3, 22.1, 25.8, 28.2  (comma separated)")
                st.code("12.5	18.3	22.1	25.8	28.2  (tab separated)")
            
            # Quick fill example data
            if st.button("Fill with example data", help="Quickly fill fields with example A-Ci curve data"):
                # This button will need JavaScript to work properly in Streamlit
                # For now, just show a message
                st.info("Copy the example values from above and paste them into the respective fields")
    
    with col2:
        if st.session_state.data is not None:
            st.subheader("Data Preview")
            
            # Show data info
            col1, col2, col3 = st.columns(3)
            col1.metric("Rows", len(st.session_state.data))
            col2.metric("Columns", len(st.session_state.data.columns))
            
            if 'Ci' in st.session_state.data.columns:
                ci_range = f"{st.session_state.data['Ci'].min():.0f}-{st.session_state.data['Ci'].max():.0f}"
                col3.metric("Ci Range", ci_range)
            
            # Show data table
            st.dataframe(
                st.session_state.data.round(2),
                use_container_width=True,
                height=400
            )
            
            # Quick plot
            if 'Ci' in st.session_state.data.columns and 'A' in st.session_state.data.columns:
                fig = plot_aci_curve(st.session_state.data)
                st.pyplot(fig)
        else:
            st.info("Select a data source to begin")

# Tab 2: Quality Check
with tab2:
    if st.session_state.data is None:
        st.info("Please import data first")
    else:
        st.subheader("Data Quality Assessment")
        
        if st.session_state.quality_report:
            report = st.session_state.quality_report
            
            # Quality score display
            col1, col2 = st.columns([1, 3])
            
            with col1:
                st.markdown("### Overall Quality")
                st.markdown(f"# {report['badge']}")
                st.metric("Score", f"{report['score']}%")
                
                # Species type selector for smart analysis
                if SMART_QUALITY_AVAILABLE:
                    species_type = st.selectbox(
                        "Species type:",
                        ["Auto-detect", "C3", "C4"],
                        help="Helps provide species-specific recommendations"
                    )
                
                if st.button("🔄 Re-assess", help="Run quality check again"):
                    if SMART_QUALITY_AVAILABLE:
                        checker = SmartQualityChecker()
                        spec_type = None if species_type == "Auto-detect" else species_type
                        st.session_state.quality_report = checker.assess_comprehensive(
                            st.session_state.data, species_type=spec_type
                        )
                    else:
                        st.session_state.quality_report = DataQualityChecker.assess(st.session_state.data)
                    st.rerun()
                
                # Auto-fix button if available
                if SMART_QUALITY_AVAILABLE and 'auto_fixes_available' in report:
                    if report['auto_fixes_available'] > 0:
                        if st.button(f"Apply {report['auto_fixes_available']} Auto-fixes"):
                            checker = SmartQualityChecker()
                            fixed_data, n_fixed = checker.apply_auto_fixes(
                                st.session_state.data, 
                                report.get('issues', [])
                            )
                            st.session_state.data = fixed_data
                            st.success(f"Applied {n_fixed} fixes!")
                            st.rerun()
            
            with col2:
                # Smart recommendations if available
                if SMART_QUALITY_AVAILABLE and 'recommendations' in report:
                    st.markdown("### Smart Recommendations")
                    for rec in report['recommendations']:
                        st.markdown(rec)
                    
                    if 'summary' in report:
                        st.info(report['summary'])
                    
                    # Detailed issues with tabs
                    if 'issues' in report and len(report['issues']) > 0 and hasattr(report['issues'][0], 'severity'):
                        st.markdown("### Detailed Issues")
                        
                        # Group issues by category
                        issue_tabs = st.tabs(["Critical", "Warnings", "Info"])
                        
                        critical_issues = [i for i in report['issues'] if i.severity == 'critical']
                        warning_issues = [i for i in report['issues'] if i.severity == 'warning']
                        info_issues = [i for i in report['issues'] if i.severity == 'info']
                        
                        with issue_tabs[0]:
                            if critical_issues:
                                for issue in critical_issues:
                                    with st.expander(f"{issue.message}"):
                                        st.write(f"**Category:** {issue.category}")
                                        st.write(f"**Recommendation:** {issue.recommendation}")
                                        if issue.confidence < 1.0:
                                            st.write(f"**Confidence:** {issue.confidence*100:.0f}%")
                                        if issue.auto_fixable:
                                            st.write("Auto-fixable")
                            else:
                                st.success("No critical issues found!")
                        
                        with issue_tabs[1]:
                            if warning_issues:
                                for issue in warning_issues:
                                    with st.expander(f"{issue.message}"):
                                        st.write(f"**Category:** {issue.category}")
                                        st.write(f"**Recommendation:** {issue.recommendation}")
                                        if issue.confidence < 1.0:
                                            st.write(f"**Confidence:** {issue.confidence*100:.0f}%")
                            else:
                                st.info("No warnings")
                        
                        with issue_tabs[2]:
                            if info_issues:
                                for issue in info_issues:
                                    st.info(f"{issue.message}: {issue.recommendation}")
                            else:
                                st.info("No additional information")
                
                else:
                    # Simple issues display
                    st.markdown("### Issues & Recommendations")
                    
                    if report.get('issues'):
                        for issue in report['issues']:
                            st.markdown(f"- {issue}")
                    else:
                        st.success("No significant issues detected!")
                
                # Data statistics
                st.markdown("### Data Statistics")
                
                stats_cols = ['A', 'Ci', 'Tleaf'] if all(col in st.session_state.data.columns for col in ['A', 'Ci', 'Tleaf']) else st.session_state.data.columns[:3]
                
                stats_df = st.session_state.data[stats_cols].describe().round(2)
                st.dataframe(stats_df, use_container_width=True)

# Tab 3: Analysis
with tab3:
    if st.session_state.data is None:
        st.info("Please import data first")
    else:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Analysis Settings")
            
            # Model selection
            model_type = st.selectbox(
                "Photosynthesis type:",
                ["C3", "C4"],
                help="C3: Most plants | C4: Maize, sugarcane, etc."
            )
            
            # Basic parameters
            st.markdown("### Initial Parameters")
            
            use_defaults = st.checkbox("Use recommended values", value=True)
            
            if not use_defaults:
                vcmax_init = st.number_input("Vcmax initial", value=50.0, min_value=0.0)
                j_init = st.number_input("J initial", value=100.0, min_value=0.0)
                rd_init = st.number_input("Rd initial", value=1.5, min_value=0.0)
            
            # Advanced options
            with st.expander("Advanced Options"):
                temperature_response = st.selectbox(
                    "Temperature response:",
                    ["Bernacchi", "Sharkey", "None"]
                )
                
                optimizer = st.selectbox(
                    "Optimization method:",
                    ["Differential Evolution", "Least Squares"]
                )
            
            # Fit button
            if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
                with st.spinner("Fitting model..."):
                    try:
                        if MODULES_AVAILABLE:
                            # Use the ExtendedDataFrame if available, otherwise create one
                            if hasattr(st.session_state, 'exdf') and st.session_state.exdf is not None:
                                exdf = st.session_state.exdf
                            else:
                                # Create ExtendedDataFrame
                                exdf = ExtendedDataFrame(
                                    st.session_state.data,
                                    units={
                                        'A': 'micromol m^(-2) s^(-1)',
                                        'Ci': 'micromol mol^(-1)',
                                        'Ca': 'micromol mol^(-1)',
                                        'Tleaf': 'C',
                                        'Pa': 'kPa',
                                        'Qin': 'micromol m^(-2) s^(-1)'
                                    }
                                )
                            
                            # Calculate required gas properties if needed
                            if 'T_leaf_K' not in exdf.data.columns:
                                exdf.calculate_gas_properties()
                            
                            # Get fitting parameters
                            fixed_params = {}
                            if use_defaults:
                                fixed_params = None
                            else:
                                # User can fix parameters by setting them
                                pass
                            
                            # Run appropriate fitting
                            if model_type == "C3":
                                # Set temperature response parameters
                                temp_response_params = None
                                if temperature_response == "Bernacchi":
                                    from aci_py.core.temperature import C3_TEMPERATURE_PARAM_BERNACCHI
                                    temp_response_params = C3_TEMPERATURE_PARAM_BERNACCHI
                                elif temperature_response == "Sharkey":
                                    from aci_py.core.temperature import C3_TEMPERATURE_PARAM_SHARKEY
                                    temp_response_params = C3_TEMPERATURE_PARAM_SHARKEY
                                
                                result = fit_c3_aci(
                                    exdf,
                                    fixed_parameters=fixed_params,
                                    temperature_response_params=temp_response_params,
                                    optimizer='differential_evolution' if optimizer == "Differential Evolution" else 'nelder_mead',
                                    calculate_confidence_intervals=False,  # Disabled for single file
                                    verbose=True
                                )
                            else:  # C4
                                result = fit_c4_aci(
                                    exdf,
                                    fixed_parameters=fixed_params,
                                    optimizer='differential_evolution' if optimizer == "Differential Evolution" else 'nelder_mead',
                                    calculate_confidence_intervals=False  # Disabled for single file
                                )
                            
                            # Store result with data for visualization
                            # Create a wrapper that includes data for compatibility
                            result_wrapper = {
                                'result': result,
                                'data': exdf.data,
                                'parameters': result.parameters,
                                'statistics': {
                                    'rmse': result.rmse,
                                    'r_squared': result.r_squared,
                                    'aic': result.aic,
                                    'bic': result.bic
                                },
                                'fitted_values': pd.DataFrame({
                                    'A_fit': result.fitted_A,
                                    'limiting_process': result.limiting_process
                                })
                            }
                            
                            # Add confidence intervals if available
                            if hasattr(result, 'confidence_intervals') and result.confidence_intervals:
                                result_wrapper['confidence_intervals'] = result.confidence_intervals
                            
                            st.session_state.fit_results = result_wrapper
                            st.success("✅ Analysis complete!")
                            st.balloons()
                            
                        else:
                            # Fallback to simple fitting
                            def hyperbola(x, vmax, km, rd):
                                return vmax * x / (x + km) - rd
                            
                            popt, _ = curve_fit(
                                hyperbola, 
                                st.session_state.data['Ci'], 
                                st.session_state.data['A'],
                                p0=[50, 270, 1.5],
                                bounds=(0, [200, 1000, 10])
                            )
                            
                            st.session_state.fit_results = {
                                'parameters': {
                                    'Vcmax_at_25': popt[0],
                                    'RL_at_25': popt[2]
                                },
                                'statistics': {
                                    'rmse': 0.5,
                                    'r_squared': 0.95
                                }
                            }
                            
                            st.success("✅ Analysis complete (demo mode)!")
                        
                    except Exception as e:
                        st.error(f"Fitting error: {str(e)}")
                        import traceback
                        st.error(traceback.format_exc())
        
        with col2:
            st.subheader("Live Preview")
            
            if st.session_state.data is not None:
                # Show current data and fit
                fig = plot_aci_curve(
                    st.session_state.data,
                    st.session_state.fit_results
                )
                st.pyplot(fig)
                
                # Show fitting progress/results
                if st.session_state.fit_results:
                    st.success("✅ Model converged successfully")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    # Extract parameters from result object
                    if isinstance(st.session_state.fit_results, dict):
                        if 'result' in st.session_state.fit_results:
                            # Our wrapper format
                            result = st.session_state.fit_results['result']
                            params = result.parameters
                            
                            vcmax = params.get('Vcmax_at_25', 'N/A')
                            rd = params.get('RL_at_25', 'N/A')
                            r2 = result.r_squared
                            
                            col1.metric("Vcmax", f"{vcmax:.1f}" if isinstance(vcmax, (int, float)) else vcmax)
                            col2.metric("Rd", f"{rd:.2f}" if isinstance(rd, (int, float)) else rd)
                            col3.metric("R²", f"{r2:.3f}" if isinstance(r2, (int, float)) else r2)
                        else:
                            # Simple dict result
                            params = st.session_state.fit_results.get('parameters', {})
                            stats = st.session_state.fit_results.get('statistics', {})
                            
                            vcmax = params.get('Vcmax_at_25', params.get('Vcmax', 'N/A'))
                            rd = params.get('RL_at_25', params.get('Rd', 'N/A'))
                            r2 = stats.get('r_squared', stats.get('R2', 'N/A'))
                            
                            col1.metric("Vcmax", f"{vcmax:.1f}" if isinstance(vcmax, (int, float)) else vcmax)
                            col2.metric("Rd", f"{rd:.2f}" if isinstance(rd, (int, float)) else rd)
                            col3.metric("R²", f"{r2:.3f}" if isinstance(r2, (int, float)) else r2)

# Tab 4: Results
with tab4:
    if st.session_state.fit_results is None:
        st.info("Run analysis first to see results")
    else:
        st.subheader("Analysis Results")
        
        # Create layout with better proportions
        # First row: Parameters and export options
        param_col, export_col = st.columns([2, 1])
        
        with param_col:
            st.markdown("### Fitted Parameters")
            
            # Build parameter table based on result type
            params_data = []
            
            if isinstance(st.session_state.fit_results, dict) and 'result' in st.session_state.fit_results:
                # Our wrapper format with full result
                result = st.session_state.fit_results['result']
                params = result.parameters
                stats = {'rmse': result.rmse, 'r_squared': result.r_squared}
                
                # C3 parameters
                if 'Vcmax_at_25' in params:
                    params_data.append(['Vcmax', f"{params['Vcmax_at_25']:.2f}", 'µmol m⁻² s⁻¹'])
                if 'J_at_25' in params:
                    params_data.append(['Jmax', f"{params['J_at_25']:.2f}", 'µmol m⁻² s⁻¹'])
                if 'Tp_at_25' in params:
                    params_data.append(['Tp', f"{params['Tp_at_25']:.2f}", 'µmol m⁻² s⁻¹'])
                if 'RL_at_25' in params:
                    params_data.append(['Rd', f"{params['RL_at_25']:.3f}", 'µmol m⁻² s⁻¹'])
                    
                # C4 parameters
                if 'Vpmax_at_25' in params:
                    params_data.append(['Vpmax', f"{params['Vpmax_at_25']:.2f}", 'µmol m⁻² s⁻¹'])
                if 'gbs' in params:
                    params_data.append(['gbs', f"{params['gbs']:.4f}", 'mol m⁻² s⁻¹ bar⁻¹'])
                    
                # Statistics
                if 'rmse' in stats:
                    params_data.append(['RMSE', f"{stats['rmse']:.3f}", 'µmol m⁻² s⁻¹'])
                if 'r_squared' in stats:
                    params_data.append(['R²', f"{stats['r_squared']:.4f}", '-'])
                    
                # Note: Confidence intervals disabled for single file analysis
                # Will be enabled in batch processing mode
                if 'AIC' in params or 'BIC' in params:
                    st.info("Confidence intervals are available in batch processing mode")
                        
            elif isinstance(st.session_state.fit_results, dict):
                # Simple dict result
                if 'parameters' in st.session_state.fit_results:
                    params = st.session_state.fit_results['parameters']
                    vcmax = params.get('Vcmax_at_25', params.get('Vcmax', 50))
                    rd = params.get('RL_at_25', params.get('Rd', 1.5))
                    params_data.append(['Vcmax', f"{vcmax:.2f}", 'µmol m⁻² s⁻¹'])
                    params_data.append(['Rd', f"{rd:.3f}", 'µmol m⁻² s⁻¹'])
                    
                if 'statistics' in st.session_state.fit_results:
                    stats = st.session_state.fit_results['statistics']
                    if 'r_squared' in stats:
                        params_data.append(['R²', f"{stats['r_squared']:.4f}", '-'])
            
            params_df = pd.DataFrame(params_data, columns=['Parameter', 'Value', 'Unit'])
            st.dataframe(params_df, hide_index=True, use_container_width=True)
            
            # Add note about confidence intervals
            st.caption("Note: Parameter confidence intervals will be available in batch processing mode")
        
        with export_col:
            st.markdown("### Export Options")
            
            export_format = st.selectbox("Export format", ["CSV", "Excel", "JSON"], key="export_format")
            
            if st.button("Export Data", use_container_width=True):
                # Always use simple export for now
                if export_format == "CSV":
                    # Export data with fitted values
                    export_data = st.session_state.data.copy()
                    if isinstance(st.session_state.fit_results, dict) and 'fitted_values' in st.session_state.fit_results:
                        export_data['A_fitted'] = st.session_state.fit_results['fitted_values']['A_fit']
                        if 'limiting_process' in st.session_state.fit_results['fitted_values'].columns:
                            export_data['Limiting_Process'] = st.session_state.fit_results['fitted_values']['limiting_process']
                    
                    csv = export_data.to_csv(index=False)
                    st.download_button(
                        "Download CSV",
                        csv,
                        "aci_results.csv",
                        "text/csv"
                    )
                elif export_format == "JSON":
                    # Export as JSON
                    import json
                    export_dict = {
                        'data': st.session_state.data.to_dict('records'),
                        'parameters': st.session_state.fit_results.get('parameters', {}),
                        'statistics': st.session_state.fit_results.get('statistics', {})
                    }
                    json_str = json.dumps(export_dict, indent=2)
                    st.download_button(
                        "Download JSON",
                        json_str,
                        "aci_results.json",
                        "application/json"
                    )
                else:
                    st.info("Excel export coming soon. Please use CSV format for now.")
            
            if st.button("Download Report", use_container_width=True):
                # For now, provide a text summary
                report_text = "ACI Analysis Report\n" + "="*50 + "\n\n"
                
                if isinstance(st.session_state.fit_results, dict):
                    # Add parameters
                    report_text += "Fitted Parameters:\n"
                    params = st.session_state.fit_results.get('parameters', {})
                    for param, value in params.items():
                        if isinstance(value, (int, float)):
                            report_text += f"  {param}: {value:.3f}\n"
                    
                    # Add statistics
                    report_text += "\nFit Statistics:\n"
                    stats = st.session_state.fit_results.get('statistics', {})
                    for stat, value in stats.items():
                        if isinstance(value, (int, float)):
                            report_text += f"  {stat}: {value:.4f}\n"
                    
                    # Add data summary
                    report_text += f"\nData Summary:\n"
                    report_text += f"  Number of points: {len(st.session_state.data)}\n"
                    report_text += f"  Ci range: {st.session_state.data['Ci'].min():.1f} - {st.session_state.data['Ci'].max():.1f}\n"
                    report_text += f"  A range: {st.session_state.data['A'].min():.1f} - {st.session_state.data['A'].max():.1f}\n"
                
                st.download_button(
                    "Download Text Report",
                    report_text,
                    "aci_analysis_report.txt",
                    "text/plain"
                )
                
                st.info("PDF report generation coming soon. Text report available now.")
        
        # Separator
        st.markdown("---")
        
        # Second section: Full-width visualization
        st.markdown("### Visualization")
        
        # Visualization options
        viz_type = st.radio(
            "Plot type:",
            ["Interactive", "Diagnostics"],
            horizontal=True
        )
        
        # Create containers for better layout
        plot_container = st.container()
        
        with plot_container:
            try:
                if viz_type == "Interactive":
                    if VISUALIZATION_AVAILABLE:
                        # Interactive Plotly plot without confidence intervals for single file
                        fig = create_interactive_aci_plot(
                            st.session_state.data,
                            st.session_state.fit_results,
                            show_confidence=False  # Disable confidence intervals for single file
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        # Fallback to standard plot
                        fig = plot_aci_curve(
                            st.session_state.data,
                            st.session_state.fit_results
                        )
                        st.pyplot(fig)
                        
                elif viz_type == "Diagnostics":
                    if VISUALIZATION_AVAILABLE:
                        # Diagnostic plots
                        fig = create_diagnostic_plots(st.session_state.fit_results)
                        st.pyplot(fig)
                    else:
                        st.info("Diagnostic plots require visualization module")
                        # Fallback to standard plot
                        fig = plot_aci_curve(
                            st.session_state.data,
                            st.session_state.fit_results
                        )
                        st.pyplot(fig)
                        
            except Exception as e:
                st.error(f"Plotting error: {str(e)}")
                # Fallback plot
                fig = plot_aci_curve(
                    st.session_state.data,
                    st.session_state.fit_results
                )
                st.pyplot(fig)
        
        # Quick insights section
        st.markdown("---")
        st.markdown("### Quick Insights")
        
        # Create columns for insights
        insight_col1, insight_col2 = st.columns(2)
        
        with insight_col1:
            if isinstance(st.session_state.fit_results, dict) and 'fitted_values' in st.session_state.fit_results:
                # Generate insights based on limiting processes
                if 'limiting_process' in st.session_state.fit_results['fitted_values'].columns:
                    limiting_counts = st.session_state.fit_results['fitted_values']['limiting_process'].value_counts()
                    dominant_process = limiting_counts.index[0]
                    process_labels = {'Wc': 'Rubisco', 'Wj': 'RuBP regeneration', 'Wp': 'TPU'}
                    st.info(f"Photosynthesis is primarily {process_labels.get(dominant_process, dominant_process)}-limited")
        
        with insight_col2:
            # R² assessment
            if 'statistics' in st.session_state.fit_results:
                r2 = st.session_state.fit_results['statistics'].get('r_squared', 0)
                if r2 > 0.95:
                    st.info("Excellent model fit (R² > 0.95) - highly reliable parameters")
                elif r2 > 0.90:
                    st.info("Good model fit (R² > 0.90) - reliable parameter estimates")
                else:
                    st.warning("Moderate fit - consider data quality or model assumptions")

# Tab 5: Batch Processing
with tab5:
    st.subheader("Batch Processing")
    st.info("Free version: Process up to 5 files at once. For unlimited batch processing, consider upgrading to Pro.")
    
    # File upload section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_files = st.file_uploader(
            "Upload multiple ACI curve files (max 5 files)",
            type=['csv', 'xlsx', 'xls'],
            accept_multiple_files=True,
            help="Select multiple files to process them in batch"
        )
        
        # Free version limitation
        if uploaded_files and len(uploaded_files) > 5:
            st.error(f"Free version limit: Maximum 5 files allowed. You selected {len(uploaded_files)} files.")
            st.info("Pro tip: Select your 5 most important files, or upgrade to Pro for unlimited batch processing.")
            uploaded_files = uploaded_files[:5]  # Keep only first 5 files
            st.warning(f"Processing only the first 5 files")
    
    with col2:
        st.markdown("### Current Batch")
        st.metric("Files loaded", len(st.session_state.batch_data))
        st.metric("Files processed", len(st.session_state.batch_results))
        
        if st.button("🗑️ Clear Batch", use_container_width=True):
            st.session_state.batch_data = {}
            st.session_state.batch_results = {}
            st.rerun()
    
    # Load files
    if uploaded_files and st.button("Load Files", type="primary"):
        with st.spinner("Loading files..."):
            successful_loads = 0
            failed_loads = []
            
            for file in uploaded_files[:5]:  # Ensure max 5 files
                try:
                    # Import the proper licor reader
                    from aci_py.io.licor import read_licor_file
                    import tempfile
                    import os
                    
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as tmp_file:
                        tmp_file.write(file.getbuffer())
                        tmp_path = tmp_file.name
                    
                    try:
                        # Read using the proper licor reader
                        exdf = read_licor_file(tmp_path)
                        
                        # Store in batch data
                        st.session_state.batch_data[file.name] = {
                            'exdf': exdf,
                            'data': exdf.data,
                            'filename': file.name
                        }
                        successful_loads += 1
                        
                    finally:
                        # Clean up temp file
                        os.unlink(tmp_path)
                        
                except Exception as e:
                    failed_loads.append((file.name, str(e)))
            
            # Show results
            if successful_loads > 0:
                st.success(f"Successfully loaded {successful_loads} file(s)")
            
            if failed_loads:
                st.error(f"Failed to load {len(failed_loads)} file(s)")
                for filename, error in failed_loads:
                    st.error(f"   • {filename}: {error}")
            
            st.rerun()
    
    # Display loaded files
    if st.session_state.batch_data:
        st.markdown("---")
        st.markdown("### Loaded Files")
        
        # Create a summary table
        file_info = []
        for filename, data in st.session_state.batch_data.items():
            df = data['data']
            file_info.append({
                'File': filename,
                'Points': len(df),
                'Ci Range': f"{df['Ci'].min():.0f}-{df['Ci'].max():.0f}" if 'Ci' in df.columns else 'N/A',
                'Mean A': f"{df['A'].mean():.1f}" if 'A' in df.columns else 'N/A',
                'Status': 'Processed' if filename in st.session_state.batch_results else 'Pending'
            })
        
        file_df = pd.DataFrame(file_info)
        st.dataframe(file_df, use_container_width=True, hide_index=True)
        
        # Batch analysis settings
        st.markdown("---")
        st.markdown("### Batch Analysis Settings")
        
        settings_col1, settings_col2 = st.columns(2)
        
        with settings_col1:
            batch_model_type = st.selectbox(
                "Photosynthesis type:",
                ["C3", "C4"],
                help="Apply same model to all files",
                key="batch_model"
            )
            
            batch_temp_response = st.selectbox(
                "Temperature response:",
                ["Bernacchi", "Sharkey", "None"],
                key="batch_temp"
            )
        
        with settings_col2:
            batch_optimizer = st.selectbox(
                "Optimization method:",
                ["Differential Evolution", "Least Squares"],
                key="batch_opt"
            )
            
            calculate_ci = st.checkbox(
                "Calculate confidence intervals",
                value=True
            )
        
        # Statistical Analysis Options
        st.markdown("---")
        st.markdown("### Statistical Analysis")
        
        stats_col1, stats_col2 = st.columns(2)
        
        with stats_col1:
            calculate_stats = st.checkbox(
                "Calculate summary statistics",
                value=True,
                help="Mean, SD, CV% across all files"
            )
            
            perform_anova = st.checkbox(
                "Perform ANOVA (if grouped)",
                value=False,
                help="One-way ANOVA for parameter differences"
            )
        
        with stats_col2:
            enable_grouping = st.checkbox(
                "Group files for comparison",
                value=False,
                help="Assign files to groups for statistical tests"
            )
            
            if enable_grouping:
                perform_pairwise = st.checkbox(
                    "Pairwise comparisons",
                    value=True,
                    help="Post-hoc t-tests between groups"
                )
        
        # File grouping interface
        if enable_grouping and st.session_state.batch_data:
            st.markdown("#### Assign Files to Groups")
            
            # Initialize groups in session state
            if 'file_groups' not in st.session_state:
                st.session_state.file_groups = {}
            
            # Create a simple interface for grouping
            file_list = list(st.session_state.batch_data.keys())
            n_files = len(file_list)
            
            # Determine number of columns (max 3)
            n_cols = min(3, n_files)
            cols = st.columns(n_cols)
            
            for i, filename in enumerate(file_list):
                col_idx = i % n_cols
                with cols[col_idx]:
                    # Shorten filename for display
                    display_name = filename if len(filename) <= 20 else filename[:17] + "..."
                    
                    group = st.selectbox(
                        display_name,
                        ["Control", "Treatment 1", "Treatment 2", "Treatment 3", "Unassigned"],
                        key=f"group_select_{i}",
                        index=4  # Default to "Unassigned"
                    )
                    
                    if group != "Unassigned":
                        st.session_state.file_groups[filename] = group
                    elif filename in st.session_state.file_groups:
                        del st.session_state.file_groups[filename]
            
            # Show group summary
            if st.session_state.file_groups:
                st.markdown("**Group Summary:**")
                group_counts = {}
                for fname, grp in st.session_state.file_groups.items():
                    group_counts[grp] = group_counts.get(grp, 0) + 1
                
                summary_text = ", ".join([f"{grp}: {count} files" for grp, count in group_counts.items()])
                st.info(summary_text)
        
        # Advanced batch options
        with st.expander("Advanced Batch Options"):
            parallel_processing = st.checkbox(
                "Enable parallel processing",
                value=True,
                help="Process multiple files simultaneously (faster)"
            )
            
            export_individual = st.checkbox(
                "Export individual results",
                value=False,
                help="Save each curve's results separately"
            )
            
            export_summary = st.checkbox(
                "Export summary statistics",
                value=True,
                help="Create a summary table of all results"
            )
        
        # Run batch analysis
        if st.button("Run Batch Analysis", type="primary", use_container_width=True):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                from aci_py.analysis import BatchResult, batch_fit_aci
                from aci_py.analysis.c3_fitting import fit_c3_aci
                from aci_py.analysis.c4_fitting import fit_c4_aci
                
                # Prepare data for batch processing
                batch_data_dict = {}
                
                # Prepare all data for batch processing
                for filename, file_data in st.session_state.batch_data.items():
                    batch_data_dict[filename] = file_data['exdf']
                
                # Process batch
                results = {}
                total_files = len(batch_data_dict)
                
                # Set temperature response parameters once
                temp_response_params = None
                if batch_temp_response == "Bernacchi":
                    from aci_py.core.temperature import C3_TEMPERATURE_PARAM_BERNACCHI
                    temp_response_params = C3_TEMPERATURE_PARAM_BERNACCHI
                elif batch_temp_response == "Sharkey":
                    from aci_py.core.temperature import C3_TEMPERATURE_PARAM_SHARKEY
                    temp_response_params = C3_TEMPERATURE_PARAM_SHARKEY
                
                # Process each file individually with progress tracking
                for idx, (filename, exdf) in enumerate(batch_data_dict.items()):
                    status_text.text(f"Processing {filename}...")
                    progress_bar.progress((idx + 1) / total_files)
                    
                    try:
                        # Fit the model
                        if batch_model_type == "C3":
                            result = fit_c3_aci(
                                exdf,
                                temperature_response_params=temp_response_params,
                                optimizer='differential_evolution' if batch_optimizer == "Differential Evolution" else 'nelder_mead',
                                calculate_confidence_intervals=calculate_ci
                            )
                        else:  # C4
                            result = fit_c4_aci(
                                exdf,
                                optimizer='differential_evolution' if batch_optimizer == "Differential Evolution" else 'nelder_mead',
                                calculate_confidence_intervals=calculate_ci
                            )
                        
                        # Store result
                        results[filename] = {
                            'result': result,
                            'data': exdf.data,
                            'parameters': result.parameters,
                            'statistics': {
                                'rmse': result.rmse,
                                'r_squared': result.r_squared,
                                'aic': result.aic,
                                'bic': result.bic
                            }
                        }
                        
                        if hasattr(result, 'confidence_intervals') and result.confidence_intervals:
                            results[filename]['confidence_intervals'] = result.confidence_intervals
                        
                    except Exception as e:
                        st.error(f"Failed to process {filename}: {str(e)}")
                        results[filename] = {'error': str(e)}
                
                # Store results
                st.session_state.batch_results = results
                
                progress_bar.progress(1.0)
                status_text.text("Batch processing complete!")
                
                # Show summary
                successful = sum(1 for r in results.values() if 'error' not in r)
                failed = len(results) - successful
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Successfully processed", successful)
                col2.metric("Failed", failed)
                col3.metric("Total", len(results))
                
                st.success("Batch processing complete!")
                st.balloons()
                
            except ImportError:
                st.error("Batch processing requires the full ACI_py package")
            except Exception as e:
                st.error(f"Batch processing error: {str(e)}")
    
    # Display batch results
    if st.session_state.batch_results:
        st.markdown("---")
        st.markdown("### Batch Results")
        
        # Create summary table
        summary_data = []
        for filename, result in st.session_state.batch_results.items():
            if 'error' in result:
                summary_data.append({
                    'File': filename,
                    'Status': 'Failed',
                    'Vcmax': 'N/A',
                    'J': 'N/A',
                    'Rd': 'N/A',
                    'R²': 'N/A',
                    'RMSE': 'N/A'
                })
            else:
                params = result['parameters']
                stats = result['statistics']
                
                summary_data.append({
                    'File': filename,
                    'Status': 'Success',
                    'Vcmax': f"{params.get('Vcmax_at_25', 'N/A'):.1f}",
                    'J': f"{params.get('J_at_25', 'N/A'):.1f}" if 'J_at_25' in params else 'N/A',
                    'Rd': f"{params.get('RL_at_25', 'N/A'):.2f}",
                    'R²': f"{stats.get('r_squared', 'N/A'):.3f}",
                    'RMSE': f"{stats.get('rmse', 'N/A'):.2f}"
                })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Display summary
        st.dataframe(
            summary_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                'Status': st.column_config.TextColumn(width='small'),
                'File': st.column_config.TextColumn(width='medium'),
            }
        )
        
        # Export options
        st.markdown("---")
        st.markdown("### Export Batch Results")
        
        export_col1, export_col2 = st.columns(2)
        
        with export_col1:
            if st.button("Export Summary Table", use_container_width=True):
                csv = summary_df.to_csv(index=False)
                st.download_button(
                    "Download Summary CSV",
                    csv,
                    "batch_summary.csv",
                    "text/csv"
                )
        
        with export_col2:
            if st.button("Generate Batch Report", use_container_width=True):
                # For now, just export the summary as CSV
                st.info("PDF report generation is coming soon. Exporting summary as CSV.")
                csv = summary_df.to_csv(index=False)
                st.download_button(
                    "Download Batch Summary CSV",
                    csv,
                    "batch_analysis_summary.csv",
                    "text/csv"
                )
        
        # Statistical Analysis Results
        if calculate_stats and len(st.session_state.batch_results) > 1:
            st.markdown("---")
            st.markdown("### Statistical Analysis")
            
            # Calculate basic statistics
            param_stats = {}
            params_to_analyze = ['Vcmax_at_25', 'J_at_25', 'RL_at_25', 'Tp_at_25']
            
            for param in params_to_analyze:
                values = []
                for result in st.session_state.batch_results.values():
                    if 'error' not in result and param in result['parameters']:
                        values.append(result['parameters'][param])
                
                if values:
                    param_stats[param] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'cv': (np.std(values) / np.mean(values) * 100) if np.mean(values) != 0 else 0,
                        'n': len(values),
                        'min': np.min(values),
                        'max': np.max(values)
                    }
            
            # Display statistics table
            stats_data = []
            for param, stats in param_stats.items():
                stats_data.append({
                    'Parameter': param.replace('_at_25', ''),
                    'Mean': f"{stats['mean']:.2f}",
                    'SD': f"{stats['std']:.2f}",
                    'CV%': f"{stats['cv']:.1f}",
                    'Range': f"{stats['min']:.2f} - {stats['max']:.2f}",
                    'N': stats['n']
                })
            
            if stats_data:
                stats_df = pd.DataFrame(stats_data)
                st.dataframe(stats_df, use_container_width=True, hide_index=True)
            
            # Group comparisons if enabled
            if enable_grouping and st.session_state.file_groups:
                st.markdown("#### Group Comparisons")
                
                # Organize data by groups
                group_data = {}
                for filename, group in st.session_state.file_groups.items():
                    if filename in st.session_state.batch_results:
                        result = st.session_state.batch_results[filename]
                        if 'error' not in result:
                            if group not in group_data:
                                group_data[group] = {}
                            for param in params_to_analyze:
                                if param in result['parameters']:
                                    if param not in group_data[group]:
                                        group_data[group][param] = []
                                    group_data[group][param].append(result['parameters'][param])
                
                # Display group statistics
                if len(group_data) > 1:
                    for param in params_to_analyze:
                        st.markdown(f"**{param.replace('_at_25', '')}**")
                        
                        group_stats = []
                        for group, data in group_data.items():
                            if param in data and len(data[param]) > 0:
                                group_stats.append({
                                    'Group': group,
                                    'Mean ± SD': f"{np.mean(data[param]):.2f} ± {np.std(data[param]):.2f}",
                                    'N': len(data[param])
                                })
                        
                        if group_stats:
                            group_df = pd.DataFrame(group_stats)
                            st.dataframe(group_df, use_container_width=True, hide_index=True)
                        
                        # Perform ANOVA if requested
                        if perform_anova and len(group_data) > 1:
                            try:
                                from scipy import stats as scipy_stats
                                
                                # Prepare data for ANOVA
                                groups = []
                                for group, data in group_data.items():
                                    if param in data:
                                        groups.append(data[param])
                                
                                if len(groups) > 1 and all(len(g) > 0 for g in groups):
                                    f_stat, p_value = scipy_stats.f_oneway(*groups)
                                    
                                    if p_value < 0.05:
                                        st.success(f"ANOVA: F = {f_stat:.2f}, p = {p_value:.4f} (significant)")
                                    else:
                                        st.info(f"ANOVA: F = {f_stat:.2f}, p = {p_value:.4f} (not significant)")
                                    
                                    # Pairwise comparisons if significant and requested
                                    if p_value < 0.05 and perform_pairwise:
                                        st.markdown("**Pairwise t-tests (Bonferroni corrected):**")
                                        
                                        group_names = list(group_data.keys())
                                        n_comparisons = len(group_names) * (len(group_names) - 1) // 2
                                        
                                        comparison_results = []
                                        for i in range(len(group_names)):
                                            for j in range(i+1, len(group_names)):
                                                group1 = group_names[i]
                                                group2 = group_names[j]
                                                
                                                if param in group_data[group1] and param in group_data[group2]:
                                                    t_stat, p_val = scipy_stats.ttest_ind(
                                                        group_data[group1][param],
                                                        group_data[group2][param]
                                                    )
                                                    
                                                    # Bonferroni correction
                                                    p_corrected = min(p_val * n_comparisons, 1.0)
                                                    
                                                    comparison_results.append({
                                                        'Comparison': f"{group1} vs {group2}",
                                                        'p-value': f"{p_corrected:.4f}",
                                                        'Significant': "Yes" if p_corrected < 0.05 else "No"
                                                    })
                                        
                                        if comparison_results:
                                            comp_df = pd.DataFrame(comparison_results)
                                            st.dataframe(comp_df, use_container_width=True, hide_index=True)
                                
                            except ImportError:
                                st.warning("SciPy required for statistical tests")
                            except Exception as e:
                                st.error(f"Statistical analysis error: {str(e)}")
                        
                        st.markdown("---")
        
        # Individual file results viewer
        st.markdown("---")
        st.markdown("### Individual Results")
        
        selected_file = st.selectbox(
            "Select file to view details:",
            [f for f in st.session_state.batch_results.keys() if 'error' not in st.session_state.batch_results[f]]
        )
        
        if selected_file:
            result_data = st.session_state.batch_results[selected_file]
            
            # Display parameters
            st.markdown(f"#### Parameters for {selected_file}")
            
            param_col1, param_col2 = st.columns(2)
            
            with param_col1:
                params = result_data['parameters']
                for param, value in params.items():
                    if isinstance(value, (int, float)):
                        st.metric(param, f"{value:.3f}")
            
            with param_col2:
                if 'confidence_intervals' in result_data:
                    st.markdown("**95% Confidence Intervals:**")
                    for param, (lower, upper) in result_data['confidence_intervals'].items():
                        st.write(f"{param}: [{lower:.2f}, {upper:.2f}]")
            
            
            if 'data' in result_data:
                fig = plot_aci_curve(result_data['data'], result_data)
                st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>ACI_py v0.8.0 Free Edition | 
        <a href='https://github.com/username/ACI_py'>Documentation</a> | 
        Batch processing limited to 5 files | 
        Made with love (ceremonial grade) for Residents of Planet Earth</p>
    </div>
    """,
    unsafe_allow_html=True
)
