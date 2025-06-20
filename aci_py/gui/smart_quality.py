import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings


@dataclass
class QualityIssue:
    """Represents a data quality issue with recommendations."""
    severity: str  # 'critical', 'warning', 'info'
    category: str  # 'data_coverage', 'environmental', 'measurement', 'outlier'
    message: str
    recommendation: str
    auto_fixable: bool = False
    fix_function: Optional[callable] = None
    confidence: float = 1.0  # Confidence in the issue detection (0-1)


class SmartQualityChecker:
    """Advanced data quality assessment with ML-based insights."""
    
    def __init__(self):
        """Initialize the quality checker with known parameter ranges."""
        # Typical parameter ranges for common species
        self.parameter_ranges = {
            'C3': {
                'Vcmax': (10, 200),  # µmol m⁻² s⁻¹
                'J': (20, 400),
                'Tp': (5, 50),
                'Rd': (0.5, 5),
                'A_max': (5, 50),
                'Ci_comp': (30, 100)  # Compensation point
            },
            'C4': {
                'Vcmax': (20, 100),
                'Vpmax': (40, 200),
                'J': (50, 500),
                'Rd': (0.5, 5),
                'A_max': (20, 80),
                'Ci_comp': (0, 10)
            }
        }
        
        # Environmental tolerances
        self.environmental_tolerances = {
            'temperature_drift': 1.0,  # °C
            'rh_drift': 5.0,  # %
            'pressure_drift': 2.0,  # kPa
            'flow_drift': 50.0  # µmol/s
        }
    
    def assess_comprehensive(self, data: pd.DataFrame, 
                           species_type: Optional[str] = None,
                           measurement_conditions: Optional[Dict] = None) -> Dict:
        """
        Perform comprehensive data quality assessment.
        
        Parameters
        ----------
        data : pd.DataFrame
            The ACI curve data
        species_type : str, optional
            'C3' or 'C4' if known
        measurement_conditions : dict, optional
            Expected measurement conditions
            
        Returns
        -------
        dict
            Comprehensive quality report with issues, score, and recommendations
        """
        issues = []
        
        # 1. Data Coverage Assessment
        issues.extend(self._check_data_coverage(data))
        
        # 2. Environmental Stability
        issues.extend(self._check_environmental_stability(data))
        
        # 3. Measurement Quality
        issues.extend(self._check_measurement_quality(data))
        
        # 4. Outlier Detection
        issues.extend(self._check_outliers(data))
        
        # 5. Curve Shape Analysis
        if species_type:
            issues.extend(self._check_curve_shape(data, species_type))
        else:
            # Try to infer species type
            inferred_type = self._infer_species_type(data)
            if inferred_type:
                issues.append(QualityIssue(
                    severity='info',
                    category='data_coverage',
                    message=f'Inferred species type: {inferred_type}',
                    recommendation='Verify this is correct for accurate analysis',
                    confidence=0.8
                ))
                issues.extend(self._check_curve_shape(data, inferred_type))
        
        # 6. Technical Issues
        issues.extend(self._check_technical_issues(data))
        
        # Calculate overall score
        score = self._calculate_quality_score(issues)
        
        # Generate smart recommendations
        recommendations = self._generate_recommendations(issues, data)
        
        # Determine overall badge
        badge = self._determine_badge(score, issues)
        
        return {
            'score': score,
            'badge': badge,
            'issues': issues,
            'recommendations': recommendations,
            'auto_fixes_available': sum(1 for i in issues if i.auto_fixable),
            'summary': self._generate_summary(issues, score)
        }
    
    def _check_data_coverage(self, data: pd.DataFrame) -> List[QualityIssue]:
        """Check data coverage and point distribution."""
        issues = []
        
        if 'Ci' not in data.columns:
            issues.append(QualityIssue(
                severity='critical',
                category='data_coverage',
                message='Missing Ci column',
                recommendation='Ensure CO2 data is properly labeled'
            ))
            return issues
        
        # Number of points
        n_points = len(data)
        if n_points < 5:
            issues.append(QualityIssue(
                severity='critical',
                category='data_coverage',
                message=f'Only {n_points} data points (minimum 5 required)',
                recommendation='Collect more measurements for reliable fitting'
            ))
        elif n_points < 8:
            issues.append(QualityIssue(
                severity='warning',
                category='data_coverage',
                message=f'Limited data points ({n_points})',
                recommendation='8-12 points recommended for robust parameter estimation'
            ))
        
        # Ci range
        ci_min, ci_max = data['Ci'].min(), data['Ci'].max()
        ci_range = ci_max - ci_min
        
        if ci_range < 300:
            issues.append(QualityIssue(
                severity='critical',
                category='data_coverage',
                message=f'Narrow Ci range ({ci_range:.0f} ppm)',
                recommendation='Extend measurements to at least 50-1500 ppm range',
                confidence=1.0
            ))
        elif ci_range < 800:
            issues.append(QualityIssue(
                severity='warning',
                category='data_coverage',
                message=f'Limited Ci range ({ci_range:.0f} ppm)',
                recommendation='Consider extending to 1500-2000 ppm for better Vcmax estimation'
            ))
        
        # Check for low Ci points
        low_ci_points = len(data[data['Ci'] < 100])
        if low_ci_points == 0:
            issues.append(QualityIssue(
                severity='warning',
                category='data_coverage',
                message='No low Ci measurements (<100 ppm)',
                recommendation='Add 2-3 points below 100 ppm for accurate Rd estimation',
                confidence=0.9
            ))
        
        # Check for high Ci points
        high_ci_points = len(data[data['Ci'] > 800])
        if high_ci_points == 0:
            issues.append(QualityIssue(
                severity='warning',
                category='data_coverage',
                message='No high Ci measurements (>800 ppm)',
                recommendation='Add points above 800 ppm to capture CO2 saturation'
            ))
        
        # Check point distribution
        ci_sorted = np.sort(data['Ci'].values)
        if len(ci_sorted) > 3:
            gaps = np.diff(ci_sorted)
            max_gap = gaps.max()
            median_gap = np.median(gaps)
            
            if max_gap > 5 * median_gap:
                gap_start = ci_sorted[np.argmax(gaps)]
                gap_end = ci_sorted[np.argmax(gaps) + 1]
                issues.append(QualityIssue(
                    severity='info',
                    category='data_coverage',
                    message=f'Large gap in Ci coverage ({gap_start:.0f}-{gap_end:.0f} ppm)',
                    recommendation='Consider adding measurements in this range',
                    confidence=0.8
                ))
        
        return issues
    
    def _check_environmental_stability(self, data: pd.DataFrame) -> List[QualityIssue]:
        """Check environmental parameter stability."""
        issues = []
        
        # Temperature stability
        if 'Tleaf' in data.columns:
            temp_range = data['Tleaf'].max() - data['Tleaf'].min()
            temp_std = data['Tleaf'].std()
            
            if temp_range > 2.0:
                issues.append(QualityIssue(
                    severity='critical',
                    category='environmental',
                    message=f'Large temperature variation ({temp_range:.1f}°C)',
                    recommendation='Ensure temperature control is working properly',
                    auto_fixable=False
                ))
            elif temp_range > 1.0:
                issues.append(QualityIssue(
                    severity='warning',
                    category='environmental',
                    message=f'Temperature drift detected ({temp_range:.1f}°C)',
                    recommendation='Allow more equilibration time between measurements'
                ))
            
            # Check for temperature trends
            if len(data) > 5:
                temp_correlation = np.abs(np.corrcoef(np.arange(len(data)), data['Tleaf'])[0, 1])
                if temp_correlation > 0.7:
                    issues.append(QualityIssue(
                        severity='warning',
                        category='environmental',
                        message='Systematic temperature drift detected',
                        recommendation='Check chamber temperature control',
                        confidence=0.85
                    ))
        
        # RH stability
        if 'RHcham' in data.columns:
            rh_range = data['RHcham'].max() - data['RHcham'].min()
            if rh_range > 10:
                issues.append(QualityIssue(
                    severity='warning',
                    category='environmental',
                    message=f'Variable humidity ({rh_range:.0f}% range)',
                    recommendation='Consider using a desiccant or humidifier for stability'
                ))
        
        # Pressure stability
        if 'Pa' in data.columns:
            pa_range = data['Pa'].max() - data['Pa'].min()
            if pa_range > 2.0:
                issues.append(QualityIssue(
                    severity='info',
                    category='environmental',
                    message=f'Atmospheric pressure variation ({pa_range:.1f} kPa)',
                    recommendation='Normal for extended measurements',
                    confidence=0.7
                ))
        
        return issues
    
    def _check_measurement_quality(self, data: pd.DataFrame) -> List[QualityIssue]:
        """Check measurement quality indicators."""
        issues = []
        
        if 'A' not in data.columns:
            return issues
        
        # Check for negative assimilation at high Ci
        high_ci_mask = data['Ci'] > 500 if 'Ci' in data.columns else np.ones(len(data), dtype=bool)
        negative_high_ci = data.loc[high_ci_mask, 'A'] < -2
        
        if negative_high_ci.any():
            n_negative = negative_high_ci.sum()
            issues.append(QualityIssue(
                severity='critical',
                category='measurement',
                message=f'{n_negative} negative A values at high Ci',
                recommendation='Check for leaks or measurement artifacts',
                auto_fixable=True,
                fix_function=lambda d: d[d['A'] > -2]  # Remove strongly negative values
            ))
        
        # Check for unrealistic A values
        if data['A'].max() > 100:
            issues.append(QualityIssue(
                severity='warning',
                category='measurement',
                message=f'Unrealistically high A values (max: {data["A"].max():.1f})',
                recommendation='Check calibration and units'
            ))
        
        # Check stomatal conductance if available
        if 'gsw' in data.columns:
            if (data['gsw'] < 0.01).any():
                issues.append(QualityIssue(
                    severity='warning',
                    category='measurement',
                    message='Very low stomatal conductance detected',
                    recommendation='Ensure stomata are open (check light, humidity)'
                ))
        
        # Check measurement stability (coefficient of variation)
        if 'A' in data.columns and len(data) > 3:
            # Group similar Ci values and check A variation
            ci_bins = pd.cut(data['Ci'], bins=5)
            for bin_label, group in data.groupby(ci_bins):
                if len(group) > 1:
                    cv = group['A'].std() / group['A'].mean() if group['A'].mean() != 0 else 0
                    if cv > 0.2:  # 20% variation
                        ci_range = f"{group['Ci'].min():.0f}-{group['Ci'].max():.0f}"
                        issues.append(QualityIssue(
                            severity='info',
                            category='measurement',
                            message=f'High variability in A at Ci {ci_range} ppm',
                            recommendation='Consider longer equilibration time',
                            confidence=0.7
                        ))
        
        return issues
    
    def _check_outliers(self, data: pd.DataFrame) -> List[QualityIssue]:
        """Detect outliers using multiple methods."""
        issues = []
        
        if 'A' not in data.columns or 'Ci' not in data.columns:
            return issues
        
        # Method 1: Modified Z-score using MAD
        from scipy import stats
        
        def modified_z_score(x):
            median = np.median(x)
            mad = np.median(np.abs(x - median))
            modified_z = 0.6745 * (x - median) / mad if mad > 0 else np.zeros_like(x)
            return np.abs(modified_z)
        
        a_outliers = modified_z_score(data['A'].values) > 3.5
        
        if a_outliers.any():
            n_outliers = a_outliers.sum()
            outlier_indices = np.where(a_outliers)[0]
            issues.append(QualityIssue(
                severity='warning',
                category='outlier',
                message=f'{n_outliers} potential outliers detected in A',
                recommendation=f'Review points at indices: {outlier_indices.tolist()}',
                auto_fixable=True,
                fix_function=lambda d: d[~a_outliers],
                confidence=0.8
            ))
        
        # Method 2: Check for non-monotonic behavior at low Ci
        low_ci_data = data[data['Ci'] < 200].sort_values('Ci')
        if len(low_ci_data) > 2:
            # A should generally increase with Ci at low values
            a_diff = np.diff(low_ci_data['A'].values)
            ci_diff = np.diff(low_ci_data['Ci'].values)
            
            # Check for significant decreases
            decreases = (a_diff < -1) & (ci_diff > 0)
            if decreases.any():
                issues.append(QualityIssue(
                    severity='info',
                    category='outlier',
                    message='Non-monotonic behavior at low Ci',
                    recommendation='Check for measurement errors or stomatal issues',
                    confidence=0.7
                ))
        
        return issues
    
    def _check_curve_shape(self, data: pd.DataFrame, species_type: str) -> List[QualityIssue]:
        """Analyze curve shape for species-specific patterns."""
        issues = []
        
        if 'A' not in data.columns or 'Ci' not in data.columns:
            return issues
        
        # Sort by Ci for analysis
        sorted_data = data.sort_values('Ci')
        
        # Check if curve shows expected saturation
        high_ci_data = sorted_data[sorted_data['Ci'] > 800]
        if len(high_ci_data) > 2:
            # Calculate slope at high Ci
            ci_high = high_ci_data['Ci'].values
            a_high = high_ci_data['A'].values
            
            if len(ci_high) > 1:
                slope = np.polyfit(ci_high, a_high, 1)[0]
                
                # For saturating curve, slope should be small
                if slope > 0.01:
                    issues.append(QualityIssue(
                        severity='info',
                        category='measurement',
                        message='No clear CO2 saturation observed',
                        recommendation='Extend measurements to higher Ci or check for TPU limitation',
                        confidence=0.7
                    ))
        
        # Check compensation point
        if sorted_data['A'].min() < 0 and sorted_data['A'].max() > 0:
            # Estimate compensation point
            try:
                # Linear interpolation near zero
                near_zero = sorted_data[np.abs(sorted_data['A']) < 5]
                if len(near_zero) > 1:
                    ci_comp = np.interp(0, near_zero['A'], near_zero['Ci'])
                    
                    expected_range = self.parameter_ranges[species_type]['Ci_comp']
                    if not (expected_range[0] <= ci_comp <= expected_range[1]):
                        issues.append(QualityIssue(
                            severity='info',
                            category='measurement',
                            message=f'Unusual CO2 compensation point ({ci_comp:.0f} ppm)',
                            recommendation=f'Expected {expected_range[0]}-{expected_range[1]} ppm for {species_type}',
                            confidence=0.6
                        ))
            except:
                pass
        
        return issues
    
    def _check_technical_issues(self, data: pd.DataFrame) -> List[QualityIssue]:
        """Check for technical measurement issues."""
        issues = []
        
        # Check for duplicate Ci values
        if 'Ci' in data.columns:
            duplicates = data['Ci'].duplicated()
            if duplicates.any():
                n_dup = duplicates.sum()
                issues.append(QualityIssue(
                    severity='info',
                    category='measurement',
                    message=f'{n_dup} duplicate Ci values found',
                    recommendation='This is fine for replicates, but check if unintentional'
                ))
        
        # Check for data gaps (missing values)
        essential_cols = ['A', 'Ci', 'Tleaf']
        for col in essential_cols:
            if col in data.columns:
                n_missing = data[col].isna().sum()
                if n_missing > 0:
                    issues.append(QualityIssue(
                        severity='critical',
                        category='measurement',
                        message=f'{n_missing} missing values in {col}',
                        recommendation='Fill or remove incomplete measurements',
                        auto_fixable=True,
                        fix_function=lambda d: d.dropna(subset=[col])
                    ))
        
        return issues
    
    def _infer_species_type(self, data: pd.DataFrame) -> Optional[str]:
        """Try to infer C3 vs C4 from curve characteristics."""
        if 'A' not in data.columns or 'Ci' not in data.columns:
            return None
        
        # Simple heuristic based on compensation point and max A
        try:
            # Find approximate compensation point
            sorted_data = data.sort_values('Ci')
            if sorted_data['A'].min() < 0 and sorted_data['A'].max() > 0:
                near_zero = sorted_data[np.abs(sorted_data['A']) < 3]
                if len(near_zero) > 0:
                    ci_comp_estimate = near_zero['Ci'].mean()
                    
                    # C4 plants have very low compensation points
                    if ci_comp_estimate < 20:
                        return 'C4'
                    else:
                        return 'C3'
            
            # If no compensation point, use max A as hint
            max_a = data['A'].max()
            if max_a > 40:
                return 'C4'  # C4 plants often have higher max rates
            else:
                return 'C3'
                
        except:
            return None
    
    def _calculate_quality_score(self, issues: List[QualityIssue]) -> float:
        """Calculate overall quality score from issues."""
        score = 100.0
        
        # Deduct points based on severity
        severity_penalties = {
            'critical': 25,
            'warning': 10,
            'info': 3
        }
        
        for issue in issues:
            penalty = severity_penalties.get(issue.severity, 0)
            # Adjust by confidence
            penalty *= issue.confidence
            score -= penalty
        
        return max(0, min(100, score))
    
    def _generate_recommendations(self, issues: List[QualityIssue], 
                                 data: pd.DataFrame) -> List[str]:
        """Generate prioritized recommendations."""
        recommendations = []
        
        # Group by severity
        critical_issues = [i for i in issues if i.severity == 'critical']
        warning_issues = [i for i in issues if i.severity == 'warning']
        
        # Top priority recommendations
        if critical_issues:
            recommendations.append("🚨 **Critical Issues to Address:**")
            for issue in critical_issues[:3]:  # Top 3
                recommendations.append(f"  • {issue.recommendation}")
        
        # Secondary recommendations
        if warning_issues and len(recommendations) < 5:
            recommendations.append("\n⚠️ **Suggested Improvements:**")
            for issue in warning_issues[:2]:
                recommendations.append(f"  • {issue.recommendation}")
        
        # Auto-fix suggestion
        auto_fixable = [i for i in issues if i.auto_fixable]
        if auto_fixable:
            recommendations.append(f"\n🔧 **{len(auto_fixable)} issues can be auto-fixed**")
        
        return recommendations
    
    def _determine_badge(self, score: float, issues: List[QualityIssue]) -> str:
        """Determine quality badge with emoji."""
        critical_count = sum(1 for i in issues if i.severity == 'critical')
        
        if critical_count > 0:
            return "🔴 Poor - Critical Issues"
        elif score >= 85:
            return "🟢 Excellent"
        elif score >= 70:
            return "🟢 Good"
        elif score >= 50:
            return "🟡 Fair"
        else:
            return "🔴 Poor"
    
    def _generate_summary(self, issues: List[QualityIssue], score: float) -> str:
        """Generate a natural language summary."""
        n_critical = sum(1 for i in issues if i.severity == 'critical')
        n_warning = sum(1 for i in issues if i.severity == 'warning')
        n_info = sum(1 for i in issues if i.severity == 'info')
        
        if score >= 85:
            summary = "Your data quality is excellent! "
        elif score >= 70:
            summary = "Your data quality is good. "
        elif score >= 50:
            summary = "Your data quality is fair. "
        else:
            summary = "Your data quality needs attention. "
        
        if n_critical > 0:
            summary += f"Found {n_critical} critical issue(s) that should be addressed. "
        if n_warning > 0:
            summary += f"{n_warning} warnings to consider. "
        if n_info > 0:
            summary += f"{n_info} informational notes. "
        
        return summary
    
    def apply_auto_fixes(self, data: pd.DataFrame, issues: List[QualityIssue]) -> pd.DataFrame:
        """Apply available auto-fixes to the data."""
        fixed_data = data.copy()
        fixes_applied = 0
        
        for issue in issues:
            if issue.auto_fixable and issue.fix_function:
                try:
                    fixed_data = issue.fix_function(fixed_data)
                    fixes_applied += 1
                except:
                    pass
        
        return fixed_data, fixes_applied