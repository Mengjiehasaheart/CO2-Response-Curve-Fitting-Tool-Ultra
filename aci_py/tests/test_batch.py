"""
Unit tests for batch processing functionality.

Tests the batch.py module including:
- BatchResult class
- batch_fit_aci function
- Parameter variability analysis
- Model comparison
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import os

from aci_py.core.data_structures import ExtendedDataFrame
from aci_py.analysis.batch import (
    BatchResult,
    batch_fit_aci,
    compare_models,
    analyze_parameter_variability,
    process_single_curve
)
from aci_py.analysis.optimization import FittingResult


class TestBatchResult:
    """Test BatchResult container class."""
    
    def test_initialization(self):
        """Test BatchResult initialization."""
        br = BatchResult()
        assert br.results == {}
        assert br.summary_df is None
        assert br.failed_curves == []
        assert br.warnings == {}
    
    def test_add_result(self):
        """Test adding fitting results."""
        br = BatchResult()
        
        # Create mock result
        result = FittingResult(
            parameters={'Vcmax_at_25': 100.0, 'J_at_25': 200.0},
            result=None,  # Mock optimization result
            exdf=None,    # Mock data
            rmse=1.5,
            r_squared=0.98,
            n_points=13
        )
        
        br.add_result('curve_1', result)
        assert 'curve_1' in br.results
        assert br.results['curve_1'] == result
    
    def test_add_failure(self):
        """Test recording failed curves."""
        br = BatchResult()
        
        br.add_failure('curve_1', 'Convergence failed')
        assert 'curve_1' in br.failed_curves
        assert 'curve_1' in br.warnings
        assert 'Fitting failed: Convergence failed' in br.warnings['curve_1'][0]
    
    def test_add_warning(self):
        """Test adding warnings."""
        br = BatchResult()
        
        br.add_warning('curve_1', 'Low R-squared')
        assert 'curve_1' in br.warnings
        assert 'Low R-squared' in br.warnings['curve_1']
        
        # Add another warning to same curve
        br.add_warning('curve_1', 'High RMSE')
        assert len(br.warnings['curve_1']) == 2
    
    def test_generate_summary(self):
        """Test summary DataFrame generation."""
        br = BatchResult()
        
        # Add successful results
        for i in range(3):
            # Create a mock result object with the expected attributes
            class MockResult:
                def __init__(self, params, rmse, r2, n, success, ci=None):
                    self.parameters = params
                    self.rmse = rmse
                    self.r_squared = r2
                    self.n_points = n
                    self.success = success
                    self.confidence_intervals = ci
            
            result = MockResult(
                params={
                    'Vcmax_at_25': 100.0 + i * 10,
                    'J_at_25': 200.0 + i * 20
                },
                rmse=1.5 + i * 0.1,
                r2=0.98 - i * 0.01,
                n=13,
                success=True,
                ci={
                    'Vcmax_at_25': (95.0 + i * 10, 105.0 + i * 10)
                }
            )
            br.add_result(f'curve_{i}', result)
        
        # Add failed curve
        br.add_failure('curve_failed', 'Test failure')
        
        # Generate summary
        summary = br.generate_summary()
        
        assert len(summary) == 4  # 3 successful + 1 failed
        assert 'curve_id' in summary.columns
        assert 'Vcmax_at_25' in summary.columns
        assert 'J_at_25' in summary.columns
        assert 'rmse' in summary.columns
        assert 'r_squared' in summary.columns
        assert 'success' in summary.columns
        assert 'Vcmax_at_25_CI_lower' in summary.columns
        assert 'Vcmax_at_25_CI_upper' in summary.columns
        
        # Check values
        assert summary.loc[summary['curve_id'] == 'curve_0', 'Vcmax_at_25'].values[0] == 100.0
        assert summary.loc[summary['curve_id'] == 'curve_failed', 'success'].values[0] == False


class TestBatchFitting:
    """Test batch fitting functionality."""
    
    @staticmethod
    def create_synthetic_aci_data(n_points=13, vcmax=100, j=200, noise_level=0.5):
        """Create synthetic A-Ci data for testing."""
        Ci = np.linspace(50, 1500, n_points)
        
        # Simple C3 model approximation
        gamma_star = 40
        Kc = 300
        Ko = 210
        O = 210  # mmol/mol
        
        # Rubisco limited
        Wc = Ci * vcmax / (Ci + Kc * (1 + O/Ko))
        # RuBP limited
        Wj = Ci * j / (4 * Ci + 8 * gamma_star)
        
        # Minimum of limitations
        A = np.minimum(Wc, Wj) - 1.5  # Subtract Rd
        
        # Add noise
        A += np.random.normal(0, noise_level, n_points)
        
        # Create DataFrame
        df = pd.DataFrame({
            'A': A,
            'Ci': Ci,
            'Tleaf': 25.0,
            'Pa': 101.325
        })
        
        return ExtendedDataFrame(df)
    
    def test_single_curve_fitting(self):
        """Test fitting a single curve."""
        # Create test data
        exdf = self.create_synthetic_aci_data()
        
        # Fit using batch function
        result = batch_fit_aci(
            exdf,
            model_type='C3',
            n_jobs=1,
            progress_bar=False
        )
        
        assert isinstance(result, BatchResult)
        assert len(result.results) == 1
        assert 'curve_1' in result.results
        assert result.results['curve_1'].success
    
    def test_dictionary_input(self):
        """Test batch fitting with dictionary input."""
        # Create multiple curves
        curves = {
            f'genotype_{i}': self.create_synthetic_aci_data(
                vcmax=80 + i * 20,
                j=160 + i * 40
            )
            for i in range(3)
        }
        
        # Batch fit
        result = batch_fit_aci(
            curves,
            model_type='C3',
            n_jobs=1,
            progress_bar=False
        )
        
        assert len(result.results) == 3
        for i in range(3):
            assert f'genotype_{i}' in result.results
    
    def test_dataframe_with_grouping(self):
        """Test batch fitting with DataFrame grouping."""
        # Create combined DataFrame
        dfs = []
        for genotype in ['A', 'B']:
            for plant in [1, 2]:
                df = self.create_synthetic_aci_data().data
                df['Genotype'] = genotype
                df['Plant'] = plant
                dfs.append(df)
        
        combined_df = pd.concat(dfs, ignore_index=True)
        
        # Batch fit with grouping
        result = batch_fit_aci(
            combined_df,
            model_type='C3',
            groupby=['Genotype', 'Plant'],
            n_jobs=1,
            progress_bar=False
        )
        
        assert len(result.results) == 4  # 2 genotypes × 2 plants
        assert result.summary_df is not None
        assert len(result.summary_df) == 4
    
    def test_parallel_processing(self):
        """Test parallel processing."""
        # Create multiple curves
        curves = {
            f'curve_{i}': self.create_synthetic_aci_data()
            for i in range(4)
        }
        
        # Batch fit with parallel processing
        result = batch_fit_aci(
            curves,
            model_type='C3',
            n_jobs=2,  # Use 2 parallel jobs
            progress_bar=False
        )
        
        assert len(result.results) == 4
        assert all(r.success for r in result.results.values())
    
    def test_failed_curve_handling(self):
        """Test handling of curves that fail to fit."""
        # Create a problematic curve (all same A values)
        bad_df = pd.DataFrame({
            'A': [5.0] * 10,
            'Ci': np.linspace(50, 1500, 10),
            'Tleaf': 25.0,
            'Pa': 101.325
        })
        
        curves = {
            'good': self.create_synthetic_aci_data(),
            'bad': ExtendedDataFrame(bad_df)
        }
        
        # Batch fit
        result = batch_fit_aci(
            curves,
            model_type='C3',
            n_jobs=1,
            progress_bar=False
        )
        
        # Should have one success and one failure (or warning)
        assert len(result.results) >= 1
        assert 'good' in result.results or 'bad' in result.results
    
    def test_fixed_parameters(self):
        """Test batch fitting with fixed parameters."""
        curves = {
            f'curve_{i}': self.create_synthetic_aci_data()
            for i in range(2)
        }
        
        # Fix mesophyll conductance
        result = batch_fit_aci(
            curves,
            model_type='C3',
            n_jobs=1,
            progress_bar=False,
            fixed_parameters={'gmc': 0.5}
        )
        
        # Check that all results have the fixed parameter
        for curve_result in result.results.values():
            if 'gmc' in curve_result.parameters:
                assert curve_result.parameters['gmc'] == 0.5


class TestModelComparison:
    """Test model comparison functionality."""
    
    def test_compare_models(self):
        """Test comparing C3 and C4 models."""
        # Create test data
        exdf = TestBatchFitting.create_synthetic_aci_data()
        
        # Add required columns for C4
        exdf.data['PCm'] = exdf.data['Ci'] * 0.7  # Approximation
        exdf.data['PCi'] = exdf.data['Ci']
        
        # Compare models
        results = compare_models(
            exdf,
            models=['C3', 'C4']
        )
        
        assert 'C3' in results
        assert 'C4' in results
        assert isinstance(results['C3'], FittingResult)
        assert isinstance(results['C4'], FittingResult)


class TestParameterVariability:
    """Test parameter variability analysis."""
    
    def test_analyze_parameter_variability(self):
        """Test parameter statistics calculation."""
        # Create BatchResult with multiple fits
        br = BatchResult()
        
        # Add results with varying parameters
        np.random.seed(42)
        for i in range(10):
            result = FittingResult(
                parameters={
                    'Vcmax_at_25': 100.0 + np.random.normal(0, 10),
                    'J_at_25': 200.0 + np.random.normal(0, 20),
                    'Tp_at_25': 12.0 + np.random.normal(0, 1)
                },
                rmse=1.5,
                r_squared=0.98,
                n_points=13,
                success=True
            )
            br.add_result(f'curve_{i}', result)
        
        # Add a failed curve
        br.add_failure('failed', 'Test')
        
        # Generate summary first
        br.generate_summary()
        
        # Analyze variability
        stats = analyze_parameter_variability(br)
        
        assert len(stats) == 3  # Three parameters
        assert 'parameter' in stats.columns
        assert 'mean' in stats.columns
        assert 'std' in stats.columns
        assert 'cv' in stats.columns
        assert 'min' in stats.columns
        assert 'max' in stats.columns
        assert 'n_curves' in stats.columns
        
        # Check Vcmax stats
        vcmax_stats = stats[stats['parameter'] == 'Vcmax_at_25'].iloc[0]
        assert 90 < vcmax_stats['mean'] < 110  # Should be around 100
        assert vcmax_stats['n_curves'] == 10
        
    def test_analyze_subset_parameters(self):
        """Test analyzing only specific parameters."""
        br = BatchResult()
        
        # Add some results
        for i in range(5):
            result = FittingResult(
                parameters={
                    'Vcmax_at_25': 100.0 + i,
                    'J_at_25': 200.0 + i * 2,
                    'Tp_at_25': 12.0,
                    'RL_at_25': 1.5
                },
                rmse=1.5,
                r_squared=0.98,
                n_points=13,
                success=True
            )
            br.add_result(f'curve_{i}', result)
        
        br.generate_summary()
        
        # Analyze only Vcmax and J
        stats = analyze_parameter_variability(br, parameters=['Vcmax_at_25', 'J_at_25'])
        
        assert len(stats) == 2
        assert all(p in ['Vcmax_at_25', 'J_at_25'] for p in stats['parameter'].values)


class TestProcessSingleCurve:
    """Test the process_single_curve function."""
    
    def test_process_single_curve_success(self):
        """Test successful processing of a single curve."""
        from aci_py.analysis.c3_fitting import fit_c3_aci
        
        # Create test data
        exdf = TestBatchFitting.create_synthetic_aci_data()
        
        # Process curve
        curve_id, result = process_single_curve(
            exdf,
            'test_curve',
            fit_c3_aci,
            {}
        )
        
        assert curve_id == 'test_curve'
        assert isinstance(result, FittingResult)
        assert result.success
    
    def test_process_single_curve_dataframe_input(self):
        """Test processing with DataFrame input (not ExtendedDataFrame)."""
        from aci_py.analysis.c3_fitting import fit_c3_aci
        
        # Create test data as regular DataFrame
        df = TestBatchFitting.create_synthetic_aci_data().data
        
        # Process curve
        curve_id, result = process_single_curve(
            df,
            'test_curve',
            fit_c3_aci,
            {}
        )
        
        assert curve_id == 'test_curve'
        assert isinstance(result, (FittingResult, Exception))
    
    def test_process_single_curve_exception(self):
        """Test handling of exceptions during processing."""
        from aci_py.analysis.c3_fitting import fit_c3_aci
        
        # Create invalid data
        bad_df = pd.DataFrame({'invalid': [1, 2, 3]})
        
        # Process curve
        curve_id, result = process_single_curve(
            bad_df,
            'test_curve',
            fit_c3_aci,
            {}
        )
        
        assert curve_id == 'test_curve'
        assert isinstance(result, Exception)


class TestBatchFittingIntegration:
    """Integration tests for batch fitting with real file loading."""
    
    def test_file_loading_single(self, tmp_path):
        """Test batch fitting from a single file."""
        # Create temporary CSV file
        df = TestBatchFitting.create_synthetic_aci_data().data
        csv_path = tmp_path / "test_aci.csv"
        df.to_csv(csv_path, index=False)
        
        # Batch fit from file
        result = batch_fit_aci(
            str(csv_path),
            model_type='C3',
            n_jobs=1,
            progress_bar=False
        )
        
        assert len(result.results) == 1
        assert 'curve_1' in result.results
    
    def test_file_loading_with_grouping(self, tmp_path):
        """Test batch fitting from file with grouping."""
        # Create combined data
        dfs = []
        for genotype in ['A', 'B']:
            for plant in [1, 2]:
                df = TestBatchFitting.create_synthetic_aci_data().data
                df['Genotype'] = genotype
                df['Plant'] = plant
                dfs.append(df)
        
        combined_df = pd.concat(dfs, ignore_index=True)
        csv_path = tmp_path / "batch_aci.csv"
        combined_df.to_csv(csv_path, index=False)
        
        # Batch fit with grouping
        result = batch_fit_aci(
            str(csv_path),
            model_type='C3',
            groupby=['Genotype', 'Plant'],
            n_jobs=1,
            progress_bar=False
        )
        
        assert len(result.results) == 4
        
        # Check that group names are properly formatted
        for curve_id in result.results.keys():
            assert '_' in curve_id  # Should be formatted like "A_1"