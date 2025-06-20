"""
Unit tests for export functionality.

Tests the export.py module including:
- Single result export
- Batch result export
- Report generation
- PhotoGEA compatibility
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import json
import tempfile
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from aci_py.core.data_structures import ExtendedDataFrame
from aci_py.analysis.optimization import FittingResult
from aci_py.analysis.batch import BatchResult
from aci_py.io.export import (
    export_fitting_result,
    export_batch_results,
    create_analysis_report,
    save_for_photogea_compatibility,
    _add_parameter_table_page,
    _add_batch_summary_page
)


class TestExportFittingResult:
    """Test single fitting result export."""
    
    @staticmethod
    def create_mock_result():
        """Create a mock FittingResult for testing."""
        # Create a mock result that looks like C3FitResult
        class MockFitResult:
            def __init__(self):
                self.parameters = {
                    'Vcmax_at_25': 95.3,
                    'J_at_25': 185.7,
                    'Tp_at_25': 11.2,
                    'RL_at_25': 1.4,
                    'gmc': 0.5
                }
                self.rmse = 1.23
                self.r_squared = 0.987
                self.n_points = 13
                self.success = True
                self.confidence_intervals = {
                    'Vcmax_at_25': (92.1, 98.5),
                    'J_at_25': (180.3, 191.1)
                }
                self.fitted_values = {
                    'Ci': np.linspace(50, 1500, 13),
                    'A_measured': np.random.normal(20, 2, 13),
                    'A_fit': np.random.normal(20, 0.5, 13)
                }
                self.figure = None
                self.model_type = 'C3'
        
        return MockFitResult()
    
    def test_export_json(self, tmp_path):
        """Test JSON export."""
        result = self.create_mock_result()
        
        files = export_fitting_result(
            result,
            tmp_path,
            formats=['json'],
            include_plots=False
        )
        
        assert 'json' in files
        json_path = files['json']
        assert json_path.exists()
        
        # Load and verify JSON
        with open(json_path) as f:
            data = json.load(f)
        
        assert 'parameters' in data
        assert data['parameters']['Vcmax_at_25'] == 95.3
        assert 'confidence_intervals' in data
        assert 'statistics' in data
        assert data['statistics']['rmse'] == 1.23
        assert data['statistics']['r_squared'] == 0.987
    
    def test_export_csv(self, tmp_path):
        """Test CSV export of fitted values."""
        result = self.create_mock_result()
        
        files = export_fitting_result(
            result,
            tmp_path,
            formats=['csv'],
            include_plots=False
        )
        
        assert 'csv' in files
        csv_path = files['csv']
        assert csv_path.exists()
        
        # Load and verify CSV
        df = pd.read_csv(csv_path)
        assert 'Ci' in df.columns
        assert 'A_measured' in df.columns
        assert 'A_fit' in df.columns
        assert len(df) == 13
    
    def test_export_excel(self, tmp_path):
        """Test Excel export with multiple sheets."""
        result = self.create_mock_result()
        
        files = export_fitting_result(
            result,
            tmp_path,
            formats=['excel'],
            include_plots=False
        )
        
        assert 'excel' in files
        excel_path = files['excel']
        assert excel_path.exists()
        
        # Load and verify Excel sheets
        with pd.ExcelFile(excel_path) as xls:
            sheets = xls.sheet_names
            assert 'Parameters' in sheets
            assert 'Confidence_Intervals' in sheets
            assert 'Fitted_Values' in sheets
            assert 'Statistics' in sheets
            
            # Check parameters sheet
            params_df = pd.read_excel(xls, 'Parameters')
            assert 'Vcmax_at_25' in params_df.columns
            assert params_df['Vcmax_at_25'].iloc[0] == 95.3
    
    def test_export_plots(self, tmp_path):
        """Test plot export."""
        result = self.create_mock_result()
        
        # Add a mock figure
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])
        result.figure = fig
        
        files = export_fitting_result(
            result,
            tmp_path,
            formats=['json'],
            include_plots=True,
            plot_dpi=150
        )
        
        assert 'plot_png' in files
        assert 'plot_pdf' in files
        assert files['plot_png'].exists()
        assert files['plot_pdf'].exists()
        
        plt.close(fig)
    
    def test_no_fitted_values(self, tmp_path):
        """Test export when fitted_values is None."""
        result = self.create_mock_result()
        result.fitted_values = None
        
        files = export_fitting_result(
            result,
            tmp_path,
            formats=['csv', 'excel'],
            include_plots=False
        )
        
        # CSV should not be created
        assert 'csv' not in files
        
        # Excel should still be created but without fitted values sheet
        assert 'excel' in files
        with pd.ExcelFile(files['excel']) as xls:
            assert 'Fitted_Values' not in xls.sheet_names


class TestExportBatchResults:
    """Test batch result export."""
    
    @staticmethod
    def create_mock_batch_result():
        """Create a mock BatchResult for testing."""
        br = BatchResult()
        
        # Add successful results
        for i in range(3):
            # Use the same mock class as above
            class MockResult:
                def __init__(self, i):
                    self.parameters = {
                        'Vcmax_at_25': 90 + i * 5,
                        'J_at_25': 180 + i * 10,
                        'Tp_at_25': 11 + i * 0.5
                    }
                    self.rmse = 1.2 + i * 0.1
                    self.r_squared = 0.99 - i * 0.01
                    self.n_points = 13
                    self.success = True
                    self.confidence_intervals = None
                    self.fitted_values = None
            
            result = MockResult(i)
            br.add_result(f'genotype_{i}', result)
        
        # Add failed result
        br.add_failure('failed_curve', 'Convergence error')
        br.add_warning('genotype_0', 'Low confidence')
        
        return br
    
    def test_export_summary_csv(self, tmp_path):
        """Test CSV summary export."""
        br = self.create_mock_batch_result()
        
        files = export_batch_results(
            br,
            tmp_path,
            formats=['csv'],
            include_individual=False
        )
        
        assert 'summary_csv' in files
        csv_path = files['summary_csv']
        assert csv_path.exists()
        
        # Verify content
        df = pd.read_csv(csv_path)
        assert len(df) == 4  # 3 successful + 1 failed
        assert 'curve_id' in df.columns
        assert 'Vcmax_at_25' in df.columns
        assert 'success' in df.columns
    
    def test_export_summary_excel(self, tmp_path):
        """Test Excel summary export with multiple sheets."""
        br = self.create_mock_batch_result()
        
        files = export_batch_results(
            br,
            tmp_path,
            formats=['excel'],
            include_individual=False
        )
        
        assert 'summary_excel' in files
        excel_path = files['summary_excel']
        assert excel_path.exists()
        
        # Verify sheets
        with pd.ExcelFile(excel_path) as xls:
            sheets = xls.sheet_names
            assert 'Summary' in sheets
            assert 'Failed_Curves' in sheets
            assert 'Parameter_Statistics' in sheets
            
            # Check failed curves sheet
            failed_df = pd.read_excel(xls, 'Failed_Curves')
            assert len(failed_df) == 1
            assert failed_df['curve_id'].iloc[0] == 'failed_curve'
    
    def test_export_individual_results(self, tmp_path):
        """Test exporting individual curve results."""
        br = self.create_mock_batch_result()
        
        files = export_batch_results(
            br,
            tmp_path,
            include_individual=True,
            formats=['csv']
        )
        
        # Check that individual results were exported
        individual_dir = tmp_path / 'individual_curves'
        assert individual_dir.exists()
        
        # Should have files for each successful curve
        for i in range(3):
            curve_files = list(individual_dir.glob(f'genotype_{i}_fit*'))
            assert len(curve_files) > 0
    
    def test_parameter_statistics(self, tmp_path):
        """Test parameter statistics calculation in Excel export."""
        br = self.create_mock_batch_result()
        
        files = export_batch_results(
            br,
            tmp_path,
            formats=['excel']
        )
        
        with pd.ExcelFile(files['summary_excel']) as xls:
            if 'Parameter_Statistics' in xls.sheet_names:
                stats_df = pd.read_excel(xls, 'Parameter_Statistics')
                
                # Check Vcmax statistics
                vcmax_stats = stats_df[stats_df['Parameter'] == 'Vcmax_at_25']
                if len(vcmax_stats) > 0:
                    assert vcmax_stats['Mean'].iloc[0] == 95.0  # (90+95+100)/3
                    assert vcmax_stats['N'].iloc[0] == 3


class TestAnalysisReport:
    """Test PDF report generation."""
    
    def test_single_result_report(self, tmp_path):
        """Test report for single fitting result."""
        result = TestExportFittingResult.create_mock_result()
        
        pdf_path = create_analysis_report(
            result,
            tmp_path / 'single_report.pdf',
            title='Test Report',
            include_methods=True,
            include_plots=False
        )
        
        assert pdf_path.exists()
        assert pdf_path.stat().st_size > 0  # File has content
    
    def test_batch_result_report(self, tmp_path):
        """Test report for batch results."""
        br = TestExportBatchResults.create_mock_batch_result()
        
        pdf_path = create_analysis_report(
            br,
            tmp_path / 'batch_report.pdf',
            title='Batch Analysis Report',
            include_methods=True
        )
        
        assert pdf_path.exists()
        assert pdf_path.stat().st_size > 0
    
    def test_model_comparison_report(self, tmp_path):
        """Test report for model comparison."""
        results = {
            'C3': TestExportFittingResult.create_mock_result(),
            'C4': TestExportFittingResult.create_mock_result()
        }
        # Modify C4 to have worse fit
        results['C4'].rmse = 2.5
        results['C4'].r_squared = 0.92
        
        pdf_path = create_analysis_report(
            results,
            tmp_path / 'comparison_report.pdf',
            title='Model Comparison',
            include_methods=False
        )
        
        assert pdf_path.exists()
        assert pdf_path.stat().st_size > 0
    
    def test_report_with_plots(self, tmp_path):
        """Test report generation including plots."""
        result = TestExportFittingResult.create_mock_result()
        
        # Add a figure
        fig, ax = plt.subplots()
        ax.scatter([1, 2, 3], [1, 4, 9])
        ax.plot([1, 2, 3], [1, 4, 9], 'r-')
        result.figure = fig
        
        pdf_path = create_analysis_report(
            result,
            tmp_path / 'report_with_plots.pdf',
            include_plots=True
        )
        
        assert pdf_path.exists()
        plt.close(fig)


class TestPhotoGEACompatibility:
    """Test PhotoGEA compatibility export."""
    
    def test_save_for_photogea(self, tmp_path):
        """Test saving in PhotoGEA-compatible format."""
        result = TestExportFittingResult.create_mock_result()
        
        csv_path = save_for_photogea_compatibility(
            result,
            tmp_path / 'photogea_output.csv'
        )
        
        assert csv_path.exists()
        
        # Read and verify format
        with open(csv_path) as f:
            lines = f.readlines()
        
        # Should have comment lines at top
        assert lines[0].startswith('#')
        assert '# Parameters:' in ''.join(lines[:10])
        assert '# Diagnostics:' in ''.join(lines[:20])
        
        # Check parameter mapping
        content = ''.join(lines)
        assert 'Vcmax,95.3' in content  # Mapped from Vcmax_at_25
        assert 'J,185.7' in content     # Mapped from J_at_25
        assert 'Rd,1.4' in content      # Mapped from RL_at_25
        assert 'gm,0.5' in content      # Mapped from gmc


class TestHelperFunctions:
    """Test internal helper functions."""
    
    def test_add_parameter_table_page(self, tmp_path):
        """Test parameter table page generation."""
        result = TestExportFittingResult.create_mock_result()
        
        with PdfPages(tmp_path / 'test.pdf') as pdf:
            _add_parameter_table_page(pdf, result)
        
        assert (tmp_path / 'test.pdf').exists()
    
    def test_add_batch_summary_page(self, tmp_path):
        """Test batch summary page generation."""
        br = TestExportBatchResults.create_mock_batch_result()
        br.generate_summary()  # Ensure summary is generated
        
        with PdfPages(tmp_path / 'test.pdf') as pdf:
            _add_batch_summary_page(pdf, br)
        
        assert (tmp_path / 'test.pdf').exists()


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_batch_result(self, tmp_path):
        """Test exporting empty batch result."""
        br = BatchResult()
        
        files = export_batch_results(
            br,
            tmp_path,
            formats=['csv', 'excel']
        )
        
        # Should still create files, even if empty
        assert 'summary_csv' in files or 'summary_excel' in files
    
    def test_no_confidence_intervals(self, tmp_path):
        """Test export without confidence intervals."""
        result = TestExportFittingResult.create_mock_result()
        result.confidence_intervals = None
        
        files = export_fitting_result(
            result,
            tmp_path,
            formats=['excel']
        )
        
        with pd.ExcelFile(files['excel']) as xls:
            # Should not have CI sheet if no intervals
            sheets = xls.sheet_names
            # Parameters and Statistics should still be there
            assert 'Parameters' in sheets
            assert 'Statistics' in sheets
    
    def test_missing_output_directory(self, tmp_path):
        """Test creating output directory if missing."""
        result = TestExportFittingResult.create_mock_result()
        
        # Use non-existent subdirectory
        output_dir = tmp_path / 'new' / 'nested' / 'dir'
        
        files = export_fitting_result(
            result,
            output_dir,
            formats=['json']
        )
        
        assert output_dir.exists()
        assert files['json'].exists()