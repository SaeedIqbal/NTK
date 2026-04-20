#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Suite for Evaluation Metrics

Tests for:
- Unlearning efficacy metrics (FA, RA, ES)
- Efficiency metrics (time, rounds, FLOPs)
- Theoretical metrics (alignment, sensitivity)

All tests are deterministic with no random operations.
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from metrics.unlearning_metrics import UnlearningMetrics, UnlearningEvaluator, MetricValidator
from metrics.efficiency_metrics import EfficiencyMetrics, ComplexityAnalyzer, PerformanceTracker
from metrics.theoretical_metrics import TheoreticalMetrics, NTKAnalyzer, SensitivityAnalyzer
from models.mlp import MLP, MLPConfig


class TestUnlearningMetrics(unittest.TestCase):
    """Test cases for unlearning efficacy metrics."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.num_classes = 2
        self.metrics = UnlearningMetrics(num_classes=self.num_classes, device='cpu')
        
        # Create deterministic model
        config = MLPConfig(
            input_dim=8,
            num_classes=self.num_classes,
            hidden_dims=[16]
        )
        self.model = MLP(config)
        
        # Create deterministic data
        self.X_forget = np.ones((5, 8), dtype=np.float32)
        self.y_forget = np.array([0, 1, 0, 1, 0], dtype=np.int64)
        self.X_retain = np.ones((10, 8), dtype=np.float32) * 2
        self.y_retain = np.array([0, 1] * 5, dtype=np.int64)
    
    def test_forget_accuracy_computation(self):
        """Test forget accuracy computation."""
        fa = self.metrics.forget_accuracy(
            self.model, self.X_forget, self.y_forget
        )
        
        # Should be in [0, 1] range
        self.assertGreaterEqual(fa, 0, "FA should be >= 0")
        self.assertLessEqual(fa, 1, "FA should be <= 1")
    
    def test_retain_accuracy_computation(self):
        """Test retain accuracy computation."""
        ra = self.metrics.retain_accuracy(
            self.model, self.X_retain, self.y_retain
        )
        
        # Should be in [0, 1] range
        self.assertGreaterEqual(ra, 0, "RA should be >= 0")
        self.assertLessEqual(ra, 1, "RA should be <= 1")
    
    def test_exactness_score_computation(self):
        """Test exactness score computation."""
        # Create deterministic predictions
        pred_surgery = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
        pred_scratch = np.array([0.9, 0.8, 0.7, 0.6, 0.5])  # Identical
        
        # Perfect match should give ES = 1.0
        es = self.metrics.exactness_score(pred_surgery, pred_scratch)
        self.assertAlmostEqual(es, 1.0, places=5, "Identical predictions should give ES=1.0")
        
        # Different predictions
        pred_surgery2 = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
        es2 = self.metrics.exactness_score(pred_surgery2, pred_scratch)
        
        # Should be less than 1.0
        self.assertLess(es2, 1.0, "Different predictions should give ES<1.0")
        self.assertGreaterEqual(es2, 0, "ES should be >= 0")
    
    def test_exactness_error_computation(self):
        """Test exactness error computation."""
        pred_surgery = np.array([0.9, 0.8, 0.7])
        pred_scratch = np.array([0.9, 0.8, 0.7])
        
        ee = self.metrics.exactness_error(pred_surgery, pred_scratch)
        
        # Identical predictions should give EE = 0
        self.assertAlmostEqual(ee, 0.0, places=5, "Identical predictions should give EE=0.0")
    
    def test_metric_quality_evaluation(self):
        """Test metric quality evaluation."""
        # Excellent metrics
        quality = self.metrics.evaluate_metric_quality(
            fa=0.1,  # Close to random (0.5 for 2 classes)
            ra=0.95,
            es=0.98
        )
        
        self.assertIn(quality.value, ['excellent', 'good', 'acceptable'], 
                     "Quality should be valid")
    
    def test_all_metrics_computation(self):
        """Test computation of all metrics together."""
        metrics = self.metrics.compute_all_metrics(
            model_surgery=self.model,
            model_scratch=self.model,
            X_forget=self.X_forget,
            y_forget=self.y_forget,
            X_retain=self.X_retain,
            y_retain=self.y_retain
        )
        
        # Verify all metrics present
        expected_keys = ['forget_accuracy', 'retain_accuracy', 'exactness_score', 
                        'exactness_error', 'random_chance', 'fa_vs_random', 'ra_vs_scratch']
        for key in expected_keys:
            self.assertIn(key, metrics, f"Missing metric: {key}")


class TestMetricValidator(unittest.TestCase):
    """Test cases for metric validation."""
    
    def test_accuracy_range_validation(self):
        """Test accuracy range validation."""
        # Valid accuracy
        self.assertTrue(MetricValidator.validate_accuracy_range(0.5))
        self.assertTrue(MetricValidator.validate_accuracy_range(0.0))
        self.assertTrue(MetricValidator.validate_accuracy_range(1.0))
        
        # Invalid accuracy
        self.assertFalse(MetricValidator.validate_accuracy_range(-0.1))
        self.assertFalse(MetricValidator.validate_accuracy_range(1.1))
    
    def test_exactness_score_validation(self):
        """Test exactness score validation."""
        # Valid scores
        self.assertTrue(MetricValidator.validate_exactness_score(0.5))
        self.assertTrue(MetricValidator.validate_exactness_score(0.0))
        self.assertTrue(MetricValidator.validate_exactness_score(1.0))
        
        # Invalid scores
        self.assertFalse(MetricValidator.validate_exactness_score(-0.1))
        self.assertFalse(MetricValidator.validate_exactness_score(1.1))
    
    def test_forget_accuracy_validation(self):
        """Test forget accuracy validation."""
        random_chance = 0.5
        
        # Good forget accuracy (close to random)
        self.assertTrue(MetricValidator.validate_forget_accuracy(
            0.5, random_chance, acceptable_deviation=0.15
        ))
        
        # Bad forget accuracy (far from random)
        self.assertFalse(MetricValidator.validate_forget_accuracy(
            0.9, random_chance, acceptable_deviation=0.15
        ))
    
    def test_metrics_consistency_validation(self):
        """Test consistency validation across all metrics."""
        validations = MetricValidator.validate_metrics_consistency(
            fa=0.5,
            ra=0.8,
            es=0.9,
            num_classes=2
        )
        
        # All validations should pass
        self.assertTrue(validations['all_valid'], "All validations should pass")


class TestEfficiencyMetrics(unittest.TestCase):
    """Test cases for efficiency metrics."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.efficiency = EfficiencyMetrics()
        
        config = MLPConfig(
            input_dim=8,
            num_classes=2,
            hidden_dims=[16]
        )
        self.model = MLP(config)
    
    def test_execution_time_measurement(self):
        """Test execution time measurement."""
        def dummy_function():
            return 1 + 1
        
        time_val, result = self.efficiency.measure_execution_time(dummy_function)
        
        # Time should be positive
        self.assertGreater(time_val, 0, "Execution time should be positive")
        
        # Result should be correct
        self.assertEqual(result, 2, "Function result should be correct")
    
    def test_communication_rounds_computation(self):
        """Test communication rounds computation."""
        # NTK-SURGERY should have 1 round
        rounds = self.efficiency.compute_communication_rounds('NTK-SURGERY')
        self.assertEqual(rounds, 1, "NTK-SURGERY should have 1 round")
        
        # SIFU should have more rounds
        rounds_sifu = self.efficiency.compute_communication_rounds('SIFU')
        self.assertGreater(rounds_sifu, 1, "SIFU should have > 1 round")
    
    def test_flops_computation(self):
        """Test FLOPs computation."""
        N = 100
        n_c = 10
        
        # NTK-SURGERY FLOPs
        flops_ntk = self.efficiency.compute_flops(
            self.model, N, 'NTK-SURGERY', n_c
        )
        
        # SIFU FLOPs
        flops_sifu = self.efficiency.compute_flops(
            self.model, N, 'SIFU', n_c
        )
        
        # Both should be positive
        self.assertGreater(flops_ntk, 0, "FLOPs should be positive")
        self.assertGreater(flops_sifu, 0, "FLOPs should be positive")
    
    def test_speedup_computation(self):
        """Test speedup computation."""
        ntk_time = 1.0
        baseline_time = 100.0
        
        # Mock baseline
        self.efficiency.baseline_times['TestBaseline'] = baseline_time
        
        speedup = self.efficiency.compute_speedup(ntk_time, 'TestBaseline')
        
        # Speedup should be > 1
        self.assertGreater(speedup, 1, "Speedup should be > 1")
    
    def test_complexity_analysis(self):
        """Test complexity analysis."""
        N = 100
        n_c = 10
        P = 1000
        
        analysis = ComplexityAnalyzer.analyze_ntk_surgery_complexity(N, n_c, P)
        
        # Verify all metrics present
        expected_keys = ['jacobian_flops', 'woodbury_flops', 'inversion_flops', 
                        'total_flops', 'dominant_operation', 'complexity_class']
        for key in expected_keys:
            self.assertIn(key, analysis, f"Missing metric: {key}")
        
        # Verify complexity class
        self.assertEqual(analysis['complexity_class'], 'O(N² n_c)', 
                        "Incorrect complexity class")


class TestTheoreticalMetrics(unittest.TestCase):
    """Test cases for theoretical metrics."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.theoretical = TheoreticalMetrics()
    
    def test_ntk_alignment_computation(self):
        """Test NTK alignment computation."""
        # Create deterministic kernel matrices
        K1 = np.eye(10) * 0.9
        K2 = np.eye(10) * 0.9
        
        alignment = self.theoretical.ntk_alignment_score(K1, K2)
        
        # Identical kernels should have high alignment
        self.assertGreater(alignment, 0.9, "Identical kernels should have high alignment")
        self.assertLessEqual(alignment, 1.0, "Alignment should be <= 1")
    
    def test_sensitivity_bound_ratio_computation(self):
        """Test sensitivity bound ratio computation."""
        zeta_sifu = 100.0
        zeta_ntk = 1.0
        
        ratio = self.theoretical.sensitivity_bound_ratio(zeta_sifu, zeta_ntk)
        
        # Ratio should be > 1 (NTK-SURGERY has lower sensitivity)
        self.assertGreater(ratio, 1, "Ratio should be > 1")
        self.assertAlmostEqual(ratio, 100.0, places=5, "Ratio should match")
    
    def test_sifu_sensitivity_bound_computation(self):
        """Test SIFU sensitivity bound computation."""
        gradient_norms = [1.0] * 10
        
        # With B > 1, should grow exponentially
        zeta_5 = self.theoretical.compute_sifu_sensitivity_bound(5, gradient_norms, B_factor=1.05)
        zeta_10 = self.theoretical.compute_sifu_sensitivity_bound(10, gradient_norms, B_factor=1.05)
        
        self.assertGreater(zeta_10, zeta_5, "Sensitivity should grow with rounds")
    
    def test_condition_number_analysis(self):
        """Test condition number analysis."""
        # Well-conditioned matrix
        K_well = np.eye(10) * 2
        
        analysis_well = self.theoretical.condition_number_analysis(K_well, 'well')
        
        self.assertTrue(analysis_well['well_conditioned'], "Should be well-conditioned")
        
        # Ill-conditioned matrix
        K_ill = np.diag([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1e-10])
        
        analysis_ill = self.theoretical.condition_number_analysis(K_ill, 'ill')
        
        self.assertTrue(analysis_ill['ill_conditioned'], "Should be ill-conditioned")
    
    def test_effective_rank_computation(self):
        """Test effective rank computation."""
        # Full rank matrix
        K_full = np.eye(10)
        
        rank_full = self.theoretical.effective_rank(K_full)
        
        self.assertEqual(rank_full, 10, "Full rank matrix should have rank 10")
        
        # Low rank matrix
        K_low = np.zeros((10, 10))
        K_low[:5, :5] = np.eye(5)
        
        rank_low = self.theoretical.effective_rank(K_low)
        
        self.assertEqual(rank_low, 5, "Low rank matrix should have rank 5")
    
    def test_all_theoretical_metrics(self):
        """Test computation of all theoretical metrics."""
        K_global = np.eye(20) * 0.9 + 0.1
        gradient_norms = [1.0] * 50
        
        result = self.theoretical.compute_all_theoretical_metrics(
            K_global, gradient_norms, 50
        )
        
        # Verify all metrics present
        expected_keys = ['ntk_alignment', 'sensitivity_bound_ratio', 'condition_number', 
                        'effective_rank', 'spectral_gap', 'quality']
        for key in expected_keys:
            self.assertIn(key, vars(result), f"Missing metric: {key}")


def run_tests():
    """Run all metrics tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestUnlearningMetrics))
    suite.addTests(loader.loadTestsFromTestCase(TestMetricValidator))
    suite.addTests(loader.loadTestsFromTestCase(TestEfficiencyMetrics))
    suite.addTests(loader.loadTestsFromTestCase(TestTheoreticalMetrics))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)