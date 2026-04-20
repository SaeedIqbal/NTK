#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Suite for Baseline Methods

Tests for:
- SIFU (Sequential Informed Federated Unlearning)
- FedEraser
- Fine-Tuning
- FedSGD
- Other baseline methods

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

from baselines.sifu import SIFU, SIFUConfig
from baselines.federaser import FedEraser, FedEraserConfig
from baselines.fine_tuning import FineTuning, FineTuningConfig
from baselines.fedsgd import FedSGD, FedSGDConfig
from baselines.bfu import BFU, BFUConfig
from models.mlp import MLP, MLPConfig


class TestSIFU(unittest.TestCase):
    """Test cases for SIFU baseline."""
    
    def setUp(self):
        """Set up test fixtures."""
        config = MLPConfig(
            input_dim=8,
            num_classes=2,
            hidden_dims=[16]
        )
        self.model = MLP(config)
        
        self.sifu_config = SIFUConfig(
            learning_rate=0.01,
            local_epochs=2,
            batch_size=4,
            epsilon=1.0,
            delta=1e-5,
            max_rollback_rounds=10,
            device='cpu'
        )
        
        self.sifu = SIFU(self.model, self.sifu_config)
    
    def test_sensitivity_bound_computation(self):
        """Test sensitivity bound computation."""
        gradient_norms = [1.0, 1.0, 1.0, 1.0, 1.0]  # Deterministic
        
        zeta = self.sifu.compute_sensitivity_bound(
            n_rounds=5,
            gradient_norms=gradient_norms,
            B_factor=1.05
        )
        
        # Sensitivity should be positive
        self.assertGreater(zeta, 0, "Sensitivity bound should be positive")
        
        # With B > 1, should grow exponentially
        zeta_10 = self.sifu.compute_sensitivity_bound(
            n_rounds=10,
            gradient_norms=gradient_norms * 2,
            B_factor=1.05
        )
        
        self.assertGreater(zeta_10, zeta, "Sensitivity should grow with rounds")
    
    def test_noise_scale_computation(self):
        """Test Gaussian noise scale computation."""
        zeta = 1.0
        sigma = self.sifu.compute_noise_scale(zeta)
        
        # Noise scale should be positive
        self.assertGreater(sigma, 0, "Noise scale should be positive")
    
    def test_rollback_checkpoint_finding(self):
        """Test rollback checkpoint finding."""
        gradient_norms = [1.0] * 20
        sensitivity_budget = 10.0
        
        T = self.sifu.find_rollback_checkpoint(
            sensitivity_budget=sensitivity_budget,
            gradient_norms=gradient_norms,
            B_factor=1.05
        )
        
        # Checkpoint should be valid
        self.assertGreaterEqual(T, 0, "Checkpoint should be >= 0")
        self.assertLessEqual(T, len(gradient_norms), "Checkpoint should be <= num rounds")
    
    def test_checkpoint_storage(self):
        """Test checkpoint storage and retrieval."""
        theta = {
            name: param.clone().detach()
            for name, param in self.model.named_parameters()
        }
        
        # Store checkpoint
        self.sifu.store_checkpoint(0, theta)
        
        # Verify checkpoint stored
        self.assertIn(0, self.sifu.checkpoints, "Checkpoint should be stored")
    
    def test_efficiency_metrics(self):
        """Test efficiency metrics computation."""
        metrics = self.sifu.get_efficiency_metrics()
        
        # Verify all metrics are present
        expected_keys = ['method', 'unlearning_time', 'rollback_rounds', 
                        'max_rollback_rounds', 'sensitivity_bounds', 
                        'checkpoints_stored', 'complexity_class']
        for key in expected_keys:
            self.assertIn(key, metrics, f"Missing metric: {key}")


class TestFedEraser(unittest.TestCase):
    """Test cases for FedEraser baseline."""
    
    def setUp(self):
        """Set up test fixtures."""
        config = MLPConfig(
            input_dim=8,
            num_classes=2,
            hidden_dims=[16]
        )
        self.model = MLP(config)
        
        self.federaser_config = FedEraserConfig(
            learning_rate=0.01,
            local_epochs=2,
            batch_size=4,
            unlearning_epochs=5,
            device='cpu'
        )
        
        self.federaser = FedEraser(self.model, self.federaser_config)
    
    def test_gradient_storage(self):
        """Test gradient history storage."""
        gradient = {
            name: torch.ones_like(param)
            for name, param in self.model.named_parameters()
        }
        
        # Store gradient
        self.federaser.store_client_gradient(0, gradient, 0)
        
        # Verify gradient stored
        self.assertIn(0, self.federaser.gradient_history, "Gradient should be stored")
        self.assertEqual(len(self.federaser.gradient_history[0]), 1, 
                        "Should have 1 gradient entry")
    
    def test_gradient_aggregation(self):
        """Test gradient aggregation."""
        # Store multiple gradients
        for i in range(3):
            gradient = {
                name: torch.ones_like(param) * (i + 1)
                for name, param in self.model.named_parameters()
            }
            self.federaser.store_client_gradient(0, gradient, i)
        
        # Aggregate
        aggregated = self.federaser.aggregate_client_gradients(0)
        
        # Verify aggregation
        self.assertIsNotNone(aggregated, "Aggregated gradient should not be None")
    
    def test_efficiency_metrics(self):
        """Test efficiency metrics computation."""
        metrics = self.federaser.get_efficiency_metrics()
        
        # Verify theoretical guarantees flag
        self.assertFalse(metrics['theoretical_guarantees'], 
                        "FedEraser should not have theoretical guarantees")


class TestFineTuning(unittest.TestCase):
    """Test cases for Fine-Tuning baseline."""
    
    def setUp(self):
        """Set up test fixtures."""
        config = MLPConfig(
            input_dim=8,
            num_classes=2,
            hidden_dims=[16]
        )
        self.model = MLP(config)
        
        self.ft_config = FineTuningConfig(
            learning_rate=0.01,
            epochs=5,
            batch_size=4,
            device='cpu'
        )
        
        self.fine_tuning = FineTuning(self.model, self.ft_config)
    
    def test_efficiency_metrics(self):
        """Test efficiency metrics computation."""
        metrics = self.fine_tuning.get_efficiency_metrics()
        
        # Verify parameter dependence note
        self.assertIn('parameter_dependence', metrics, 
                     "Should note parameter dependence")
    
    def test_no_theoretical_guarantees(self):
        """Test that Fine-Tuning has no theoretical guarantees."""
        metrics = self.fine_tuning.get_efficiency_metrics()
        
        self.assertFalse(metrics['theoretical_guarantees'], 
                        "Fine-Tuning should not have theoretical guarantees")


class TestFedSGD(unittest.TestCase):
    """Test cases for FedSGD baseline."""
    
    def setUp(self):
        """Set up test fixtures."""
        config = MLPConfig(
            input_dim=8,
            num_classes=2,
            hidden_dims=[16]
        )
        self.model = MLP(config)
        
        self.fedsgd_config = FedSGDConfig(
            learning_rate=0.01,
            communication_rounds=5,
            local_epochs=1,
            batch_size=4,
            device='cpu'
        )
        
        self.fedsgd = FedSGD(self.model, self.fedsgd_config)
    
    def test_efficiency_metrics(self):
        """Test efficiency metrics computation."""
        metrics = self.fedsgd.get_efficiency_metrics()
        
        # Verify communication overhead note
        self.assertIn('communication_overhead', metrics, 
                     "Should note communication overhead")


class TestBFU(unittest.TestCase):
    """Test cases for BFU baseline."""
    
    def setUp(self):
        """Set up test fixtures."""
        config = MLPConfig(
            input_dim=8,
            num_classes=2,
            hidden_dims=[16]
        )
        self.model = MLP(config)
        
        self.bfu_config = BFUConfig(
            prior_variance=1.0,
            likelihood_variance=0.1,
            num_samples=10,
            device='cpu'
        )
        
        self.bfu = BFU(self.model, self.bfu_config)
    
    def test_efficiency_metrics(self):
        """Test efficiency metrics computation."""
        metrics = self.bfu.get_efficiency_metrics()
        
        # Verify scalability note
        self.assertIn('scalability', metrics, "Should note scalability")


def run_tests():
    """Run all baseline tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestSIFU))
    suite.addTests(loader.loadTestsFromTestCase(TestFedEraser))
    suite.addTests(loader.loadTestsFromTestCase(TestFineTuning))
    suite.addTests(loader.loadTestsFromTestCase(TestFedSGD))
    suite.addTests(loader.loadTestsFromTestCase(TestBFU))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)