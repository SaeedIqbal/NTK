#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Suite for NTK-SURGERY Core Components

Tests for:
- Federated NTK computation (Section 4.1)
- Influence Matrix construction (Section 4.2)
- Surgery Operator implementation (Section 4.3)
- Finite-Width Projection (Section 4.4)

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

from ntk_surgery.federated_ntk import FederatedNTK, NTKKernelComputer
from ntk_surgery.influence_matrix import InfluenceMatrix, ResolventMatrix
from ntk_surgery.surgery_operator import SurgeryOperator, WoodburyUpdater
from ntk_surgery.finite_width_projection import FiniteWidthProjector, JacobianComputer
from models.cnn import CNN, CNNConfig
from models.mlp import MLP, MLPConfig


class TestNTKKernelComputer(unittest.TestCase):
    """Test cases for NTK kernel computation."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create deterministic model
        config = MLPConfig(
            input_dim=10,
            num_classes=2,
            hidden_dims=[32, 16],
            use_batch_norm=False,
            dropout_rate=0.0
        )
        self.model = MLP(config)
        self.model.eval()
        
        # Create deterministic input data
        self.X = np.ones((5, 10), dtype=np.float32)
        self.X_tensor = torch.tensor(self.X, dtype=torch.float32)
        
        # Initialize kernel computer
        self.kernel_computer = NTKKernelComputer(self.model, device='cpu')
    
    def test_jacobian_computation(self):
        """Test Jacobian matrix computation."""
        J = self.kernel_computer.compute_jacobian(self.X_tensor)
        
        # Verify shape
        P = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.assertEqual(J.shape, (5, P), f"Expected shape (5, {P}), got {J.shape}")
        
        # Verify no NaN or Inf values
        self.assertFalse(np.any(np.isnan(J)), "Jacobian contains NaN values")
        self.assertFalse(np.any(np.isinf(J)), "Jacobian contains Inf values")
    
    def test_ntk_matrix_computation(self):
        """Test NTK matrix computation."""
        K = self.kernel_computer.compute_ntk_matrix(self.X_tensor)
        
        # Verify shape
        self.assertEqual(K.shape, (5, 5), f"Expected shape (5, 5), got {K.shape}")
        
        # Verify symmetry
        self.assertTrue(np.allclose(K, K.T), "NTK matrix is not symmetric")
        
        # Verify positive semi-definite (all eigenvalues >= 0)
        eigenvalues = np.linalg.eigvalsh(K)
        self.assertTrue(np.all(eigenvalues >= -1e-6), "NTK matrix is not positive semi-definite")
        
        # Verify normalized to [0, 1] range
        self.assertTrue(np.all(K >= 0), "NTK matrix has negative values")
        self.assertTrue(np.all(K <= 1), "NTK matrix has values > 1")
    
    def test_kernel_positive_definite_verification(self):
        """Test kernel positive definite verification."""
        K = self.kernel_computer.compute_ntk_matrix(self.X_tensor)
        is_psd = self.kernel_computer.verify_kernel_positive_definite(K)
        
        self.assertTrue(is_psd, "Kernel should be positive semi-definite")


class TestFederatedNTK(unittest.TestCase):
    """Test cases for Federated NTK representation."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create deterministic model
        config = MLPConfig(
            input_dim=8,
            num_classes=2,
            hidden_dims=[16],
            use_batch_norm=False
        )
        self.model = MLP(config)
        
        # Create deterministic data
        self.N = 20
        self.X = np.ones((self.N, 8), dtype=np.float32)
        self.y = np.array([0, 1] * 10, dtype=np.float32)
        
        # Create deterministic client partitions (no random)
        self.clients = {
            0: np.arange(0, 10),
            1: np.arange(10, 20)
        }
        
        # Initialize FederatedNTK
        self.federated_ntk = FederatedNTK(
            model=self.model,
            X=self.X,
            y=self.y,
            clients=self.clients,
            lambda_reg=0.05,
            device='cpu'
        )
    
    def test_selection_matrices(self):
        """Test client selection matrix construction."""
        # Verify orthogonality: S_c^T S_c' = δ_cc' I
        for c1 in self.clients:
            for c2 in self.clients:
                S_c1 = self.federated_ntk.S_matrices[c1]
                S_c2 = self.federated_ntk.S_matrices[c2]
                product = S_c1.T @ S_c2
                
                if c1 == c2:
                    # Should be identity
                    self.assertTrue(
                        np.allclose(product, np.eye(len(self.clients[c1]))),
                        f"S_{c1}^T S_{c1} should be identity"
                    )
                else:
                    # Should be zero
                    self.assertTrue(
                        np.allclose(product, np.zeros_like(product)),
                        f"S_{c1}^T S_{c2} should be zero for c1 != c2"
                    )
    
    def test_global_kernel_aggregation(self):
        """Test global kernel aggregation from client kernels."""
        # Compute global kernel
        K_global = self.federated_ntk.compute_global_kernel()
        
        # Compute client kernels
        K_client = self.federated_ntk.compute_client_kernels()
        
        # Reconstruct global kernel from client kernels
        K_reconstructed = self.federated_ntk.aggregate_kernels()
        
        # Verify reconstruction matches
        self.assertTrue(
            np.allclose(K_global, K_reconstructed),
            "Reconstructed kernel should match global kernel"
        )
    
    def test_kernel_ridge_predictions(self):
        """Test kernel ridge regression predictions."""
        K_global = self.federated_ntk.compute_global_kernel()
        Y_hat = self.federated_ntk.get_kernel_ridge_predictions(self.y)
        
        # Verify shape
        self.assertEqual(Y_hat.shape, (self.N,), f"Expected shape ({self.N},), got {Y_hat.shape}")
        
        # Verify no NaN or Inf
        self.assertFalse(np.any(np.isnan(Y_hat)), "Predictions contain NaN")
        self.assertFalse(np.any(np.isinf(Y_hat)), "Predictions contain Inf")


class TestInfluenceMatrix(unittest.TestCase):
    """Test cases for Influence Matrix computation."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create deterministic kernel matrix
        N = 10
        self.K_global = np.eye(N) * 0.9 + 0.1  # Positive definite
        self.lambda_reg = 0.05
        
        # Initialize Influence Matrix
        self.inf_mat = InfluenceMatrix(self.K_global, self.lambda_reg)
    
    def test_influence_matrix_construction(self):
        """Test Influence Matrix construction."""
        # Verify shape
        self.assertEqual(
            self.inf_mat.Influence.shape, (10, 10),
            "Influence Matrix shape mismatch"
        )
        
        # Verify symmetry
        self.assertTrue(
            np.allclose(self.inf_mat.Influence, self.inf_mat.Influence.T),
            "Influence Matrix should be symmetric"
        )
    
    def test_eigenvalue_bounds(self):
        """Test eigenvalue bounds [0, 1)."""
        eigenvalues = self.inf_mat.compute_eigenvalues()
        
        # All eigenvalues should be in [0, 1)
        self.assertTrue(
            np.all(eigenvalues >= 0),
            "Eigenvalues should be >= 0"
        )
        self.assertTrue(
            np.all(eigenvalues < 1),
            "Eigenvalues should be < 1"
        )
    
    def test_spectral_properties(self):
        """Test spectral property computation."""
        props = self.inf_mat.get_spectral_properties()
        
        # Verify all properties are present
        expected_keys = ['min_eigenvalue', 'max_eigenvalue', 'condition_number', 
                        'effective_rank', 'trace', 'nuclear_norm']
        for key in expected_keys:
            self.assertIn(key, props, f"Missing property: {key}")
        
        # Verify condition number is positive
        self.assertGreater(props['condition_number'], 0, "Condition number should be positive")
    
    def test_influence_properties_verification(self):
        """Test Influence Matrix property verification."""
        props = self.inf_mat.verify_influence_properties()
        
        self.assertTrue(props['eigenvalues_in_range'], "Eigenvalues should be in [0, 1)")
        self.assertTrue(props['symmetric'], "Matrix should be symmetric")
        self.assertTrue(props['positive_semi_definite'], "Matrix should be PSD")


class TestResolventMatrix(unittest.TestCase):
    """Test cases for Resolvent Matrix computation."""
    
    def setUp(self):
        """Set up test fixtures."""
        N = 10
        self.K_global = np.eye(N) * 0.9 + 0.1
        self.lambda_reg = 0.05
        self.resolvent = ResolventMatrix(self.K_global, self.lambda_reg)
    
    def test_resolvent_computation(self):
        """Test resolvent matrix computation."""
        # Verify G_λ = (K + λI)^{-1}
        expected = np.linalg.inv(self.K_global + self.lambda_reg * np.eye(10))
        
        self.assertTrue(
            np.allclose(self.resolvent.G_lambda, expected),
            "Resolvent computation incorrect"
        )
    
    def test_condition_number(self):
        """Test condition number computation."""
        cond = self.resolvent.get_condition_number()
        
        self.assertGreater(cond, 0, "Condition number should be positive")
        self.assertFalse(np.isinf(cond), "Condition number should be finite")


class TestSurgeryOperator(unittest.TestCase):
    """Test cases for Surgery Operator implementation."""
    
    def setUp(self):
        """Set up test fixtures."""
        N = 20
        n_c = 10
        
        # Create deterministic kernel matrices
        self.K_global = np.eye(N) * 0.9 + 0.1
        
        # Create client selection matrix (deterministic)
        self.S_c = np.zeros((N, n_c))
        for i in range(n_c):
            self.S_c[i, i] = 1
        
        # Create client kernel
        self.K_c = np.eye(n_c) * 0.9 + 0.1
        
        # Create Influence Matrix
        self.inf_mat = InfluenceMatrix(self.K_global, 0.05)
        
        # Create Surgery Operator
        self.surgery = SurgeryOperator(
            inf_mat=self.inf_mat,
            K_global=self.K_global,
            S_matrices={0: self.S_c},
            K_client={0: self.K_c},
            woodbury_jitter=1e-5
        )
    
    def test_woodbury_identity(self):
        """Test Woodbury identity implementation."""
        # Test with known matrices
        A = np.eye(5) * 2
        U = np.eye(5, 3)
        C = np.eye(3)
        V = U.T
        
        A_inv = np.linalg.inv(A)
        updated_inv = WoodburyUpdater.apply_woodbury(A_inv, U, C, V)
        
        # Verify against direct computation
        expected = np.linalg.inv(A + U @ C @ V)
        
        self.assertTrue(
            np.allclose(updated_inv, expected),
            "Woodbury identity implementation incorrect"
        )
    
    def test_client_unlearning(self):
        """Test client unlearning operation."""
        I_unlearned, G_unlearned = self.surgery.unlearn_client(0)
        
        # Verify shapes
        self.assertEqual(I_unlearned.shape, (20, 20), "Influence Matrix shape mismatch")
        self.assertEqual(G_unlearned.shape, (20, 20), "Resolvent shape mismatch")
        
        # Verify no NaN or Inf
        self.assertFalse(np.any(np.isnan(I_unlearned)), "Unlearned Influence contains NaN")
        self.assertFalse(np.any(np.isinf(I_unlearned)), "Unlearned Influence contains Inf")
    
    def test_exactness_error_computation(self):
        """Test exactness error computation."""
        I_unlearned, _ = self.surgery.unlearn_client(0)
        
        # Create reference Influence Matrix (simulating retraining)
        I_retrain = I_unlearned * 0.99  # Slight difference
        
        # Create label vector
        Y = np.ones(20)
        
        # Compute exactness error
        error = self.surgery.compute_exactness_error(I_unlearned, I_retrain, Y)
        
        # Error should be small
        self.assertLess(error, 0.1, f"Exactness error too large: {error}")
    
    def test_spectral_stability(self):
        """Test spectral stability metrics."""
        metrics = self.surgery.compute_spectral_stability(0)
        
        # Verify all metrics are present
        expected_keys = ['cond_G_original', 'cond_G_updated', 'cond_M_c', 
                        'condition_bound', 'sigma_min_G', 'stability_ratio']
        for key in expected_keys:
            self.assertIn(key, metrics, f"Missing metric: {key}")


class TestFiniteWidthProjector(unittest.TestCase):
    """Test cases for Finite-Width Projection."""
    
    def setUp(self):
        """Set up test fixtures."""
        config = MLPConfig(
            input_dim=8,
            num_classes=2,
            hidden_dims=[16],
            use_batch_norm=False
        )
        self.model = MLP(config)
        
        self.X = np.ones((10, 8), dtype=np.float32)
        self.theta_t = {
            name: param.clone().detach()
            for name, param in self.model.named_parameters()
        }
        
        self.projector = FiniteWidthProjector(
            model=self.model,
            theta_t=self.theta_t,
            X=self.X,
            lambda_reg=0.05,
            device='cpu'
        )
    
    def test_jacobian_computation(self):
        """Test Jacobian computation for finite-width network."""
        J = self.projector.compute_jacobian()
        
        P = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # Verify shape
        self.assertEqual(J.shape, (10, P), f"Expected shape (10, {P}), got {J.shape}")
        
        # Verify no NaN or Inf
        self.assertFalse(np.any(np.isnan(J)), "Jacobian contains NaN")
        self.assertFalse(np.any(np.isinf(J)), "Jacobian contains Inf")
    
    def test_weight_projection(self):
        """Test weight projection from function space."""
        J = self.projector.compute_jacobian()
        
        # Create target predictions
        Y_target = np.ones(10) * 0.5
        
        # Create resolvent
        K = J @ J.T
        G_lambda = np.linalg.inv(K + 0.05 * np.eye(10))
        
        # Project weights
        theta_new = self.projector.project_weights(Y_target, G_lambda)
        
        # Verify all parameters are updated
        for name, param in self.model.named_parameters():
            self.assertIn(name, theta_new, f"Missing parameter: {name}")
            self.assertEqual(theta_new[name].shape, param.shape, 
                           f"Shape mismatch for {name}")
    
    def test_linearization_error_bound(self):
        """Test linearization error bound computation."""
        self.projector.compute_jacobian()
        
        bound = self.projector.compute_linearization_error_bound(L_J=2.3)
        
        # Bound should be positive and finite
        self.assertGreater(bound, 0, "Error bound should be positive")
        self.assertFalse(np.isinf(bound), "Error bound should be finite")
    
    def test_width_dependent_error(self):
        """Test width-dependent error computation."""
        P = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        metrics = self.projector.compute_width_dependent_error(P)
        
        # Verify all metrics are present
        expected_keys = ['theoretical_error', 'higher_order_term', 
                        'total_estimated_error', 'error_bound', 'width', 'decay_rate']
        for key in expected_keys:
            self.assertIn(key, metrics, f"Missing metric: {key}")
        
        # Verify decay rate is O(P^{-1/2})
        self.assertEqual(metrics['decay_rate'], 'O(P^{-1/2})', "Incorrect decay rate")


class TestNTKSurgeyIntegration(unittest.TestCase):
    """Integration tests for complete NTK-SURGERY pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        config = MLPConfig(
            input_dim=8,
            num_classes=2,
            hidden_dims=[16],
            use_batch_norm=False
        )
        self.model = MLP(config)
        
        self.N = 20
        self.X = np.ones((self.N, 8), dtype=np.float32)
        self.y = np.array([0, 1] * 10, dtype=np.float32)
        
        self.clients = {
            0: np.arange(0, 10),
            1: np.arange(10, 20)
        }
    
    def test_complete_unlearning_pipeline(self):
        """Test complete unlearning pipeline."""
        from unlearning.unlearn_client import UnlearnClient, UnlearningConfig
        
        # Initialize unlearner
        config = UnlearningConfig(
            lambda_reg=0.05,
            width_multiplier=1,
            device='cpu',
            save_intermediate=False
        )
        
        unlearner = UnlearnClient(self.model, config)
        unlearner.set_client_partitions(self.clients)
        
        # Run unlearning
        result = unlearner.unlearn(client_id=0, X=self.X, y=self.y)
        
        # Verify success
        self.assertTrue(result.success, "Unlearning should succeed")
        
        # Verify communication rounds = 1
        self.assertEqual(result.communication_rounds, 1, "Should require 1 round")
        
        # Verify unlearning time is positive
        self.assertGreater(result.unlearning_time, 0, "Unlearning time should be positive")


def run_tests():
    """Run all NTK-SURGERY tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestNTKKernelComputer))
    suite.addTests(loader.loadTestsFromTestCase(TestFederatedNTK))
    suite.addTests(loader.loadTestsFromTestCase(TestInfluenceMatrix))
    suite.addTests(loader.loadTestsFromTestCase(TestResolventMatrix))
    suite.addTests(loader.loadTestsFromTestCase(TestSurgeryOperator))
    suite.addTests(loader.loadTestsFromTestCase(TestFiniteWidthProjector))
    suite.addTests(loader.loadTestsFromTestCase(TestNTKSurgeyIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)