#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Suite for Data Loading Utilities

Tests for:
- Dataset loading
- Federated partitioning
- Data preprocessing
- Data validation

All tests are deterministic with no random operations.
"""

import unittest
import torch
import numpy as np
from pathlib import Path
import sys
import os

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data.data_loader import DataLoaderManager, FederatedDataset
from data.dataset_utils import DatasetUtils, DatasetValidator
from data.federated_partition import FederatedPartitioner, PartitionStrategy
from data.preprocessor import DataPreprocessor, NormalizationStrategy


class TestFederatedDataset(unittest.TestCase):
    """Test cases for FederatedDataset."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.data = np.ones((10, 8), dtype=np.float32)
        self.targets = np.array([0, 1] * 5, dtype=np.int64)
        
        self.dataset = FederatedDataset(self.data, self.targets, transform=None)
    
    def test_dataset_length(self):
        """Test dataset length."""
        self.assertEqual(len(self.dataset), 10, "Dataset length should be 10")
    
    def test_dataset_getitem(self):
        """Test dataset item retrieval."""
        data, target = self.dataset[0]
        
        # Verify data shape
        if isinstance(data, torch.Tensor):
            self.assertEqual(len(data), 8, "Data should have 8 features")
        
        # Verify target
        self.assertIn(target, [0, 1], "Target should be 0 or 1")
    
    def test_class_distribution(self):
        """Test class distribution computation."""
        dist = self.dataset.get_class_distribution()
        
        # Should have 2 classes with equal distribution
        self.assertEqual(len(dist), 2, "Should have 2 classes")
        self.assertEqual(dist[0], 5, "Class 0 should have 5 samples")
        self.assertEqual(dist[1], 5, "Class 1 should have 5 samples")
    
    def test_data_statistics(self):
        """Test data statistics computation."""
        stats = self.dataset.get_data_statistics()
        
        # Verify all statistics present
        expected_keys = ['mean', 'std', 'min', 'max', 'shape']
        for key in expected_keys:
            self.assertIn(key, stats, f"Missing statistic: {key}")


class TestDataLoaderManager(unittest.TestCase):
    """Test cases for DataLoaderManager."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.manager = DataLoaderManager()
    
    def test_supported_datasets(self):
        """Test supported datasets list."""
        expected_datasets = ['MNIST', 'FashionMNIST', 'CIFAR-10', 
                           'CIFAR-100', 'CelebA', 'TinyImageNet']
        
        for dataset in expected_datasets:
            self.assertIn(dataset, self.manager.SUPPORTED_DATASETS, 
                         f"Missing dataset: {dataset}")
    
    def test_dataset_config(self):
        """Test dataset configuration retrieval."""
        config = self.manager.get_dataset_config('CIFAR-10')
        
        # Verify configuration
        self.assertEqual(config['num_classes'], 10, "CIFAR-10 should have 10 classes")
        self.assertEqual(config['input_channels'], 3, "CIFAR-10 should have 3 channels")
    
    def test_federated_partition_creation(self):
        """Test federated partition creation."""
        # Create deterministic data
        data = np.ones((100, 8), dtype=np.float32)
        targets = np.array([0, 1] * 50, dtype=np.int64)
        
        # Create partitions (deterministic)
        clients = self.manager.create_federated_partitions(
            data, targets,
            num_clients=10,
            partition_type='uniform',
            alpha=0.1,
            seed=42
        )
        
        # Verify number of clients
        self.assertEqual(len(clients), 10, "Should have 10 clients")
        
        # Verify each client has data
        for cid, dataset in clients.items():
            self.assertGreater(len(dataset), 0, f"Client {cid} should have data")


class TestFederatedPartitioner(unittest.TestCase):
    """Test cases for FederatedPartitioner."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.targets = np.array([0, 1, 2, 3] * 25, dtype=np.int64)  # 100 samples, 4 classes
    
    def test_label_shard_partition(self):
        """Test label shard partitioning."""
        partitioner = FederatedPartitioner(
            strategy='label_shard',
            alpha=0.1,
            seed=42
        )
        
        client_indices = partitioner.partition(self.targets, num_clients=4)
        
        # Verify number of clients
        self.assertEqual(len(client_indices), 4, "Should have 4 clients")
        
        # Verify no overlap
        all_indices = []
        for indices in client_indices.values():
            all_indices.extend(indices.tolist())
        
        self.assertEqual(len(all_indices), len(set(all_indices)), 
                        "Should have no overlapping indices")
    
    def test_dirichlet_partition(self):
        """Test Dirichlet partitioning."""
        partitioner = FederatedPartitioner(
            strategy='dirichlet',
            alpha=0.5,
            seed=42
        )
        
        client_indices = partitioner.partition(self.targets, num_clients=4)
        
        # Verify number of clients
        self.assertEqual(len(client_indices), 4, "Should have 4 clients")
        
        # Verify coverage
        all_indices = []
        for indices in client_indices.values():
            all_indices.extend(indices.tolist())
        
        self.assertEqual(len(all_indices), len(self.targets), 
                        "Should cover all samples")
    
    def test_uniform_partition(self):
        """Test uniform partitioning."""
        partitioner = FederatedPartitioner(
            strategy='uniform',
            alpha=0.1,
            seed=42
        )
        
        client_indices = partitioner.partition(self.targets, num_clients=4)
        
        # Verify equal sizes (approximately)
        sizes = [len(indices) for indices in client_indices.values()]
        
        # All sizes should be close to equal
        max_diff = max(sizes) - min(sizes)
        self.assertLessEqual(max_diff, 1, "Client sizes should be nearly equal")
    
    def test_partition_statistics(self):
        """Test partition statistics computation."""
        partitioner = FederatedPartitioner(
            strategy='uniform',
            seed=42
        )
        
        client_indices = partitioner.partition(self.targets, num_clients=4)
        
        stats = partitioner.compute_partition_statistics(client_indices, self.targets)
        
        # Verify all statistics present
        expected_keys = ['num_clients', 'total_samples', 'min_client_size', 
                        'max_client_size', 'mean_client_size', 'std_client_size']
        for key in expected_keys:
            self.assertIn(key, stats, f"Missing statistic: {key}")


class TestDatasetUtils(unittest.TestCase):
    """Test cases for DatasetUtils."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.targets = np.array([0, 0, 0, 1, 1, 2], dtype=np.int64)
    
    def test_class_distribution(self):
        """Test class distribution computation."""
        dist = DatasetUtils.compute_class_distribution(self.targets)
        
        # Verify distribution
        self.assertEqual(len(dist), 3, "Should have 3 classes")
        self.assertAlmostEqual(dist[0], 0.5, places=2, "Class 0 should be 50%")
        self.assertAlmostEqual(dist[1], 0.33, places=2, "Class 1 should be 33%")
        self.assertAlmostEqual(dist[2], 0.17, places=2, "Class 2 should be 17%")
    
    def test_label_imbalance(self):
        """Test label imbalance computation."""
        imbalance = DatasetUtils.compute_label_imbalance(self.targets)
        
        # Should be > 1 (imbalanced)
        self.assertGreater(imbalance, 1, "Should detect imbalance")
    
    def test_data_statistics(self):
        """Test data statistics computation."""
        data = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        
        stats = DatasetUtils.compute_data_statistics(data)
        
        # Verify statistics
        expected_keys = ['mean', 'std', 'variance', 'min', 'max', 'median', 
                        'skewness', 'kurtosis']
        for key in expected_keys:
            self.assertIn(key, stats, f"Missing statistic: {key}")
        
        # Verify mean
        self.assertAlmostEqual(stats['mean'], 3.0, places=5, "Mean should be 3.0")


class TestDatasetValidator(unittest.TestCase):
    """Test cases for DatasetValidator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.validator = DatasetValidator(min_samples_per_class=10)
    
    def test_federated_partition_validation(self):
        """Test federated partition validation."""
        # Create valid partition
        clients = {
            0: (np.ones((20, 8)), np.array([0, 1] * 10)),
            1: (np.ones((20, 8)), np.array([0, 1] * 10))
        }
        
        results = self.validator.validate_federated_partition(clients, 2)
        
        # All validations should pass
        self.assertTrue(results['all_clients_have_data'], "All clients should have data")
        self.assertTrue(results['all_classes_represented'], "All classes should be represented")
    
    def test_data_quality_validation(self):
        """Test data quality validation."""
        # Valid data
        data = np.ones((20, 8), dtype=np.float32)
        targets = np.array([0, 1] * 10, dtype=np.int64)
        
        results = self.validator.validate_data_quality(data, targets, 'TestDataset')
        
        # All validations should pass
        self.assertTrue(results['no_nan_values'], "Should have no NaN values")
        self.assertTrue(results['no_inf_values'], "Should have no Inf values")
        self.assertTrue(results['consistent_lengths'], "Lengths should be consistent")
    
    def test_validation_summary(self):
        """Test validation summary generation."""
        # Run validation
        data = np.ones((20, 8), dtype=np.float32)
        targets = np.array([0, 1] * 10, dtype=np.int64)
        
        self.validator.validate_data_quality(data, targets, 'TestDataset')
        
        summary = self.validator.get_validation_summary()
        
        # Summary should not be empty
        self.assertGreater(len(summary), 0, "Summary should not be empty")


class TestDataPreprocessor(unittest.TestCase):
    """Test cases for DataPreprocessor."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.data = np.random.randn(20, 8).astype(np.float32)
    
    def test_standard_normalization(self):
        """Test standard normalization."""
        preprocessor = DataPreprocessor(strategy='standard')
        
        # Fit and transform
        transformed = preprocessor.fit_transform(self.data)
        
        # Verify shape
        self.assertEqual(transformed.shape, self.data.shape, "Shape should be preserved")
        
        # Verify mean is close to 0
        mean = np.mean(transformed, axis=0)
        self.assertTrue(np.allclose(mean, 0, atol=1e-6), "Mean should be close to 0")
    
    def test_minmax_normalization(self):
        """Test min-max normalization."""
        preprocessor = DataPreprocessor(strategy='minmax')
        
        # Fit and transform
        transformed = preprocessor.fit_transform(self.data)
        
        # Verify shape
        self.assertEqual(transformed.shape, self.data.shape, "Shape should be preserved")
        
        # Verify values are in [0, 1]
        self.assertTrue(np.all(transformed >= 0), "Values should be >= 0")
        self.assertTrue(np.all(transformed <= 1), "Values should be <= 1")
    
    def test_inverse_transform(self):
        """Test inverse transformation."""
        preprocessor = DataPreprocessor(strategy='standard')
        
        # Fit and transform
        transformed = preprocessor.fit_transform(self.data)
        
        # Inverse transform
        original = preprocessor.inverse_transform(transformed)
        
        # Verify reconstruction
        self.assertTrue(np.allclose(original, self.data), "Should reconstruct original")
    
    def test_normalization_params(self):
        """Test normalization parameter retrieval."""
        preprocessor = DataPreprocessor(strategy='standard')
        preprocessor.fit(self.data)
        
        params = preprocessor.get_normalization_params()
        
        # Verify all parameters present
        expected_keys = ['strategy', 'fitted', 'mean', 'std', 'min', 'max']
        for key in expected_keys:
            self.assertIn(key, params, f"Missing parameter: {key}")


def run_tests():
    """Run all data loader tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestFederatedDataset))
    suite.addTests(loader.loadTestsFromTestCase(TestDataLoaderManager))
    suite.addTests(loader.loadTestsFromTestCase(TestFederatedPartitioner))
    suite.addTests(loader.loadTestsFromTestCase(TestDatasetUtils))
    suite.addTests(loader.loadTestsFromTestCase(TestDatasetValidator))
    suite.addTests(loader.loadTestsFromTestCase(TestDataPreprocessor))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)