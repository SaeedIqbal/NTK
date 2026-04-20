#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset Utilities for NTK-SURGERY
Provides utility functions for dataset manipulation and analysis
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
import logging
from collections import Counter

logger = logging.getLogger(__name__)


class DatasetUtils:
    """
    Utility class for dataset operations and analysis.
    
    Provides static methods for common dataset operations used in
    federated learning and unlearning scenarios.
    """
    
    @staticmethod
    def compute_class_distribution(targets: np.ndarray) -> Dict[int, float]:
        """
        Compute class distribution as percentages.
        
        Args:
            targets: Array of class labels
            
        Returns:
            Dictionary mapping class labels to their percentage of total
        """
        total = len(targets)
        counter = Counter(targets)
        
        distribution = {
            int(cls): count / total for cls, count in counter.items()
        }
        
        return distribution
    
    @staticmethod
    def compute_label_imbalance(targets: np.ndarray) -> float:
        """
        Compute label imbalance ratio.
        
        Ratio = max_class_count / min_class_count
        
        Args:
            targets: Array of class labels
            
        Returns:
            Imbalance ratio (1.0 = perfectly balanced)
        """
        counter = Counter(targets)
        counts = list(counter.values())
        
        if len(counts) == 0:
            return 1.0
        
        return float(max(counts) / (min(counts) + 1e-8))
    
    @staticmethod
    def split_data_by_client(
        data: np.ndarray, 
        targets: np.ndarray,
        client_indices: Dict[int, np.ndarray]
    ) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """
        Split global data into client-specific subsets.
        
        Args:
            data: Global data array
            targets: Global targets array
            client_indices: Dict mapping client_id to sample indices
            
        Returns:
            Dict mapping client_id to (data, targets) tuple
        """
        client_data = {}
        
        for client_id, indices in client_indices.items():
            client_data[client_id] = (data[indices], targets[indices])
        
        return client_data
    
    @staticmethod
    def merge_client_data(
        client_data: Dict[int, Tuple[np.ndarray, np.ndarray]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Merge client-specific data into global arrays.
        
        Args:
            client_data: Dict mapping client_id to (data, targets)
            
        Returns:
            Tuple of (merged_data, merged_targets)
        """
        all_data = []
        all_targets = []
        
        for client_id in sorted(client_data.keys()):
            data, targets = client_data[client_id]
            all_data.append(data)
            all_targets.append(targets)
        
        return np.vstack(all_data), np.hstack(all_targets)
    
    @staticmethod
    def compute_data_statistics(data: np.ndarray) -> Dict[str, float]:
        """
        Compute statistical properties of data.
        
        Args:
            data: Data array
            
        Returns:
            Dictionary with statistical measures
        """
        return {
            'mean': float(np.mean(data)),
            'std': float(np.std(data)),
            'variance': float(np.var(data)),
            'min': float(np.min(data)),
            'max': float(np.max(data)),
            'median': float(np.median(data)),
            'skewness': float(np.mean(((data - np.mean(data)) / (np.std(data) + 1e-8)) ** 3)),
            'kurtosis': float(np.mean(((data - np.mean(data)) / (np.std(data) + 1e-8)) ** 4) - 3)
        }
    
    @staticmethod
    def compute_feature_correlations(data: np.ndarray) -> np.ndarray:
        """
        Compute feature correlation matrix.
        
        Args:
            data: Data array of shape (N, D)
            
        Returns:
            Correlation matrix of shape (D, D)
        """
        if len(data.shape) > 2:
            # Flatten spatial dimensions for images
            data = data.reshape(data.shape[0], -1)
        
        return np.corrcoef(data.T)
    
    @staticmethod
    def detect_outliers(
        data: np.ndarray, 
        threshold: float = 3.0
    ) -> np.ndarray:
        """
        Detect outliers using z-score method.
        
        Args:
            data: Data array
            threshold: Z-score threshold for outlier detection
            
        Returns:
            Boolean array indicating outlier positions
        """
        z_scores = np.abs((data - np.mean(data)) / (np.std(data) + 1e-8))
        return z_scores > threshold
    
    @staticmethod
    def normalize_targets(targets: np.ndarray, mapping: Optional[Dict] = None) -> np.ndarray:
        """
        Normalize target labels to consecutive integers starting from 0.
        
        Args:
            targets: Array of target labels
            mapping: Optional existing mapping to use
            
        Returns:
            Normalized targets and mapping dictionary
        """
        if mapping is None:
            unique = np.unique(targets)
            mapping = {old: new for new, old in enumerate(unique)}
        
        normalized = np.array([mapping[t] for t in targets])
        
        return normalized, mapping
    
    @staticmethod
    def stratified_sample(
        data: np.ndarray, 
        targets: np.ndarray,
        sample_size: int,
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform stratified sampling to maintain class distribution.
        
        Args:
            data: Data array
            targets: Targets array
            sample_size: Number of samples to draw
            random_state: Random seed
            
        Returns:
            Sampled (data, targets) tuple
        """
        np.random.seed(random_state)
        
        classes = np.unique(targets)
        class_counts = {cls: np.sum(targets == cls) for cls in classes}
        total = len(targets)
        
        sampled_indices = []
        
        for cls in classes:
            cls_indices = np.where(targets == cls)[0]
            cls_sample_size = int((class_counts[cls] / total) * sample_size)
            
            sampled_cls_indices = np.random.choice(
                cls_indices, 
                size=min(cls_sample_size, len(cls_indices)),
                replace=False
            )
            sampled_indices.extend(sampled_cls_indices)
        
        sampled_indices = np.array(sampled_indices)
        
        return data[sampled_indices], targets[sampled_indices]


class DatasetValidator:
    """
    Validator class for dataset integrity and quality checks.
    
    Provides methods to validate dataset properties before training
    or unlearning operations.
    """
    
    def __init__(self, min_samples_per_class: int = 10):
        """
        Initialize DatasetValidator.
        
        Args:
            min_samples_per_class: Minimum required samples per class
        """
        self.min_samples_per_class = min_samples_per_class
        self.validation_results = {}
    
    def validate_federated_partition(
        self,
        clients: Dict[int, Tuple[np.ndarray, np.ndarray]],
        expected_num_classes: int
    ) -> Dict[str, bool]:
        """
        Validate federated data partition quality.
        
        Args:
            clients: Dict of client data
            expected_num_classes: Expected number of classes
            
        Returns:
            Dictionary of validation results
        """
        results = {
            'all_clients_have_data': True,
            'all_classes_represented': True,
            'min_samples_satisfied': True,
            'no_empty_clients': True
        }
        
        all_targets = []
        
        for client_id, (data, targets) in clients.items():
            # Check for empty clients
            if len(data) == 0:
                results['no_empty_clients'] = False
                logger.warning(f"Client {client_id} has no data")
            
            # Check minimum samples
            if len(data) < self.min_samples_per_class:
                results['min_samples_satisfied'] = False
                logger.warning(
                    f"Client {client_id} has {len(data)} samples, "
                    f"below minimum {self.min_samples_per_class}"
                )
            
            all_targets.extend(targets)
        
        # Check class coverage
        all_classes = np.unique(all_targets)
        if len(all_classes) < expected_num_classes:
            results['all_classes_represented'] = False
            logger.warning(
                f"Only {len(all_classes)} of {expected_num_classes} "
                f"classes represented"
            )
        
        self.validation_results = results
        return results
    
    def validate_data_quality(
        self,
        data: np.ndarray,
        targets: np.ndarray,
        dataset_name: str
    ) -> Dict[str, bool]:
        """
        Validate data quality metrics.
        
        Args:
            data: Data array
            targets: Targets array
            dataset_name: Name of dataset for logging
            
        Returns:
            Dictionary of validation results
        """
        results = {
            'no_nan_values': True,
            'no_inf_values': True,
            'consistent_lengths': True,
            'valid_targets': True
        }
        
        # Check for NaN values
        if np.any(np.isnan(data)):
            results['no_nan_values'] = False
            logger.warning(f"{dataset_name}: Contains NaN values")
        
        # Check for Inf values
        if np.any(np.isinf(data)):
            results['no_inf_values'] = False
            logger.warning(f"{dataset_name}: Contains Inf values")
        
        # Check consistent lengths
        if len(data) != len(targets):
            results['consistent_lengths'] = False
            logger.warning(
                f"{dataset_name}: Data length {len(data)} != "
                f"targets length {len(targets)}"
            )
        
        # Check valid targets (non-negative integers)
        if np.any(targets < 0) or not np.all(targets == targets.astype(int)):
            results['valid_targets'] = False
            logger.warning(f"{dataset_name}: Invalid target values")
        
        self.validation_results = results
        return results
    
    def get_validation_summary(self) -> str:
        """
        Get human-readable validation summary.
        
        Returns:
            Formatted string summary of validation results
        """
        if not self.validation_results:
            return "No validation performed yet"
        
        passed = sum(self.validation_results.values())
        total = len(self.validation_results)
        
        summary = f"Validation: {passed}/{total} checks passed\n"
        
        for check, result in self.validation_results.items():
            status = "✓" if result else "✗"
            summary += f"  {status} {check}\n"
        
        return summary