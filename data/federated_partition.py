#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Federated Data Partitioning for NTK-SURGERY
Implements label shard and Dirichlet partitioning strategies
Implements Section 5.1 of the manuscript
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from enum import Enum
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class PartitionStrategy(Enum):
    """Enumeration of partition strategies."""
    LABEL_SHARD = "label_shard"
    DIRICHLET = "dirichlet"
    UNIFORM = "uniform"
    PATHOLOGICAL = "pathological"


class FederatedPartitioner:
    """
    Federated data partitioner for creating client datasets.
    
    Implements various partitioning strategies to simulate realistic
    federated learning scenarios with non-IID data distributions.
    
    Attributes:
        strategy (PartitionStrategy): Partition strategy to use
        alpha (float): Dirichlet concentration parameter
        seed (int): Random seed for reproducibility
    """
    
    def __init__(
        self, 
        strategy: str = "label_shard",
        alpha: float = 0.1,
        seed: int = 42
    ):
        """
        Initialize FederatedPartitioner.
        
        Args:
            strategy: Partition strategy ('label_shard' or 'dirichlet')
            alpha: Dirichlet concentration parameter (for dirichlet)
            seed: Random seed for reproducibility
        """
        self.strategy = self._parse_strategy(strategy)
        self.alpha = alpha
        self.seed = seed
        np.random.seed(seed)
        
        logger.info(
            f"Initialized FederatedPartitioner with strategy={strategy}, "
            f"alpha={alpha}, seed={seed}"
        )
    
    def _parse_strategy(self, strategy: str) -> PartitionStrategy:
        """Parse string strategy to PartitionStrategy enum."""
        strategy_map = {
            'label_shard': PartitionStrategy.LABEL_SHARD,
            'dirichlet': PartitionStrategy.DIRICHLET,
            'uniform': PartitionStrategy.UNIFORM,
            'pathological': PartitionStrategy.PATHOLOGICAL
        }
        
        if strategy not in strategy_map:
            raise ValueError(
                f"Unknown strategy: {strategy}. "
                f"Available: {list(strategy_map.keys())}"
            )
        
        return strategy_map[strategy]
    
    def partition(
        self, 
        targets: np.ndarray, 
        num_clients: int
    ) -> Dict[int, np.ndarray]:
        """
        Partition data indices among clients.
        
        Args:
            targets: Array of class labels
            num_clients: Number of clients to partition into
            
        Returns:
            Dictionary mapping client_id to array of sample indices
        """
        if num_clients <= 0:
            raise ValueError(f"num_clients must be positive, got {num_clients}")
        
        if len(targets) == 0:
            raise ValueError("targets array is empty")
        
        logger.info(
            f"Partitioning {len(targets)} samples across {num_clients} clients "
            f"using {self.strategy.value} strategy"
        )
        
        if self.strategy == PartitionStrategy.LABEL_SHARD:
            client_indices = self._label_shard_partition(targets, num_clients)
        elif self.strategy == PartitionStrategy.DIRICHLET:
            client_indices = self._dirichlet_partition(targets, num_clients)
        elif self.strategy == PartitionStrategy.UNIFORM:
            client_indices = self._uniform_partition(targets, num_clients)
        elif self.strategy == PartitionStrategy.PATHOLOGICAL:
            client_indices = self._pathological_partition(targets, num_clients)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        
        # Validate partition
        self._validate_partition(client_indices, len(targets))
        
        return client_indices
    
    def _label_shard_partition(
        self, 
        targets: np.ndarray, 
        num_clients: int
    ) -> Dict[int, np.ndarray]:
        """
        Partition data by sharding labels across clients.
        
        Each client receives samples from a subset of classes.
        Commonly used for MNIST and FashionMNIST.
        
        Args:
            targets: Array of class labels
            num_clients: Number of clients
            
        Returns:
            Dictionary mapping client_id to sample indices
        """
        num_classes = len(np.unique(targets))
        samples_per_client = len(targets) // num_clients
        
        # Sort indices by class
        class_indices = defaultdict(list)
        for idx, label in enumerate(targets):
            class_indices[label].append(idx)
        
        client_indices = defaultdict(list)
        
        # Distribute classes to clients
        classes_per_client = max(1, num_classes // num_clients)
        
        for client_id in range(num_clients):
            start_class = (client_id * classes_per_client) % num_classes
            
            for class_offset in range(classes_per_client):
                class_label = (start_class + class_offset) % num_classes
                
                if class_label in class_indices:
                    available = class_indices[class_label]
                    num_samples = min(
                        samples_per_client // classes_per_client,
                        len(available)
                    )
                    
                    selected = np.random.choice(
                        available, 
                        size=num_samples, 
                        replace=False
                    ).tolist()
                    
                    client_indices[client_id].extend(selected)
                    
                    # Remove selected from available
                    class_indices[class_label] = [
                        i for i in available if i not in selected
                    ]
        
        # Convert to numpy arrays
        return {
            cid: np.array(indices) for cid, indices in client_indices.items()
        }
    
    def _dirichlet_partition(
        self, 
        targets: np.ndarray, 
        num_clients: int
    ) -> Dict[int, np.ndarray]:
        """
        Partition data using Dirichlet distribution for non-IID simulation.
        
        Each client receives a mixture of all classes with proportions
        drawn from Dirichlet(α). Lower α creates more heterogeneous partitions.
        
        Args:
            targets: Array of class labels
            num_clients: Number of clients
            
        Returns:
            Dictionary mapping client_id to sample indices
        """
        num_classes = len(np.unique(targets))
        
        # Sort indices by class
        class_indices = defaultdict(list)
        for idx, label in enumerate(targets):
            class_indices[label].append(idx)
        
        client_indices = defaultdict(list)
        
        # Sample Dirichlet distribution for each client
        dirichlet_samples = np.random.dirichlet(
            np.repeat(self.alpha, num_classes), 
            num_clients
        )
        
        for class_label in range(num_classes):
            available = class_indices[class_label]
            num_samples = len(available)
            
            if num_samples == 0:
                continue
            
            # Get proportions for this class across clients
            proportions = dirichlet_samples[:, class_label]
            
            # Normalize to sum to 1
            proportions = proportions / proportions.sum()
            
            # Allocate samples to clients
            samples_allocated = np.random.multinomial(
                num_samples, 
                proportions
            )
            
            # Shuffle available indices
            np.random.shuffle(available)
            
            start_idx = 0
            for client_id, num_alloc in enumerate(samples_allocated):
                if num_alloc > 0:
                    end_idx = start_idx + num_alloc
                    selected = available[start_idx:end_idx]
                    client_indices[client_id].extend(selected)
                    start_idx = end_idx
        
        # Convert to numpy arrays
        return {
            cid: np.array(indices) for cid, indices in client_indices.items()
        }
    
    def _uniform_partition(
        self, 
        targets: np.ndarray, 
        num_clients: int
    ) -> Dict[int, np.ndarray]:
        """
        Partition data uniformly at random (IID baseline).
        
        Args:
            targets: Array of class labels
            num_clients: Number of clients
            
        Returns:
            Dictionary mapping client_id to sample indices
        """
        num_samples = len(targets)
        samples_per_client = num_samples // num_clients
        
        # Shuffle all indices
        all_indices = np.arange(num_samples)
        np.random.shuffle(all_indices)
        
        client_indices = {}
        
        for client_id in range(num_clients):
            start_idx = client_id * samples_per_client
            end_idx = start_idx + samples_per_client
            
            if client_id == num_clients - 1:
                # Last client gets remaining samples
                end_idx = num_samples
            
            client_indices[client_id] = all_indices[start_idx:end_idx]
        
        return client_indices
    
    def _pathological_partition(
        self, 
        targets: np.ndarray, 
        num_clients: int
    ) -> Dict[int, np.ndarray]:
        """
        Pathological non-IID partition (extreme case).
        
        Each client receives samples from only 1-2 classes.
        Creates highly heterogeneous partitions for stress testing.
        
        Args:
            targets: Array of class labels
            num_clients: Number of clients
            
        Returns:
            Dictionary mapping client_id to sample indices
        """
        num_classes = len(np.unique(targets))
        
        # Sort indices by class
        class_indices = defaultdict(list)
        for idx, label in enumerate(targets):
            class_indices[label].append(idx)
        
        client_indices = defaultdict(list)
        classes = list(class_indices.keys())
        
        # Assign 1-2 classes per client
        for client_id in range(num_clients):
            num_classes_for_client = np.random.randint(1, 3)
            selected_classes = np.random.choice(
                classes, 
                size=min(num_classes_for_client, len(classes)),
                replace=False
            )
            
            for class_label in selected_classes:
                available = class_indices[class_label]
                
                if len(available) > 0:
                    # Take all or subset
                    if len(available) > 100:
                        selected = np.random.choice(
                            available, 
                            size=100, 
                            replace=False
                        ).tolist()
                    else:
                        selected = available
                    
                    client_indices[client_id].extend(selected)
        
        # Convert to numpy arrays
        return {
            cid: np.array(indices) for cid, indices in client_indices.items()
        }
    
    def _validate_partition(
        self, 
        client_indices: Dict[int, np.ndarray], 
        total_samples: int
    ):
        """
        Validate partition integrity.
        
        Args:
            client_indices: Partition dictionary
            total_samples: Total number of samples
            
        Raises:
            ValueError: If partition is invalid
        """
        # Check no overlap
        all_indices = []
        for indices in client_indices.values():
            all_indices.extend(indices.tolist())
        
        if len(all_indices) != len(set(all_indices)):
            raise ValueError("Partition has overlapping indices")
        
        # Check coverage
        if len(all_indices) != total_samples:
            logger.warning(
                f"Partition covers {len(all_indices)} of {total_samples} samples"
            )
        
        # Check no empty clients
        for client_id, indices in client_indices.items():
            if len(indices) == 0:
                logger.warning(f"Client {client_id} has no samples")
        
        logger.debug("Partition validation passed")
    
    def compute_partition_statistics(
        self, 
        client_indices: Dict[int, np.ndarray],
        targets: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute statistics about the partition.
        
        Args:
            client_indices: Partition dictionary
            targets: Array of class labels
            
        Returns:
            Dictionary with partition statistics
        """
        client_sizes = [len(indices) for indices in client_indices.values()]
        
        # Compute class distribution per client
        client_class_counts = []
        for client_id, indices in client_indices.items():
            client_targets = targets[indices]
            unique_classes = len(np.unique(client_targets))
            client_class_counts.append(unique_classes)
        
        stats = {
            'num_clients': len(client_indices),
            'total_samples': sum(client_sizes),
            'min_client_size': min(client_sizes),
            'max_client_size': max(client_sizes),
            'mean_client_size': float(np.mean(client_sizes)),
            'std_client_size': float(np.std(client_sizes)),
            'min_classes_per_client': min(client_class_counts),
            'max_classes_per_client': max(client_class_counts),
            'mean_classes_per_client': float(np.mean(client_class_counts))
        }
        
        return stats
    
    def visualize_partition(
        self, 
        client_indices: Dict[int, np.ndarray],
        targets: np.ndarray,
        save_path: Optional[str] = None
    ):
        """
        Visualize partition distribution (requires matplotlib).
        
        Args:
            client_indices: Partition dictionary
            targets: Array of class labels
            save_path: Optional path to save figure
        """
        try:
            import matplotlib.pyplot as plt
            
            num_clients = len(client_indices)
            num_classes = len(np.unique(targets))
            
            # Create heatmap of class distribution
            distribution = np.zeros((num_clients, num_classes))
            
            for client_id, indices in client_indices.items():
                client_targets = targets[indices]
                for cls in range(num_classes):
                    distribution[client_id, cls] = np.sum(client_targets == cls)
            
            # Normalize
            row_sums = distribution.sum(axis=1, keepdims=True)
            distribution = distribution / (row_sums + 1e-8)
            
            # Plot
            fig, ax = plt.subplots(figsize=(10, 6))
            im = ax.imshow(distribution, cmap='viridis', aspect='auto')
            
            ax.set_xlabel('Class')
            ax.set_ylabel('Client')
            ax.set_title('Class Distribution Across Clients')
            
            plt.colorbar(im, ax=ax, label='Proportion')
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Partition visualization saved to {save_path}")
            
            plt.close()
            
        except ImportError:
            logger.warning("matplotlib not available. Skipping visualization.")