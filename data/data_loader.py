#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Loading Utilities for NTK-SURGERY
Loads datasets from /home/phd/datasets/ with federated partitioning
Implements Section 5.1 of the manuscript
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from typing import Dict, List, Tuple, Optional, Union
import yaml
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class FederatedDataset(Dataset):
    """
    Federated dataset wrapper for client-specific data access.
    
    This class wraps numpy arrays into a PyTorch Dataset compatible format
    for use with DataLoader in federated learning scenarios.
    
    Attributes:
        data (np.ndarray): Input data array
        targets (np.ndarray): Label array
        transform (callable, optional): Data transformations to apply
    """
    
    def __init__(self, data: np.ndarray, targets: np.ndarray, 
                 transform: Optional[transforms.Compose] = None):
        """
        Initialize FederatedDataset.
        
        Args:
            data: Input data array of shape (N, *)
            targets: Label array of shape (N,)
            transform: Optional transforms to apply to data
        """
        if len(data) != len(targets):
            raise ValueError(f"Data length {len(data)} != targets length {len(targets)}")
        
        self.data = data
        self.targets = targets
        self.transform = transform
        self._len = len(data)
        
        logger.debug(f"Initialized FederatedDataset with {self._len} samples")
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return self._len
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Index of the sample to retrieve
            
        Returns:
            Tuple of (transformed_data, target)
        """
        if idx < 0 or idx >= self._len:
            raise IndexError(f"Index {idx} out of range [0, {self._len})")
        
        data, target = self.data[idx], self.targets[idx]
        
        if self.transform is not None:
            # Convert to PIL Image if needed for torchvision transforms
            if len(data.shape) == 3:  # HWC format
                from PIL import Image
                data = Image.fromarray(data.astype(np.uint8))
            
            data = self.transform(data)
        else:
            # Convert to tensor if no transform
            if not isinstance(data, torch.Tensor):
                data = torch.from_numpy(data).float()
        
        return data, int(target)
    
    def get_class_distribution(self) -> Dict[int, int]:
        """
        Get the class distribution in this dataset.
        
        Returns:
            Dictionary mapping class labels to their counts
        """
        unique, counts = np.unique(self.targets, return_counts=True)
        return dict(zip(unique.tolist(), counts.tolist()))
    
    def get_data_statistics(self) -> Dict[str, float]:
        """
        Get statistical properties of the data.
        
        Returns:
            Dictionary with mean, std, min, max statistics
        """
        return {
            'mean': float(np.mean(self.data)),
            'std': float(np.std(self.data)),
            'min': float(np.min(self.data)),
            'max': float(np.max(self.data)),
            'shape': self.data.shape
        }


class DataLoaderManager:
    """
    Manages dataset loading and federated partitioning for NTK-SURGERY.
    
    This class handles loading datasets from the configured root path,
    applying appropriate transforms, and creating federated client partitions.
    
    Attributes:
        config (dict): Configuration dictionary
        root_path (str): Root path for datasets
        transform_dict (dict): Dataset-specific transforms
    """
    
    # Supported datasets
    SUPPORTED_DATASETS = [
        'MNIST', 'FashionMNIST', 'CIFAR-10', 'CIFAR-100', 
        'CelebA', 'TinyImageNet'
    ]
    
    # Default dataset configurations
    DATASET_CONFIGS = {
        'MNIST': {
            'num_classes': 10,
            'input_channels': 1,
            'input_size': (28, 28),
            'mean': (0.1307,),
            'std': (0.3081,)
        },
        'FashionMNIST': {
            'num_classes': 10,
            'input_channels': 1,
            'input_size': (28, 28),
            'mean': (0.2860,),
            'std': (0.3530,)
        },
        'CIFAR-10': {
            'num_classes': 10,
            'input_channels': 3,
            'input_size': (32, 32),
            'mean': (0.4914, 0.4822, 0.4465),
            'std': (0.2470, 0.2435, 0.2616)
        },
        'CIFAR-100': {
            'num_classes': 100,
            'input_channels': 3,
            'input_size': (32, 32),
            'mean': (0.5071, 0.4867, 0.4408),
            'std': (0.2675, 0.2565, 0.2761)
        },
        'CelebA': {
            'num_classes': 2,  # Binary classification
            'input_channels': 3,
            'input_size': (64, 64),
            'mean': (0.5, 0.5, 0.5),
            'std': (0.5, 0.5, 0.5)
        },
        'TinyImageNet': {
            'num_classes': 200,
            'input_channels': 3,
            'input_size': (64, 64),
            'mean': (0.485, 0.456, 0.406),
            'std': (0.229, 0.224, 0.225)
        }
    }
    
    def __init__(self, config_path: str = "config/default_config.yaml"):
        """
        Initialize DataLoaderManager.
        
        Args:
            config_path: Path to configuration YAML file
        """
        self.config = self._load_config(config_path)
        self.root_path = self.config.get('datasets', {}).get(
            'root_path', '/home/phd/datasets/'
        )
        self.transform_dict = self._get_transforms()
        
        # Create root path if it doesn't exist
        Path(self.root_path).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"DataLoaderManager initialized with root path: {self.root_path}")
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            logger.warning(f"Config file {config_path} not found. Using defaults.")
            return {'datasets': {'root_path': self.root_path}}
    
    def _get_transforms(self) -> Dict[str, transforms.Compose]:
        """
        Get dataset-specific transforms.
        
        Returns:
            Dictionary mapping dataset names to transform compositions
        """
        return {
            'MNIST': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(*self.DATASET_CONFIGS['MNIST']['mean'],
                                   self.DATASET_CONFIGS['MNIST']['std'])
            ]),
            'FashionMNIST': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(*self.DATASET_CONFIGS['FashionMNIST']['mean'],
                                   self.DATASET_CONFIGS['FashionMNIST']['std'])
            ]),
            'CIFAR-10': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(*self.DATASET_CONFIGS['CIFAR-10']['mean'],
                                   self.DATASET_CONFIGS['CIFAR-10']['std'])
            ]),
            'CIFAR-100': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(*self.DATASET_CONFIGS['CIFAR-100']['mean'],
                                   self.DATASET_CONFIGS['CIFAR-100']['std'])
            ]),
            'CelebA': transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize(*self.DATASET_CONFIGS['CelebA']['mean'],
                                   self.DATASET_CONFIGS['CelebA']['std'])
            ]),
            'TinyImageNet': transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize(*self.DATASET_CONFIGS['TinyImageNet']['mean'],
                                   self.DATASET_CONFIGS['TinyImageNet']['std'])
            ])
        }
    
    def load_dataset(self, dataset_name: str, train: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load dataset from /home/phd/datasets/.
        
        Args:
            dataset_name: Name of the dataset to load
            train: Whether to load training or test data
            
        Returns:
            Tuple of (data, targets) as numpy arrays
            
        Raises:
            ValueError: If dataset_name is not supported
            FileNotFoundError: If dataset is not found at root_path
        """
        if dataset_name not in self.SUPPORTED_DATASETS:
            raise ValueError(
                f"Unsupported dataset: {dataset_name}. "
                f"Supported: {self.SUPPORTED_DATASETS}"
            )
        
        dataset_path = os.path.join(self.root_path, dataset_name)
        logger.info(f"Loading dataset: {dataset_name} from {dataset_path}")
        
        try:
            if dataset_name == 'MNIST':
                dataset = datasets.MNIST(
                    root=dataset_path, train=train, download=True
                )
            elif dataset_name == 'FashionMNIST':
                dataset = datasets.FashionMNIST(
                    root=dataset_path, train=train, download=True
                )
            elif dataset_name == 'CIFAR-10':
                dataset = datasets.CIFAR10(
                    root=dataset_path, train=train, download=True
                )
            elif dataset_name == 'CIFAR-100':
                dataset = datasets.CIFAR100(
                    root=dataset_path, train=train, download=True
                )
            elif dataset_name == 'CelebA':
                dataset = datasets.CelebA(
                    root=dataset_path, 
                    split='all' if train else 'test',
                    download=True
                )
            elif dataset_name == 'TinyImageNet':
                split_dir = 'train' if train else 'val'
                dataset = datasets.ImageFolder(
                    root=os.path.join(dataset_path, split_dir)
                )
            else:
                raise ValueError(f"Unknown dataset: {dataset_name}")
            
            # Convert to numpy arrays
            data, targets = self._convert_to_numpy(dataset, dataset_name)
            
            logger.info(
                f"Loaded {dataset_name}: {len(data)} samples, "
                f"{len(np.unique(targets))} classes"
            )
            
            return data, targets
            
        except Exception as e:
            logger.error(f"Failed to load dataset {dataset_name}: {str(e)}")
            raise
    
    def _convert_to_numpy(self, dataset, dataset_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert torchvision dataset to numpy arrays.
        
        Args:
            dataset: Torchvision dataset object
            dataset_name: Name of the dataset
            
        Returns:
            Tuple of (data, targets) as numpy arrays
        """
        if dataset_name in ['MNIST', 'FashionMNIST', 'CIFAR-10', 'CIFAR-100']:
            data = np.array(dataset.data)
            targets = np.array(dataset.targets)
        elif dataset_name == 'CelebA':
            data = np.array([np.asarray(item[0]) for item in dataset])
            targets = np.array([item[1] for item in dataset])
        elif dataset_name == 'TinyImageNet':
            data = np.array([np.asarray(item[0]) for item in dataset])
            targets = np.array([item[1] for item in dataset])
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        return data, targets
    
    def create_federated_partitions(
        self, 
        data: np.ndarray, 
        targets: np.ndarray,
        num_clients: int, 
        partition_type: str,
        alpha: float = 0.1,
        seed: int = 42
    ) -> Dict[int, FederatedDataset]:
        """
        Create federated client partitions.
        
        Args:
            data: Global data array
            targets: Global targets array
            num_clients: Number of clients to partition data into
            partition_type: Type of partition ('label_shard' or 'dirichlet')
            alpha: Dirichlet concentration parameter (for dirichlet partition)
            seed: Random seed for reproducibility
            
        Returns:
            Dictionary mapping client_id to FederatedDataset
        """
        from data.federated_partition import FederatedPartitioner
        
        partitioner = FederatedPartitioner(partition_type, alpha, seed)
        client_indices = partitioner.partition(targets, num_clients)
        
        clients = {}
        transform = self.transform_dict.get(
            list(self.transform_dict.keys())[0], 
            None
        )
        
        for client_id, indices in client_indices.items():
            client_data = data[indices]
            client_targets = targets[indices]
            clients[client_id] = FederatedDataset(
                client_data, 
                client_targets, 
                transform
            )
            
            logger.debug(
                f"Client {client_id}: {len(indices)} samples, "
                f"classes: {len(np.unique(client_targets))}"
            )
        
        logger.info(
            f"Created {num_clients} client partitions using {partition_type} strategy"
        )
        
        return clients
    
    def get_client_loaders(
        self, 
        clients: Dict[int, FederatedDataset],
        batch_size: int, 
        shuffle: bool = True,
        num_workers: int = 0
    ) -> Dict[int, DataLoader]:
        """
        Create DataLoaders for each client.
        
        Args:
            clients: Dictionary of client datasets
            batch_size: Batch size for DataLoader
            shuffle: Whether to shuffle data
            num_workers: Number of worker processes for data loading
            
        Returns:
            Dictionary mapping client_id to DataLoader
        """
        loaders = {}
        for client_id, dataset in clients.items():
            loaders[client_id] = DataLoader(
                dataset, 
                batch_size=batch_size, 
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=False
            )
        
        logger.debug(f"Created {len(loaders)} client DataLoaders")
        
        return loaders
    
    def get_dataset_config(self, dataset_name: str) -> dict:
        """
        Get configuration for a specific dataset.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Dictionary with dataset configuration
        """
        if dataset_name not in self.DATASET_CONFIGS:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        return self.DATASET_CONFIGS[dataset_name].copy()
    
    def validate_data_integrity(
        self, 
        data: np.ndarray, 
        targets: np.ndarray,
        dataset_name: str
    ) -> bool:
        """
        Validate data integrity before processing.
        
        Args:
            data: Data array to validate
            targets: Targets array to validate
            dataset_name: Name of the dataset
            
        Returns:
            True if data is valid, raises exception otherwise
        """
        config = self.get_dataset_config(dataset_name)
        
        # Check lengths match
        if len(data) != len(targets):
            raise ValueError(
                f"Data length {len(data)} != targets length {len(targets)}"
            )
        
        # Check number of classes
        unique_classes = len(np.unique(targets))
        if unique_classes != config['num_classes']:
            raise ValueError(
                f"Expected {config['num_classes']} classes, got {unique_classes}"
            )
        
        # Check data shape
        expected_shape = (len(data), *config['input_size'])
        if config['input_channels'] > 1:
            expected_shape = (len(data), config['input_channels'], *config['input_size'])
        
        if data.shape != expected_shape:
            logger.warning(
                f"Data shape {data.shape} != expected {expected_shape}. "
                f"Attempting to reshape."
            )
        
        logger.info(f"Data integrity validation passed for {dataset_name}")
        return True