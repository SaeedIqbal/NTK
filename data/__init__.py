#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Module for NTK-SURGERY
Provides data loading, partitioning, and preprocessing utilities
"""

from data.data_loader import DataLoaderManager, FederatedDataset
from data.dataset_utils import DatasetUtils, DatasetValidator
from data.federated_partition import FederatedPartitioner, PartitionStrategy
from data.preprocessor import DataPreprocessor, NormalizationStrategy

__all__ = [
    'DataLoaderManager',
    'FederatedDataset',
    'DatasetUtils',
    'DatasetValidator',
    'FederatedPartitioner',
    'PartitionStrategy',
    'DataPreprocessor',
    'NormalizationStrategy'
]

__version__ = '1.0.0'
__author__ = 'NTK-SURGERY Team'