#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Preprocessing for NTK-SURGERY
Provides preprocessing utilities for federated learning datasets
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Union
from enum import Enum
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler

logger = logging.getLogger(__name__)


class NormalizationStrategy(Enum):
    """Enumeration of normalization strategies."""
    STANDARD = "standard"  # Z-score normalization
    MINMAX = "minmax"  # Min-Max scaling
    NONE = "none"  # No normalization


class DataPreprocessor:
    """
    Data preprocessor for NTK-SURGERY datasets.
    
    Provides methods for normalizing, augmenting, and transforming
    data before training or unlearning operations.
    
    Attributes:
        strategy (NormalizationStrategy): Normalization strategy
        fitted (bool): Whether preprocessor has been fitted
    """
    
    def __init__(self, strategy: str = "standard"):
        """
        Initialize DataPreprocessor.
        
        Args:
            strategy: Normalization strategy ('standard', 'minmax', 'none')
        """
        self.strategy = self._parse_strategy(strategy)
        self.fitted = False
        self.scaler = None
        self.data_mean = None
        self.data_std = None
        self.data_min = None
        self.data_max = None
        
        logger.info(f"Initialized DataPreprocessor with strategy={strategy}")
    
    def _parse_strategy(self, strategy: str) -> NormalizationStrategy:
        """Parse string strategy to NormalizationStrategy enum."""
        strategy_map = {
            'standard': NormalizationStrategy.STANDARD,
            'minmax': NormalizationStrategy.MINMAX,
            'none': NormalizationStrategy.NONE
        }
        
        if strategy not in strategy_map:
            raise ValueError(
                f"Unknown strategy: {strategy}. "
                f"Available: {list(strategy_map.keys())}"
            )
        
        return strategy_map[strategy]
    
    def fit(self, data: np.ndarray) -> 'DataPreprocessor':
        """
        Fit preprocessor on training data.
        
        Args:
            data: Training data array
            
        Returns:
            self for method chaining
        """
        if self.strategy == NormalizationStrategy.NONE:
            logger.debug("No normalization applied")
            self.fitted = True
            return self
        
        # Flatten data for fitting (preserve original shape)
        original_shape = data.shape
        flat_data = data.reshape(len(data), -1)
        
        if self.strategy == NormalizationStrategy.STANDARD:
            self.scaler = StandardScaler()
            self.scaler.fit(flat_data)
            self.data_mean = self.scaler.mean_
            self.data_std = self.scaler.scale_
        
        elif self.strategy == NormalizationStrategy.MINMAX:
            self.scaler = MinMaxScaler()
            self.scaler.fit(flat_data)
            self.data_min = self.scaler.data_min_
            self.data_max = self.scaler.data_max_
        
        self.fitted = True
        logger.info(f"Fitted preprocessor on data shape {original_shape}")
        
        return self
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Transform data using fitted preprocessor.
        
        Args:
            data: Data array to transform
            
        Returns:
            Transformed data array
            
        Raises:
            RuntimeError: If preprocessor not fitted
        """
        if not self.fitted:
            raise RuntimeError("Preprocessor not fitted. Call fit() first.")
        
        if self.strategy == NormalizationStrategy.NONE:
            return data
        
        original_shape = data.shape
        flat_data = data.reshape(len(data), -1)
        
        transformed = self.scaler.transform(flat_data)
        
        # Restore original shape
        return transformed.reshape(original_shape)
    
    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Fit and transform data in one step.
        
        Args:
            data: Data array
            
        Returns:
            Transformed data array
        """
        self.fit(data)
        return self.transform(data)
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Inverse transform normalized data.
        
        Args:
            data: Normalized data array
            
        Returns:
            Original scale data array
        """
        if not self.fitted:
            raise RuntimeError("Preprocessor not fitted. Call fit() first.")
        
        if self.strategy == NormalizationStrategy.NONE:
            return data
        
        original_shape = data.shape
        flat_data = data.reshape(len(data), -1)
        
        inverse = self.scaler.inverse_transform(flat_data)
        
        return inverse.reshape(original_shape)
    
    def get_normalization_params(self) -> Dict[str, Optional[np.ndarray]]:
        """
        Get normalization parameters.
        
        Returns:
            Dictionary with normalization parameters
        """
        return {
            'strategy': self.strategy.value,
            'fitted': self.fitted,
            'mean': self.data_mean,
            'std': self.data_std,
            'min': self.data_min,
            'max': self.data_max
        }
    
    @staticmethod
    def apply_data_augmentation(
        data: np.ndarray,
        augmentation_type: str = 'flip',
        seed: int = 42
    ) -> np.ndarray:
        """
        Apply data augmentation to increase dataset size.
        
        Args:
            data: Data array
            augmentation_type: Type of augmentation ('flip', 'rotate', 'noise')
            seed: Random seed
            
        Returns:
            Augmented data array
        """
        np.random.seed(seed)
        augmented = data.copy()
        
        if augmentation_type == 'flip':
            # Random horizontal flip
            if len(data.shape) == 4:  # NCHW or NHWC
                flip_mask = np.random.rand(len(data)) > 0.5
                augmented[flip_mask] = np.flip(augmented[flip_mask], axis=2)
        
        elif augmentation_type == 'rotate':
            # Random rotation (simplified)
            for i in range(len(augmented)):
                angle = np.random.uniform(-15, 15)
                # Rotation would require scipy.ndimage or similar
                logger.debug(f"Rotation augmentation not fully implemented")
        
        elif augmentation_type == 'noise':
            # Add Gaussian noise
            noise = np.random.normal(0, 0.01, augmented.shape)
            augmented = np.clip(augmented + noise, 0, 1)
        
        logger.info(f"Applied {augmentation_type} augmentation to {len(data)} samples")
        
        return augmented
    
    @staticmethod
    def balance_dataset(
        data: np.ndarray,
        targets: np.ndarray,
        method: str = 'undersample'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Balance dataset to address class imbalance.
        
        Args:
            data: Data array
            targets: Targets array
            method: Balancing method ('undersample', 'oversample')
            
        Returns:
            Balanced (data, targets) tuple
        """
        unique, counts = np.unique(targets, return_counts=True)
        min_count = counts.min()
        max_count = counts.max()
        
        logger.info(
            f"Original class distribution: min={min_count}, max={max_count}"
        )
        
        if method == 'undersample':
            # Reduce majority classes
            balanced_indices = []
            
            for cls in unique:
                cls_indices = np.where(targets == cls)[0]
                
                if len(cls_indices) > min_count:
                    selected = np.random.choice(
                        cls_indices, 
                        size=min_count, 
                        replace=False
                    )
                else:
                    selected = cls_indices
                
                balanced_indices.extend(selected)
            
            balanced_indices = np.array(balanced_indices)
            
        elif method == 'oversample':
            # Increase minority classes
            balanced_indices = []
            
            for cls in unique:
                cls_indices = np.where(targets == cls)[0]
                
                if len(cls_indices) < max_count:
                    # Sample with replacement
                    additional = np.random.choice(
                        cls_indices, 
                        size=max_count - len(cls_indices),
                        replace=True
                    )
                    balanced_indices.extend(cls_indices.tolist())
                    balanced_indices.extend(additional.tolist())
                else:
                    balanced_indices.extend(cls_indices.tolist())
            
            balanced_indices = np.array(balanced_indices)
        
        else:
            raise ValueError(f"Unknown balancing method: {method}")
        
        # Shuffle
        np.random.shuffle(balanced_indices)
        
        return data[balanced_indices], targets[balanced_indices]
    
    @staticmethod
    def remove_outliers(
        data: np.ndarray,
        targets: np.ndarray,
        threshold: float = 3.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Remove outliers from dataset using z-score method.
        
        Args:
            data: Data array
            targets: Targets array
            threshold: Z-score threshold
            
        Returns:
            Tuple of (cleaned_data, cleaned_targets, outlier_mask)
        """
        # Compute z-scores
        flat_data = data.reshape(len(data), -1)
        z_scores = np.abs((flat_data - np.mean(flat_data, axis=0)) / 
                         (np.std(flat_data, axis=0) + 1e-8))
        
        # Identify outliers
        outlier_mask = np.any(z_scores > threshold, axis=1)
        
        # Remove outliers
        clean_mask = ~outlier_mask
        cleaned_data = data[clean_mask]
        cleaned_targets = targets[clean_mask]
        
        logger.info(
            f"Removed {np.sum(outlier_mask)} outliers "
            f"({100 * np.mean(outlier_mask):.2f}%)"
        )
        
        return cleaned_data, cleaned_targets, outlier_mask
    
    def save_preprocessor(self, path: str):
        """
        Save preprocessor state to file.
        
        Args:
            path: Path to save preprocessor
        """
        import pickle
        
        state = {
            'strategy': self.strategy.value,
            'fitted': self.fitted,
            'scaler': self.scaler,
            'data_mean': self.data_mean,
            'data_std': self.data_std,
            'data_min': self.data_min,
            'data_max': self.data_max
        }
        
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        
        logger.info(f"Preprocessor saved to {path}")
    
    @classmethod
    def load_preprocessor(cls, path: str) -> 'DataPreprocessor':
        """
        Load preprocessor state from file.
        
        Args:
            path: Path to load preprocessor from
            
        Returns:
            Loaded DataPreprocessor instance
        """
        import pickle
        
        with open(path, 'rb') as f:
            state = pickle.load(f)
        
        preprocessor = cls(strategy=state['strategy'])
        preprocessor.fitted = state['fitted']
        preprocessor.scaler = state['scaler']
        preprocessor.data_mean = state['data_mean']
        preprocessor.data_std = state['data_std']
        preprocessor.data_min = state['data_min']
        preprocessor.data_max = state['data_max']
        
        logger.info(f"Preprocessor loaded from {path}")
        
        return preprocessor