#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convolutional Neural Network for NTK-SURGERY
Implements CNN architecture compatible with NTK analysis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import logging
from dataclasses import dataclass

from models.ntk_model import NTKModel

logger = logging.getLogger(__name__)


@dataclass
class CNNConfig:
    """
    Configuration class for CNN architecture.
    
    Attributes:
        input_channels: Number of input channels
        num_classes: Number of output classes
        width_multiplier: Width multiplier for network
        use_batch_norm: Whether to use batch normalization
        dropout_rate: Dropout rate
    """
    input_channels: int = 3
    num_classes: int = 10
    width_multiplier: int = 4
    use_batch_norm: bool = True
    dropout_rate: float = 0.0
    
    def __post_init__(self):
        """Validate configuration."""
        if self.input_channels <= 0:
            raise ValueError("input_channels must be positive")
        if self.num_classes <= 0:
            raise ValueError("num_classes must be positive")
        if self.width_multiplier <= 0:
            raise ValueError("width_multiplier must be positive")
        if not 0 <= self.dropout_rate < 1:
            raise ValueError("dropout_rate must be in [0, 1)")


class CNNBlock(nn.Module):
    """
    Convolutional block with optional batch normalization.
    
    Attributes:
        conv: Convolutional layer
        bn: Batch normalization layer (optional)
        dropout: Dropout layer (optional)
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        use_batch_norm: bool = True,
        dropout_rate: float = 0.0
    ):
        """
        Initialize CNNBlock.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Convolution kernel size
            padding: Convolution padding
            use_batch_norm: Whether to use batch normalization
            dropout_rate: Dropout rate
        """
        super().__init__()
        
        self.conv = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=kernel_size,
            padding=padding,
            bias=not use_batch_norm
        )
        
        self.bn = nn.BatchNorm2d(out_channels) if use_batch_norm else None
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None
        
        logger.debug(
            f"Created CNNBlock: {in_channels} -> {out_channels}, "
            f"BN={use_batch_norm}, Dropout={dropout_rate}"
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through block.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        x = self.conv(x)
        
        if self.bn is not None:
            x = self.bn(x)
        
        x = F.relu(x)
        
        if self.dropout is not None:
            x = self.dropout(x)
        
        return x


class CNN(NTKModel):
    """
    Convolutional Neural Network for NTK-SURGERY.
    
    Implements a CNN architecture compatible with NTK analysis
    and federated unlearning operations.
    
    Architecture:
        - 4 Convolutional blocks
        - 2 Fully connected layers
        - Optional batch normalization
        - Optional dropout
    
    Attributes:
        config (CNNConfig): Network configuration
        features (nn.Sequential): Convolutional feature extractor
        classifier (nn.Sequential): Classification head
    """
    
    def __init__(self, config: Optional[CNNConfig] = None):
        """
        Initialize CNN.
        
        Args:
            config: CNN configuration (uses defaults if None)
        """
        if config is None:
            config = CNNConfig()
        
        self.config = config
        
        # Build network
        model = self._build_network()
        
        super().__init__(model)
        
        logger.info(
            f"Initialized CNN with config: "
            f"channels={config.input_channels}, classes={config.num_classes}, "
            f"width={config.width_multiplier}"
        )
    
    def _build_network(self) -> nn.Module:
        """
        Build CNN architecture.
        
        Returns:
            PyTorch module
        """
        # Width multiplier for network capacity
        w = self.config.width_multiplier
        
        # Convolutional feature extractor
        features = nn.Sequential(
            # Block 1: 3 -> 64*w
            CNNBlock(
                self.config.input_channels, 
                64 * w,
                use_batch_norm=self.config.use_batch_norm,
                dropout_rate=self.config.dropout_rate
            ),
            nn.MaxPool2d(2),
            
            # Block 2: 64*w -> 128*w
            CNNBlock(
                64 * w, 
                128 * w,
                use_batch_norm=self.config.use_batch_norm,
                dropout_rate=self.config.dropout_rate
            ),
            nn.MaxPool2d(2),
            
            # Block 3: 128*w -> 256*w
            CNNBlock(
                128 * w, 
                256 * w,
                use_batch_norm=self.config.use_batch_norm,
                dropout_rate=self.config.dropout_rate
            ),
            nn.MaxPool2d(2),
            
            # Block 4: 256*w -> 512*w
            CNNBlock(
                256 * w, 
                512 * w,
                use_batch_norm=self.config.use_batch_norm,
                dropout_rate=self.config.dropout_rate
            ),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Classification head
        classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * w, 256 * w),
            nn.ReLU(),
            nn.Dropout(self.config.dropout_rate),
            nn.Linear(256 * w, self.config.num_classes)
        )
        
        # Combine into single module
        model = nn.Module()
        model.features = features
        model.classifier = classifier
        model.forward = self._model_forward
        
        return model
    
    def _model_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Model forward pass.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        x = self.features(x)
        x = self.classifier(x)
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through CNN.
        
        Args:
            x: Input tensor of shape (N, C, H, W)
            
        Returns:
            Output tensor of shape (N, num_classes)
        """
        return self._model_forward(x)
    
    def functional_forward(
        self, 
        x: torch.Tensor, 
        params: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Forward pass with explicit parameters for Jacobian computation.
        
        This is critical for NTK computation where we need to compute
        gradients with respect to all parameters.
        
        Args:
            x: Input tensor
            params: Dictionary of parameter tensors
            
        Returns:
            Output tensor
        """
        # This is a simplified version - for full functional forward,
        # we would need to implement parameterized layers
        # For NTK-SURGERY, we use the standard forward with set_parameters
        
        self.set_parameters(params)
        return self.forward(x)
    
    def get_feature_extractor(self) -> nn.Sequential:
        """
        Get feature extractor module.
        
        Returns:
            Feature extractor (convolutional layers)
        """
        return self.model.features
    
    def get_classifier(self) -> nn.Sequential:
        """
        Get classifier module.
        
        Returns:
            Classifier (fully connected layers)
        """
        return self.model.classifier
    
    def freeze_features(self):
        """Freeze feature extractor parameters."""
        for param in self.model.features.parameters():
            param.requires_grad = False
        
        logger.info("Feature extractor frozen")
    
    def unfreeze_features(self):
        """Unfreeze feature extractor parameters."""
        for param in self.model.features.parameters():
            param.requires_grad = True
        
        logger.info("Feature extractor unfrozen")
    
    def get_architecture_summary(self) -> Dict[str, any]:
        """
        Get summary of network architecture.
        
        Returns:
            Dictionary with architecture details
        """
        return {
            'input_channels': self.config.input_channels,
            'num_classes': self.config.num_classes,
            'width_multiplier': self.config.width_multiplier,
            'use_batch_norm': self.config.use_batch_norm,
            'dropout_rate': self.config.dropout_rate,
            'total_parameters': self._parameter_count,
            'trainable_parameters': sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            )
        }
    
    @classmethod
    def create_for_dataset(cls, dataset_name: str) -> 'CNN':
        """
        Create CNN configured for specific dataset.
        
        Args:
            dataset_name: Name of dataset
            
        Returns:
            Configured CNN instance
        """
        dataset_configs = {
            'MNIST': CNNConfig(input_channels=1, num_classes=10, width_multiplier=2),
            'FashionMNIST': CNNConfig(input_channels=1, num_classes=10, width_multiplier=2),
            'CIFAR-10': CNNConfig(input_channels=3, num_classes=10, width_multiplier=4),
            'CIFAR-100': CNNConfig(input_channels=3, num_classes=100, width_multiplier=4),
            'CelebA': CNNConfig(input_channels=3, num_classes=2, width_multiplier=4),
            'TinyImageNet': CNNConfig(input_channels=3, num_classes=200, width_multiplier=4)
        }
        
        if dataset_name not in dataset_configs:
            logger.warning(
                f"Unknown dataset: {dataset_name}. Using CIFAR-10 config."
            )
            config = dataset_configs['CIFAR-10']
        else:
            config = dataset_configs[dataset_name]
        
        logger.info(f"Created CNN for dataset: {dataset_name}")
        
        return cls(config)