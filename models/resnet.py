#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ResNet Architecture for NTK-SURGERY
Implements ResNet with residual connections compatible with NTK analysis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Type, Union
import logging
from dataclasses import dataclass

from models.ntk_model import NTKModel

logger = logging.getLogger(__name__)


@dataclass
class ResNetConfig:
    """
    Configuration class for ResNet architecture.
    
    Attributes:
        input_channels: Number of input channels
        num_classes: Number of output classes
        block_type: Type of residual block ('basic' or 'bottleneck')
        layers: Number of blocks per layer [2, 2, 2, 2] for ResNet-18
        width_multiplier: Width multiplier for network
        use_batch_norm: Whether to use batch normalization
        dropout_rate: Dropout rate
    """
    input_channels: int = 3
    num_classes: int = 10
    block_type: str = 'basic'
    layers: List[int] = None
    width_multiplier: int = 1
    use_batch_norm: bool = True
    dropout_rate: float = 0.0
    
    def __post_init__(self):
        """Validate configuration."""
        if self.layers is None:
            self.layers = [2, 2, 2, 2]  # ResNet-18 default
        
        if self.input_channels <= 0:
            raise ValueError("input_channels must be positive")
        if self.num_classes <= 0:
            raise ValueError("num_classes must be positive")
        if self.block_type not in ['basic', 'bottleneck']:
            raise ValueError(f"Unknown block type: {self.block_type}")


class BasicBlock(nn.Module):
    """
    Basic residual block for ResNet-18/34.
    
    Architecture:
        - Conv3x3 -> BN -> ReLU
        - Conv3x3 -> BN
        - Skip connection
        - ReLU
    
    Attributes:
        conv1: First convolution
        bn1: First batch norm
        conv2: Second convolution
        bn2: Second batch norm
        shortcut: Skip connection (if needed)
    """
    
    expansion = 1
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        use_batch_norm: bool = True,
        dropout_rate: float = 0.0
    ):
        """
        Initialize BasicBlock.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            stride: Convolution stride
            use_batch_norm: Whether to use batch normalization
            dropout_rate: Dropout rate
        """
        super().__init__()
        
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3,
            stride=stride, padding=1, bias=not use_batch_norm
        )
        self.bn1 = nn.BatchNorm2d(out_channels) if use_batch_norm else None
        
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3,
            stride=1, padding=1, bias=not use_batch_norm
        )
        self.bn2 = nn.BatchNorm2d(out_channels) if use_batch_norm else None
        
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None
        
        # Shortcut for dimension matching
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, 
                    out_channels * self.expansion,
                    kernel_size=1, stride=stride, bias=not use_batch_norm
                ),
                nn.BatchNorm2d(out_channels * self.expansion) if use_batch_norm else nn.Identity()
            )
        else:
            self.shortcut = nn.Identity()
        
        logger.debug(
            f"Created BasicBlock: {in_channels} -> {out_channels}, stride={stride}"
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through residual block.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        identity = x
        
        out = self.conv1(x)
        if self.bn1 is not None:
            out = self.bn1(out)
        out = F.relu(out)
        
        if self.dropout is not None:
            out = self.dropout(out)
        
        out = self.conv2(out)
        if self.bn2 is not None:
            out = self.bn2(out)
        
        # Add skip connection
        out += self.shortcut(identity)
        out = F.relu(out)
        
        return out


class Bottleneck(nn.Module):
    """
    Bottleneck residual block for ResNet-50/101.
    
    Architecture:
        - Conv1x1 -> BN -> ReLU
        - Conv3x3 -> BN -> ReLU
        - Conv1x1 -> BN
        - Skip connection
        - ReLU
    
    Attributes:
        conv1: 1x1 convolution (reduction)
        conv2: 3x3 convolution
        conv3: 1x1 convolution (expansion)
        bn1, bn2, bn3: Batch normalization layers
        shortcut: Skip connection (if needed)
    """
    
    expansion = 4
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        use_batch_norm: bool = True,
        dropout_rate: float = 0.0
    ):
        """
        Initialize Bottleneck.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels (before expansion)
            stride: Convolution stride
            use_batch_norm: Whether to use batch normalization
            dropout_rate: Dropout rate
        """
        super().__init__()
        
        mid_channels = out_channels * self.expansion // 4
        
        self.conv1 = nn.Conv2d(
            in_channels, mid_channels, kernel_size=1,
            bias=not use_batch_norm
        )
        self.bn1 = nn.BatchNorm2d(mid_channels) if use_batch_norm else None
        
        self.conv2 = nn.Conv2d(
            mid_channels, mid_channels, kernel_size=3,
            stride=stride, padding=1, bias=not use_batch_norm
        )
        self.bn2 = nn.BatchNorm2d(mid_channels) if use_batch_norm else None
        
        self.conv3 = nn.Conv2d(
            mid_channels, out_channels * self.expansion, kernel_size=1,
            bias=not use_batch_norm
        )
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion) if use_batch_norm else None
        
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None
        
        # Shortcut for dimension matching
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, 
                    out_channels * self.expansion,
                    kernel_size=1, stride=stride, bias=not use_batch_norm
                ),
                nn.BatchNorm2d(out_channels * self.expansion) if use_batch_norm else nn.Identity()
            )
        else:
            self.shortcut = nn.Identity()
        
        logger.debug(
            f"Created Bottleneck: {in_channels} -> {out_channels * self.expansion}, stride={stride}"
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through bottleneck block.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        identity = x
        
        out = self.conv1(x)
        if self.bn1 is not None:
            out = self.bn1(out)
        out = F.relu(out)
        
        out = self.conv2(out)
        if self.bn2 is not None:
            out = self.bn2(out)
        out = F.relu(out)
        
        if self.dropout is not None:
            out = self.dropout(out)
        
        out = self.conv3(out)
        if self.bn3 is not None:
            out = self.bn3(out)
        
        # Add skip connection
        out += self.shortcut(identity)
        out = F.relu(out)
        
        return out


class ResNet(NTKModel):
    """
    ResNet for NTK-SURGERY.
    
    Implements ResNet architecture with residual connections
    compatible with NTK analysis and federated unlearning.
    
    Supported variants:
        - ResNet-18: [2, 2, 2, 2] basic blocks
        - ResNet-34: [3, 4, 6, 3] basic blocks
        - ResNet-50: [3, 4, 6, 3] bottleneck blocks
        - ResNet-101: [3, 4, 23, 3] bottleneck blocks
    
    Attributes:
        config (ResNetConfig): Network configuration
        conv1: Initial convolution
        bn1: Initial batch norm
        layers: Residual layers
        avgpool: Average pooling
        fc: Fully connected classifier
    """
    
    # Block type mapping
    BLOCK_TYPES = {
        'basic': BasicBlock,
        'bottleneck': Bottleneck
    }
    
    # Channel progression
    CHANNELS = [64, 128, 256, 512]
    
    def __init__(self, config: Optional[ResNetConfig] = None):
        """
        Initialize ResNet.
        
        Args:
            config: ResNet configuration (uses defaults if None)
        """
        if config is None:
            config = ResNetConfig()
        
        self.config = config
        
        # Build network
        model = self._build_network()
        
        super().__init__(model)
        
        logger.info(
            f"Initialized ResNet with config: "
            f"channels={config.input_channels}, classes={config.num_classes}, "
            f"block_type={config.block_type}, layers={config.layers}"
        )
    
    def _build_network(self) -> nn.Module:
        """
        Build ResNet architecture.
        
        Returns:
            PyTorch module
        """
        block = self.BLOCK_TYPES[self.config.block_type]
        w = self.config.width_multiplier
        
        # Initial convolution
        conv1 = nn.Conv2d(
            self.config.input_channels, 
            64 * w, 
            kernel_size=7, 
            stride=2, 
            padding=3,
            bias=not self.config.use_batch_norm
        )
        bn1 = nn.BatchNorm2d(64 * w) if self.config.use_batch_norm else None
        
        # Residual layers
        layers = nn.ModuleList()
        
        in_channels = 64 * w
        for layer_idx, num_blocks in enumerate(self.config.layers):
            out_channels = self.CHANNELS[layer_idx] * w
            stride = 2 if layer_idx > 0 else 1
            
            layer_blocks = nn.Sequential(*[
                block(
                    in_channels if i == 0 else out_channels * block.expansion,
                    out_channels,
                    stride=stride if i == 0 else 1,
                    use_batch_norm=self.config.use_batch_norm,
                    dropout_rate=self.config.dropout_rate
                )
                for i in range(num_blocks)
            ])
            
            layers.append(layer_blocks)
            in_channels = out_channels * block.expansion
        
        # Average pooling and classifier
        avgpool = nn.AdaptiveAvgPool2d((1, 1))
        fc = nn.Linear(self.CHANNELS[-1] * block.expansion * w, self.config.num_classes)
        
        # Combine into single module
        model = nn.Module()
        model.conv1 = conv1
        model.bn1 = bn1
        model.layers = layers
        model.avgpool = avgpool
        model.fc = fc
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
        x = self.model.conv1(x)
        if self.model.bn1 is not None:
            x = self.model.bn1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        
        for layer in self.model.layers:
            x = layer(x)
        
        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.model.fc(x)
        
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through ResNet.
        
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
        
        Args:
            x: Input tensor
            params: Dictionary of parameter tensors
            
        Returns:
            Output tensor
        """
        self.set_parameters(params)
        return self.forward(x)
    
    def get_architecture_summary(self) -> Dict[str, any]:
        """
        Get summary of network architecture.
        
        Returns:
            Dictionary with architecture details
        """
        block = self.BLOCK_TYPES[self.config.block_type]
        
        # Calculate total layers
        total_blocks = sum(self.config.layers)
        
        # Estimate depth based on block type
        if self.config.block_type == 'basic':
            depth = 2 * total_blocks + 1  # ResNet-18, 34
        else:
            depth = 3 * total_blocks + 1  # ResNet-50, 101
        
        return {
            'input_channels': self.config.input_channels,
            'num_classes': self.config.num_classes,
            'block_type': self.config.block_type,
            'layers': self.config.layers,
            'depth': depth,
            'width_multiplier': self.config.width_multiplier,
            'use_batch_norm': self.config.use_batch_norm,
            'dropout_rate': self.config.dropout_rate,
            'total_parameters': self._parameter_count,
            'trainable_parameters': sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            )
        }
    
    @classmethod
    def create_resnet18(cls, num_classes: int = 10, **kwargs) -> 'ResNet':
        """Create ResNet-18."""
        config = ResNetConfig(
            num_classes=num_classes,
            block_type='basic',
            layers=[2, 2, 2, 2],
            **kwargs
        )
        logger.info("Created ResNet-18")
        return cls(config)
    
    @classmethod
    def create_resnet34(cls, num_classes: int = 10, **kwargs) -> 'ResNet':
        """Create ResNet-34."""
        config = ResNetConfig(
            num_classes=num_classes,
            block_type='basic',
            layers=[3, 4, 6, 3],
            **kwargs
        )
        logger.info("Created ResNet-34")
        return cls(config)
    
    @classmethod
    def create_resnet50(cls, num_classes: int = 10, **kwargs) -> 'ResNet':
        """Create ResNet-50."""
        config = ResNetConfig(
            num_classes=num_classes,
            block_type='bottleneck',
            layers=[3, 4, 6, 3],
            **kwargs
        )
        logger.info("Created ResNet-50")
        return cls(config)
    
    @classmethod
    def create_for_dataset(cls, dataset_name: str) -> 'ResNet':
        """
        Create ResNet configured for specific dataset.
        
        Args:
            dataset_name: Name of dataset
            
        Returns:
            Configured ResNet instance
        """
        dataset_configs = {
            'MNIST': {'num_classes': 10, 'input_channels': 1},
            'FashionMNIST': {'num_classes': 10, 'input_channels': 1},
            'CIFAR-10': {'num_classes': 10, 'input_channels': 3},
            'CIFAR-100': {'num_classes': 100, 'input_channels': 3},
            'CelebA': {'num_classes': 2, 'input_channels': 3},
            'TinyImageNet': {'num_classes': 200, 'input_channels': 3}
        }
        
        if dataset_name not in dataset_configs:
            logger.warning(
                f"Unknown dataset: {dataset_name}. Using CIFAR-10 config."
            )
            config_params = dataset_configs['CIFAR-10']
        else:
            config_params = dataset_configs[dataset_name]
        
        logger.info(f"Created ResNet for dataset: {dataset_name}")
        
        return cls.create_resnet18(**config_params)