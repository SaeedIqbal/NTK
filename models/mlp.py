#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Layer Perceptron for NTK-SURGERY
Implements MLP architecture compatible with NTK analysis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass

from models.ntk_model import NTKModel

logger = logging.getLogger(__name__)


@dataclass
class MLPConfig:
    """
    Configuration class for MLP architecture.
    
    Attributes:
        input_dim: Input feature dimension
        num_classes: Number of output classes
        hidden_dims: List of hidden layer dimensions
        use_batch_norm: Whether to use batch normalization
        dropout_rate: Dropout rate
        activation: Activation function ('relu', 'tanh', 'sigmoid')
    """
    input_dim: int = 512
    num_classes: int = 10
    hidden_dims: List[int] = None
    use_batch_norm: bool = True
    dropout_rate: float = 0.0
    activation: str = 'relu'
    
    def __post_init__(self):
        """Validate configuration."""
        if self.hidden_dims is None:
            self.hidden_dims = [512, 256, 128]
        
        if self.input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if self.num_classes <= 0:
            raise ValueError("num_classes must be positive")
        if self.activation not in ['relu', 'tanh', 'sigmoid']:
            raise ValueError(f"Unknown activation: {self.activation}")


class MLPBlock(nn.Module):
    """
    MLP block with linear layer, optional batch norm, and activation.
    
    Attributes:
        linear: Linear layer
        bn: Batch normalization layer (optional)
        dropout: Dropout layer (optional)
        activation: Activation function
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        use_batch_norm: bool = True,
        dropout_rate: float = 0.0,
        activation: str = 'relu'
    ):
        """
        Initialize MLPBlock.
        
        Args:
            in_features: Number of input features
            out_features: Number of output features
            use_batch_norm: Whether to use batch normalization
            dropout_rate: Dropout rate
            activation: Activation function name
        """
        super().__init__()
        
        self.linear = nn.Linear(in_features, out_features, bias=not use_batch_norm)
        self.bn = nn.BatchNorm1d(out_features) if use_batch_norm else None
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None
        
        # Select activation function
        activation_map = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid()
        }
        self.activation = activation_map.get(activation, nn.ReLU())
        
        logger.debug(
            f"Created MLPBlock: {in_features} -> {out_features}, "
            f"BN={use_batch_norm}, Activation={activation}"
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through block.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        x = self.linear(x)
        
        if self.bn is not None:
            x = self.bn(x)
        
        x = self.activation(x)
        
        if self.dropout is not None:
            x = self.dropout(x)
        
        return x


class MLP(NTKModel):
    """
    Multi-Layer Perceptron for NTK-SURGERY.
    
    Implements an MLP architecture compatible with NTK analysis
    and federated unlearning operations.
    
    Architecture:
        - Input layer
        - Multiple hidden layers with configurable dimensions
        - Output layer
        - Optional batch normalization
        - Optional dropout
    
    Attributes:
        config (MLPConfig): Network configuration
        layers (nn.ModuleList): Network layers
    """
    
    def __init__(self, config: Optional[MLPConfig] = None):
        """
        Initialize MLP.
        
        Args:
            config: MLP configuration (uses defaults if None)
        """
        if config is None:
            config = MLPConfig()
        
        self.config = config
        
        # Build network
        model = self._build_network()
        
        super().__init__(model)
        
        logger.info(
            f"Initialized MLP with config: "
            f"input={config.input_dim}, classes={config.num_classes}, "
            f"hidden={config.hidden_dims}"
        )
    
    def _build_network(self) -> nn.Module:
        """
        Build MLP architecture.
        
        Returns:
            PyTorch module
        """
        layers = nn.ModuleList()
        
        # Input layer
        layers.append(MLPBlock(
            self.config.input_dim,
            self.config.hidden_dims[0],
            use_batch_norm=self.config.use_batch_norm,
            dropout_rate=self.config.dropout_rate,
            activation=self.config.activation
        ))
        
        # Hidden layers
        for i in range(len(self.config.hidden_dims) - 1):
            layers.append(MLPBlock(
                self.config.hidden_dims[i],
                self.config.hidden_dims[i + 1],
                use_batch_norm=self.config.use_batch_norm,
                dropout_rate=self.config.dropout_rate,
                activation=self.config.activation
            ))
        
        # Output layer (no activation, no batch norm, no dropout)
        layers.append(nn.Linear(
            self.config.hidden_dims[-1],
            self.config.num_classes
        ))
        
        # Combine into single module
        model = nn.Module()
        model.layers = layers
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
        for layer in self.model.layers:
            x = layer(x)
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through MLP.
        
        Args:
            x: Input tensor of shape (N, input_dim)
            
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
    
    def get_hidden_representations(
        self, 
        x: torch.Tensor
    ) -> Dict[int, torch.Tensor]:
        """
        Get hidden layer representations.
        
        Args:
            x: Input tensor
            
        Returns:
            Dictionary mapping layer index to representations
        """
        self.model.eval()
        representations = {}
        
        h = x
        for i, layer in enumerate(self.model.layers):
            h = layer(h)
            representations[i] = h.detach().cpu()
        
        return representations
    
    def get_architecture_summary(self) -> Dict[str, any]:
        """
        Get summary of network architecture.
        
        Returns:
            Dictionary with architecture details
        """
        return {
            'input_dim': self.config.input_dim,
            'num_classes': self.config.num_classes,
            'hidden_dims': self.config.hidden_dims,
            'num_layers': len(self.config.hidden_dims) + 1,
            'use_batch_norm': self.config.use_batch_norm,
            'dropout_rate': self.config.dropout_rate,
            'activation': self.config.activation,
            'total_parameters': self._parameter_count,
            'trainable_parameters': sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            )
        }
    
    @classmethod
    def create_for_ntk_analysis(
        cls,
        input_dim: int = 512,
        num_classes: int = 10,
        width: int = 256,
        depth: int = 3
    ) -> 'MLP':
        """
        Create MLP optimized for NTK analysis.
        
        For NTK theory, wider networks better approximate the infinite-width limit.
        
        Args:
            input_dim: Input dimension
            num_classes: Number of classes
            width: Width of hidden layers
            depth: Number of hidden layers
            
        Returns:
            Configured MLP instance
        """
        hidden_dims = [width] * depth
        
        config = MLPConfig(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dims=hidden_dims,
            use_batch_norm=False,  # No BN for cleaner NTK analysis
            dropout_rate=0.0,  # No dropout for NTK
            activation='relu'
        )
        
        logger.info(
            f"Created MLP for NTK analysis: width={width}, depth={depth}"
        )
        
        return cls(config)