#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Local Training for Federated Learning in NTK-SURGERY
Implements client-side local SGD training

Each client performs K local epochs on their data before sending
updates to the server for aggregation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional
import logging
import time
from dataclasses import dataclass
import copy

logger = logging.getLogger(__name__)


@dataclass
class LocalTrainingConfig:
    """
    Configuration for local client training.
    
    Attributes:
        learning_rate: Learning rate for SGD
        epochs: Number of local epochs
        batch_size: Mini-batch size
        device: Computing device
        optimizer_type: Optimizer type ('sgd', 'adam')
        momentum: SGD momentum
        weight_decay: L2 regularization
        clip_grad_norm: Gradient clipping threshold (None for no clipping)
    """
    learning_rate: float = 0.01
    epochs: int = 5
    batch_size: int = 64
    device: str = 'cpu'
    optimizer_type: str = 'sgd'
    momentum: float = 0.9
    weight_decay: float = 1e-4
    clip_grad_norm: Optional[float] = None
    
    def __post_init__(self):
        """Validate configuration."""
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.optimizer_type not in ['sgd', 'adam']:
            raise ValueError(f"Unknown optimizer: {self.optimizer_type}")


class ClientModel:
    """
    Wrapper for client-side model with local state.
    
    Maintains separate optimizer state per client.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: LocalTrainingConfig
    ):
        """
        Initialize ClientModel.
        
        Args:
            model: Neural network model
            config: Local training configuration
        """
        self.model = model
        self.config = config
        self.device = config.device
        
        # Move model to device
        self.model.to(self.device)
        
        # Initialize optimizer
        self.optimizer = self._create_optimizer()
        
        # Training state
        self.current_epoch = 0
        self.training_history = []
        
        logger.debug(f"Initialized ClientModel on {self.device}")
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer based on config."""
        if self.config.optimizer_type == 'sgd':
            return optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer_type == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer_type}")
    
    def get_state_dict(self) -> Dict[str, torch.Tensor]:
        """Get model state dict."""
        return {
            name: param.clone().detach().cpu()
            for name, param in self.model.named_parameters()
        }
    
    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]):
        """Load model state dict."""
        self.model.load_state_dict(state_dict)
    
    def reset_optimizer(self):
        """Reset optimizer state."""
        self.optimizer = self._create_optimizer()
        logger.debug("Optimizer reset")


class LocalTrainer:
    """
    Local trainer for federated clients.
    
    Performs K local epochs of SGD on client data.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Optional[LocalTrainingConfig] = None
    ):
        """
        Initialize LocalTrainer.
        
        Args:
            model: Neural network model
            config: Local training configuration
        """
        self.config = config if config is not None else LocalTrainingConfig()
        self.device = self.config.device
        
        # Create client model
        self.client_model = ClientModel(model, self.config)
        
        # Gradient tracking
        self.gradient_norms = {}
        
        logger.info(
            f"Initialized LocalTrainer: epochs={self.config.epochs}, "
            f"lr={self.config.learning_rate}, optimizer={self.config.optimizer_type}"
        )
    
    def compute_gradient_norm(
        self,
        model: nn.Module
    ) -> float:
        """
        Compute L2 norm of all gradients.
        
        Args:
            model: Model with computed gradients
            
        Returns:
            Gradient norm
        """
        total_norm = 0.0
        
        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        
        return float(np.sqrt(total_norm))
    
    def train_epoch(
        self,
        client_id: int,
        loader: DataLoader,
        epoch: int
    ) -> Tuple[float, float]:
        """
        Train for one epoch on client data.
        
        Args:
            client_id: Client identifier
            loader: Client's DataLoader
            epoch: Current epoch number
            
        Returns:
            Tuple of (average_loss, gradient_norm)
        """
        self.client_model.model.train()
        
        total_loss = 0.0
        num_batches = 0
        gradient_norms = []
        
        for batch_idx, (data, target) in enumerate(loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Forward pass
            self.client_model.optimizer.zero_grad()
            output = self.client_model.model(data)
            loss = nn.CrossEntropyLoss()(output, target)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config.clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.client_model.model.parameters(),
                    self.config.clip_grad_norm
                )
            
            # Track gradient norm
            grad_norm = self.compute_gradient_norm(self.client_model.model)
            gradient_norms.append(grad_norm)
            
            # Update parameters
            self.client_model.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        avg_grad_norm = float(np.mean(gradient_norms)) if gradient_norms else 0.0
        
        logger.debug(
            f"Client {client_id}, Epoch {epoch}: loss={avg_loss:.4f}, "
            f"grad_norm={avg_grad_norm:.4f}"
        )
        
        return avg_loss, avg_grad_norm
    
    def train(
        self,
        client_id: int,
        loader: DataLoader,
        round_num: int
    ) -> Tuple[float, float]:
        """
        Perform complete local training for a client.
        
        Args:
            client_id: Client identifier
            loader: Client's DataLoader
            round_num: Federated round number
            
        Returns:
            Tuple of (final_loss, final_gradient_norm)
        """
        logger.info(f"Client {client_id} starting local training (round {round_num})")
        
        start_time = time.perf_counter()
        
        epoch_losses = []
        epoch_grad_norms = []
        
        for epoch in range(self.config.epochs):
            self.client_model.current_epoch = epoch
            
            loss, grad_norm = self.train_epoch(client_id, loader, epoch)
            
            epoch_losses.append(loss)
            epoch_grad_norms.append(grad_norm)
            
            # Store in history
            self.client_model.training_history.append({
                'round': round_num,
                'epoch': epoch,
                'loss': loss,
                'gradient_norm': grad_norm
            })
        
        # Store gradient norm for this round
        final_grad_norm = epoch_grad_norms[-1] if epoch_grad_norms else 0.0
        self.gradient_norms[round_num] = final_grad_norm
        
        elapsed_time = time.perf_counter() - start_time
        
        logger.info(
            f"Client {client_id} completed local training: "
            f"{self.config.epochs} epochs, {elapsed_time:.2f}s, "
            f"final_loss={epoch_losses[-1]:.4f}"
        )
        
        return epoch_losses[-1], final_grad_norm
    
    def get_model_state(self) -> Dict[str, torch.Tensor]:
        """
        Get current model state.
        
        Returns:
            Model state dict
        """
        return self.client_model.get_state_dict()
    
    def load_model_state(self, state_dict: Dict[str, torch.Tensor]):
        """
        Load model state.
        
        Args:
            state_dict: Model state dict
        """
        self.client_model.load_state_dict(state_dict)
    
    def get_training_history(self) -> List[Dict]:
        """
        Get training history.
        
        Returns:
            List of training metrics per epoch
        """
        return self.client_model.training_history.copy()
    
    def reset(self):
        """Reset trainer state."""
        self.client_model.reset_optimizer()
        self.client_model.training_history = []
        self.gradient_norms = {}
        logger.info("LocalTrainer reset")