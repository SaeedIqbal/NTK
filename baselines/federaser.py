#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FedEraser: Federated Unlearning via Gradient Erasure
Implements FedEraser from Liu et al., 2021

Key limitation: Requires excessive iterations, often exceeding Scratch cost
without theoretical unlearning guarantees
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional
import logging
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class FedEraserConfig:
    """Configuration for FedEraser."""
    learning_rate: float = 0.01
    local_epochs: int = 5
    batch_size: int = 64
    unlearning_epochs: int = 100
    gradient_history_size: int = 50
    device: str = 'cpu'


class FedEraser:
    """
    Federated Unlearning via Gradient Erasure (FedEraser).
    
    Uses stored gradient history to reverse client contributions.
    
    Critical limitations:
    - No theoretical unlearning guarantees
    - Requires excessive iterations (often > Scratch)
    - Gradient storage overhead
    """
    
    def __init__(self, model: nn.Module, config: Optional[FedEraserConfig] = None):
        """
        Initialize FedEraser.
        
        Args:
            model: Neural network model
            config: FedEraser configuration
        """
        self.model = model
        self.config = config if config is not None else FedEraserConfig()
        self.device = self.config.device
        
        # Gradient history storage
        self.gradient_history = {}  # client_id -> list of gradients
        self.update_history = []
        
        # Unlearning metrics
        self.unlearning_time = 0.0
        self.unlearning_iterations = 0
        
        logger.info(f"Initialized FedEraser with {self.config.unlearning_epochs} unlearning epochs")
    
    def store_client_gradient(
        self,
        client_id: int,
        gradient: Dict[str, torch.Tensor],
        round_num: int
    ):
        """
        Store client gradient for later erasure.
        
        Args:
            client_id: Client identifier
            gradient: Gradient dictionary
            round_num: Training round number
        """
        if client_id not in self.gradient_history:
            self.gradient_history[client_id] = []
        
        self.gradient_history[client_id].append({
            'round': round_num,
            'gradient': {
                name: grad.clone().detach().cpu()
                for name, grad in gradient.items()
            }
        })
        
        logger.debug(
            f"Stored gradient for client {client_id}, round {round_num}. "
            f"Total stored: {len(self.gradient_history[client_id])}"
        )
    
    def aggregate_client_gradients(
        self,
        client_id: int
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate all stored gradients for a client.
        
        Args:
            client_id: Client identifier
            
        Returns:
            Aggregated gradient dictionary
        """
        if client_id not in self.gradient_history:
            logger.warning(f"No gradient history for client {client_id}")
            return None
        
        aggregated = {}
        
        for entry in self.gradient_history[client_id]:
            gradient = entry['gradient']
            
            for name, grad in gradient.items():
                if name not in aggregated:
                    aggregated[name] = grad.clone()
                else:
                    aggregated[name] += grad
        
        # Average across stored gradients
        num_entries = len(self.gradient_history[client_id])
        for name in aggregated:
            aggregated[name] /= num_entries
        
        logger.info(
            f"Aggregated {num_entries} gradients for client {client_id}"
        )
        
        return aggregated
    
    def unlearn_client(
        self,
        client_id: int,
        remaining_clients: Dict[int, torch.utils.data.DataLoader]
    ) -> nn.Module:
        """
        Unlearn a client using FedEraser method.
        
        Args:
            client_id: Client to remove
            remaining_clients: DataLoaders for remaining clients
            
        Returns:
            Unlearned model
        """
        start_time = time.perf_counter()
        
        logger.info(f"FedEraser unlearning client {client_id}")
        
        # Step 1: Get aggregated gradient for client
        client_gradient = self.aggregate_client_gradients(client_id)
        
        if client_gradient is None:
            logger.warning("No gradient history. Falling back to fine-tuning.")
            return self._fallback_fine_tune(remaining_clients)
        
        # Step 2: Reverse gradient contribution
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in client_gradient:
                    # Subtract client's gradient contribution
                    param.sub_(client_gradient[name])
        
        # Step 3: Fine-tune on remaining data (many iterations)
        self._fine_tune_remaining(remaining_clients)
        
        self.unlearning_time = time.perf_counter() - start_time
        
        logger.info(
            f"FedEraser unlearning completed in {self.unlearning_time:.2f}s. "
            f"Iterations: {self.unlearning_iterations}"
        )
        
        return self.model
    
    def _fine_tune_remaining(
        self,
        remaining_clients: Dict[int, torch.utils.data.DataLoader]
    ):
        """
        Fine-tune on remaining clients' data.
        
        Args:
            remaining_clients: DataLoaders for remaining clients
        """
        self.model.train()
        optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.config.learning_rate
        )
        
        total_loss = 0.0
        self.unlearning_iterations = 0
        
        for epoch in range(self.config.unlearning_epochs):
            epoch_loss = 0.0
            epoch_batches = 0
            
            for client_id, loader in remaining_clients.items():
                for data, target in loader:
                    data, target = data.to(self.device), target.to(self.device)
                    
                    optimizer.zero_grad()
                    output = self.model(data)
                    loss = nn.CrossEntropyLoss()(output, target)
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    epoch_batches += 1
                    self.unlearning_iterations += 1
            
            avg_epoch_loss = epoch_loss / max(epoch_batches, 1)
            
            if epoch % 10 == 0:
                logger.info(f"FedEraser epoch {epoch}: loss = {avg_epoch_loss:.4f}")
            
            total_loss += avg_epoch_loss
        
        avg_loss = total_loss / self.config.unlearning_epochs
        logger.info(f"FedEraser fine-tuning completed. Average loss: {avg_loss:.4f}")
    
    def _fallback_fine_tune(
        self,
        remaining_clients: Dict[int, torch.utils.data.DataLoader]
    ) -> nn.Module:
        """
        Fallback to fine-tuning when no gradient history.
        
        Args:
            remaining_clients: DataLoaders for remaining clients
            
        Returns:
            Fine-tuned model
        """
        logger.warning("FedEraser falling back to fine-tuning (no gradient history)")
        self._fine_tune_remaining(remaining_clients)
        return self.model
    
    def clear_client_history(self, client_id: int):
        """
        Clear gradient history for a client after unlearning.
        
        Args:
            client_id: Client identifier
        """
        if client_id in self.gradient_history:
            del self.gradient_history[client_id]
            logger.info(f"Cleared gradient history for client {client_id}")
    
    def get_efficiency_metrics(self) -> Dict[str, any]:
        """
        Get efficiency metrics for FedEraser.
        
        Returns:
            Dictionary with efficiency metrics
        """
        return {
            'method': 'FedEraser',
            'unlearning_time': self.unlearning_time,
            'unlearning_iterations': self.unlearning_iterations,
            'gradient_history_size': sum(
                len(v) for v in self.gradient_history.values()
            ),
            'clients_stored': len(self.gradient_history),
            'complexity_class': 'O(unlearning_epochs · M · K · P)',
            'theoretical_guarantees': False
        }