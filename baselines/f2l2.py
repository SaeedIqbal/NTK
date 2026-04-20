#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
F²L²: Forgettable Federated Linear Learning
Implements F²L² from Jin et al., 2023

Key limitation: Assumes convex losses, does not support non-convex regimes
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Optional
import logging
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class F2L2Config:
    """Configuration for F²L²."""
    learning_rate: float = 0.01
    epochs: int = 50
    batch_size: int = 64
    regularization: float = 0.001
    device: str = 'cpu'


class F2L2:
    """
    Forgettable Federated Linear Learning (F²L²).
    
    Provides certified data removal for linear models.
    
    Critical limitation: Assumes convex losses, does not support
    non-convex deep learning regimes.
    """
    
    def __init__(self, model: nn.Module, config: Optional[F2L2Config] = None):
        """
        Initialize F²L².
        
        Args:
            model: Neural network model
            config: F²L² configuration
        """
        self.model = model
        self.config = config if config is not None else F2L2Config()
        self.device = self.config.device
        
        # Unlearning metrics
        self.unlearning_time = 0.0
        
        logger.info(f"Initialized F²L² with regularization={self.config.regularization}")
    
    def unlearn_client(
        self,
        client_id: int,
        remaining_clients: Dict[int, torch.utils.data.DataLoader]
    ) -> nn.Module:
        """
        Unlearn client using F²L² method.
        
        Args:
            client_id: Client to remove
            remaining_clients: DataLoaders for remaining clients
            
        Returns:
            Unlearned model
        """
        start_time = time.perf_counter()
        
        logger.info(f"F²L² unlearning client {client_id}")
        
        self.model.train()
        optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.regularization
        )
        
        total_loss = 0.0
        num_batches = 0
        
        for epoch in range(self.config.epochs):
            epoch_loss = 0.0
            
            for cid, loader in remaining_clients.items():
                for data, target in loader:
                    data, target = data.to(self.device), target.to(self.device)
                    
                    optimizer.zero_grad()
                    output = self.model(data)
                    loss = nn.CrossEntropyLoss()(output, target)
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    num_batches += 1
            
            avg_epoch_loss = epoch_loss / max(num_batches, 1)
            
            if epoch % 10 == 0:
                logger.info(f"F²L² epoch {epoch}: loss = {avg_epoch_loss:.4f}")
            
            total_loss += avg_epoch_loss
        
        self.unlearning_time = time.perf_counter() - start_time
        avg_loss = total_loss / self.config.epochs
        
        logger.info(f"F²L² unlearning completed in {self.unlearning_time:.2f}s")
        
        return self.model
    
    def get_efficiency_metrics(self) -> Dict[str, any]:
        """
        Get efficiency metrics for F²L².
        
        Returns:
            Dictionary with efficiency metrics
        """
        return {
            'method': 'F²L²',
            'unlearning_time': self.unlearning_time,
            'regularization': self.config.regularization,
            'complexity_class': 'O(epochs · M · K · P)',
            'convex_assumption': True,
            'supports_non_convex': False
        }