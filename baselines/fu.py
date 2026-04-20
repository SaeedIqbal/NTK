#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FU: Federated Unlearning (Generic)
Generic federated unlearning baseline

Key limitation: Often assumes convex losses or independent server data
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
class FUConfig:
    """Configuration for FU."""
    learning_rate: float = 0.01
    epochs: int = 50
    batch_size: int = 64
    device: str = 'cpu'


class FU:
    """
    Generic Federated Unlearning baseline.
    
    Critical limitation: Assumes convex losses or independent server data,
    violating standard FL constraints.
    """
    
    def __init__(self, model: nn.Module, config: Optional[FUConfig] = None):
        """
        Initialize FU.
        
        Args:
            model: Neural network model
            config: FU configuration
        """
        self.model = model
        self.config = config if config is not None else FUConfig()
        self.device = self.config.device
        
        # Unlearning metrics
        self.unlearning_time = 0.0
        
        logger.info(f"Initialized FU baseline")
    
    def unlearn_client(
        self,
        client_id: int,
        remaining_clients: Dict[int, torch.utils.data.DataLoader]
    ) -> nn.Module:
        """
        Unlearn client using generic FU approach.
        
        Args:
            client_id: Client to remove
            remaining_clients: DataLoaders for remaining clients
            
        Returns:
            Unlearned model
        """
        start_time = time.perf_counter()
        
        logger.info(f"FU unlearning client {client_id}")
        
        self.model.train()
        optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.config.learning_rate
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
                logger.info(f"FU epoch {epoch}: loss = {avg_epoch_loss:.4f}")
            
            total_loss += avg_epoch_loss
        
        self.unlearning_time = time.perf_counter() - start_time
        avg_loss = total_loss / self.config.epochs
        
        logger.info(f"FU unlearning completed in {self.unlearning_time:.2f}s")
        
        return self.model
    
    def get_efficiency_metrics(self) -> Dict[str, any]:
        """
        Get efficiency metrics for FU.
        
        Returns:
            Dictionary with efficiency metrics
        """
        return {
            'method': 'FU',
            'unlearning_time': self.unlearning_time,
            'complexity_class': 'O(epochs · M · K · P)',
            'assumptions': 'Convex losses or independent server data',
            'violates_fl_constraints': True
        }