#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FedSGD: Federated Stochastic Gradient Descent with Unlearning
Implements FedSGD baseline with high communication overhead
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
class FedSGDConfig:
    """Configuration for FedSGD."""
    learning_rate: float = 0.01
    communication_rounds: int = 100
    local_epochs: int = 1
    batch_size: int = 64
    device: str = 'cpu'


class FedSGD:
    """
    Federated SGD with unlearning capability.
    
    Critical limitation: High communication overhead compared to FedAvg.
    """
    
    def __init__(self, model: nn.Module, config: Optional[FedSGDConfig] = None):
        """
        Initialize FedSGD.
        
        Args:
            model: Neural network model
            config: FedSGD configuration
        """
        self.model = model
        self.config = config if config is not None else FedSGDConfig()
        self.device = self.config.device
        
        # Unlearning metrics
        self.unlearning_time = 0.0
        self.communication_rounds = 0
        
        logger.info(f"Initialized FedSGD with {self.config.communication_rounds} rounds")
    
    def unlearn_client(
        self,
        client_id: int,
        remaining_clients: Dict[int, torch.utils.data.DataLoader]
    ) -> nn.Module:
        """
        Unlearn client via FedSGD retraining.
        
        Args:
            client_id: Client to remove
            remaining_clients: DataLoaders for remaining clients
            
        Returns:
            Unlearned model
        """
        start_time = time.perf_counter()
        
        logger.info(f"FedSGD unlearning client {client_id}")
        
        self.model.train()
        
        for round_num in range(self.config.communication_rounds):
            round_updates = []
            
            # Local SGD for each remaining client
            for cid, loader in remaining_clients.items():
                client_update = self._local_sgd(cid, loader)
                round_updates.append(client_update)
            
            # Aggregate updates
            self._aggregate_updates(round_updates)
            self.communication_rounds += 1
            
            if round_num % 20 == 0:
                logger.info(f"FedSGD round {round_num} completed")
        
        self.unlearning_time = time.perf_counter() - start_time
        
        logger.info(
            f"FedSGD unlearning completed in {self.unlearning_time:.2f}s. "
            f"Rounds: {self.communication_rounds}"
        )
        
        return self.model
    
    def _local_sgd(
        self,
        client_id: int,
        loader: torch.utils.data.DataLoader
    ) -> Dict[str, torch.Tensor]:
        """
        Perform local SGD for a client.
        
        Args:
            client_id: Client identifier
            loader: Client's DataLoader
            
        Returns:
            Parameter updates
        """
        optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.config.learning_rate
        )
        
        updates = {
            name: torch.zeros_like(param)
            for name, param in self.model.named_parameters()
        }
        
        num_batches = 0
        
        for data, target in loader:
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            output = self.model(data)
            loss = nn.CrossEntropyLoss()(output, target)
            loss.backward()
            
            # Accumulate gradients
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    updates[name] += param.grad.clone()
            
            optimizer.step()
            num_batches += 1
        
        # Average updates
        for name in updates:
            updates[name] /= max(num_batches, 1)
        
        return updates
    
    def _aggregate_updates(
        self,
        round_updates: List[Dict[str, torch.Tensor]]
    ):
        """
        Aggregate client updates.
        
        Args:
            round_updates: List of client updates
        """
        if not round_updates:
            return
        
        # Average updates across clients
        avg_updates = {}
        
        for name in round_updates[0].keys():
            avg_updates[name] = torch.stack([
                update[name] for update in round_updates
            ]).mean(dim=0)
        
        # Apply averaged updates
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in avg_updates:
                    param.sub_(self.config.learning_rate * avg_updates[name])
    
    def get_efficiency_metrics(self) -> Dict[str, any]:
        """
        Get efficiency metrics for FedSGD.
        
        Returns:
            Dictionary with efficiency metrics
        """
        return {
            'method': 'FedSGD',
            'unlearning_time': self.unlearning_time,
            'communication_rounds': self.communication_rounds,
            'complexity_class': 'O(rounds · M · K · P)',
            'communication_overhead': 'High (per-sample aggregation)'
        }