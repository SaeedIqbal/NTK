#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Server Aggregation for Federated Learning in NTK-SURGERY
Implements server-side weighted averaging of client updates

Implements manuscript Eq. 5:
θ_{n+1} = θ_n + Σ_{i∈I} (|D_i|/|D_I|) (θ_{n+1}^i - θ_n)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from enum import Enum
import logging
from dataclasses import dataclass, field
import time

logger = logging.getLogger(__name__)


class AggregationStrategy(Enum):
    """Aggregation strategy types."""
    FEDAVG = "fedavg"  # Weighted by dataset size
    FEDAVG_UNIFORM = "fedavg_uniform"  # Uniform weighting
    TRIMMED_MEAN = "trimmed_mean"  # Robust aggregation
    KOERED_MEAN = "koered_mean"  # Coordinate-wise median


@dataclass
class ServerState:
    """
    Server state for federated learning.
    
    Attributes:
        round_num: Current round number
        model_state_dict: Current model parameters
        gradient_norms: History of gradient norms
        checkpoint_history: Checkpoint file paths
        aggregation_history: History of aggregation metrics
    """
    round_num: int = 0
    model_state_dict: Dict[str, torch.Tensor] = field(default_factory=dict)
    gradient_norms: List[float] = field(default_factory=list)
    checkpoint_history: Dict[int, str] = field(default_factory=dict)
    aggregation_history: List[Dict] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'round_num': self.round_num,
            'gradient_norms': self.gradient_norms,
            'checkpoint_history': self.checkpoint_history,
            'aggregation_history': self.aggregation_history
        }
    
    def from_dict(self, data: Dict):
        """Load from dictionary."""
        self.round_num = data.get('round_num', 0)
        self.gradient_norms = data.get('gradient_norms', [])
        self.checkpoint_history = data.get('checkpoint_history', {})
        self.aggregation_history = data.get('aggregation_history', [])


class ServerAggregator:
    """
    Server-side aggregation for federated learning.
    
    Implements multiple aggregation strategies with support for:
    - Standard FedAvg (weighted by dataset size)
    - Uniform weighting
    - Robust aggregation (trimmed mean, coordinate-wise median)
    - Gradient norm tracking for sensitivity analysis
    
    Attributes:
        model (nn.Module): Global model
        device (str): Computing device
        strategy (AggregationStrategy): Aggregation strategy
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cpu',
        strategy: AggregationStrategy = AggregationStrategy.FEDAVG
    ):
        """
        Initialize ServerAggregator.
        
        Args:
            model: Global neural network model
            device: Computing device
            strategy: Aggregation strategy
        """
        self.model = model
        self.device = device
        self.strategy = strategy
        
        # Aggregation metrics
        self.aggregation_history = []
        self.update_norms = []
        
        logger.info(f"Initialized ServerAggregator with strategy={strategy.value}")
    
    def aggregate(
        self,
        client_updates: Dict[int, Dict[str, torch.Tensor]],
        client_sizes: Dict[int, int]
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate client updates.
        
        Args:
            client_updates: Dictionary of client parameter updates
            client_sizes: Dictionary of client dataset sizes
            
        Returns:
            Aggregated global parameters
        """
        if not client_updates:
            logger.warning("No client updates to aggregate")
            return {
                name: param.clone().detach()
                for name, param in self.model.named_parameters()
            }
        
        start_time = time.perf_counter()
        
        if self.strategy == AggregationStrategy.FEDAVG:
            aggregated = self._aggregate_fedavg(client_updates, client_sizes)
        elif self.strategy == AggregationStrategy.FEDAVG_UNIFORM:
            aggregated = self._aggregate_uniform(client_updates)
        elif self.strategy == AggregationStrategy.TRIMMED_MEAN:
            aggregated = self._aggregate_trimmed_mean(client_updates)
        elif self.strategy == AggregationStrategy.KOERED_MEAN:
            aggregated = self._aggregate_coordinate_median(client_updates)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        
        elapsed_time = time.perf_counter() - start_time
        
        # Compute update norm for tracking
        update_norm = self._compute_update_norm(aggregated)
        self.update_norms.append(update_norm)
        
        # Store aggregation metrics
        metrics = {
            'num_clients': len(client_updates),
            'update_norm': update_norm,
            'aggregation_time': elapsed_time,
            'strategy': self.strategy.value
        }
        self.aggregation_history.append(metrics)
        
        logger.debug(
            f"Aggregation completed: {len(client_updates)} clients, "
            f"update_norm={update_norm:.4f}, time={elapsed_time:.4f}s"
        )
        
        return aggregated
    
    def _aggregate_fedavg(
        self,
        client_updates: Dict[int, Dict[str, torch.Tensor]],
        client_sizes: Dict[int, int]
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate using FedAvg (weighted by dataset size).
        
        Implements: ω(I, θ_n) = Σ_{i∈I} (|D_i|/|D_I|) (θ_{n+1}^i - θ_n)
        
        Args:
            client_updates: Client parameter updates
            client_sizes: Client dataset sizes
            
        Returns:
            Aggregated parameters
        """
        total_size = sum(client_sizes.get(cid, 0) for cid in client_updates)
        
        if total_size == 0:
            logger.warning("Total client size is 0, using uniform weighting")
            return self._aggregate_uniform(client_updates)
        
        aggregated = {}
        
        # Get parameter names from first client
        first_cid = list(client_updates.keys())[0]
        param_names = list(client_updates[first_cid].keys())
        
        for name in param_names:
            # Weighted sum of updates
            weighted_sum = torch.zeros_like(
                client_updates[first_cid][name],
                device=self.device
            )
            
            for cid, update in client_updates.items():
                if name in update:
                    weight = client_sizes.get(cid, 0) / total_size
                    weighted_sum += weight * update[name].to(self.device)
            
            aggregated[name] = weighted_sum
        
        # Add aggregated update to current model
        current_state = {
            name: param.clone().detach()
            for name, param in self.model.named_parameters()
        }
        
        for name in aggregated:
            if name in current_state:
                aggregated[name] = current_state[name] + aggregated[name]
        
        return aggregated
    
    def _aggregate_uniform(
        self,
        client_updates: Dict[int, Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate using uniform weighting.
        
        Args:
            client_updates: Client parameter updates
            
        Returns:
            Aggregated parameters
        """
        num_clients = len(client_updates)
        
        if num_clients == 0:
            return {}
        
        aggregated = {}
        first_cid = list(client_updates.keys())[0]
        param_names = list(client_updates[first_cid].keys())
        
        for name in param_names:
            uniform_sum = torch.zeros_like(
                client_updates[first_cid][name],
                device=self.device
            )
            
            for update in client_updates.values():
                if name in update:
                    uniform_sum += update[name].to(self.device)
            
            aggregated[name] = uniform_sum / num_clients
        
        # Add to current model
        current_state = {
            name: param.clone().detach()
            for name, param in self.model.named_parameters()
        }
        
        for name in aggregated:
            if name in current_state:
                aggregated[name] = current_state[name] + aggregated[name]
        
        return aggregated
    
    def _aggregate_trimmed_mean(
        self,
        client_updates: Dict[int, Dict[str, torch.Tensor]],
        trim_fraction: float = 0.1
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate using trimmed mean (robust to outliers).
        
        Args:
            client_updates: Client parameter updates
            trim_fraction: Fraction to trim from each end
            
        Returns:
            Aggregated parameters
        """
        num_clients = len(client_updates)
        trim_count = max(1, int(num_clients * trim_fraction))
        
        if num_clients <= 2 * trim_count:
            logger.warning("Too few clients for trimmed mean, using uniform")
            return self._aggregate_uniform(client_updates)
        
        aggregated = {}
        first_cid = list(client_updates.keys())[0]
        param_names = list(client_updates[first_cid].keys())
        
        for name in param_names:
            # Stack all updates for this parameter
            updates_stack = torch.stack([
                update[name].to(self.device).flatten()
                for update in client_updates.values()
                if name in update
            ])
            
            # Sort along client dimension
            sorted_updates, _ = torch.sort(updates_stack, dim=0)
            
            # Trim and average
            trimmed = sorted_updates[trim_count:-trim_count]
            trimmed_mean = trimmed.mean(dim=0)
            
            aggregated[name] = trimmed_mean.reshape(
                client_updates[first_cid][name].shape
            )
        
        # Add to current model
        current_state = {
            name: param.clone().detach()
            for name, param in self.model.named_parameters()
        }
        
        for name in aggregated:
            if name in current_state:
                aggregated[name] = current_state[name] + aggregated[name]
        
        return aggregated
    
    def _aggregate_coordinate_median(
        self,
        client_updates: Dict[int, Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate using coordinate-wise median (robust to outliers).
        
        Args:
            client_updates: Client parameter updates
            
        Returns:
            Aggregated parameters
        """
        aggregated = {}
        first_cid = list(client_updates.keys())[0]
        param_names = list(client_updates[first_cid].keys())
        
        for name in param_names:
            # Stack all updates for this parameter
            updates_stack = torch.stack([
                update[name].to(self.device).flatten()
                for update in client_updates.values()
                if name in update
            ])
            
            # Compute median along client dimension
            median, _ = torch.median(updates_stack, dim=0)
            
            aggregated[name] = median.reshape(
                client_updates[first_cid][name].shape
            )
        
        # Add to current model
        current_state = {
            name: param.clone().detach()
            for name, param in self.model.named_parameters()
        }
        
        for name in aggregated:
            if name in current_state:
                aggregated[name] = current_state[name] + aggregated[name]
        
        return aggregated
    
    def _compute_update_norm(
        self,
        aggregated: Dict[str, torch.Tensor]
    ) -> float:
        """
        Compute L2 norm of aggregated update.
        
        Args:
            aggregated: Aggregated parameters
            
        Returns:
            Update norm
        """
        total_norm = 0.0
        
        current_state = {
            name: param.clone().detach()
            for name, param in self.model.named_parameters()
        }
        
        for name in aggregated:
            if name in current_state:
                diff = aggregated[name] - current_state[name]
                total_norm += torch.norm(diff, 2).item() ** 2
        
        return float(np.sqrt(total_norm))
    
    def get_aggregation_statistics(self) -> Dict[str, any]:
        """
        Get aggregation statistics.
        
        Returns:
            Dictionary with aggregation metrics
        """
        if not self.aggregation_history:
            return {}
        
        update_norms = [m['update_norm'] for m in self.aggregation_history]
        agg_times = [m['aggregation_time'] for m in self.aggregation_history]
        
        stats = {
            'num_aggregations': len(self.aggregation_history),
            'strategy': self.strategy.value,
            'avg_update_norm': float(np.mean(update_norms)),
            'std_update_norm': float(np.std(update_norms)),
            'max_update_norm': float(np.max(update_norms)),
            'avg_aggregation_time': float(np.mean(agg_times)),
            'total_aggregation_time': float(np.sum(agg_times))
        }
        
        return stats
    
    def set_strategy(self, strategy: AggregationStrategy):
        """
        Set aggregation strategy.
        
        Args:
            strategy: New aggregation strategy
        """
        self.strategy = strategy
        logger.info(f"Aggregation strategy set to {strategy.value}")