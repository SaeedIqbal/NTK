#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Federated Averaging (FedAvg) Implementation for NTK-SURGERY
Implements Section 3.1 (Preliminaries) of the manuscript

FedAvg Update Rule (Eq. 5 from SIFU):
θ_{n+1} = θ_n + ω(I, θ_n)
where ω(I, θ_n) = Σ_{i∈I} (|D_i|/|D_I|) (θ_{n+1}^i - θ_n)

This module orchestrates the complete federated training process.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
import logging
import time
from dataclasses import dataclass
from pathlib import Path
import copy
import json

from training.local_training import LocalTrainer, LocalTrainingConfig
from training.server_aggregation import ServerAggregator, ServerState

logger = logging.getLogger(__name__)


@dataclass
class FedAvgConfig:
    """
    Configuration for Federated Averaging.
    
    Attributes:
        learning_rate: Global learning rate
        local_epochs: Number of local epochs per client
        batch_size: Mini-batch size for local training
        communication_rounds: Total federated rounds
        fraction_clients: Fraction of clients per round
        num_clients: Total number of clients
        device: Computing device
        checkpoint_dir: Directory for saving checkpoints
        save_checkpoints: Whether to save checkpoints (for SIFU baseline)
        checkpoint_interval: Save checkpoint every N rounds
    """
    learning_rate: float = 0.01
    local_epochs: int = 5
    batch_size: int = 64
    communication_rounds: int = 50
    fraction_clients: float = 1.0
    num_clients: int = 100
    device: str = 'cpu'
    checkpoint_dir: str = 'checkpoints'
    save_checkpoints: bool = True
    checkpoint_interval: int = 10
    
    def __post_init__(self):
        """Validate configuration."""
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.local_epochs <= 0:
            raise ValueError("local_epochs must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.communication_rounds <= 0:
            raise ValueError("communication_rounds must be positive")
        if not 0 < self.fraction_clients <= 1:
            raise ValueError("fraction_clients must be in (0, 1]")


class FedAvg:
    """
    Federated Averaging (FedAvg) orchestrator.
    
    Implements the standard FedAvg algorithm with support for:
    - Partial client participation
    - Checkpoint saving for unlearning baselines (SIFU)
    - Gradient norm tracking for sensitivity analysis
    - Comprehensive logging and metrics
    
    Implements manuscript Eq. 5:
    θ_{n+1} = θ_n + Σ_{i∈I} (|D_i|/|D_I|) (θ_{n+1}^i - θ_n)
    
    Attributes:
        model (nn.Module): Global model
        config (FedAvgConfig): Training configuration
        server_aggregator (ServerAggregator): Server-side aggregation
        local_trainer (LocalTrainer): Client-side training
        server_state (ServerState): Current server state
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Optional[FedAvgConfig] = None
    ):
        """
        Initialize FedAvg.
        
        Args:
            model: Global neural network model
            config: FedAvg configuration
        """
        self.model = model
        self.config = config if config is not None else FedAvgConfig()
        self.device = self.config.device
        
        # Move model to device
        self.model.to(self.device)
        
        # Initialize components
        self.server_aggregator = ServerAggregator(self.model, self.device)
        self.local_trainer = LocalTrainer(
            model,
            LocalTrainingConfig(
                learning_rate=self.config.learning_rate,
                epochs=self.config.local_epochs,
                batch_size=self.config.batch_size,
                device=self.device
            )
        )
        
        # Server state
        self.server_state = ServerState(
            round_num=0,
            model_state_dict=self._get_model_state(),
            gradient_norms=[],
            checkpoint_history={}
        )
        
        # Training metrics
        self.training_history = []
        self.gradient_norm_history = []
        
        # Create checkpoint directory
        if self.config.save_checkpoints:
            Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info(
            f"Initialized FedAvg: rounds={self.config.communication_rounds}, "
            f"clients={self.config.num_clients}, lr={self.config.learning_rate}"
        )
    
    def _get_model_state(self) -> Dict[str, torch.Tensor]:
        """Get current model state dict."""
        return {
            name: param.clone().detach().cpu()
            for name, param in self.model.named_parameters()
        }
    
    def _set_model_state(self, state_dict: Dict[str, torch.Tensor]):
        """Set model state dict."""
        self.model.load_state_dict(state_dict)
    
    def select_clients_for_round(
        self,
        round_num: int,
        client_loaders: Dict[int, torch.utils.data.DataLoader]
    ) -> Dict[int, torch.utils.data.DataLoader]:
        """
        Select clients for current round.
        
        Args:
            round_num: Current round number
            client_loaders: All client DataLoaders
            
        Returns:
            Dictionary of selected client DataLoaders
        """
        num_clients = len(client_loaders)
        num_selected = max(1, int(num_clients * self.config.fraction_clients))
        
        # Deterministic selection based on round for reproducibility
        np.random.seed(round_num)
        selected_ids = np.random.choice(
            list(client_loaders.keys()),
            size=num_selected,
            replace=False
        ).tolist()
        
        selected_loaders = {
            cid: client_loaders[cid] for cid in selected_ids
        }
        
        logger.info(
            f"Round {round_num}: Selected {len(selected_loaders)}/{num_clients} clients"
        )
        
        return selected_loaders
    
    def train_client(
        self,
        client_id: int,
        loader: torch.utils.data.DataLoader,
        round_num: int
    ) -> Tuple[Dict[str, torch.Tensor], float, float]:
        """
        Train a single client locally.
        
        Args:
            client_id: Client identifier
            loader: Client's DataLoader
            round_num: Current round number
            
        Returns:
            Tuple of (updated_params, loss, gradient_norm)
        """
        # Set client model to current global state
        self._set_model_state(self.server_state.model_state_dict)
        
        # Train locally
        loss, gradient_norm = self.local_trainer.train(
            client_id,
            loader,
            round_num
        )
        
        # Get updated parameters
        updated_params = self._get_model_state()
        
        logger.debug(
            f"Client {client_id}: loss={loss:.4f}, grad_norm={gradient_norm:.4f}"
        )
        
        return updated_params, loss, gradient_norm
    
    def aggregate_client_updates(
        self,
        client_updates: Dict[int, Dict[str, torch.Tensor]],
        client_sizes: Dict[int, int]
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate client updates using weighted averaging.
        
        Implements manuscript Eq. 5:
        ω(I, θ_n) = Σ_{i∈I} (|D_i|/|D_I|) (θ_{n+1}^i - θ_n)
        
        Args:
            client_updates: Dictionary of client parameter updates
            client_sizes: Dictionary of client dataset sizes
            
        Returns:
            Aggregated global parameters
        """
        aggregated = self.server_aggregator.aggregate(
            client_updates,
            client_sizes
        )
        
        return aggregated
    
    def compute_gradient_norm(
        self,
        old_params: Dict[str, torch.Tensor],
        new_params: Dict[str, torch.Tensor]
    ) -> float:
        """
        Compute gradient norm for sensitivity analysis.
        
        Implements ||ω(I, θ_n)||_2 from SIFU Eq. 9.
        
        Args:
            old_params: Parameters before update
            new_params: Parameters after update
            
        Returns:
            Gradient norm (L2 norm of parameter change)
        """
        total_norm = 0.0
        
        for name in old_params:
            if name in new_params:
                diff = new_params[name] - old_params[name]
                total_norm += torch.norm(diff, 2).item() ** 2
        
        gradient_norm = np.sqrt(total_norm)
        
        return float(gradient_norm)
    
    def save_checkpoint(self, round_num: int):
        """
        Save model checkpoint for potential rollback (SIFU baseline).
        
        Args:
            round_num: Current round number
        """
        if not self.config.save_checkpoints:
            return
        
        checkpoint_path = Path(self.config.checkpoint_dir) / f"round_{round_num:03d}.pt"
        
        checkpoint = {
            'round': round_num,
            'model_state_dict': self._get_model_state(),
            'server_state': self.server_state.to_dict(),
            'gradient_norms': self.gradient_norm_history.copy(),
            'config': {
                'learning_rate': self.config.learning_rate,
                'local_epochs': self.config.local_epochs,
                'batch_size': self.config.batch_size
            }
        }
        
        torch.save(checkpoint, checkpoint_path)
        
        # Store in memory for quick access
        self.server_state.checkpoint_history[round_num] = checkpoint_path
        
        logger.info(f"Saved checkpoint for round {round_num} at {checkpoint_path}")
    
    def load_checkpoint(self, round_num: int) -> Optional[Dict]:
        """
        Load model checkpoint from specific round.
        
        Args:
            round_num: Round number to load
            
        Returns:
            Checkpoint dictionary or None if not found
        """
        if round_num not in self.server_state.checkpoint_history:
            logger.warning(f"Checkpoint for round {round_num} not found")
            return None
        
        checkpoint_path = self.server_state.checkpoint_history[round_num]
        
        if not Path(checkpoint_path).exists():
            logger.warning(f"Checkpoint file {checkpoint_path} not found")
            return None
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Restore model state
        self._set_model_state(checkpoint['model_state_dict'])
        
        logger.info(f"Loaded checkpoint from round {round_num}")
        
        return checkpoint
    
    def train(
        self,
        client_loaders: Dict[int, torch.utils.data.DataLoader],
        client_sizes: Optional[Dict[int, int]] = None,
        callback: Optional[Callable[[int, Dict], None]] = None
    ) -> Dict[str, any]:
        """
        Run complete federated training.
        
        Args:
            client_loaders: Dictionary of client DataLoaders
            client_sizes: Dictionary of client dataset sizes (optional)
            callback: Optional callback function(round_num, metrics)
            
        Returns:
            Training results dictionary
        """
        logger.info(
            f"Starting FedAvg training for {self.config.communication_rounds} rounds"
        )
        
        start_time = time.perf_counter()
        
        # Initialize client sizes if not provided
        if client_sizes is None:
            client_sizes = {
                cid: len(loader.dataset)
                for cid, loader in client_loaders.items()
            }
        
        total_size = sum(client_sizes.values())
        
        for round_num in range(self.config.communication_rounds):
            round_start = time.perf_counter()
            
            # Store old parameters for gradient norm computation
            old_params = self._get_model_state()
            
            # Select clients for this round
            selected_loaders = self.select_clients_for_round(
                round_num,
                client_loaders
            )
            
            # Train selected clients
            client_updates = {}
            round_losses = []
            
            for cid, loader in selected_loaders.items():
                updated_params, loss, _ = self.train_client(
                    cid, loader, round_num
                )
                
                # Compute update: θ_{n+1}^i - θ_n
                update = {
                    name: updated_params[name] - old_params[name]
                    for name in old_params
                }
                
                client_updates[cid] = update
                round_losses.append(loss)
            
            # Aggregate updates
            new_params = self.aggregate_client_updates(
                client_updates,
                {cid: client_sizes[cid] for cid in client_updates}
            )
            
            # Apply aggregated update
            self._set_model_state(new_params)
            
            # Compute gradient norm for sensitivity analysis
            gradient_norm = self.compute_gradient_norm(old_params, new_params)
            self.gradient_norm_history.append(gradient_norm)
            
            # Update server state
            self.server_state.round_num = round_num + 1
            self.server_state.model_state_dict = self._get_model_state()
            
            # Compute round metrics
            round_time = time.perf_counter() - round_start
            avg_loss = float(np.mean(round_losses))
            
            round_metrics = {
                'round': round_num,
                'avg_loss': avg_loss,
                'gradient_norm': gradient_norm,
                'num_clients': len(selected_loaders),
                'round_time': round_time
            }
            
            self.training_history.append(round_metrics)
            
            # Save checkpoint
            if self.config.save_checkpoints and (
                round_num % self.config.checkpoint_interval == 0 or
                round_num == self.config.communication_rounds - 1
            ):
                self.save_checkpoint(round_num)
            
            # Callback
            if callback is not None:
                callback(round_num, round_metrics)
            
            # Log progress
            if round_num % 10 == 0 or round_num == self.config.communication_rounds - 1:
                logger.info(
                    f"Round {round_num}: loss={avg_loss:.4f}, "
                    f"grad_norm={gradient_norm:.4f}, time={round_time:.2f}s"
                )
        
        total_time = time.perf_counter() - start_time
        
        results = {
            'total_rounds': self.config.communication_rounds,
            'total_time': total_time,
            'avg_loss_per_round': float(np.mean([m['avg_loss'] for m in self.training_history])),
            'avg_gradient_norm': float(np.mean(self.gradient_norm_history)),
            'final_gradient_norm': self.gradient_norm_history[-1] if self.gradient_norm_history else 0.0,
            'checkpoints_saved': len(self.server_state.checkpoint_history),
            'training_history': self.training_history
        }
        
        logger.info(
            f"FedAvg training completed: {self.config.communication_rounds} rounds, "
            f"{total_time:.2f}s, avg_loss={results['avg_loss_per_round']:.4f}"
        )
        
        return results
    
    def get_training_summary(self) -> Dict[str, any]:
        """
        Get comprehensive training summary.
        
        Returns:
            Dictionary with training summary
        """
        if not self.training_history:
            return {}
        
        losses = [m['avg_loss'] for m in self.training_history]
        grad_norms = self.gradient_norm_history
        
        summary = {
            'config': {
                'learning_rate': self.config.learning_rate,
                'local_epochs': self.config.local_epochs,
                'batch_size': self.config.batch_size,
                'communication_rounds': self.config.communication_rounds,
                'fraction_clients': self.config.fraction_clients
            },
            'performance': {
                'total_rounds': len(self.training_history),
                'avg_loss': float(np.mean(losses)),
                'std_loss': float(np.std(losses)),
                'min_loss': float(np.min(losses)),
                'max_loss': float(np.max(losses)),
                'avg_gradient_norm': float(np.mean(grad_norms)),
                'max_gradient_norm': float(np.max(grad_norms))
            },
            'checkpoints': {
                'num_saved': len(self.server_state.checkpoint_history),
                'checkpoint_rounds': sorted(list(self.server_state.checkpoint_history.keys()))
            }
        }
        
        return summary
    
    def export_training_metrics(self, filepath: str):
        """
        Export training metrics to JSON file.
        
        Args:
            filepath: Path to save metrics
        """
        metrics = {
            'config': self.config.__dict__,
            'summary': self.get_training_summary(),
            'history': self.training_history,
            'gradient_norms': self.gradient_norm_history
        }
        
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        
        logger.info(f"Exported training metrics to {filepath}")


class FedAvgTrainer:
    """
    High-level trainer wrapper for FedAvg.
    
    Provides simplified interface for training with FedAvg.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Optional[FedAvgConfig] = None
    ):
        """
        Initialize FedAvgTrainer.
        
        Args:
            model: Neural network model
            config: FedAvg configuration
        """
        self.fedavg = FedAvg(model, config)
        self.is_trained = False
        self.results = None
    
    def train(
        self,
        client_loaders: Dict[int, torch.utils.data.DataLoader],
        client_sizes: Optional[Dict[int, int]] = None
    ) -> Dict[str, any]:
        """
        Train using FedAvg.
        
        Args:
            client_loaders: Client DataLoaders
            client_sizes: Client dataset sizes
            
        Returns:
            Training results
        """
        self.results = self.fedavg.train(client_loaders, client_sizes)
        self.is_trained = True
        
        return self.results
    
    def get_model(self) -> nn.Module:
        """
        Get trained model.
        
        Returns:
            Trained model
        """
        return self.fedavg.model
    
    def get_checkpoint(self, round_num: int) -> Optional[Dict]:
        """
        Get checkpoint from specific round.
        
        Args:
            round_num: Round number
            
        Returns:
            Checkpoint dictionary
        """
        return self.fedavg.load_checkpoint(round_num)
    
    def get_gradient_norms(self) -> List[float]:
        """
        Get gradient norm history for sensitivity analysis.
        
        Returns:
            List of gradient norms per round
        """
        return self.fedavg.gradient_norm_history