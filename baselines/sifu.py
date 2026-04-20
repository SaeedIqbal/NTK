#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SIFU: Sequential Informed Federated Unlearning
Implements the SIFU baseline from Fraboni et al., 2024

Key limitation: Sensitivity bound ζ(n, c) grows exponentially in non-convex settings
when contraction factor B(f_I, η) > 1 (Eq. 26, SIFU paper)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional
import logging
import time
from dataclasses import dataclass
import copy

logger = logging.getLogger(__name__)


@dataclass
class SIFUConfig:
    """Configuration for SIFU unlearning."""
    learning_rate: float = 0.01
    local_epochs: int = 5
    batch_size: int = 64
    epsilon: float = 1.0
    delta: float = 1e-5
    max_rollback_rounds: int = 50
    sensitivity_threshold: float = 1.0
    device: str = 'cpu'


class SIFU:
    """
    Sequential Informed Federated Unlearning (SIFU).
    
    Provides (ε, δ)-unlearning guarantees via Gaussian perturbation
    proportional to sensitivity bound ζ(n, c).
    
    Critical limitation: In non-convex regimes, B(f_I, η) > 1 causes
    exponential growth of ζ(n, c), forcing rollback to T ≈ 0.
    
    Implements SIFU Eq. 9: ζ(n, c) = Σ_{s=0}^{n-1} B(f_I, η)^{(n-s-1)K} · ||ω(I, θ_s)||_2
    """
    
    def __init__(self, model: nn.Module, config: Optional[SIFUConfig] = None):
        """
        Initialize SIFU.
        
        Args:
            model: Neural network model
            config: SIFU configuration
        """
        self.model = model
        self.config = config if config is not None else SIFUConfig()
        self.device = self.config.device
        
        # Training history for rollback
        self.checkpoints = {}
        self.gradient_history = []
        self.sensitivity_bounds = []
        
        # Unlearning metrics
        self.unlearning_time = 0.0
        self.rollback_rounds = 0
        
        logger.info(f"Initialized SIFU with ε={self.config.epsilon}, δ={self.config.delta}")
    
    def store_checkpoint(self, round_num: int, theta: Dict[str, torch.Tensor]):
        """
        Store model checkpoint for potential rollback.
        
        Args:
            round_num: Training round number
            theta: Model parameters
        """
        self.checkpoints[round_num] = {
            name: param.clone().detach().cpu() 
            for name, param in theta.items()
        }
        logger.debug(f"Stored checkpoint for round {round_num}")
    
    def compute_sensitivity_bound(
        self,
        n_rounds: int,
        gradient_norms: List[float],
        B_factor: float = 1.05
    ) -> float:
        """
        Compute sensitivity bound ζ(n, c).
        
        Implements SIFU Eq. 9:
        ζ(n, c) = Σ_{s=0}^{n-1} B(f_I, η)^{(n-s-1)K} · ||ω(I, θ_s)||_2
        
        In non-convex settings, B(f_I, η) > 1 (Eq. 26), causing exponential growth.
        
        Args:
            n_rounds: Number of training rounds
            gradient_norms: List of gradient norms per round
            B_factor: Contraction factor B(f_I, η)
            
        Returns:
            Sensitivity bound value
        """
        K = self.config.local_epochs
        zeta = 0.0
        
        for s in range(min(n_rounds, len(gradient_norms))):
            exponent = (n_rounds - s - 1) * K
            contribution = (B_factor ** exponent) * gradient_norms[s]
            zeta += contribution
        
        self.sensitivity_bounds.append(zeta)
        
        logger.info(
            f"Sensitivity bound ζ({n_rounds}, c) = {zeta:.4f} "
            f"(B={B_factor}, exponential growth: {B_factor > 1})"
        )
        
        return zeta
    
    def compute_noise_scale(self, zeta: float) -> float:
        """
        Compute Gaussian noise scale for (ε, δ)-unlearning.
        
        Implements SIFU Theorem 2: σ = ζ · ε^{-1} · √(2(ln(1.25) - ln(δ)))
        
        Args:
            zeta: Sensitivity bound
            
        Returns:
            Noise standard deviation
        """
        epsilon = self.config.epsilon
        delta = self.config.delta
        
        threshold = np.sqrt(2 * (np.log(1.25) - np.log(delta))) / epsilon
        sigma = zeta * threshold
        
        logger.info(f"Noise scale σ = {sigma:.4f} (ζ={zeta:.4f}, ε={epsilon}, δ={delta})")
        
        return sigma
    
    def find_rollback_checkpoint(
        self,
        sensitivity_budget: float,
        gradient_norms: List[float],
        B_factor: float = 1.05
    ) -> int:
        """
        Find earliest checkpoint satisfying sensitivity budget.
        
        Implements SIFU Eq. 12: T = max{n : ζ(n, c) ≤ Δ}
        
        In non-convex settings, this forces T → 0, reducing to retraining.
        
        Args:
            sensitivity_budget: Maximum allowed sensitivity Δ
            gradient_norms: Historical gradient norms
            B_factor: Contraction factor
            
        Returns:
            Checkpoint round number T
        """
        T = 0
        
        for n in range(len(gradient_norms)):
            zeta_n = self.compute_sensitivity_bound(n, gradient_norms, B_factor)
            
            if zeta_n <= sensitivity_budget:
                T = n
            else:
                # Sensitivity exceeded budget
                logger.warning(
                    f"Sensitivity ζ({n}) = {zeta_n:.4f} exceeds budget {sensitivity_budget}. "
                    f"Rollback to T = {T}"
                )
                break
        
        self.rollback_rounds = T
        
        if T == 0:
            logger.critical(
                "SIFU reduced to retraining from scratch (T=0) due to "
                "exponential sensitivity growth in non-convex regime"
            )
        
        return T
    
    def unlearn_client(
        self,
        client_id: int,
        remaining_clients: Dict[int, torch.utils.data.DataLoader],
        gradient_norms: List[float],
        sensitivity_budget: float = 1.0
    ) -> nn.Module:
        """
        Unlearn a client using SIFU method.
        
        Args:
            client_id: Client to remove
            remaining_clients: DataLoaders for remaining clients
            gradient_norms: Historical gradient norms
            sensitivity_budget: Maximum sensitivity Δ
            
        Returns:
            Unlearned model
        """
        start_time = time.perf_counter()
        
        logger.info(f"SIFU unlearning client {client_id}")
        
        # Step 1: Find rollback checkpoint
        T = self.find_rollback_checkpoint(
            sensitivity_budget, gradient_norms
        )
        
        # Step 2: Load checkpoint
        if T in self.checkpoints:
            checkpoint = self.checkpoints[T]
            self.model.load_state_dict(checkpoint)
            logger.info(f"Loaded checkpoint from round {T}")
        else:
            logger.warning(f"Checkpoint {T} not found. Using current model.")
        
        # Step 3: Compute sensitivity bound
        zeta = self.compute_sensitivity_bound(
            len(gradient_norms), gradient_norms
        )
        
        # Step 4: Compute noise scale
        sigma = self.compute_noise_scale(zeta)
        
        # Step 5: Apply Gaussian perturbation to weights
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                noise = torch.randn_like(param) * sigma
                param.add_(noise)
        
        # Step 6: Fine-tune on remaining data (SIFU Algorithm 2)
        self._fine_tune_remaining(remaining_clients)
        
        self.unlearning_time = time.perf_counter() - start_time
        
        logger.info(
            f"SIFU unlearning completed in {self.unlearning_time:.2f}s. "
            f"Rollback rounds: {self.rollback_rounds}, Noise σ: {sigma:.4f}"
        )
        
        return self.model
    
    def _fine_tune_remaining(
        self,
        remaining_clients: Dict[int, torch.utils.data.DataLoader]
    ):
        """
        Fine-tune model on remaining clients' data.
        
        Args:
            remaining_clients: DataLoaders for remaining clients
        """
        self.model.train()
        optimizer = optim.SGD(
            self.model.parameters(), 
            lr=self.config.learning_rate
        )
        
        total_loss = 0.0
        num_batches = 0
        
        for client_id, loader in remaining_clients.items():
            for batch_idx, (data, target) in enumerate(loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(data)
                loss = nn.CrossEntropyLoss()(output, target)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        logger.info(f"Fine-tuning completed. Average loss: {avg_loss:.4f}")
    
    def get_efficiency_metrics(self) -> Dict[str, any]:
        """
        Get efficiency metrics for SIFU.
        
        Returns:
            Dictionary with efficiency metrics
        """
        return {
            'method': 'SIFU',
            'unlearning_time': self.unlearning_time,
            'rollback_rounds': self.rollback_rounds,
            'max_rollback_rounds': self.config.max_rollback_rounds,
            'sensitivity_bounds': self.sensitivity_bounds,
            'checkpoints_stored': len(self.checkpoints),
            'complexity_class': 'O(T · K · P) where T → 0 in non-convex'
        }