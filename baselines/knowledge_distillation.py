#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Knowledge Distillation for Federated Unlearning
Uses knowledge transfer from teacher to student model

Key limitation: Fails to guarantee exact removal of teacher influences
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Optional
import logging
import time
from dataclasses import dataclass
import copy

logger = logging.getLogger(__name__)


@dataclass
class KDConfig:
    """Configuration for Knowledge Distillation."""
    learning_rate: float = 0.01
    epochs: int = 50
    batch_size: int = 64
    temperature: float = 2.0
    alpha: float = 0.5  # Balance between hard and soft labels
    device: str = 'cpu'


class KnowledgeDistillation:
    """
    Knowledge Distillation for federated unlearning.
    
    Trains student model on remaining data using teacher's soft labels.
    
    Critical limitation: Does not guarantee exact removal of teacher influences.
    """
    
    def __init__(
        self,
        teacher_model: nn.Module,
        student_model: Optional[nn.Module] = None,
        config: Optional[KDConfig] = None
    ):
        """
        Initialize Knowledge Distillation.
        
        Args:
            teacher_model: Teacher model (original)
            student_model: Student model (optional, defaults to teacher architecture)
            config: KD configuration
        """
        self.teacher = teacher_model
        self.student = student_model if student_model is not None else copy.deepcopy(teacher_model)
        self.config = config if config is not None else KDConfig()
        self.device = self.config.device
        
        # Unlearning metrics
        self.unlearning_time = 0.0
        
        logger.info(f"Initialized KD with temperature={self.config.temperature}")
    
    def compute_soft_labels(
        self,
        data: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute soft labels from teacher model.
        
        Args:
            data: Input data
            
        Returns:
            Soft labels (probabilities)
        """
        self.teacher.eval()
        
        with torch.no_grad():
            teacher_output = self.teacher(data.to(self.device))
            soft_labels = torch.softmax(teacher_output / self.config.temperature, dim=1)
        
        return soft_labels
    
    def distillation_loss(
        self,
        student_output: torch.Tensor,
        soft_labels: torch.Tensor,
        hard_labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute knowledge distillation loss.
        
        Args:
            student_output: Student model output
            soft_labels: Teacher's soft labels
            hard_labels: True labels
            
        Returns:
            Combined loss
        """
        # Soft loss (KL divergence)
        soft_loss = nn.KLDivLoss(reduction='batchmean')(
            torch.log_softmax(student_output / self.config.temperature, dim=1),
            soft_labels
        ) * (self.config.temperature ** 2)
        
        # Hard loss (cross-entropy)
        hard_loss = nn.CrossEntropyLoss()(student_output, hard_labels)
        
        # Combined loss
        loss = self.config.alpha * soft_loss + (1 - self.config.alpha) * hard_loss
        
        return loss
    
    def unlearn_client(
        self,
        client_id: int,
        remaining_clients: Dict[int, torch.utils.data.DataLoader]
    ) -> nn.Module:
        """
        Unlearn client via knowledge distillation.
        
        Args:
            client_id: Client to remove
            remaining_clients: DataLoaders for remaining clients
            
        Returns:
            Student model (unlearned)
        """
        start_time = time.perf_counter()
        
        logger.info(f"KD unlearning client {client_id}")
        
        self.student.train()
        self.teacher.eval()
        
        optimizer = optim.SGD(
            self.student.parameters(),
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
                    
                    # Get soft labels from teacher
                    soft_labels = self.compute_soft_labels(data)
                    
                    # Student output
                    student_output = self.student(data)
                    
                    # Compute loss
                    loss = self.distillation_loss(student_output, soft_labels, target)
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    num_batches += 1
            
            avg_epoch_loss = epoch_loss / max(num_batches, 1)
            
            if epoch % 10 == 0:
                logger.info(f"KD epoch {epoch}: loss = {avg_epoch_loss:.4f}")
            
            total_loss += avg_epoch_loss
        
        self.unlearning_time = time.perf_counter() - start_time
        avg_loss = total_loss / self.config.epochs
        
        logger.info(
            f"KD unlearning completed in {self.unlearning_time:.2f}s. "
            f"Avg loss: {avg_loss:.4f}"
        )
        
        return self.student
    
    def get_efficiency_metrics(self) -> Dict[str, any]:
        """
        Get efficiency metrics for KD.
        
        Returns:
            Dictionary with efficiency metrics
        """
        return {
            'method': 'Knowledge Distillation',
            'unlearning_time': self.unlearning_time,
            'temperature': self.config.temperature,
            'alpha': self.config.alpha,
            'complexity_class': 'O(epochs · M · K · P)',
            'exact_removal_guarantee': False
        }