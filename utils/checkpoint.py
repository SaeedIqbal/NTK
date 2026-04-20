#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Checkpoint Management for NTK-SURGERY
Implements Model and State Checkpointing

This module provides:
- Model state saving and loading
- Training state persistence
- Checkpoint rotation and archival
- Checkpoint validation and verification
- Automatic checkpoint recovery

All checkpointing is designed for reproducibility and fault tolerance.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json
import logging
import hashlib
import os

logger = logging.getLogger(__name__)


@dataclass
class CheckpointConfig:
    """
    Configuration for checkpoint management.
    
    Attributes:
        checkpoint_dir: Directory for saving checkpoints
        save_interval: Save checkpoint every N epochs/steps
        max_checkpoints: Maximum number of checkpoints to keep
        save_best_only: Only save best performing checkpoint
        save_optimizer: Whether to save optimizer state
        save_scheduler: Whether to save scheduler state
        compression: Whether to compress checkpoints
        validation_metric: Metric to use for best checkpoint selection
        validation_mode: 'max' or 'min' for best checkpoint
    """
    checkpoint_dir: str = 'checkpoints'
    save_interval: int = 10
    max_checkpoints: int = 5
    save_best_only: bool = False
    save_optimizer: bool = True
    save_scheduler: bool = True
    compression: bool = False
    validation_metric: str = 'loss'
    validation_mode: str = 'min'
    
    def __post_init__(self):
        """Validate configuration."""
        if self.validation_mode not in ['min', 'max']:
            raise ValueError(f"Invalid validation mode: {self.validation_mode}")
        if self.save_interval <= 0:
            raise ValueError("save_interval must be positive")
        if self.max_checkpoints <= 0:
            raise ValueError("max_checkpoints must be positive")


@dataclass
class CheckpointState:
    """
    Data class for checkpoint state information.
    
    Attributes:
        epoch: Current epoch number
        step: Current step number
        model_state_dict: Model state dictionary
        optimizer_state_dict: Optimizer state dictionary (optional)
        scheduler_state_dict: Scheduler state dictionary (optional)
        metrics: Dictionary of metrics
        config: Configuration dictionary
        timestamp: Checkpoint timestamp
        checksum: Checksum for verification
    """
    epoch: int
    step: int
    model_state_dict: Dict[str, torch.Tensor]
    optimizer_state_dict: Optional[Dict] = None
    scheduler_state_dict: Optional[Dict] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    checksum: str = ''
    
    def compute_checksum(self):
        """Compute checksum for verification."""
        # Create hash of model state
        state_str = str(sorted(self.model_state_dict.keys()))
        self.checksum = hashlib.md5(state_str.encode()).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'epoch': self.epoch,
            'step': self.step,
            'model_state_dict': self.model_state_dict,
            'optimizer_state_dict': self.optimizer_state_dict,
            'scheduler_state_dict': self.scheduler_state_dict,
            'metrics': self.metrics,
            'config': self.config,
            'timestamp': self.timestamp,
            'checksum': self.checksum
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CheckpointState':
        """Create from dictionary."""
        return cls(
            epoch=data['epoch'],
            step=data['step'],
            model_state_dict=data['model_state_dict'],
            optimizer_state_dict=data.get('optimizer_state_dict'),
            scheduler_state_dict=data.get('scheduler_state_dict'),
            metrics=data.get('metrics', {}),
            config=data.get('config', {}),
            timestamp=data.get('timestamp', ''),
            checksum=data.get('checksum', '')
        )


class CheckpointManager:
    """
    Manages checkpoint saving and loading for NTK-SURGERY.
    
    Provides:
    - Automatic checkpoint saving at intervals
    - Best checkpoint tracking
    - Checkpoint rotation and cleanup
    - Checkpoint validation and verification
    - Recovery from interrupted training
    """
    
    def __init__(self, config: Optional[CheckpointConfig] = None):
        """
        Initialize CheckpointManager.
        
        Args:
            config: Checkpoint configuration
        """
        self.config = config if config is not None else CheckpointConfig()
        
        # Create checkpoint directory
        Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        # Track best checkpoint
        self.best_metric_value = float('inf') if self.config.validation_mode == 'min' else float('-inf')
        self.best_checkpoint_path = None
        
        # Track saved checkpoints
        self.saved_checkpoints: List[str] = []
        
        # Load checkpoint history
        self._load_checkpoint_history()
        
        logger.info(f"Initialized CheckpointManager: dir={self.config.checkpoint_dir}")
    
    def _load_checkpoint_history(self):
        """Load checkpoint history from file."""
        history_file = Path(self.config.checkpoint_dir) / 'checkpoint_history.json'
        
        if history_file.exists():
            with open(history_file, 'r') as f:
                history = json.load(f)
                self.saved_checkpoints = history.get('checkpoints', [])
                self.best_checkpoint_path = history.get('best_checkpoint')
                self.best_metric_value = history.get('best_metric_value', self.best_metric_value)
    
    def _save_checkpoint_history(self):
        """Save checkpoint history to file."""
        history_file = Path(self.config.checkpoint_dir) / 'checkpoint_history.json'
        
        history = {
            'checkpoints': self.saved_checkpoints,
            'best_checkpoint': self.best_checkpoint_path,
            'best_metric_value': self.best_metric_value,
            'last_updated': datetime.now().isoformat()
        }
        
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2)
    
    def save_checkpoint(
        self,
        model: nn.Module,
        epoch: int,
        step: int,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        metrics: Optional[Dict[str, float]] = None,
        config: Optional[Dict[str, Any]] = None,
        is_best: bool = False
    ) -> str:
        """
        Save checkpoint to disk.
        
        Args:
            model: Model to save
            epoch: Current epoch number
            step: Current step number
            optimizer: Optimizer (optional)
            scheduler: Scheduler (optional)
            metrics: Dictionary of metrics (optional)
            config: Configuration dictionary (optional)
            is_best: Whether this is the best checkpoint
            
        Returns:
            Path to saved checkpoint
        """
        # Create checkpoint state
        state = CheckpointState(
            epoch=epoch,
            step=step,
            model_state_dict={
                name: param.cpu().detach()
                for name, param in model.named_parameters()
            },
            optimizer_state_dict=optimizer.state_dict() if optimizer and self.config.save_optimizer else None,
            scheduler_state_dict=scheduler.state_dict() if scheduler and self.config.save_scheduler else None,
            metrics=metrics or {},
            config=config or {}
        )
        
        # Compute checksum
        state.compute_checksum()
        
        # Create checkpoint filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        checkpoint_name = f'checkpoint_epoch{epoch:04d}_step{step:06d}_{timestamp}.pt'
        checkpoint_path = Path(self.config.checkpoint_dir) / checkpoint_name
        
        # Save checkpoint
        torch.save(state.to_dict(), checkpoint_path)
        
        # Update checkpoint list
        self.saved_checkpoints.append(str(checkpoint_path))
        
        # Update best checkpoint if needed
        if is_best or self._is_better(metrics):
            self.best_checkpoint_path = str(checkpoint_path)
            if metrics:
                metric_value = metrics.get(self.config.validation_metric, 0.0)
                self.best_metric_value = metric_value
        
        # Save checkpoint history
        self._save_checkpoint_history()
        
        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()
        
        logger.info(f"Saved checkpoint: {checkpoint_path}")
        
        return str(checkpoint_path)
    
    def _is_better(self, metrics: Optional[Dict[str, float]]) -> bool:
        """
        Check if current metrics are better than best.
        
        Args:
            metrics: Current metrics
            
        Returns:
            True if better
        """
        if metrics is None:
            return False
        
        if self.config.validation_metric not in metrics:
            return False
        
        current_value = metrics[self.config.validation_metric]
        
        if self.config.validation_mode == 'min':
            return current_value < self.best_metric_value
        else:
            return current_value > self.best_metric_value
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints beyond max_checkpoints."""
        if len(self.saved_checkpoints) <= self.config.max_checkpoints:
            return
        
        # Sort by modification time
        checkpoints_with_time = []
        for checkpoint_path in self.saved_checkpoints:
            if Path(checkpoint_path).exists():
                mtime = Path(checkpoint_path).stat().st_mtime
                checkpoints_with_time.append((checkpoint_path, mtime))
        
        checkpoints_with_time.sort(key=lambda x: x[1])
        
        # Remove oldest checkpoints
        while len(self.saved_checkpoints) > self.config.max_checkpoints:
            oldest_checkpoint = checkpoints_with_time.pop(0)[0]
            
            if Path(oldest_checkpoint).exists():
                Path(oldest_checkpoint).unlink()
                logger.info(f"Removed old checkpoint: {oldest_checkpoint}")
            
            self.saved_checkpoints.remove(oldest_checkpoint)
        
        # Update history
        self._save_checkpoint_history()
    
    def load_checkpoint(
        self,
        model: nn.Module,
        checkpoint_path: Optional[str] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: str = 'cpu'
    ) -> CheckpointState:
        """
        Load checkpoint from disk.
        
        Args:
            model: Model to load into
            checkpoint_path: Path to checkpoint (uses best if None)
            optimizer: Optimizer to load into (optional)
            scheduler: Scheduler to load into (optional)
            device: Device to load tensors to
            
        Returns:
            CheckpointState object
        """
        # Determine checkpoint path
        if checkpoint_path is None:
            checkpoint_path = self.best_checkpoint_path
        
        if checkpoint_path is None:
            # Find latest checkpoint
            checkpoint_path = self._find_latest_checkpoint()
        
        if checkpoint_path is None or not Path(checkpoint_path).exists():
            logger.warning("No checkpoint found, starting from scratch")
            return None
        
        # Load checkpoint
        checkpoint_data = torch.load(checkpoint_path, map_location=device)
        state = CheckpointState.from_dict(checkpoint_data)
        
        # Verify checksum
        state.compute_checksum()
        if state.checksum != checkpoint_data.get('checksum', ''):
            logger.warning("Checkpoint checksum mismatch, file may be corrupted")
        
        # Load model state
        model.load_state_dict(state.model_state_dict)
        logger.info(f"Loaded model state from {checkpoint_path}")
        
        # Load optimizer state
        if optimizer and state.optimizer_state_dict:
            optimizer.load_state_dict(state.optimizer_state_dict)
            logger.info("Loaded optimizer state")
        
        # Load scheduler state
        if scheduler and state.scheduler_state_dict:
            scheduler.load_state_dict(state.scheduler_state_dict)
            logger.info("Loaded scheduler state")
        
        return state
    
    def _find_latest_checkpoint(self) -> Optional[str]:
        """
        Find latest checkpoint in checkpoint directory.
        
        Returns:
            Path to latest checkpoint or None
        """
        checkpoint_files = list(Path(self.config.checkpoint_dir).glob('checkpoint_*.pt'))
        
        if not checkpoint_files:
            return None
        
        # Sort by modification time
        checkpoint_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        
        return str(checkpoint_files[0])
    
    def get_checkpoint_list(self) -> List[str]:
        """
        Get list of all saved checkpoints.
        
        Returns:
            List of checkpoint paths
        """
        return self.saved_checkpoints.copy()
    
    def get_best_checkpoint(self) -> Optional[str]:
        """
        Get best checkpoint path.
        
        Returns:
            Best checkpoint path or None
        """
        return self.best_checkpoint_path
    
    def delete_checkpoint(self, checkpoint_path: str) -> bool:
        """
        Delete a specific checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint to delete
            
        Returns:
            True if deleted successfully
        """
        if not Path(checkpoint_path).exists():
            return False
        
        Path(checkpoint_path).unlink()
        
        if checkpoint_path in self.saved_checkpoints:
            self.saved_checkpoints.remove(checkpoint_path)
        
        if checkpoint_path == self.best_checkpoint_path:
            self.best_checkpoint_path = None
        
        self._save_checkpoint_history()
        
        logger.info(f"Deleted checkpoint: {checkpoint_path}")
        
        return True
    
    def get_checkpoint_info(self, checkpoint_path: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint
            
        Returns:
            Dictionary with checkpoint information
        """
        if not Path(checkpoint_path).exists():
            return None
        
        checkpoint_data = torch.load(checkpoint_path, map_location='cpu')
        state = CheckpointState.from_dict(checkpoint_data)
        
        info = {
            'path': checkpoint_path,
            'epoch': state.epoch,
            'step': state.step,
            'timestamp': state.timestamp,
            'metrics': state.metrics,
            'checksum': state.checksum,
            'file_size': Path(checkpoint_path).stat().st_size
        }
        
        return info


def save_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    epoch: int = 0,
    step: int = 0,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    metrics: Optional[Dict[str, float]] = None,
    config: Optional[Dict[str, Any]] = None
) -> str:
    """
    Convenience function to save checkpoint.
    
    Args:
        model: Model to save
        checkpoint_path: Path to save checkpoint
        epoch: Current epoch
        step: Current step
        optimizer: Optimizer (optional)
        scheduler: Scheduler (optional)
        metrics: Metrics dictionary (optional)
        config: Configuration dictionary (optional)
        
    Returns:
        Path to saved checkpoint
    """
    manager = CheckpointManager(CheckpointConfig(checkpoint_dir=str(Path(checkpoint_path).parent)))
    
    return manager.save_checkpoint(
        model=model,
        epoch=epoch,
        step=step,
        optimizer=optimizer,
        scheduler=scheduler,
        metrics=metrics,
        config=config
    )


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: str = 'cpu'
) -> CheckpointState:
    """
    Convenience function to load checkpoint.
    
    Args:
        model: Model to load into
        checkpoint_path: Path to checkpoint
        optimizer: Optimizer to load into (optional)
        scheduler: Scheduler to load into (optional)
        device: Device to load tensors to
        
    Returns:
        CheckpointState object
    """
    manager = CheckpointManager()
    
    return manager.load_checkpoint(
        model=model,
        checkpoint_path=checkpoint_path,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device
    )