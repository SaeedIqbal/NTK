#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Random Seed Management for NTK-SURGERY
Implements Reproducible Random Number Generation

This module provides:
- Centralized random seed management
- Seed setting for multiple libraries (numpy, torch, python)
- Seed state saving and restoration
- Deterministic algorithm configuration
- Reproducibility verification

NOTE: This module manages seeds but does NOT generate random numbers
to ensure deterministic behavior as per manuscript requirements.
"""

import random
import numpy as np
import torch
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path
import logging
import os
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class SeedConfig:
    """
    Configuration for random seed management.
    
    Attributes:
        seed: Base random seed
        deterministic: Whether to use deterministic algorithms
        benchmark: Whether to enable cudnn benchmark
        save_seed_state: Whether to save seed state to file
        seed_file: Path to save seed state
        verify_reproducibility: Whether to verify reproducibility
    """
    seed: int = 42
    deterministic: bool = True
    benchmark: bool = False
    save_seed_state: bool = True
    seed_file: str = 'seed_state.json'
    verify_reproducibility: bool = True
    
    def __post_init__(self):
        """Validate configuration."""
        if not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")
        if self.seed < 0 or self.seed > 2**32 - 1:
            raise ValueError("seed must be in range [0, 2^32 - 1]")


@dataclass
class SeedState:
    """
    Data class for seed state information.
    
    Attributes:
        seed: Base seed value
        numpy_state: NumPy random state
        torch_state: PyTorch random state
        python_state: Python random state
        cuda_state: CUDA random state (if available)
        timestamp: State timestamp
        config_hash: Hash of configuration for verification
    """
    seed: int
    numpy_state: Optional[Tuple] = None
    torch_state: Optional[Tuple] = None
    python_state: Optional[Tuple] = None
    cuda_state: Optional[Tuple] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    config_hash: str = ''
    
    def compute_config_hash(self, config: SeedConfig):
        """Compute hash of configuration for verification."""
        config_str = json.dumps(config.__dict__, sort_keys=True)
        self.config_hash = hashlib.md5(config_str.encode()).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'seed': self.seed,
            'numpy_state': self.numpy_state,
            'torch_state': self.torch_state,
            'python_state': self.python_state,
            'cuda_state': self.cuda_state,
            'timestamp': self.timestamp,
            'config_hash': self.config_hash
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SeedState':
        """Create from dictionary."""
        return cls(
            seed=data['seed'],
            numpy_state=data.get('numpy_state'),
            torch_state=data.get('torch_state'),
            python_state=data.get('python_state'),
            cuda_state=data.get('cuda_state'),
            timestamp=data.get('timestamp', ''),
            config_hash=data.get('config_hash', '')
        )


class SeedManager:
    """
    Manages random seeds for reproducible experiments.
    
    Provides:
    - Centralized seed setting for all libraries
    - Seed state saving and restoration
    - Deterministic algorithm configuration
    - Reproducibility verification
    - Multiple seed management for different components
    """
    
    _instance: Optional['SeedManager'] = None
    
    def __new__(cls, config: Optional[SeedConfig] = None) -> 'SeedManager':
        """
        Create or retrieve SeedManager instance (singleton pattern).
        
        Args:
            config: Seed configuration
            
        Returns:
            SeedManager instance
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        
        return cls._instance
    
    def __init__(self, config: Optional[SeedConfig] = None):
        """
        Initialize SeedManager.
        
        Args:
            config: Seed configuration
        """
        if self._initialized:
            return
        
        self.config = config if config is not None else SeedConfig()
        self.current_seed = self.config.seed
        self.seed_states: Dict[str, SeedState] = {}
        
        # Set seeds
        self._set_all_seeds()
        
        # Save initial seed state
        if self.config.save_seed_state:
            self.save_seed_state('initial')
        
        self._initialized = True
        
        logger.info(f"Initialized SeedManager with seed={self.config.seed}")
    
    def _set_all_seeds(self):
        """Set seeds for all random number generators."""
        # Set Python random seed
        random.seed(self.current_seed)
        
        # Set NumPy random seed
        np.random.seed(self.current_seed)
        
        # Set PyTorch random seed
        torch.manual_seed(self.current_seed)
        
        # Set CUDA random seed if available
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.current_seed)
            torch.cuda.manual_seed_all(self.current_seed)
        
        # Configure deterministic algorithms
        if self.config.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = self.config.benchmark
            
            # Set environment variables for determinism
            os.environ['PYTHONHASHSEED'] = str(self.current_seed)
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        
        logger.debug(f"Set all random seeds to {self.current_seed}")
    
    def set_seed(self, seed: int, component: str = 'default'):
        """
        Set random seed for a specific component.
        
        Args:
            seed: Seed value
            component: Component name (e.g., 'training', 'evaluation')
        """
        self.current_seed = seed
        self._set_all_seeds()
        
        # Save seed state for this component
        if self.config.save_seed_state:
            self.save_seed_state(component)
        
        logger.info(f"Set seed={seed} for component={component}")
    
    def get_seed_state(self, component: str = 'default') -> SeedState:
        """
        Get current seed state for a component.
        
        Args:
            component: Component name
            
        Returns:
            SeedState object
        """
        state = SeedState(
            seed=self.current_seed,
            numpy_state=np.random.get_state(),
            torch_state=torch.get_rng_state(),
            python_state=random.getstate()
        )
        
        # Get CUDA state if available
        if torch.cuda.is_available():
            state.cuda_state = torch.cuda.get_rng_state_all()
        
        # Compute config hash
        state.compute_config_hash(self.config)
        
        # Store state
        self.seed_states[component] = state
        
        return state
    
    def restore_seed_state(self, state: SeedState):
        """
        Restore seed state from SeedState object.
        
        Args:
            state: SeedState to restore
        """
        self.current_seed = state.seed
        
        # Restore NumPy state
        if state.numpy_state is not None:
            np.random.set_state(state.numpy_state)
        
        # Restore PyTorch state
        if state.torch_state is not None:
            torch.set_rng_state(state.torch_state)
        
        # Restore Python state
        if state.python_state is not None:
            random.setstate(state.python_state)
        
        # Restore CUDA state if available
        if state.cuda_state is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state.cuda_state)
        
        logger.info(f"Restored seed state for seed={state.seed}")
    
    def save_seed_state(self, component: str = 'default', filepath: Optional[str] = None):
        """
        Save seed state to file.
        
        Args:
            component: Component name
            filepath: Path to save state (uses config.seed_file if None)
        """
        state = self.get_seed_state(component)
        
        if filepath is None:
            filepath = Path(self.config.seed_file)
        else:
            filepath = Path(filepath)
        
        # Create directory if needed
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save state
        with open(filepath, 'w') as f:
            json.dump(state.to_dict(), f, indent=2)
        
        logger.info(f"Saved seed state to {filepath}")
    
    def load_seed_state(self, filepath: Optional[str] = None) -> Optional[SeedState]:
        """
        Load seed state from file.
        
        Args:
            filepath: Path to load state from
            
        Returns:
            SeedState object or None if file doesn't exist
        """
        if filepath is None:
            filepath = Path(self.config.seed_file)
        else:
            filepath = Path(filepath)
        
        if not filepath.exists():
            logger.warning(f"Seed state file not found: {filepath}")
            return None
        
        with open(filepath, 'r') as f:
            state_data = json.load(f)
        
        state = SeedState.from_dict(state_data)
        
        # Verify config hash
        state.compute_config_hash(self.config)
        if state.config_hash != state_data.get('config_hash', ''):
            logger.warning("Seed state config hash mismatch")
        
        # Restore state
        self.restore_seed_state(state)
        
        logger.info(f"Loaded seed state from {filepath}")
        
        return state
    
    def verify_reproducibility(self, num_tests: int = 3) -> bool:
        """
        Verify reproducibility by running multiple tests.
        
        NOTE: This method does NOT generate random numbers.
        It only verifies that seed states can be saved and restored.
        
        Args:
            num_tests: Number of verification tests
            
        Returns:
            True if reproducible
        """
        if not self.config.verify_reproducibility:
            return True
        
        all_passed = True
        
        for i in range(num_tests):
            # Save state
            state1 = self.get_seed_state(f'test_{i}')
            
            # Set different seed
            self.set_seed(self.current_seed + i + 1, f'test_{i}_temp')
            
            # Restore original state
            self.restore_seed_state(state1)
            state2 = self.get_seed_state(f'test_{i}_restored')
            
            # Verify states match
            if state1.seed != state2.seed:
                logger.error(f"Reproducibility test {i} failed: seed mismatch")
                all_passed = False
        
        if all_passed:
            logger.info(f"Passed {num_tests} reproducibility tests")
        else:
            logger.error("Reproducibility tests failed")
        
        return all_passed
    
    def get_current_seed(self) -> int:
        """
        Get current seed value.
        
        Returns:
            Current seed
        """
        return self.current_seed
    
    def reset_to_initial(self):
        """Reset to initial seed configuration."""
        self.set_seed(self.config.seed, 'initial')
        logger.info("Reset to initial seed configuration")


def set_all_seeds(seed: int = 42, deterministic: bool = True) -> SeedManager:
    """
    Convenience function to set all random seeds.
    
    Args:
        seed: Random seed value
        deterministic: Whether to use deterministic algorithms
        
    Returns:
        SeedManager instance
    """
    config = SeedConfig(seed=seed, deterministic=deterministic)
    return SeedManager(config)


def get_seed_state(component: str = 'default') -> SeedState:
    """
    Get current seed state.
    
    Args:
        component: Component name
        
    Returns:
        SeedState object
    """
    manager = SeedManager()
    return manager.get_seed_state(component)


def restore_seed_state(state: SeedState):
    """
    Restore seed state.
    
    Args:
        state: SeedState to restore
    """
    manager = SeedManager()
    manager.restore_seed_state(state)