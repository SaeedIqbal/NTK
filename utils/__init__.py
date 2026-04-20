#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities Module for NTK-SURGERY
Provides Core Utility Functions for the Entire Project

This module provides:
- Logging configuration and management
- Checkpoint saving and loading for models
- Random seed management for reproducibility
- Memory usage tracking and monitoring
- Common helper functions used across the project

All utilities follow best practices for scientific computing and are
designed for production use with comprehensive error handling.
"""

from utils.logger import (
    Logger,
    LoggerConfig,
    setup_logger,
    get_logger,
    LogFormatter
)
from utils.checkpoint import (
    CheckpointManager,
    CheckpointConfig,
    save_checkpoint,
    load_checkpoint,
    CheckpointState
)
from utils.random_seed import (
    SeedManager,
    SeedConfig,
    set_all_seeds,
    get_seed_state,
    restore_seed_state
)
from utils.memory_tracker import (
    MemoryTracker,
    MemoryConfig,
    track_memory,
    get_memory_usage,
    MemoryReport
)

__version__ = '1.0.0'
__author__ = 'NTK-SURGERY Team'
__all__ = [
    'Logger',
    'LoggerConfig',
    'setup_logger',
    'get_logger',
    'LogFormatter',
    'CheckpointManager',
    'CheckpointConfig',
    'save_checkpoint',
    'load_checkpoint',
    'CheckpointState',
    'SeedManager',
    'SeedConfig',
    'set_all_seeds',
    'get_seed_state',
    'restore_seed_state',
    'MemoryTracker',
    'MemoryConfig',
    'track_memory',
    'get_memory_usage',
    'MemoryReport'
]