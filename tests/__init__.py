#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests Module for NTK-SURGERY
Implements Comprehensive Test Suite for All Components

This module provides:
- Unit tests for NTK-SURGERY core components
- Integration tests for complete pipelines
- Baseline method validation tests
- Metrics computation tests
- Data loading and preprocessing tests

All tests are deterministic (no random operations) and follow
scientific computing best practices for reproducibility.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

__version__ = '1.0.0'
__author__ = 'NTK-SURGERY Team'
__all__ = [
    'test_ntk_surgery',
    'test_baselines',
    'test_metrics',
    'test_data_loader'
]