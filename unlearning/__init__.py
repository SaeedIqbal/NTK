#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unlearning Module for NTK-SURGERY
Implements Core Unlearning Functionality for Federated Learning

This module provides:
- Client unlearning via NTK-SURGERY methodology (Sections 4.1-4.4)
- Unlearning evaluation framework with comprehensive metrics
- Support for baseline unlearning methods comparison
- Checkpoint management and rollback capabilities

All implementations align with Sections 4.1-4.4 of the manuscript.
"""

from unlearning.unlearn_client import (
    UnlearnClient,
    UnlearningPipeline,
    UnlearningConfig,
    UnlearningResult
)
from unlearning.unlearn_evaluator import (
    UnlearningEvaluator,
    EvaluationConfig,
    EvaluationReport,
    ComparativeAnalysis
)

__version__ = '1.0.0'
__author__ = 'NTK-SURGERY Team'
__all__ = [
    'UnlearnClient',
    'UnlearningPipeline',
    'UnlearningConfig',
    'UnlearningResult',
    'UnlearningEvaluator',
    'EvaluationConfig',
    'EvaluationReport',
    'ComparativeAnalysis'
]