#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Metrics Module for NTK-SURGERY
Provides comprehensive evaluation metrics for federated unlearning

This module implements three categories of metrics:
1. Unlearning Efficacy Metrics (Forget Accuracy, Retain Accuracy, Exactness Score)
2. Efficiency Metrics (Communication Rounds, Server Time, FLOPs)
3. Theoretical Metrics (NTK Alignment, Sensitivity Bound Ratio)

All metrics align with Section 5.2 of the manuscript.
"""

from metrics.unlearning_metrics import (
    UnlearningMetrics,
    UnlearningEvaluator,
    MetricValidator
)
from metrics.efficiency_metrics import (
    EfficiencyMetrics,
    ComplexityAnalyzer,
    PerformanceTracker
)
from metrics.theoretical_metrics import (
    TheoreticalMetrics,
    NTKAnalyzer,
    SensitivityAnalyzer
)

__version__ = '1.0.0'
__author__ = 'NTK-SURGERY Team'
__all__ = [
    'UnlearningMetrics',
    'UnlearningEvaluator',
    'MetricValidator',
    'EfficiencyMetrics',
    'ComplexityAnalyzer',
    'PerformanceTracker',
    'TheoreticalMetrics',
    'NTKAnalyzer',
    'SensitivityAnalyzer'
]