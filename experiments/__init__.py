#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiments Module for NTK-SURGERY
Implements Complete Experimental Framework for Manuscript Evaluation

This module provides:
- Main experiment runner for NTK-SURGERY evaluation
- Ablation study framework for component analysis
- Domain generalization experiments across 5 datasets
- Hyperparameter search and sensitivity analysis
- Result aggregation and export utilities

All experiments align with Section 5 of the manuscript.
"""

from experiments.run_main import MainExperiment, ExperimentConfig, ExperimentRunner
from experiments.run_ablation import AblationStudy, AblationConfig, AblationAnalyzer
from experiments.run_domain_generalization import (
    DomainGeneralizationExperiment,
    DomainConfig,
    DomainAnalyzer
)
from experiments.run_hyperparameter_search import (
    HyperparameterSearch,
    SearchConfig,
    SearchResults
)

__version__ = '1.0.0'
__author__ = 'NTK-SURGERY Team'
__all__ = [
    'MainExperiment',
    'ExperimentConfig',
    'ExperimentRunner',
    'AblationStudy',
    'AblationConfig',
    'AblationAnalyzer',
    'DomainGeneralizationExperiment',
    'DomainConfig',
    'DomainAnalyzer',
    'HyperparameterSearch',
    'SearchConfig',
    'SearchResults'
]