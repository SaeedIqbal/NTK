#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualization Module for NTK-SURGERY
Implements Publication-Quality Plots for Manuscript Figures

This module provides:
- Unlearning efficacy visualization (Forget/Retain Accuracy, Exactness Score)
- Efficiency metrics visualization (Communication Rounds, Server Time)
- Theoretical compliance visualization (NTK Alignment, Sensitivity Bounds)
- Ablation study visualization (Component Contributions)

All plots follow ACCV publication standards with serif fonts and white backgrounds.
"""

from visualization.plot_efficacy import (
    EfficacyPlotter,
    EfficacyConfig,
    plot_unlearning_efficacy,
    create_efficacy_summary
)
from visualization.plot_efficiency import (
    EfficiencyPlotter,
    EfficiencyConfig,
    plot_efficiency_metrics,
    create_efficiency_comparison
)
from visualization.plot_theoretical import (
    TheoreticalPlotter,
    TheoreticalConfig,
    plot_theoretical_metrics,
    create_theoretical_analysis
)
from visualization.plot_ablation import (
    AblationPlotter,
    AblationConfig,
    plot_ablation_study,
    create_ablation_summary
)

__version__ = '1.0.0'
__author__ = 'NTK-SURGERY Team'
__all__ = [
    'EfficacyPlotter',
    'EfficacyConfig',
    'plot_unlearning_efficacy',
    'create_efficacy_summary',
    'EfficiencyPlotter',
    'EfficiencyConfig',
    'plot_efficiency_metrics',
    'create_efficiency_comparison',
    'TheoreticalPlotter',
    'TheoreticalConfig',
    'plot_theoretical_metrics',
    'create_theoretical_analysis',
    'AblationPlotter',
    'AblationConfig',
    'plot_ablation_study',
    'create_ablation_summary'
]