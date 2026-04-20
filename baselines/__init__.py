#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Baselines Module for NTK-SURGERY
Implements State-of-the-Art Federated Unlearning Methods for Comparison

This module provides implementations of:
- SIFU: Sequential Informed Federated Unlearning
- FedEraser: Federated Unlearning via Gradient Erasure
- Fine-Tuning: Simple fine-tuning on remaining data
- FedSGD: Federated SGD with unlearning
- BFU: Bayesian Federated Unlearning
- Forget-SVGD: Forgettable Steined Variational Gradient Descent
- Knowledge Distillation: Unlearning via knowledge transfer
- FU: Federated Unlearning (generic)
- F²L²: Forgettable Federated Linear Learning
"""

from baselines.sifu import SIFU, SIFUConfig
from baselines.federaser import FedEraser, FedEraserConfig
from baselines.fine_tuning import FineTuning, FineTuningConfig
from baselines.fedsgd import FedSGD, FedSGDConfig
from baselines.bfu import BFU, BFUConfig
from baselines.forget_svgd import ForgetSVGD, ForgetSVGDConfig
from baselines.knowledge_distillation import KnowledgeDistillation, KDConfig
from baselines.fu import FU, FUConfig
from baselines.f2l2 import F2L2, F2L2Config

__version__ = '1.0.0'
__author__ = 'NTK-SURGERY Team'
__all__ = [
    'SIFU', 'SIFUConfig',
    'FedEraser', 'FedEraserConfig',
    'FineTuning', 'FineTuningConfig',
    'FedSGD', 'FedSGDConfig',
    'BFU', 'BFUConfig',
    'ForgetSVGD', 'ForgetSVGDConfig',
    'KnowledgeDistillation', 'KDConfig',
    'FU', 'FUConfig',
    'F2L2', 'F2L2Config'
]