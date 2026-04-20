#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NTK-SURGERY Core Module
Implements the proposed methodology for Federated Unlearning via Neural Tangent Kernel Surgery

This module implements Sections 4.1-4.4 of the manuscript:
- Section 4.1: The Federated NTK Representation
- Section 4.2: The Influence Matrix
- Section 4.3: The Surgery Operator
- Section 4.4: Finite-Width Projection
"""

from ntk_surgery.federated_ntk import FederatedNTK, NTKKernelComputer
from ntk_surgery.influence_matrix import InfluenceMatrix, ResolventMatrix
from ntk_surgery.surgery_operator import SurgeryOperator, WoodburyUpdater
from ntk_surgery.finite_width_projection import FiniteWidthProjector, JacobianComputer

__version__ = '1.0.0'
__author__ = 'NTK-SURGERY Team'
__all__ = [
    'FederatedNTK',
    'NTKKernelComputer',
    'InfluenceMatrix',
    'ResolventMatrix',
    'SurgeryOperator',
    'WoodburyUpdater',
    'FiniteWidthProjector',
    'JacobianComputer'
]