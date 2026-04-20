#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training Module for NTK-SURGERY
Implements Federated Learning Training Infrastructure

This module provides:
- Federated Averaging (FedAvg) implementation
- Client-side local training with SGD
- Server-side aggregation with weighted averaging
- Checkpoint management for unlearning baselines

All implementations align with Section 3.1 (Preliminaries) of the manuscript.
"""

from training.fedavg import FedAvg, FedAvgConfig, FedAvgTrainer
from training.local_training import LocalTrainer, LocalTrainingConfig, ClientModel
from training.server_aggregation import ServerAggregator, AggregationStrategy, ServerState

__version__ = '1.0.0'
__author__ = 'NTK-SURGERY Team'
__all__ = [
    'FedAvg',
    'FedAvgConfig',
    'FedAvgTrainer',
    'LocalTrainer',
    'LocalTrainingConfig',
    'ClientModel',
    'ServerAggregator',
    'AggregationStrategy',
    'ServerState'
]