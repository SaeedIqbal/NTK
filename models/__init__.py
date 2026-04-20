#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Models Module for NTK-SURGERY
Provides neural network architectures and NTK-specific model utilities
"""

from models.ntk_model import NTKModel, NTKUtilities
from models.cnn import CNN, CNNConfig
from models.mlp import MLP, MLPConfig
from models.resnet import ResNet, ResNetConfig, BasicBlock, Bottleneck

__all__ = [
    'NTKModel',
    'NTKUtilities',
    'CNN',
    'CNNConfig',
    'MLP',
    'MLPConfig',
    'ResNet',
    'ResNetConfig',
    'BasicBlock',
    'Bottleneck'
]

__version__ = '1.0.0'
__author__ = 'NTK-SURGERY Team'