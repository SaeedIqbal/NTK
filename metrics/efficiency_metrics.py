#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Efficiency Metrics for NTK-SURGERY
Implements efficiency metrics from Section 5.2 of the manuscript

Metrics:
- Communication Rounds
- Server Compute Time
- FLOPs / Computational Complexity
- Speedup Factor
"""

import numpy as np
import time
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Callable
import logging
from dataclasses import dataclass
from functools import wraps
import psutil
import os

logger = logging.getLogger(__name__)


@dataclass
class EfficiencyResult:
    """Data class for efficiency evaluation results."""
    communication_rounds: int
    server_time: float
    client_time: float
    total_time: float
    flops: int
    memory_usage_mb: float
    speedup_vs_scratch: float
    speedup_vs_sifu: float
    complexity_class: str


class EfficiencyMetrics:
    """
    Computes efficiency metrics for NTK-SURGERY evaluation.
    
    Implements metrics from Section 5.2:
    1. Communication Rounds (lower is better)
    2. Server Compute Time (lower is better)
    3. FLOPs / Computational Complexity
    4. Speedup Factor vs baselines
    
    Attributes:
        baseline_times (dict): Baseline method times for comparison
    """
    
    def __init__(self):
        """Initialize EfficiencyMetrics."""
        self.baseline_times = {
            'Scratch': 1768.8,  # Average from experiments
            'SIFU': 863.1,
            'FedEraser': 1294.1,
            'Fine-Tuning': 1092.8
        }
        
        self.complexity_classes = {
            'NTK-SURGERY': 'O(M²)',
            'SIFU': 'O(T · K · P)',
            'FedEraser': 'O(epochs · M · K · P)',
            'Scratch': 'O(N · K · P)',
            'Fine-Tuning': 'O(epochs · M · K · P)'
        }
        
        logger.info("Initialized EfficiencyMetrics")
    
    def measure_execution_time(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> Tuple[float, any]:
        """
        Measure execution time of a function.
        
        Args:
            func: Function to measure
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Tuple of (execution_time, function_result)
        """
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        
        execution_time = end_time - start_time
        
        logger.debug(f"Execution time: {execution_time:.4f}s")
        
        return execution_time, result
    
    def compute_communication_rounds(
        self,
        method_name: str,
        custom_rounds: Optional[int] = None
    ) -> int:
        """
        Compute communication rounds for a method.
        
        NTK-SURGERY: 1 round (server-side only)
        SIFU: Ñ rounds (rollback + retraining)
        Scratch: N rounds (full training)
        
        Args:
            method_name: Name of unlearning method
            custom_rounds: Custom round count (optional)
            
        Returns:
            Number of communication rounds
        """
        if custom_rounds is not None:
            return custom_rounds
        
        round_mapping = {
            'NTK-SURGERY': 1,
            'SIFU': 50,  # Typical Ñ from experiments
            'FedEraser': 100,
            'Scratch': 50,
            'Fine-Tuning': 50,
            'FedSGD': 100,
            'BFU': 30,
            'Forget-SVGD': 40,
            'Knowledge Distillation': 50,
            'FU': 50,
            'F2L2': 50
        }
        
        rounds = round_mapping.get(method_name, 50)
        
        logger.info(f"{method_name}: {rounds} communication rounds")
        
        return rounds
    
    def compute_flops(
        self,
        model: nn.Module,
        N: int,
        method_name: str,
        n_c: Optional[int] = None
    ) -> int:
        """
        Compute FLOPs for unlearning operation.
        
        NTK-SURGERY: O(N² n_c) for Woodbury update
        SIFU: O(Ñ · M · K · P · B_batch)
        Scratch: O(N · K · P)
        
        Args:
            model: Neural network model
            N: Total number of samples
            method_name: Unlearning method name
            n_c: Client data size (for NTK-SURGERY)
            
        Returns:
            Estimated FLOPs
        """
        P = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        if method_name == 'NTK-SURGERY':
            # Jacobian computation: O(PN)
            # Woodbury update: O(N² n_c)
            if n_c is None:
                n_c = N // 100  # Assume 100 clients
            
            flops = P * N + (N ** 2) * n_c
            
        elif method_name == 'SIFU':
            # Retrain for Ñ rounds
            N_tilde = 50
            M = 100
            K = 5
            B_batch = 64
            
            flops = N_tilde * M * K * P * B_batch
            
        elif method_name == 'Scratch':
            # Full retraining
            N_rounds = 50
            K = 5
            B_batch = 64
            
            flops = N_rounds * (N // B_batch) * K * P
            
        else:
            # Default estimate
            flops = P * N * 10
        
        logger.info(f"{method_name}: {flops:,} FLOPs")
        
        return int(flops)
    
    def compute_speedup(
        self,
        ntk_time: float,
        baseline_name: str
    ) -> float:
        """
        Compute speedup factor vs baseline.
        
        Args:
            ntk_time: NTK-SURGERY execution time
            baseline_name: Baseline method name
            
        Returns:
            Speedup factor
        """
        baseline_time = self.baseline_times.get(baseline_name, ntk_time)
        
        if ntk_time <= 0:
            return float('inf')
        
        speedup = baseline_time / ntk_time
        
        logger.info(f"Speedup vs {baseline_name}: {speedup:.1f}×")
        
        return float(speedup)
    
    def get_memory_usage(self) -> float:
        """
        Get current memory usage in MB.
        
        Returns:
            Memory usage in MB
        """
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        return float(memory_mb)
    
    def compute_all_efficiency_metrics(
        self,
        method_name: str,
        model: nn.Module,
        N: int,
        server_time: float,
        n_c: Optional[int] = None
    ) -> EfficiencyResult:
        """
        Compute all efficiency metrics for a method.
        
        Args:
            method_name: Unlearning method name
            model: Neural network model
            N: Total samples
            server_time: Server compute time
            n_c: Client data size
            
        Returns:
            EfficiencyResult dataclass
        """
        comm_rounds = self.compute_communication_rounds(method_name)
        flops = self.compute_flops(model, N, method_name, n_c)
        memory_mb = self.get_memory_usage()
        
        speedup_scratch = self.compute_speedup(server_time, 'Scratch')
        speedup_sifu = self.compute_speedup(server_time, 'SIFU')
        
        complexity = self.complexity_classes.get(
            method_name, 'O(unknown)'
        )
        
        result = EfficiencyResult(
            communication_rounds=comm_rounds,
            server_time=server_time,
            client_time=0.0,  # NTK-SURGERY is server-side only
            total_time=server_time,
            flops=flops,
            memory_usage_mb=memory_mb,
            speedup_vs_scratch=speedup_scratch,
            speedup_vs_sifu=speedup_sifu,
            complexity_class=complexity
        )
        
        logger.info(
            f"Efficiency metrics: rounds={comm_rounds}, "
            f"time={server_time:.2f}s, speedup_scratch={speedup_scratch:.1f}×"
        )
        
        return result


class ComplexityAnalyzer:
    """
    Analyzes computational complexity of unlearning methods.
    """
    
    @staticmethod
    def analyze_ntk_surgery_complexity(
        N: int,
        n_c: int,
        P: int
    ) -> Dict[str, int]:
        """
        Analyze NTK-SURGERY computational complexity.
        
        Breaks down complexity by operation:
        - Jacobian computation: O(PN)
        - Woodbury update: O(N² n_c)
        - Client kernel inversion: O(n_c³)
        
        Args:
            N: Total samples
            n_c: Client samples
            P: Parameters
            
        Returns:
            Dictionary with complexity breakdown
        """
        jacobian_flops = P * N
        woodbury_flops = (N ** 2) * n_c
        inversion_flops = n_c ** 3
        
        total_flops = jacobian_flops + woodbury_flops + inversion_flops
        
        analysis = {
            'jacobian_flops': jacobian_flops,
            'woodbury_flops': woodbury_flops,
            'inversion_flops': inversion_flops,
            'total_flops': total_flops,
            'dominant_operation': 'Woodbury' if woodbury_flops > jacobian_flops else 'Jacobian',
            'complexity_class': 'O(N² n_c)',
            'independent_of_training_rounds': True
        }
        
        logger.info(f"NTK-SURGERY complexity analysis: {analysis}")
        
        return analysis
    
    @staticmethod
    def analyze_sifu_complexity(
        N_tilde: int,
        M: int,
        K: int,
        P: int,
        B_batch: int
    ) -> Dict[str, int]:
        """
        Analyze SIFU computational complexity.
        
        Args:
            N_tilde: Rollback rounds
            M: Clients
            K: Local steps
            P: Parameters
            B_batch: Batch size
            
        Returns:
            Dictionary with complexity breakdown
        """
        total_flops = N_tilde * M * K * P * B_batch
        
        analysis = {
            'total_flops': total_flops,
            'complexity_class': 'O(Ñ · M · K · P · B)',
            'depends_on_training_rounds': True,
            'exponential_in_nonconvex': True,
            'parameters': {
                'N_tilde': N_tilde,
                'M': M,
                'K': K,
                'P': P,
                'B': B_batch
            }
        }
        
        logger.info(f"SIFU complexity analysis: {total_flops:,} FLOPs")
        
        return analysis
    
    @staticmethod
    def compare_complexities(
        ntk_flops: int,
        sifu_flops: int
    ) -> Dict[str, float]:
        """
        Compare NTK-SURGERY and SIFU complexities.
        
        Args:
            ntk_flops: NTK-SURGERY FLOPs
            sifu_flops: SIFU FLOPs
            
        Returns:
            Comparison metrics
        """
        speedup = sifu_flops / (ntk_flops + 1e-8)
        reduction = 1.0 - (ntk_flops / (sifu_flops + 1e-8))
        
        comparison = {
            'ntk_flops': ntk_flops,
            'sifu_flops': sifu_flops,
            'speedup_factor': speedup,
            'flop_reduction': reduction,
            'ntk_more_efficient': ntk_flops < sifu_flops
        }
        
        logger.info(
            f"Complexity comparison: {speedup:.1f}× speedup, "
            f"{reduction*100:.1f}% reduction"
        )
        
        return comparison


class PerformanceTracker:
    """
    Tracks performance metrics across multiple experiments.
    """
    
    def __init__(self):
        """Initialize PerformanceTracker."""
        self.experiments = []
        self.baseline_metrics = {}
        
        logger.info("Initialized PerformanceTracker")
    
    def record_experiment(
        self,
        method_name: str,
        dataset_name: str,
        server_time: float,
        comm_rounds: int,
        flops: int,
        memory_mb: float
    ):
        """
        Record experiment performance metrics.
        
        Args:
            method_name: Unlearning method
            dataset_name: Dataset name
            server_time: Server compute time
            comm_rounds: Communication rounds
            flops: FLOPs
            memory_mb: Memory usage
        """
        experiment = {
            'method': method_name,
            'dataset': dataset_name,
            'server_time': server_time,
            'comm_rounds': comm_rounds,
            'flops': flops,
            'memory_mb': memory_mb,
            'timestamp': time.time()
        }
        
        self.experiments.append(experiment)
        
        logger.debug(
            f"Recorded experiment: {method_name} on {dataset_name}, "
            f"time={server_time:.2f}s"
        )
    
    def set_baseline(
        self,
        method_name: str,
        server_time: float,
        comm_rounds: int
    ):
        """
        Set baseline metrics for comparison.
        
        Args:
            method_name: Baseline method name
            server_time: Baseline server time
            comm_rounds: Baseline communication rounds
        """
        self.baseline_metrics[method_name] = {
            'server_time': server_time,
            'comm_rounds': comm_rounds
        }
        
        logger.info(f"Set baseline for {method_name}: {server_time:.2f}s")
    
    def compute_relative_performance(
        self,
        method_name: str,
        baseline_name: str = 'Scratch'
    ) -> Dict[str, float]:
        """
        Compute relative performance vs baseline.
        
        Args:
            method_name: Method to evaluate
            baseline_name: Baseline method
            
        Returns:
            Relative performance metrics
        """
        method_experiments = [
            e for e in self.experiments if e['method'] == method_name
        ]
        baseline_experiments = [
            e for e in self.experiments if e['method'] == baseline_name
        ]
        
        if not method_experiments or not baseline_experiments:
            return {}
        
        method_time = np.mean([e['server_time'] for e in method_experiments])
        baseline_time = np.mean([e['server_time'] for e in baseline_experiments])
        
        method_rounds = np.mean([e['comm_rounds'] for e in method_experiments])
        baseline_rounds = np.mean([e['comm_rounds'] for e in baseline_experiments])
        
        relative = {
            'time_speedup': baseline_time / (method_time + 1e-8),
            'rounds_reduction': 1.0 - (method_rounds / (baseline_rounds + 1e-8)),
            'method_avg_time': method_time,
            'baseline_avg_time': baseline_time,
            'method_avg_rounds': method_rounds,
            'baseline_avg_rounds': baseline_rounds
        }
        
        logger.info(
            f"Relative performance: {method_name} vs {baseline_name}, "
            f"speedup={relative['time_speedup']:.1f}×"
        )
        
        return relative
    
    def get_aggregate_statistics(self) -> Dict[str, any]:
        """
        Get aggregate statistics across all experiments.
        
        Returns:
            Dictionary with aggregate metrics
        """
        if not self.experiments:
            return {}
        
        by_method = {}
        
        for exp in self.experiments:
            method = exp['method']
            
            if method not in by_method:
                by_method[method] = {
                    'times': [],
                    'rounds': [],
                    'flops': [],
                    'memory': []
                }
            
            by_method[method]['times'].append(exp['server_time'])
            by_method[method]['rounds'].append(exp['comm_rounds'])
            by_method[method]['flops'].append(exp['flops'])
            by_method[method]['memory'].append(exp['memory_mb'])
        
        stats = {}
        
        for method, data in by_method.items():
            stats[method] = {
                'num_experiments': len(data['times']),
                'avg_time': float(np.mean(data['times'])),
                'std_time': float(np.std(data['times'])),
                'avg_rounds': float(np.mean(data['rounds'])),
                'avg_flops': float(np.mean(data['flops'])),
                'avg_memory': float(np.mean(data['memory']))
            }
        
        return stats