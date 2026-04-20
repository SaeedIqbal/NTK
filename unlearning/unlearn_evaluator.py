#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unlearning Evaluator for NTK-SURGERY
Implements Comprehensive Evaluation Framework for Unlearning Methods

This module provides:
- Multi-metric evaluation (efficacy, efficiency, theoretical)
- Baseline method comparison
- Statistical analysis and reporting
- Visualization-ready data export

All metrics align with Section 5.2 of the manuscript.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
import json
import copy

from metrics.unlearning_metrics import UnlearningMetrics, UnlearningEvaluator
from metrics.efficiency_metrics import EfficiencyMetrics, EfficiencyResult
from metrics.theoretical_metrics import TheoreticalMetrics, TheoreticalResult
from baselines.sifu import SIFU, SIFUConfig
from baselines.federaser import FedEraser, FedEraserConfig
from baselines.fine_tuning import FineTuning, FineTuningConfig

logger = logging.getLogger(__name__)


@dataclass
class EvaluationConfig:
    """
    Configuration for unlearning evaluation.
    
    Attributes:
        num_classes: Number of classification classes
        device: Computing device
        batch_size: Evaluation batch size
        evaluate_efficacy: Evaluate unlearning efficacy metrics
        evaluate_efficiency: Evaluate efficiency metrics
        evaluate_theoretical: Evaluate theoretical metrics
        compare_baselines: Compare with baseline methods
        baseline_methods: List of baseline methods to compare
        save_results: Save evaluation results
        results_dir: Directory for saving results
    """
    num_classes: int = 10
    device: str = 'cpu'
    batch_size: int = 64
    evaluate_efficacy: bool = True
    evaluate_efficiency: bool = True
    evaluate_theoretical: bool = True
    compare_baselines: bool = True
    baseline_methods: List[str] = field(default_factory=lambda: [
        'Scratch', 'SIFU', 'FedEraser', 'Fine-Tuning'
    ])
    save_results: bool = True
    results_dir: str = 'evaluation_results'
    
    def __post_init__(self):
        """Validate configuration."""
        if self.num_classes <= 0:
            raise ValueError("num_classes must be positive")


@dataclass
class EvaluationReport:
    """
    Comprehensive evaluation report.
    
    Attributes:
        method_name: Evaluated method name
        dataset_name: Dataset name
        efficacy_metrics: Efficacy evaluation results
        efficiency_metrics: Efficiency evaluation results
        theoretical_metrics: Theoretical evaluation results
        overall_score: Combined overall score
        ranking: Method ranking
        timestamp: Evaluation timestamp
        metadata: Additional metadata
    """
    method_name: str
    dataset_name: str
    efficacy_metrics: Optional[Dict] = None
    efficiency_metrics: Optional[Dict] = None
    theoretical_metrics: Optional[Dict] = None
    overall_score: float = 0.0
    ranking: int = 0
    timestamp: float = field(default_factory=time.time)
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'method_name': self.method_name,
            'dataset_name': self.dataset_name,
            'efficacy_metrics': self.efficacy_metrics,
            'efficiency_metrics': self.efficiency_metrics,
            'theoretical_metrics': self.theoretical_metrics,
            'overall_score': self.overall_score,
            'ranking': self.ranking,
            'timestamp': self.timestamp,
            'metadata': self.metadata
        }
    
    def to_json(self, filepath: str):
        """Save to JSON file."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        
        logger.info(f"Saved evaluation report to {filepath}")


class ComparativeAnalysis:
    """
    Comparative analysis across multiple unlearning methods.
    
    Provides statistical comparison and ranking of methods.
    """
    
    def __init__(self):
        """Initialize ComparativeAnalysis."""
        self.method_reports = {}
        self.ranking_results = {}
        
        logger.info("Initialized ComparativeAnalysis")
    
    def add_report(
        self,
        method_name: str,
        report: EvaluationReport
    ):
        """
        Add evaluation report for a method.
        
        Args:
            method_name: Method name
            report: Evaluation report
        """
        self.method_reports[method_name] = report
        logger.info(f"Added report for method: {method_name}")
    
    def compute_rankings(
        self,
        weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, any]:
        """
        Compute method rankings based on overall scores.
        
        Args:
            weights: Metric weights for scoring
            
        Returns:
            Ranking results
        """
        if weights is None:
            weights = {
                'exactness_score': 0.4,
                'forget_accuracy': 0.2,
                'retain_accuracy': 0.2,
                'efficiency': 0.1,
                'theoretical': 0.1
            }
        
        scores = {}
        
        for method_name, report in self.method_reports.items():
            score = 0.0
            
            # Efficacy score
            if report.efficacy_metrics:
                es = report.efficacy_metrics.get('exactness_score', 0)
                fa = report.efficacy_metrics.get('forget_accuracy', 1)
                ra = report.efficacy_metrics.get('retain_accuracy', 0)
                
                # Lower FA is better (convert to score)
                fa_score = 1.0 - min(fa / 0.5, 1.0)
                
                efficacy_score = (
                    weights['exactness_score'] * es +
                    weights['forget_accuracy'] * fa_score +
                    weights['retain_accuracy'] * ra
                )
                score += efficacy_score
            
            # Efficiency score
            if report.efficiency_metrics:
                # Higher speedup is better
                speedup = report.efficiency_metrics.get('speedup_vs_scratch', 1)
                efficiency_score = weights['efficiency'] * min(speedup / 100, 1.0)
                score += efficiency_score
            
            # Theoretical score
            if report.theoretical_metrics:
                alignment = report.theoretical_metrics.get('ntk_alignment', 0)
                sensitivity_ratio = report.theoretical_metrics.get('sensitivity_bound_ratio', 1)
                
                theoretical_score = weights['theoretical'] * (
                    0.5 * alignment + 0.5 * min(sensitivity_ratio / 100, 1.0)
                )
                score += theoretical_score
            
            scores[method_name] = score
        
        # Rank methods
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        self.ranking_results = {
            'scores': scores,
            'ranking': [name for name, _ in ranked],
            'best_method': ranked[0][0] if ranked else None,
            'score_spread': ranked[0][1] - ranked[-1][1] if len(ranked) > 1 else 0
        }
        
        # Update report rankings
        for rank, (method_name, _) in enumerate(ranked, 1):
            if method_name in self.method_reports:
                self.method_reports[method_name].ranking = rank
        
        logger.info(
            f"Rankings computed: Best={self.ranking_results['best_method']}, "
            f"Spread={self.ranking_results['score_spread']:.4f}"
        )
        
        return self.ranking_results
    
    def generate_comparison_table(self) -> str:
        """
        Generate comparison table for all methods.
        
        Returns:
            Formatted table string
        """
        if not self.method_reports:
            return "No reports available"
        
        lines = []
        lines.append("=" * 100)
        lines.append(f"{'Method':<20} {'Rank':<6} {'ES':<8} {'FA':<8} {'RA':<8} {'Time':<10} {'Speedup':<10}")
        lines.append("=" * 100)
        
        for method_name in self.ranking_results.get('ranking', []):
            report = self.method_reports.get(method_name)
            if report:
                es = report.efficacy_metrics.get('exactness_score', 0) if report.efficacy_metrics else 0
                fa = report.efficacy_metrics.get('forget_accuracy', 0) if report.efficacy_metrics else 0
                ra = report.efficacy_metrics.get('retain_accuracy', 0) if report.efficacy_metrics else 0
                time_val = report.efficiency_metrics.get('server_time', 0) if report.efficiency_metrics else 0
                speedup = report.efficiency_metrics.get('speedup_vs_scratch', 1) if report.efficiency_metrics else 1
                
                lines.append(
                    f"{method_name:<20} {report.ranking:<6} {es:<8.3f} {fa:<8.3f} "
                    f"{ra:<8.3f} {time_val:<10.2f} {speedup:<10.1f}"
                )
        
        lines.append("=" * 100)
        
        return "\n".join(lines)
    
    def export_comparison(self, filepath: str):
        """
        Export comparison results to JSON.
        
        Args:
            filepath: Path to save results
        """
        export_data = {
            'rankings': self.ranking_results,
            'reports': {
                name: report.to_dict()
                for name, report in self.method_reports.items()
            },
            'comparison_table': self.generate_comparison_table()
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logger.info(f"Exported comparison to {filepath}")


class UnlearningEvaluator:
    """
    Comprehensive unlearning evaluation orchestrator.
    
    Coordinates evaluation across:
    - Efficacy metrics (FA, RA, ES)
    - Efficiency metrics (time, rounds, FLOPs)
    - Theoretical metrics (alignment, sensitivity)
    - Baseline comparisons
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Optional[EvaluationConfig] = None
    ):
        """
        Initialize UnlearningEvaluator.
        
        Args:
            model: Neural network model
            config: Evaluation configuration
        """
        self.model = model
        self.config = config if config is not None else EvaluationConfig()
        self.device = self.config.device
        
        # Initialize metric evaluators
        self.efficacy_evaluator = UnlearningMetrics(
            num_classes=self.config.num_classes,
            device=self.device
        )
        self.efficiency_evaluator = EfficiencyMetrics()
        self.theoretical_evaluator = TheoreticalMetrics()
        
        # Comparative analysis
        self.comparative_analysis = ComparativeAnalysis()
        
        # Evaluation results storage
        self.evaluation_results = {}
        
        # Create results directory
        if self.config.save_results:
            Path(self.config.results_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info(
            f"Initialized UnlearningEvaluator: classes={self.config.num_classes}, "
            f"baselines={self.config.baseline_methods}"
        )
    
    def evaluate_efficacy(
        self,
        model_unlearned: nn.Module,
        model_scratch: nn.Module,
        X_forget: np.ndarray,
        y_forget: np.ndarray,
        X_retain: np.ndarray,
        y_retain: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate unlearning efficacy metrics.
        
        Args:
            model_unlearned: Unlearned model
            model_scratch: Scratch baseline model
            X_forget: Forget set data
            y_forget: Forget set labels
            X_retain: Retain set data
            y_retain: Retain set labels
            
        Returns:
            Efficacy metrics dictionary
        """
        if not self.config.evaluate_efficacy:
            return {}
        
        logger.info("Evaluating unlearning efficacy")
        
        metrics = self.efficacy_evaluator.compute_all_metrics(
            model_surgery=model_unlearned,
            model_scratch=model_scratch,
            X_forget=X_forget,
            y_forget=y_forget,
            X_retain=X_retain,
            y_retain=y_retain,
            batch_size=self.config.batch_size
        )
        
        logger.info(
            f"Efficacy evaluation: FA={metrics['forget_accuracy']:.4f}, "
            f"RA={metrics['retain_accuracy']:.4f}, ES={metrics['exactness_score']:.4f}"
        )
        
        return metrics
    
    def evaluate_efficiency(
        self,
        method_name: str,
        server_time: float,
        N: int,
        n_c: Optional[int] = None
    ) -> Dict[str, any]:
        """
        Evaluate efficiency metrics.
        
        Args:
            method_name: Unlearning method name
            server_time: Server compute time
            N: Total samples
            n_c: Client data size
            
        Returns:
            Efficiency metrics dictionary
        """
        if not self.config.evaluate_efficiency:
            return {}
        
        logger.info(f"Evaluating efficiency for {method_name}")
        
        result = self.efficiency_evaluator.compute_all_efficiency_metrics(
            method_name=method_name,
            model=self.model,
            N=N,
            server_time=server_time,
            n_c=n_c
        )
        
        metrics = {
            'communication_rounds': result.communication_rounds,
            'server_time': result.server_time,
            'flops': result.flops,
            'speedup_vs_scratch': result.speedup_vs_scratch,
            'speedup_vs_sifu': result.speedup_vs_sifu,
            'complexity_class': result.complexity_class
        }
        
        logger.info(
            f"Efficiency evaluation: rounds={result.communication_rounds}, "
            f"time={result.server_time:.2f}s, speedup={result.speedup_vs_scratch:.1f}×"
        )
        
        return metrics
    
    def evaluate_theoretical(
        self,
        K_global: np.ndarray,
        gradient_norms: List[float],
        n_rounds: int
    ) -> Dict[str, any]:
        """
        Evaluate theoretical compliance metrics.
        
        Args:
            K_global: Global kernel matrix
            gradient_norms: Historical gradient norms
            n_rounds: Number of training rounds
            
        Returns:
            Theoretical metrics dictionary
        """
        if not self.config.evaluate_theoretical:
            return {}
        
        logger.info("Evaluating theoretical compliance")
        
        result = self.theoretical_evaluator.compute_all_theoretical_metrics(
            K_global=K_global,
            gradient_norms=gradient_norms,
            n_rounds=n_rounds
        )
        
        metrics = {
            'ntk_alignment': result.ntk_alignment,
            'sensitivity_bound_ratio': result.sensitivity_bound_ratio,
            'condition_number': result.condition_number,
            'effective_rank': result.effective_rank,
            'spectral_gap': result.spectral_gap,
            'quality': result.quality
        }
        
        logger.info(
            f"Theoretical evaluation: alignment={result.ntk_alignment:.4f}, "
            f"sensitivity_ratio={result.sensitivity_bound_ratio:.2f}"
        )
        
        return metrics
    
    def evaluate_baseline(
        self,
        method_name: str,
        client_id: int,
        client_loaders: Dict[int, torch.utils.data.DataLoader],
        gradient_norms: Optional[List[float]] = None
    ) -> Tuple[nn.Module, float]:
        """
        Evaluate baseline unlearning method.
        
        Args:
            method_name: Baseline method name
            client_id: Client to remove
            client_loaders: Client DataLoaders
            gradient_norms: Gradient norms for SIFU
            
        Returns:
            Tuple of (unlearned_model, unlearning_time)
        """
        logger.info(f"Evaluating baseline: {method_name}")
        
        start_time = time.perf_counter()
        
        # Create baseline unlearner
        if method_name == 'SIFU':
            baseline = SIFU(
                copy.deepcopy(self.model),
                SIFUConfig(device=self.device)
            )
            remaining = {
                cid: loader for cid, loader in client_loaders.items()
                if cid != client_id
            }
            unlearned_model = baseline.unlearn_client(
                client_id, remaining, gradient_norms or []
            )
            
        elif method_name == 'FedEraser':
            baseline = FedEraser(
                copy.deepcopy(self.model),
                FedEraserConfig(device=self.device)
            )
            remaining = {
                cid: loader for cid, loader in client_loaders.items()
                if cid != client_id
            }
            unlearned_model = baseline.unlearn_client(client_id, remaining)
            
        elif method_name == 'Fine-Tuning':
            baseline = FineTuning(
                copy.deepcopy(self.model),
                FineTuningConfig(device=self.device)
            )
            remaining = {
                cid: loader for cid, loader in client_loaders.items()
                if cid != client_id
            }
            unlearned_model = baseline.unlearn_client(client_id, remaining)
            
        elif method_name == 'Scratch':
            # Return original model (Scratch is gold standard)
            unlearned_model = copy.deepcopy(self.model)
            
        else:
            raise ValueError(f"Unknown baseline method: {method_name}")
        
        elapsed_time = time.perf_counter() - start_time
        
        logger.info(f"{method_name} completed in {elapsed_time:.2f}s")
        
        return unlearned_model, elapsed_time
    
    def comprehensive_evaluation(
        self,
        method_name: str,
        dataset_name: str,
        model_unlearned: nn.Module,
        model_scratch: nn.Module,
        X_forget: np.ndarray,
        y_forget: np.ndarray,
        X_retain: np.ndarray,
        y_retain: np.ndarray,
        server_time: float,
        N: int,
        K_global: Optional[np.ndarray] = None,
        gradient_norms: Optional[List[float]] = None,
        n_rounds: int = 50
    ) -> EvaluationReport:
        """
        Perform comprehensive evaluation across all metric categories.
        
        Args:
            method_name: Unlearning method name
            dataset_name: Dataset name
            model_unlearned: Unlearned model
            model_scratch: Scratch baseline model
            X_forget: Forget set data
            y_forget: Forget set labels
            X_retain: Retain set data
            y_retain: Retain set labels
            server_time: Server compute time
            N: Total samples
            K_global: Global kernel matrix
            gradient_norms: Gradient norms
            n_rounds: Training rounds
            
        Returns:
            EvaluationReport dataclass
        """
        logger.info(f"Starting comprehensive evaluation for {method_name} on {dataset_name}")
        
        # Evaluate efficacy
        efficacy_metrics = self.evaluate_efficacy(
            model_unlearned, model_scratch,
            X_forget, y_forget, X_retain, y_retain
        )
        
        # Evaluate efficiency
        efficiency_metrics = self.evaluate_efficiency(
            method_name, server_time, N
        )
        
        # Evaluate theoretical
        theoretical_metrics = {}
        if K_global is not None:
            theoretical_metrics = self.evaluate_theoretical(
                K_global, gradient_norms or [], n_rounds
            )
        
        # Compute overall score
        overall_score = self._compute_overall_score(
            efficacy_metrics, efficiency_metrics, theoretical_metrics
        )
        
        # Create report
        report = EvaluationReport(
            method_name=method_name,
            dataset_name=dataset_name,
            efficacy_metrics=efficacy_metrics,
            efficiency_metrics=efficiency_metrics,
            theoretical_metrics=theoretical_metrics,
            overall_score=overall_score,
            metadata={
                'server_time': server_time,
                'num_samples': N,
                'num_classes': self.config.num_classes
            }
        )
        
        # Store result
        key = f"{method_name}_{dataset_name}"
        self.evaluation_results[key] = report
        
        # Add to comparative analysis
        if self.config.compare_baselines:
            self.comparative_analysis.add_report(method_name, report)
        
        # Save results
        if self.config.save_results:
            filepath = Path(self.config.results_dir) / f"{key}_report.json"
            report.to_json(str(filepath))
        
        logger.info(
            f"Comprehensive evaluation completed: score={overall_score:.4f}"
        )
        
        return report
    
    def _compute_overall_score(
        self,
        efficacy: Dict,
        efficiency: Dict,
        theoretical: Dict
    ) -> float:
        """
        Compute overall evaluation score.
        
        Args:
            efficacy: Efficacy metrics
            efficiency: Efficiency metrics
            theoretical: Theoretical metrics
            
        Returns:
            Overall score in [0, 1]
        """
        score = 0.0
        weight_sum = 0.0
        
        # Efficacy contribution (50%)
        if efficacy:
            es = efficacy.get('exactness_score', 0)
            fa = efficacy.get('forget_accuracy', 1)
            ra = efficacy.get('retain_accuracy', 0)
            
            # Convert FA to score (lower is better)
            fa_score = 1.0 - min(fa / 0.5, 1.0)
            
            efficacy_score = 0.5 * es + 0.25 * fa_score + 0.25 * ra
            score += 0.5 * efficacy_score
            weight_sum += 0.5
        
        # Efficiency contribution (30%)
        if efficiency:
            speedup = efficiency.get('speedup_vs_scratch', 1)
            efficiency_score = min(speedup / 100, 1.0)
            score += 0.3 * efficiency_score
            weight_sum += 0.3
        
        # Theoretical contribution (20%)
        if theoretical:
            alignment = theoretical.get('ntk_alignment', 0)
            sensitivity_ratio = theoretical.get('sensitivity_bound_ratio', 1)
            
            theoretical_score = 0.5 * alignment + 0.5 * min(sensitivity_ratio / 100, 1.0)
            score += 0.2 * theoretical_score
            weight_sum += 0.2
        
        return float(score / weight_sum) if weight_sum > 0 else 0.0
    
    def compare_all_methods(
        self,
        dataset_name: str,
        model_ntk: nn.Module,
        model_scratch: nn.Module,
        X_forget: np.ndarray,
        y_forget: np.ndarray,
        X_retain: np.ndarray,
        y_retain: np.ndarray,
        client_loaders: Dict,
        client_id: int,
        N: int,
        K_global: Optional[np.ndarray] = None,
        gradient_norms: Optional[List[float]] = None
    ) -> ComparativeAnalysis:
        """
        Compare NTK-SURGERY against all baseline methods.
        
        Args:
            dataset_name: Dataset name
            model_ntk: NTK-SURGERY unlearned model
            model_scratch: Scratch baseline model
            X_forget: Forget set data
            y_forget: Forget set labels
            X_retain: Retain set data
            y_retain: Retain set labels
            client_loaders: Client DataLoaders
            client_id: Client to remove
            N: Total samples
            K_global: Global kernel matrix
            gradient_norms: Gradient norms
            
        Returns:
            ComparativeAnalysis object
        """
        logger.info(f"Starting comprehensive method comparison on {dataset_name}")
        
        # Evaluate NTK-SURGERY
        ntk_result = self.comprehensive_evaluation(
            method_name='NTK-SURGERY',
            dataset_name=dataset_name,
            model_unlearned=model_ntk,
            model_scratch=model_scratch,
            X_forget=X_forget,
            y_forget=y_forget,
            X_retain=X_retain,
            y_retain=y_retain,
            server_time=0.0,  # Already computed
            N=N,
            K_global=K_global,
            gradient_norms=gradient_norms
        )
        
        # Evaluate baselines
        for baseline_name in self.config.baseline_methods:
            try:
                # Run baseline unlearning
                baseline_model, baseline_time = self.evaluate_baseline(
                    baseline_name, client_id, client_loaders, gradient_norms
                )
                
                # Evaluate baseline
                self.comprehensive_evaluation(
                    method_name=baseline_name,
                    dataset_name=dataset_name,
                    model_unlearned=baseline_model,
                    model_scratch=model_scratch,
                    X_forget=X_forget,
                    y_forget=y_forget,
                    X_retain=X_retain,
                    y_retain=y_retain,
                    server_time=baseline_time,
                    N=N,
                    K_global=K_global,
                    gradient_norms=gradient_norms
                )
                
            except Exception as e:
                logger.warning(f"Failed to evaluate {baseline_name}: {str(e)}")
        
        # Compute rankings
        self.comparative_analysis.compute_rankings()
        
        # Print comparison table
        print(self.comparative_analysis.generate_comparison_table())
        
        return self.comparative_analysis
    
    def export_all_results(self, filepath: str):
        """
        Export all evaluation results to JSON.
        
        Args:
            filepath: Path to save results
        """
        export_data = {
            'config': self.config.__dict__,
            'results': {
                key: report.to_dict()
                for key, report in self.evaluation_results.items()
            },
            'comparative_analysis': {
                'rankings': self.comparative_analysis.ranking_results,
                'comparison_table': self.comparative_analysis.generate_comparison_table()
            }
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logger.info(f"Exported all results to {filepath}")