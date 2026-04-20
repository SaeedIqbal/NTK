#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unlearning Efficacy Metrics for NTK-SURGERY
Implements metrics from Section 5.2 of the manuscript

Metrics:
- Forget Accuracy (FA): Should drop to random chance
- Retain Accuracy (RA): Should match Scratch baseline
- Exactness Score (ES): Functional distance to true retraining
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class MetricQuality(Enum):
    """Quality levels for metric validation."""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    INVALID = "invalid"


@dataclass
class UnlearningResult:
    """Data class for unlearning evaluation results."""
    forget_accuracy: float
    retain_accuracy: float
    exactness_score: float
    exactness_error: float
    quality: MetricQuality
    metadata: Dict[str, any]


class UnlearningMetrics:
    """
    Computes unlearning efficacy metrics for NTK-SURGERY evaluation.
    
    Implements three core metrics from Section 5.2:
    1. Forget Accuracy (FA) - Lower is better (target: random chance)
    2. Retain Accuracy (RA) - Higher is better (target: Scratch performance)
    3. Exactness Score (ES) - Higher is better (target: 1.0)
    
    Attributes:
        device (str): Computing device
        num_classes (int): Number of classification classes
    """
    
    def __init__(self, num_classes: int, device: str = 'cpu'):
        """
        Initialize UnlearningMetrics.
        
        Args:
            num_classes: Number of classification classes
            device: Computing device ('cpu' or 'cuda')
        """
        self.num_classes = num_classes
        self.device = device
        self.random_chance = 1.0 / num_classes
        
        logger.info(
            f"Initialized UnlearningMetrics: classes={num_classes}, "
            f"random_chance={self.random_chance:.4f}"
        )
    
    def forget_accuracy(
        self,
        model: nn.Module,
        X_forget: np.ndarray,
        y_forget: np.ndarray,
        batch_size: int = 64
    ) -> float:
        """
        Compute Forget Accuracy (FA).
        
        Measures model's accuracy on removed client's data.
        After successful unlearning, FA should drop to random chance.
        
        Implements: FA = (1/N_f) Σ 1[f(x_i) = y_i] for x_i ∈ D_forget
        
        Args:
            model: Unlearned model
            X_forget: Data from removed client
            y_forget: Labels from removed client
            batch_size: Batch size for evaluation
            
        Returns:
            Forget Accuracy (lower is better, target ≈ random_chance)
        """
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for i in range(0, len(X_forget), batch_size):
                end_idx = min(i + batch_size, len(X_forget))
                X_batch = torch.tensor(
                    X_forget[i:end_idx], 
                    dtype=torch.float32
                ).to(self.device)
                
                outputs = model(X_batch)
                predictions = outputs.argmax(dim=1).cpu().numpy()
                
                correct += np.sum(predictions == y_forget[i:end_idx])
                total += len(y_forget[i:end_idx])
        
        fa = correct / max(total, 1)
        
        logger.info(
            f"Forget Accuracy: {fa:.4f} (target: {self.random_chance:.4f}, "
            f"diff: {abs(fa - self.random_chance):.4f})"
        )
        
        return float(fa)
    
    def retain_accuracy(
        self,
        model: nn.Module,
        X_retain: np.ndarray,
        y_retain: np.ndarray,
        batch_size: int = 64
    ) -> float:
        """
        Compute Retain Accuracy (RA).
        
        Measures model's accuracy on remaining clients' data.
        After unlearning, RA should match Scratch baseline performance.
        
        Implements: RA = (1/N_r) Σ 1[f(x_i) = y_i] for x_i ∈ D_retain
        
        Args:
            model: Unlearned model
            X_retain: Data from remaining clients
            y_retain: Labels from remaining clients
            batch_size: Batch size for evaluation
            
        Returns:
            Retain Accuracy (higher is better, target: Scratch performance)
        """
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for i in range(0, len(X_retain), batch_size):
                end_idx = min(i + batch_size, len(X_retain))
                X_batch = torch.tensor(
                    X_retain[i:end_idx], 
                    dtype=torch.float32
                ).to(self.device)
                
                outputs = model(X_batch)
                predictions = outputs.argmax(dim=1).cpu().numpy()
                
                correct += np.sum(predictions == y_retain[i:end_idx])
                total += len(y_retain[i:end_idx])
        
        ra = correct / max(total, 1)
        
        logger.info(f"Retain Accuracy: {ra:.4f}")
        
        return float(ra)
    
    def exactness_score(
        self,
        pred_surgery: np.ndarray,
        pred_scratch: np.ndarray,
        normalization: str = 'l2'
    ) -> float:
        """
        Compute Exactness Score (ES).
        
        Measures functional distance between NTK-SURGERY and true retraining.
        ES = 1 - ||f_surgery - f_scratch|| / ||f_scratch||
        
        Implements Section 4.3 Exactness Metric:
        ℰ_exact = ||ℐ^{(-c)}Y - ℐ_retrain Y||_2
        
        Args:
            pred_surgery: Predictions from NTK-SURGERY
            pred_scratch: Predictions from retraining (Scratch baseline)
            normalization: Normalization type ('l2', 'l1', 'linf')
            
        Returns:
            Exactness Score (higher is better, 1.0 = perfect match)
        """
        if len(pred_surgery) != len(pred_scratch):
            raise ValueError(
                f"Prediction length mismatch: {len(pred_surgery)} vs {len(pred_scratch)}"
            )
        
        # Compute error norm
        diff = pred_surgery - pred_scratch
        
        if normalization == 'l2':
            error_norm = np.linalg.norm(diff, 2)
            ref_norm = np.linalg.norm(pred_scratch, 2)
        elif normalization == 'l1':
            error_norm = np.linalg.norm(diff, 1)
            ref_norm = np.linalg.norm(pred_scratch, 1)
        elif normalization == 'linf':
            error_norm = np.linalg.norm(diff, np.inf)
            ref_norm = np.linalg.norm(pred_scratch, np.inf)
        else:
            raise ValueError(f"Unknown normalization: {normalization}")
        
        # Compute exactness score
        exactness_error = error_norm / (ref_norm + 1e-8)
        exactness_score = 1.0 - exactness_error
        
        # Clip to [0, 1] range
        exactness_score = float(np.clip(exactness_score, 0.0, 1.0))
        
        logger.info(
            f"Exactness Score: {exactness_score:.4f} "
            f"(Error: {exactness_error:.4f})"
        )
        
        return exactness_score
    
    def exactness_error(
        self,
        pred_surgery: np.ndarray,
        pred_scratch: np.ndarray
    ) -> float:
        """
        Compute Exactness Error (complement of Exactness Score).
        
        Args:
            pred_surgery: Predictions from NTK-SURGERY
            pred_scratch: Predictions from retraining
            
        Returns:
            Exactness Error (lower is better, 0.0 = perfect)
        """
        return 1.0 - self.exactness_score(pred_surgery, pred_scratch)
    
    def compute_all_metrics(
        self,
        model_surgery: nn.Module,
        model_scratch: nn.Module,
        X_forget: np.ndarray,
        y_forget: np.ndarray,
        X_retain: np.ndarray,
        y_retain: np.ndarray,
        batch_size: int = 64
    ) -> Dict[str, float]:
        """
        Compute all unlearning efficacy metrics.
        
        Args:
            model_surgery: Model from NTK-SURGERY
            model_scratch: Model from retraining (gold standard)
            X_forget: Forget set data
            y_forget: Forget set labels
            X_retain: Retain set data
            y_retain: Retain set labels
            batch_size: Evaluation batch size
            
        Returns:
            Dictionary with all metrics
        """
        # Get predictions for exactness computation
        model_surgery.eval()
        model_scratch.eval()
        
        X_retain_tensor = torch.tensor(
            X_retain, dtype=torch.float32
        ).to(self.device)
        
        with torch.no_grad():
            pred_surgery = model_surgery(X_retain_tensor).cpu().numpy()
            pred_scratch = model_scratch(X_retain_tensor).cpu().numpy()
        
        # Compute metrics
        fa = self.forget_accuracy(model_surgery, X_forget, y_forget, batch_size)
        ra = self.retain_accuracy(model_surgery, X_retain, y_retain, batch_size)
        es = self.exactness_score(pred_surgery, pred_scratch)
        ee = self.exactness_error(pred_surgery, pred_scratch)
        
        metrics = {
            'forget_accuracy': fa,
            'retain_accuracy': ra,
            'exactness_score': es,
            'exactness_error': ee,
            'random_chance': self.random_chance,
            'fa_vs_random': abs(fa - self.random_chance),
            'ra_vs_scratch': abs(ra - self._compute_scratch_ra(model_scratch, X_retain, y_retain))
        }
        
        logger.info(f"All metrics computed: {metrics}")
        
        return metrics
    
    def _compute_scratch_ra(
        self,
        model: nn.Module,
        X: np.ndarray,
        y: np.ndarray
    ) -> float:
        """Compute Scratch baseline retain accuracy."""
        return self.retain_accuracy(model, X, y)
    
    def evaluate_metric_quality(
        self,
        fa: float,
        ra: float,
        es: float,
        thresholds: Optional[Dict[str, float]] = None
    ) -> MetricQuality:
        """
        Evaluate overall metric quality.
        
        Args:
            fa: Forget Accuracy
            ra: Retain Accuracy
            es: Exactness Score
            thresholds: Quality thresholds
            
        Returns:
            MetricQuality enum value
        """
        if thresholds is None:
            thresholds = {
                'fa_excellent': self.random_chance + 0.05,
                'fa_good': self.random_chance + 0.10,
                'ra_excellent': 0.95,
                'ra_good': 0.90,
                'es_excellent': 0.98,
                'es_good': 0.95
            }
        
        # Score each metric
        fa_score = 2 if fa <= thresholds['fa_excellent'] else (1 if fa <= thresholds['fa_good'] else 0)
        ra_score = 2 if ra >= thresholds['ra_excellent'] else (1 if ra >= thresholds['ra_good'] else 0)
        es_score = 2 if es >= thresholds['es_excellent'] else (1 if es >= thresholds['es_good'] else 0)
        
        total_score = fa_score + ra_score + es_score
        
        if total_score >= 5:
            return MetricQuality.EXCELLENT
        elif total_score >= 4:
            return MetricQuality.GOOD
        elif total_score >= 3:
            return MetricQuality.ACCEPTABLE
        elif total_score >= 1:
            return MetricQuality.POOR
        else:
            return MetricQuality.INVALID


class UnlearningEvaluator:
    """
    Comprehensive unlearning evaluation orchestrator.
    
    Coordinates metric computation across multiple models and datasets.
    """
    
    def __init__(
        self,
        num_classes: int,
        device: str = 'cpu'
    ):
        """
        Initialize UnlearningEvaluator.
        
        Args:
            num_classes: Number of classification classes
            device: Computing device
        """
        self.metrics = UnlearningMetrics(num_classes, device)
        self.evaluation_history = []
        
        logger.info(f"Initialized UnlearningEvaluator for {num_classes} classes")
    
    def evaluate_unlearning(
        self,
        method_name: str,
        model_surgery: nn.Module,
        model_scratch: nn.Module,
        X_forget: np.ndarray,
        y_forget: np.ndarray,
        X_retain: np.ndarray,
        y_retain: np.ndarray,
        dataset_name: str = 'unknown'
    ) -> UnlearningResult:
        """
        Perform comprehensive unlearning evaluation.
        
        Args:
            method_name: Name of unlearning method
            model_surgery: Unlearned model
            model_scratch: Scratch baseline model
            X_forget: Forget set data
            y_forget: Forget set labels
            X_retain: Retain set data
            y_retain: Retain set labels
            dataset_name: Dataset name for logging
            
        Returns:
            UnlearningResult dataclass
        """
        logger.info(f"Evaluating {method_name} on {dataset_name}")
        
        # Compute metrics
        metrics_dict = self.metrics.compute_all_metrics(
            model_surgery, model_scratch,
            X_forget, y_forget,
            X_retain, y_retain
        )
        
        # Evaluate quality
        quality = self.metrics.evaluate_metric_quality(
            metrics_dict['forget_accuracy'],
            metrics_dict['retain_accuracy'],
            metrics_dict['exactness_score']
        )
        
        # Create result
        result = UnlearningResult(
            forget_accuracy=metrics_dict['forget_accuracy'],
            retain_accuracy=metrics_dict['retain_accuracy'],
            exactness_score=metrics_dict['exactness_score'],
            exactness_error=metrics_dict['exactness_error'],
            quality=quality,
            metadata={
                'method': method_name,
                'dataset': dataset_name,
                'random_chance': metrics_dict['random_chance'],
                'fa_vs_random': metrics_dict['fa_vs_random'],
                'ra_vs_scratch': metrics_dict['ra_vs_scratch']
            }
        )
        
        # Store in history
        self.evaluation_history.append(result)
        
        logger.info(
            f"Evaluation complete: FA={result.forget_accuracy:.4f}, "
            f"RA={result.retain_accuracy:.4f}, ES={result.exactness_score:.4f}, "
            f"Quality={result.quality.value}"
        )
        
        return result
    
    def compare_methods(
        self,
        results: Dict[str, UnlearningResult]
    ) -> Dict[str, any]:
        """
        Compare multiple unlearning methods.
        
        Args:
            results: Dictionary mapping method names to results
            
        Returns:
            Comparison summary
        """
        comparison = {
            'methods': list(results.keys()),
            'forget_accuracy': {k: v.forget_accuracy for k, v in results.items()},
            'retain_accuracy': {k: v.retain_accuracy for k, v in results.items()},
            'exactness_score': {k: v.exactness_score for k, v in results.items()},
            'quality': {k: v.quality.value for k, v in results.items()}
        }
        
        # Rank methods by exactness score
        ranked = sorted(
            results.items(),
            key=lambda x: x[1].exactness_score,
            reverse=True
        )
        
        comparison['ranking'] = [name for name, _ in ranked]
        comparison['best_method'] = ranked[0][0] if ranked else None
        
        logger.info(f"Method comparison: Best = {comparison['best_method']}")
        
        return comparison
    
    def get_aggregate_statistics(self) -> Dict[str, float]:
        """
        Get aggregate statistics across all evaluations.
        
        Returns:
            Dictionary with aggregate metrics
        """
        if not self.evaluation_history:
            return {}
        
        fa_values = [r.forget_accuracy for r in self.evaluation_history]
        ra_values = [r.retain_accuracy for r in self.evaluation_history]
        es_values = [r.exactness_score for r in self.evaluation_history]
        
        stats = {
            'num_evaluations': len(self.evaluation_history),
            'forget_accuracy_mean': float(np.mean(fa_values)),
            'forget_accuracy_std': float(np.std(fa_values)),
            'retain_accuracy_mean': float(np.mean(ra_values)),
            'retain_accuracy_std': float(np.std(ra_values)),
            'exactness_score_mean': float(np.mean(es_values)),
            'exactness_score_std': float(np.std(es_values)),
            'quality_distribution': self._compute_quality_distribution()
        }
        
        return stats
    
    def _compute_quality_distribution(self) -> Dict[str, int]:
        """Compute distribution of quality ratings."""
        distribution = {q.value: 0 for q in MetricQuality}
        
        for result in self.evaluation_history:
            distribution[result.quality.value] += 1
        
        return distribution


class MetricValidator:
    """
    Validates metric computations for correctness and consistency.
    """
    
    @staticmethod
    def validate_accuracy_range(accuracy: float, tolerance: float = 1e-6) -> bool:
        """
        Validate accuracy is in valid range [0, 1].
        
        Args:
            accuracy: Accuracy value
            tolerance: Numerical tolerance
            
        Returns:
            True if valid
        """
        return -tolerance <= accuracy <= 1.0 + tolerance
    
    @staticmethod
    def validate_exactness_score(score: float, tolerance: float = 1e-6) -> bool:
        """
        Validate exactness score is in valid range [0, 1].
        
        Args:
            score: Exactness score
            tolerance: Numerical tolerance
            
        Returns:
            True if valid
        """
        return -tolerance <= score <= 1.0 + tolerance
    
    @staticmethod
    def validate_forget_accuracy(
        fa: float,
        random_chance: float,
        acceptable_deviation: float = 0.15
    ) -> bool:
        """
        Validate forget accuracy is close to random chance.
        
        Args:
            fa: Forget accuracy
            random_chance: Random chance baseline
            acceptable_deviation: Maximum acceptable deviation
            
        Returns:
            True if acceptable
        """
        return abs(fa - random_chance) <= acceptable_deviation
    
    @staticmethod
    def validate_retain_accuracy(
        ra: float,
        scratch_ra: float,
        acceptable_drop: float = 0.05
    ) -> bool:
        """
        Validate retain accuracy is close to Scratch baseline.
        
        Args:
            ra: Retain accuracy
            scratch_ra: Scratch baseline retain accuracy
            acceptable_drop: Maximum acceptable drop
            
        Returns:
            True if acceptable
        """
        return ra >= scratch_ra - acceptable_drop
    
    @staticmethod
    def validate_metrics_consistency(
        fa: float,
        ra: float,
        es: float,
        num_classes: int
    ) -> Dict[str, bool]:
        """
        Validate consistency across all metrics.
        
        Args:
            fa: Forget accuracy
            ra: Retain accuracy
            es: Exactness score
            num_classes: Number of classes
            
        Returns:
            Dictionary with validation results
        """
        random_chance = 1.0 / num_classes
        
        validations = {
            'fa_in_range': MetricValidator.validate_accuracy_range(fa),
            'ra_in_range': MetricValidator.validate_accuracy_range(ra),
            'es_in_range': MetricValidator.validate_exactness_score(es),
            'fa_near_random': MetricValidator.validate_forget_accuracy(
                fa, random_chance
            ),
            'ra_reasonable': ra >= 0.0,
            'es_reasonable': es >= 0.0
        }
        
        all_valid = all(validations.values())
        validations['all_valid'] = all_valid
        
        if not all_valid:
            failed = [k for k, v in validations.items() if not v]
            logger.warning(f"Metric validation failed: {failed}")
        
        return validations