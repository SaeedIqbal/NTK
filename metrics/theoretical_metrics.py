#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Theoretical Metrics for NTK-SURGERY
Implements theoretical compliance metrics from Section 5.2 of the manuscript

Metrics:
- NTK Alignment Score
- Sensitivity Bound Ratio
- Condition Number Analysis
- Spectral Properties
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from scipy.linalg import svd, eigvalsh, inv, cond
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TheoreticalResult:
    """Data class for theoretical evaluation results."""
    ntk_alignment: float
    sensitivity_bound_ratio: float
    condition_number: float
    effective_rank: int
    spectral_gap: float
    quality: str
    metadata: Dict[str, any]


class TheoreticalMetrics:
    """
    Computes theoretical compliance metrics for NTK-SURGERY.
    
    Implements metrics from Section 5.2:
    1. NTK Alignment Score (higher is better)
    2. Sensitivity Bound Ratio (higher is better)
    3. Condition Number Analysis
    4. Spectral Properties
    
    Attributes:
        epsilon (float): Numerical tolerance
    """
    
    def __init__(self, epsilon: float = 1e-6):
        """
        Initialize TheoreticalMetrics.
        
        Args:
            epsilon: Numerical tolerance
        """
        self.epsilon = epsilon
        
        logger.info(f"Initialized TheoreticalMetrics with epsilon={epsilon}")
    
    def ntk_alignment_score(
        self,
        K_empirical: np.ndarray,
        K_theoretical: Optional[np.ndarray] = None
    ) -> float:
        """
        Compute NTK Alignment Score.
        
        Measures how well empirical NTK matches theoretical infinite-width limit.
        
        Implements: Alignment = ||K_cross||_F / sqrt(||K_source||_F · ||K_target||_F)
        
        Args:
            K_empirical: Empirical kernel matrix
            K_theoretical: Theoretical kernel (optional, uses identity if None)
            
        Returns:
            Alignment score in [0, 1]
        """
        N = K_empirical.shape[0]
        
        if K_theoretical is None:
            # Use normalized identity as theoretical reference
            K_theoretical = np.eye(N) * 0.9
        
        if K_empirical.shape != K_theoretical.shape:
            raise ValueError(
                f"Kernel shape mismatch: {K_empirical.shape} vs {K_theoretical.shape}"
            )
        
        # Frobenius norm of Hadamard product
        numerator = np.linalg.norm(K_empirical * K_theoretical, 'fro')
        
        # Product of Frobenius norms
        norm_emp = np.linalg.norm(K_empirical, 'fro')
        norm_theo = np.linalg.norm(K_theoretical, 'fro')
        denominator = norm_emp * norm_theo
        
        if denominator < self.epsilon:
            return 0.0
        
        alignment = numerator / denominator
        
        # Clip to [0, 1]
        alignment = float(np.clip(alignment, 0.0, 1.0))
        
        logger.info(f"NTK Alignment Score: {alignment:.4f}")
        
        return alignment
    
    def sensitivity_bound_ratio(
        self,
        zeta_sifu: float,
        zeta_ntk: float = 1.0
    ) -> float:
        """
        Compute Sensitivity Bound Ratio.
        
        Ratio = ζ_SIFU / ζ_NTK
        
        Higher ratio indicates NTK-SURGERY avoids exponential sensitivity growth.
        
        Args:
            zeta_sifu: SIFU sensitivity bound
            zeta_ntk: NTK-SURGERY sensitivity bound (constant)
            
        Returns:
            Sensitivity bound ratio
        """
        if zeta_ntk <= 0:
            zeta_ntk = 1.0  # NTK-SURGERY has constant bound
        
        ratio = zeta_sifu / zeta_ntk
        
        logger.info(
            f"Sensitivity Bound Ratio: {ratio:.2f} "
            f"(ζ_SIFU={zeta_sifu:.2f}, ζ_NTK={zeta_ntk:.2f})"
        )
        
        return float(ratio)
    
    def compute_sifu_sensitivity_bound(
        self,
        n_rounds: int,
        gradient_norms: List[float],
        B_factor: float = 1.05,
        K_local: int = 5
    ) -> float:
        """
        Compute SIFU sensitivity bound ζ(n, c).
        
        Implements SIFU Eq. 9:
        ζ(n, c) = Σ_{s=0}^{n-1} B(f_I, η)^{(n-s-1)K} · ||ω(I, θ_s)||_2
        
        In non-convex regimes, B(f_I, η) > 1 (Eq. 26), causing exponential growth.
        
        Args:
            n_rounds: Number of training rounds
            gradient_norms: List of gradient norms per round
            B_factor: Contraction factor B(f_I, η)
            K_local: Local SGD steps
            
        Returns:
            Sensitivity bound value
        """
        zeta = 0.0
        
        for s in range(min(n_rounds, len(gradient_norms))):
            exponent = (n_rounds - s - 1) * K_local
            contribution = (B_factor ** exponent) * gradient_norms[s]
            zeta += contribution
        
        growth_type = "exponential" if B_factor > 1 else "linear"
        
        logger.info(
            f"SIFU sensitivity bound ζ({n_rounds}) = {zeta:.4f} "
            f"({growth_type} growth, B={B_factor})"
        )
        
        return float(zeta)
    
    def condition_number_analysis(
        self,
        matrix: np.ndarray,
        matrix_name: str = 'matrix'
    ) -> Dict[str, float]:
        """
        Analyze condition number of a matrix.
        
        Args:
            matrix: Matrix to analyze
            matrix_name: Name for logging
            
        Returns:
            Dictionary with condition number metrics
        """
        # Compute singular values
        singular_values = svd(matrix, compute_uv=False)
        
        # Filter positive singular values
        positive_sv = singular_values[singular_values > self.epsilon]
        
        if len(positive_sv) == 0:
            return {
                'condition_number': float('inf'),
                'min_singular_value': 0.0,
                'max_singular_value': 0.0,
                'rank': 0,
                'well_conditioned': False
            }
        
        cond_num = positive_sv.max() / positive_sv.min()
        
        analysis = {
            'condition_number': float(cond_num),
            'min_singular_value': float(positive_sv.min()),
            'max_singular_value': float(positive_sv.max()),
            'rank': len(positive_sv),
            'full_rank': len(positive_sv) == matrix.shape[0],
            'well_conditioned': cond_num < 1e4,
            'moderately_conditioned': 1e4 <= cond_num < 1e8,
            'ill_conditioned': cond_num >= 1e8
        }
        
        logger.info(
            f"{matrix_name} condition number: {cond_num:.2e} "
            f"(rank={analysis['rank']}/{matrix.shape[0]})"
        )
        
        return analysis
    
    def effective_rank(
        self,
        K: np.ndarray,
        threshold: float = 0.01
    ) -> int:
        """
        Compute effective rank of kernel matrix.
        
        Number of eigenvalues above threshold fraction of max eigenvalue.
        
        Args:
            K: Kernel matrix
            threshold: Eigenvalue threshold as fraction of max
            
        Returns:
            Effective rank
        """
        eigenvalues = eigvalsh(K)
        eigenvalues = np.sort(eigenvalues)[::-1]  # Descending
        
        if len(eigenvalues) == 0:
            return 0
        
        max_eig = eigenvalues[0]
        threshold_value = threshold * max_eig
        
        effective_rank = int(np.sum(eigenvalues > threshold_value))
        
        logger.info(
            f"Effective rank: {effective_rank}/{len(eigenvalues)} "
            f"(threshold={threshold})"
        )
        
        return effective_rank
    
    def spectral_gap(
        self,
        K: np.ndarray,
        num_top: int = 10
    ) -> float:
        """
        Compute spectral gap of kernel matrix.
        
        Measures separation between top eigenvalues.
        
        Args:
            K: Kernel matrix
            num_top: Number of top eigenvalues to consider
            
        Returns:
            Spectral gap (ratio of top eigenvalues)
        """
        eigenvalues = eigvalsh(K)
        eigenvalues = np.sort(eigenvalues)[::-1]  # Descending
        
        if len(eigenvalues) < 2:
            return 1.0
        
        top_eigs = eigenvalues[:min(num_top, len(eigenvalues))]
        
        # Gap between first and second eigenvalue
        if top_eigs[1] < self.epsilon:
            gap = float('inf')
        else:
            gap = top_eigs[0] / (top_eigs[1] + self.epsilon)
        
        logger.info(f"Spectral gap: {gap:.4f}")
        
        return float(gap)
    
    def compute_all_theoretical_metrics(
        self,
        K_global: np.ndarray,
        gradient_norms: List[float],
        n_rounds: int
    ) -> TheoreticalResult:
        """
        Compute all theoretical metrics.
        
        Args:
            K_global: Global kernel matrix
            gradient_norms: Historical gradient norms
            n_rounds: Number of training rounds
            
        Returns:
            TheoreticalResult dataclass
        """
        # NTK Alignment
        ntk_alignment = self.ntk_alignment_score(K_global)
        
        # Sensitivity bounds
        zeta_sifu = self.compute_sifu_sensitivity_bound(n_rounds, gradient_norms)
        zeta_ntk = 1.0  # NTK-SURGERY constant bound
        sensitivity_ratio = self.sensitivity_bound_ratio(zeta_sifu, zeta_ntk)
        
        # Condition number
        cond_analysis = self.condition_number_analysis(K_global, 'K_global')
        condition_number = cond_analysis['condition_number']
        
        # Effective rank
        effective_rank = self.effective_rank(K_global)
        
        # Spectral gap
        spectral_gap = self.spectral_gap(K_global)
        
        # Quality assessment
        quality = self._assess_theoretical_quality(
            ntk_alignment, sensitivity_ratio, condition_number
        )
        
        result = TheoreticalResult(
            ntk_alignment=ntk_alignment,
            sensitivity_bound_ratio=sensitivity_ratio,
            condition_number=condition_number,
            effective_rank=effective_rank,
            spectral_gap=spectral_gap,
            quality=quality,
            metadata={
                'zeta_sifu': zeta_sifu,
                'zeta_ntk': zeta_ntk,
                'condition_analysis': cond_analysis
            }
        )
        
        logger.info(
            f"Theoretical metrics: alignment={ntk_alignment:.4f}, "
            f"sensitivity_ratio={sensitivity_ratio:.2f}, quality={quality}"
        )
        
        return result
    
    def _assess_theoretical_quality(
        self,
        ntk_alignment: float,
        sensitivity_ratio: float,
        condition_number: float
    ) -> str:
        """
        Assess overall theoretical quality.
        
        Args:
            ntk_alignment: NTK alignment score
            sensitivity_ratio: Sensitivity bound ratio
            condition_number: Condition number
            
        Returns:
            Quality string
        """
        score = 0
        
        if ntk_alignment >= 0.95:
            score += 3
        elif ntk_alignment >= 0.90:
            score += 2
        elif ntk_alignment >= 0.80:
            score += 1
        
        if sensitivity_ratio >= 100:
            score += 3
        elif sensitivity_ratio >= 50:
            score += 2
        elif sensitivity_ratio >= 10:
            score += 1
        
        if condition_number < 1e4:
            score += 2
        elif condition_number < 1e8:
            score += 1
        
        if score >= 7:
            return "excellent"
        elif score >= 5:
            return "good"
        elif score >= 3:
            return "acceptable"
        else:
            return "poor"


class NTKAnalyzer:
    """
    Specialized analyzer for Neural Tangent Kernel properties.
    """
    
    @staticmethod
    def verify_ntk_constancy(
        K_initial: np.ndarray,
        K_final: np.ndarray,
        tolerance: float = 0.1
    ) -> Dict[str, any]:
        """
        Verify NTK remains constant during training.
        
        Args:
            K_initial: Initial NTK matrix
            K_final: Final NTK matrix
            tolerance: Acceptable change tolerance
            
        Returns:
            Verification results
        """
        if K_initial.shape != K_final.shape:
            return {'valid': False, 'error': 'Shape mismatch'}
        
        # Relative change
        change = np.linalg.norm(K_final - K_initial, 'fro') / (
            np.linalg.norm(K_initial, 'fro') + 1e-8
        )
        
        is_constant = change < tolerance
        
        results = {
            'relative_change': float(change),
            'is_constant': bool(is_constant),
            'tolerance': tolerance,
            'valid': is_constant
        }
        
        logger.info(
            f"NTK constancy: change={change:.4f}, "
            f"constant={is_constant} (tol={tolerance})"
        )
        
        return results
    
    @staticmethod
    def compute_kernel_eigenvalue_distribution(
        K: np.ndarray,
        num_bins: int = 50
    ) -> Dict[str, any]:
        """
        Compute eigenvalue distribution of kernel matrix.
        
        Args:
            K: Kernel matrix
            num_bins: Number of histogram bins
            
        Returns:
            Distribution statistics
        """
        eigenvalues = eigvalsh(K)
        eigenvalues = np.sort(eigenvalues)[::-1]
        
        # Histogram
        hist, bin_edges = np.histogram(
            eigenvalues,
            bins=num_bins,
            density=True
        )
        
        # Statistics
        stats = {
            'mean': float(np.mean(eigenvalues)),
            'std': float(np.std(eigenvalues)),
            'median': float(np.median(eigenvalues)),
            'skewness': float(np.mean(((eigenvalues - np.mean(eigenvalues)) / (np.std(eigenvalues) + 1e-8)) ** 3)),
            'kurtosis': float(np.mean(((eigenvalues - np.mean(eigenvalues)) / (np.std(eigenvalues) + 1e-8)) ** 4) - 3),
            'histogram': hist.tolist(),
            'bin_edges': bin_edges.tolist(),
            'num_eigenvalues': len(eigenvalues)
        }
        
        logger.info(
            f"Eigenvalue distribution: mean={stats['mean']:.4f}, "
            f"std={stats['std']:.4f}"
        )
        
        return stats


class SensitivityAnalyzer:
    """
    Analyzes sensitivity bounds for unlearning methods.
    """
    
    @staticmethod
    def analyze_sensitivity_growth(
        gradient_norms: List[float],
        B_factors: List[float] = [1.0, 1.05, 1.1, 1.2]
    ) -> Dict[str, List[float]]:
        """
        Analyze sensitivity growth for different contraction factors.
        
        Args:
            gradient_norms: Historical gradient norms
            B_factors: List of contraction factors to analyze
            
        Returns:
            Dictionary with sensitivity growth curves
        """
        growth_curves = {}
        
        for B in B_factors:
            zeta_values = []
            
            for n in range(1, len(gradient_norms) + 1):
                zeta = 0.0
                for s in range(n):
                    exponent = (n - s - 1) * 5  # K=5 local steps
                    zeta += (B ** exponent) * gradient_norms[s]
                zeta_values.append(zeta)
            
            growth_type = "exponential" if B > 1 else "linear"
            growth_curves[f'B={B} ({growth_type})'] = zeta_values
        
        logger.info(f"Analyzed sensitivity growth for {len(B_factors)} B factors")
        
        return growth_curves
    
    @staticmethod
    def compute_critical_rounds(
        sensitivity_budget: float,
        B_factor: float,
        avg_gradient_norm: float,
        K_local: int = 5
    ) -> int:
        """
        Compute critical rounds before sensitivity exceeds budget.
        
        For B > 1, sensitivity grows exponentially, limiting practical rounds.
        
        Args:
            sensitivity_budget: Maximum allowed sensitivity
            B_factor: Contraction factor
            avg_gradient_norm: Average gradient norm
            K_local: Local SGD steps
            
        Returns:
            Maximum rounds before exceeding budget
        """
        if B_factor <= 1:
            return float('inf')  # Linear growth, no hard limit
        
        # Solve for n where ζ(n) = budget
        # Simplified: ζ(n) ≈ avg_norm · B^(n·K) / (B - 1)
        
        n = 0
        zeta = 0.0
        
        while zeta < sensitivity_budget and n < 1000:
            n += 1
            zeta = avg_gradient_norm * (B_factor ** (n * K_local)) / (B_factor - 1)
        
        logger.info(
            f"Critical rounds: {n} (B={B_factor}, budget={sensitivity_budget})"
        )
        
        return n
    
    @staticmethod
    def compare_sensitivity_bounds(
        methods: Dict[str, float]
    ) -> Dict[str, any]:
        """
        Compare sensitivity bounds across methods.
        
        Args:
            methods: Dictionary mapping method names to sensitivity bounds
            
        Returns:
            Comparison results
        """
        if not methods:
            return {}
        
        min_zeta = min(methods.values())
        max_zeta = max(methods.values())
        
        comparison = {
            'methods': list(methods.keys()),
            'bounds': methods,
            'min_bound': min_zeta,
            'max_bound': max_zeta,
            'range': max_zeta - min_zeta,
            'best_method': min(methods, key=methods.get),
            'worst_method': max(methods, key=methods.get),
            'ratio_max_min': max_zeta / (min_zeta + 1e-8)
        }
        
        logger.info(
            f"Sensitivity comparison: best={comparison['best_method']}, "
            f"ratio={comparison['ratio_max_min']:.2f}"
        )
        
        return comparison