#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Section 4.3: The Surgery Operator
Implements Equation (5), (6), and (7) from the manuscript

This module handles:
- Client removal via Sherman-Morrison-Woodbury identity
- Exactness error computation
- Spectral stability analysis
- Computational complexity tracking
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from scipy.linalg import inv, svd
import time

logger = logging.getLogger(__name__)


class WoodburyUpdater:
    """
    Implements Sherman-Morrison-Woodbury identity for efficient matrix updates.
    
    (A + UCV)^{-1} = A^{-1} - A^{-1}U(C^{-1} + VA^{-1}U)^{-1}VA^{-1}
    
    Used for efficient client removal without full matrix inversion.
    """
    
    @staticmethod
    def apply_woodbury(
        A_inv: np.ndarray,
        U: np.ndarray,
        C: np.ndarray,
        V: np.ndarray,
        jitter: float = 1e-5
    ) -> np.ndarray:
        """
        Apply Woodbury identity to compute (A + UCV)^{-1}.
        
        Args:
            A_inv: Inverse of A (pre-computed)
            U: Left update matrix
            C: Middle update matrix
            V: Right update matrix
            jitter: Numerical stability jitter
            
        Returns:
            Updated inverse matrix
        """
        # Compute VA^{-1}U
        VAU = V @ A_inv @ U
        
        # Compute (C^{-1} + VA^{-1}U)
        C_inv = inv(C + jitter * np.eye(C.shape[0]))
        middle = C_inv + VAU
        
        # Invert middle term
        middle_inv = inv(middle + jitter * np.eye(middle.shape[0]))
        
        # Compute update: A^{-1}U(C^{-1} + VA^{-1}U)^{-1}VA^{-1}
        update = A_inv @ U @ middle_inv @ V @ A_inv
        
        # Updated inverse: A^{-1} - update
        updated_inv = A_inv - update
        
        return updated_inv
    
    @staticmethod
    def compute_condition_number_bound(
        G_lambda: np.ndarray,
        S_c: np.ndarray,
        M_c_inv: np.ndarray,
        sigma_min_G: float
    ) -> float:
        """
        Compute bound on condition number of updated resolvent.
        
        Implements: κ(G_λ^{(-c)}) ≤ κ(G_λ) · (1 + ||G_λ S_c||² ||M_c^{-1}|| / σ_min(G_λ))
        
        Args:
            G_lambda: Original resolvent
            S_c: Client selection matrix
            M_c_inv: Inverse of middle term
            sigma_min_G: Minimum singular value of G_λ
            
        Returns:
            Upper bound on condition number
        """
        # Compute ||G_λ S_c||_2
        G_S_norm = np.linalg.norm(G_lambda @ S_c, 2)
        
        # Compute ||M_c^{-1}||_2
        M_inv_norm = np.linalg.norm(M_c_inv, 2)
        
        # Original condition number
        kappa_G = np.linalg.cond(G_lambda)
        
        # Bound
        bound = kappa_G * (1 + (G_S_norm ** 2) * M_inv_norm / (sigma_min_G + 1e-10))
        
        return float(bound)


class SurgeryOperator:
    """
    Surgery Operator 𝒮_c for client removal.
    
    Implements Section 4.3 of the manuscript.
    Maps ℐ to ℐ^{(-c)} via closed-form update.
    
    Implements Equation (6):
    ℐ^{(-c)} = ℐ + λ[G_λ S_c (S_c^T G_λ S_c - N/n_c K_c^{-1})^{-1} S_c^T G_λ]
    
    Attributes:
        inf_mat: InfluenceMatrix object
        K_global: Global kernel matrix
        S_matrices: Client selection matrices
        K_client: Client kernel matrices
    """
    
    def __init__(
        self,
        inf_mat,  # InfluenceMatrix
        K_global: np.ndarray,
        S_matrices: Dict[int, np.ndarray],
        K_client: Dict[int, np.ndarray],
        woodbury_jitter: float = 1e-5
    ):
        """
        Initialize SurgeryOperator.
        
        Args:
            inf_mat: InfluenceMatrix object
            K_global: Global kernel matrix
            S_matrices: Dict of client selection matrices
            K_client: Dict of client kernel matrices
            woodbury_jitter: Numerical stability jitter
        """
        self.inf_mat = inf_mat
        self.K_global = K_global
        self.S_matrices = S_matrices
        self.K_client = K_client
        self.woodbury_jitter = woodbury_jitter
        self.N = inf_mat.N
        
        # Performance tracking
        self.computation_times = {}
        
        logger.info(f"Initialized SurgeryOperator for N={self.N} samples")
    
    def unlearn_client(
        self,
        client_id: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Remove client c from Influence Matrix.
        
        Implements Equation (6) via Woodbury update.
        
        Args:
            client_id: Client identifier to remove
            
        Returns:
            Tuple of (I_new, G_new) updated matrices
        """
        start_time = time.perf_counter()
        
        S_c = self.S_matrices[client_id]
        K_c = self.K_client[client_id]
        n_c = K_c.shape[0]
        
        logger.info(f"Unlearning client {client_id} with {n_c} samples")
        
        # Step 1: Compute projected resolvent S_c^T G_λ S_c
        G_sub = S_c.T @ self.inf_mat.G_lambda @ S_c
        
        # Step 2: Compute K_c^{-1}
        K_c_inv = inv(K_c + self.woodbury_jitter * np.eye(n_c))
        
        # Step 3: Compute middle term M_c = S_c^T G_λ S_c - N/n_c K_c^{-1}
        M_c = G_sub - (self.N / n_c) * K_c_inv
        
        # Check condition number for stability
        cond_M = np.linalg.cond(M_c + self.woodbury_jitter * np.eye(n_c))
        
        if cond_M > 1e10:
            logger.warning(
                f"High condition number {cond_M:.2e} for client {client_id}. "
                f"Results may be numerically unstable."
            )
        
        # Step 4: Invert middle term
        M_c_inv = inv(M_c + self.woodbury_jitter * np.eye(n_c))
        
        # Step 5: Compute update term using Equation (6)
        G_left = self.inf_mat.G_lambda @ S_c
        G_right = S_c.T @ self.inf_mat.G_lambda
        update = G_left @ M_c_inv @ G_right
        
        # Step 6: Updated resolvent G_λ^{(-c)} = G_λ - update
        G_new = self.inf_mat.G_lambda - update
        
        # Step 7: Updated Influence Matrix ℐ^{(-c)} = I_N - λG_λ^{(-c)}
        I_new = self.inf_mat.I_N - self.inf_mat.lambda_reg * G_new
        
        # Track computation time
        elapsed = time.perf_counter() - start_time
        self.computation_times[client_id] = elapsed
        
        logger.info(
            f"Client {client_id} unlearned in {elapsed:.4f}s. "
            f"Condition number of M_c: {cond_M:.2e}"
        )
        
        return I_new, G_new
    
    def apply_surgery_operator(
        self,
        I: np.ndarray,
        client_id: int
    ) -> np.ndarray:
        """
        Apply Surgery Operator 𝒮_c to Influence Matrix.
        
        Implements: ℐ^{(-c)} = 𝒮_c(ℐ)
        
        Args:
            I: Current Influence Matrix
            client_id: Client to remove
            
        Returns:
            Updated Influence Matrix
        """
        I_new, _ = self.unlearn_client(client_id)
        return I_new
    
    def compute_exactness_error(
        self,
        I_unlearned: np.ndarray,
        I_retrain: np.ndarray,
        Y: np.ndarray
    ) -> float:
        """
        Compute exactness error ℰ_exact.
        
        Implements: ℰ_exact = ||ℐ^{(-c)}Y - ℐ_retrain Y||_2 / ||ℐ_retrain Y||_2
        
        Args:
            I_unlearned: Influence Matrix after surgery
            I_retrain: Influence Matrix from retraining on D \ D_c
            Y: Label vector
            
        Returns:
            Normalized exactness error
        """
        # Predictions from surgery
        pred_surgery = I_unlearned @ Y
        
        # Predictions from retraining
        pred_retrain = I_retrain @ Y
        
        # Compute error
        error = np.linalg.norm(pred_surgery - pred_retrain)
        norm = np.linalg.norm(pred_retrain) + 1e-8
        
        exactness_error = error / norm
        
        # Convert to exactness score (higher is better)
        exactness_score = 1.0 - exactness_error
        
        logger.info(
            f"Exactness error: {exactness_error:.6f}, "
            f"Exactness score: {exactness_score:.6f}"
        )
        
        return float(exactness_error)
    
    def compute_exactness_score(
        self,
        I_unlearned: np.ndarray,
        I_retrain: np.ndarray,
        Y: np.ndarray
    ) -> float:
        """
        Compute exactness score (1 - error).
        
        Args:
            I_unlearned: Influence Matrix after surgery
            I_retrain: Influence Matrix from retraining
            Y: Label vector
            
        Returns:
            Exactness score (1.0 = perfect)
        """
        error = self.compute_exactness_error(I_unlearned, I_retrain, Y)
        return 1.0 - error
    
    def compute_spectral_stability(
        self,
        client_id: int
    ) -> Dict[str, float]:
        """
        Compute spectral stability metrics for client removal.
        
        Args:
            client_id: Client identifier
            
        Returns:
            Dictionary with stability metrics
        """
        S_c = self.S_matrices[client_id]
        K_c = self.K_client[client_id]
        n_c = K_c.shape[0]
        
        # Compute M_c and its inverse
        G_sub = S_c.T @ self.inf_mat.G_lambda @ S_c
        K_c_inv = inv(K_c + self.woodbury_jitter * np.eye(n_c))
        M_c = G_sub - (self.N / n_c) * K_c_inv
        M_c_inv = inv(M_c + self.woodbury_jitter * np.eye(n_c))
        
        # Compute condition number bound
        sigma_min_G = np.min(svd(self.inf_mat.G_lambda, compute_uv=False))
        cond_bound = WoodburyUpdater.compute_condition_number_bound(
            self.inf_mat.G_lambda, S_c, M_c_inv, sigma_min_G
        )
        
        # Actual condition numbers
        cond_G_original = np.linalg.cond(self.inf_mat.G_lambda)
        
        # Compute updated G and its condition number
        G_left = self.inf_mat.G_lambda @ S_c
        G_right = S_c.T @ self.inf_mat.G_lambda
        update = G_left @ M_c_inv @ G_right
        G_new = self.inf_mat.G_lambda - update
        cond_G_new = np.linalg.cond(G_new)
        
        metrics = {
            'cond_G_original': float(cond_G_original),
            'cond_G_updated': float(cond_G_new),
            'cond_M_c': float(np.linalg.cond(M_c)),
            'condition_bound': float(cond_bound),
            'sigma_min_G': float(sigma_min_G),
            'stability_ratio': float(cond_G_new / (cond_G_original + 1e-8))
        }
        
        logger.info(f"Spectral stability metrics: {metrics}")
        
        return metrics
    
    def get_computation_complexity(self) -> Dict[str, int]:
        """
        Compute computational complexity of surgery operation.
        
        NTK-SURGERY: O(N² n_c) vs Full inversion: O(N³)
        
        Returns:
            Dictionary with complexity estimates
        """
        # Dominant operations
        # 1. G_λ S_c: O(N² n_c)
        # 2. M_c inversion: O(n_c³)
        # 3. Update computation: O(N² n_c)
        
        n_c_avg = np.mean([K.shape[0] for K in self.K_client.values()])
        
        surgery_complexity = 2 * (self.N ** 2) * n_c_avg + n_c_avg ** 3
        full_inversion_complexity = self.N ** 3
        
        speedup = full_inversion_complexity / (surgery_complexity + 1e-8)
        
        complexity = {
            'surgery_flops': int(surgery_complexity),
            'full_inversion_flops': int(full_inversion_complexity),
            'speedup_factor': float(speedup),
            'complexity_class': f'O(N² n_c) vs O(N³)'
        }
        
        logger.info(f"Computational complexity: {complexity}")
        
        return complexity
    
    def unlearn_multiple_clients(
        self,
        client_ids: List[int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Unlearn multiple clients sequentially.
        
        Args:
            client_ids: List of client identifiers to remove
            
        Returns:
            Tuple of (I_final, G_final) after all removals
        """
        I_current = self.inf_mat.Influence.copy()
        G_current = self.inf_mat.G_lambda.copy()
        
        for client_id in client_ids:
            logger.info(f"Unlearning client {client_id} ({len(client_ids)} total)")
            
            # Update surgery operator with current matrices
            temp_inf_mat = self.inf_mat.__class__.__new__(self.inf_mat.__class__)
            temp_inf_mat.Influence = I_current
            temp_inf_mat.G_lambda = G_current
            temp_inf_mat.lambda_reg = self.inf_mat.lambda_reg
            temp_inf_mat.I_N = self.inf_mat.I_N
            temp_inf_mat.N = self.inf_mat.N
            
            self.inf_mat = temp_inf_mat
            
            # Perform unlearning
            I_current, G_current = self.unlearn_client(client_id)
        
        logger.info(f"Completed unlearning {len(client_ids)} clients")
        
        return I_current, G_current