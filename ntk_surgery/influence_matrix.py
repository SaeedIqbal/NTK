#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Section 4.2: The Influence Matrix
Implements Equation (3) and (4) from the manuscript

This module handles:
- Influence Matrix construction
- Resolvent matrix computation
- Spectral analysis of influence
- Client removal preparation
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from scipy.linalg import inv, eigvalsh
import warnings

logger = logging.getLogger(__name__)


class ResolventMatrix:
    """
    Resolvent matrix G_λ = (K_global + λI)^{-1}.
    
    Central component for Influence Matrix computation.
    
    Attributes:
        K_global (np.ndarray): Global kernel matrix
        lambda_reg (float): Regularization parameter
        G_lambda (np.ndarray): Resolvent matrix
    """
    
    def __init__(
        self,
        K_global: np.ndarray,
        lambda_reg: float = 0.05
    ):
        """
        Initialize ResolventMatrix.
        
        Args:
            K_global: Global kernel matrix (N × N)
            lambda_reg: Regularization parameter λ
        """
        self.K_global = K_global
        self.lambda_reg = lambda_reg
        self.N = K_global.shape[0]
        self.I_N = np.eye(self.N)
        
        # Compute resolvent
        self.G_lambda = self._compute_resolvent()
        
        logger.info(
            f"Initialized ResolventMatrix: N={self.N}, λ={lambda_reg}"
        )
    
    def _compute_resolvent(self) -> np.ndarray:
        """
        Compute G_λ = (K_global + λI)^{-1}.
        
        Returns:
            Resolvent matrix
        """
        # Add regularization and numerical stability term
        regularized_K = self.K_global + self.lambda_reg * self.I_N + 1e-6 * self.I_N
        
        # Compute inverse
        G_lambda = inv(regularized_K)
        
        # Verify positive definiteness
        eigenvalues = eigvalsh(G_lambda)
        min_eig = eigenvalues.min()
        max_eig = eigenvalues.max()
        
        logger.info(
            f"Resolvent computed: eigenvalue range [{min_eig:.4f}, {max_eig:.4f}]"
        )
        
        return G_lambda
    
    def get_condition_number(self) -> float:
        """
        Compute condition number of resolvent.
        
        Returns:
            Condition number κ(G_λ)
        """
        eigenvalues = eigvalsh(self.G_lambda)
        positive_eigs = eigenvalues[eigenvalues > 1e-10]
        
        if len(positive_eigs) == 0:
            return float('inf')
        
        condition_number = positive_eigs.max() / positive_eigs.min()
        
        return float(condition_number)
    
    def get_eigenvalues(self) -> np.ndarray:
        """
        Get eigenvalues of resolvent.
        
        Returns:
            Array of eigenvalues (sorted descending)
        """
        eigenvalues = eigvalsh(self.G_lambda)
        return np.sort(eigenvalues)[::-1]
    
    def update_for_client_removal(
        self,
        client_id: int,
        S_c: np.ndarray,
        K_c: np.ndarray,
        n_c: int,
        N_total: int,
        woodbury_jitter: float = 1e-5
    ) -> 'ResolventMatrix':
        """
        Update resolvent for client removal using Woodbury identity.
        
        Implements Equation (4):
        G_λ^{(-c)} = G_λ - G_λ S_c (-N/n_c K_c^{-1} + S_c^T G_λ S_c)^{-1} S_c^T G_λ
        
        Args:
            client_id: Client identifier
            S_c: Client selection matrix
            K_c: Client kernel matrix
            n_c: Number of samples for client
            N_total: Total number of samples
            woodbury_jitter: Numerical stability jitter
            
        Returns:
            New ResolventMatrix with updated G_λ
        """
        # Compute projected resolvent: S_c^T G_λ S_c
        G_sub = S_c.T @ self.G_lambda @ S_c
        
        # Compute K_c^{-1}
        K_c_inv = inv(K_c + woodbury_jitter * np.eye(n_c))
        
        # Compute middle term: M_c = -N/n_c K_c^{-1} + S_c^T G_λ S_c
        M_c = -(N_total / n_c) * K_c_inv + G_sub
        
        # Check condition number
        cond_M = np.linalg.cond(M_c + woodbury_jitter * np.eye(n_c))
        
        if cond_M > 1e10:
            logger.warning(f"High condition number {cond_M:.2e} for client {client_id}")
        
        # Invert middle term
        M_c_inv = inv(M_c + woodbury_jitter * np.eye(n_c))
        
        # Compute update term
        G_left = self.G_lambda @ S_c
        G_right = S_c.T @ self.G_lambda
        update = G_left @ M_c_inv @ G_right
        
        # Updated resolvent: G_λ^{(-c)} = G_λ - update
        G_new = self.G_lambda - update
        
        # Create new ResolventMatrix with updated G_λ
        new_resolvent = ResolventMatrix.__new__(ResolventMatrix)
        new_resolvent.K_global = self.K_global
        new_resolvent.lambda_reg = self.lambda_reg
        new_resolvent.N = self.N
        new_resolvent.I_N = self.I_N
        new_resolvent.G_lambda = G_new
        
        logger.info(f"Resolvent updated for client {client_id} removal")
        
        return new_resolvent


class InfluenceMatrix:
    """
    Influence Matrix ℐ = I_N - λG_λ.
    
    Implements Section 4.2 of the manuscript.
    Maps labels Y to predictions Ŷ via Ŷ = ℐY.
    
    Attributes:
        K_global (np.ndarray): Global kernel matrix
        lambda_reg (float): Regularization parameter
        Influence (np.ndarray): Influence matrix
        resolvent (ResolventMatrix): Resolvent matrix object
    """
    
    def __init__(
        self,
        K_global: np.ndarray,
        lambda_reg: float = 0.05
    ):
        """
        Initialize InfluenceMatrix.
        
        Args:
            K_global: Global kernel matrix (N × N)
            lambda_reg: Regularization parameter λ
        """
        self.K_global = K_global
        self.lambda_reg = lambda_reg
        self.N = K_global.shape[0]
        self.I_N = np.eye(self.N)
        
        # Compute resolvent
        self.resolvent = ResolventMatrix(K_global, lambda_reg)
        self.G_lambda = self.resolvent.G_lambda
        
        # Compute Influence Matrix: ℐ = I_N - λG_λ
        self.Influence = self.I_N - self.lambda_reg * self.G_lambda
        
        # Cache for eigenvalues
        self._eigenvalues = None
        
        logger.info(f"Initialized InfluenceMatrix: N={self.N}, λ={lambda_reg}")
    
    def get_predictions(self, Y: np.ndarray) -> np.ndarray:
        """
        Get predictions using Influence Matrix.
        
        Implements: Ŷ = ℐY
        
        Args:
            Y: Label vector
            
        Returns:
            Predicted function values
        """
        return self.Influence @ Y
    
    def compute_eigenvalues(self) -> np.ndarray:
        """
        Compute eigenvalues of Influence Matrix.
        
        Implements: μ_k(ℐ) = σ_k(K_global) / (σ_k(K_global) + λ)
        
        Returns:
            Array of eigenvalues (sorted descending)
        """
        if self._eigenvalues is None:
            self._eigenvalues = eigvalsh(self.Influence)
            self._eigenvalues = np.sort(self._eigenvalues)[::-1]
            
            logger.info(
                f"Influence eigenvalues: range [{self._eigenvalues.min():.4f}, "
                f"{self._eigenvalues.max():.4f}]"
            )
        
        return self._eigenvalues
    
    def get_spectral_properties(self) -> Dict[str, float]:
        """
        Get spectral properties of Influence Matrix.
        
        Returns:
            Dictionary with spectral metrics
        """
        eigenvalues = self.compute_eigenvalues()
        
        # Filter positive eigenvalues
        positive_eigs = eigenvalues[eigenvalues > 1e-10]
        
        if len(positive_eigs) == 0:
            return {
                'min_eigenvalue': 0.0,
                'max_eigenvalue': 0.0,
                'condition_number': float('inf'),
                'effective_rank': 0,
                'trace': float(np.trace(self.Influence))
            }
        
        # Effective rank (eigenvalues > 1% of max)
        threshold = 0.01 * positive_eigs.max()
        effective_rank = np.sum(positive_eigs > threshold)
        
        properties = {
            'min_eigenvalue': float(positive_eigs.min()),
            'max_eigenvalue': float(positive_eigs.max()),
            'condition_number': float(positive_eigs.max() / positive_eigs.min()),
            'effective_rank': int(effective_rank),
            'trace': float(np.trace(self.Influence)),
            'nuclear_norm': float(np.sum(positive_eigs))
        }
        
        logger.debug(f"Spectral properties: {properties}")
        
        return properties
    
    def get_influence_strength(self, i: int, j: int) -> float:
        """
        Get influence of sample j on prediction i.
        
        ℐ_{ij} quantifies influence of y_j on ŷ_i.
        
        Args:
            i: Prediction index
            j: Label index
            
        Returns:
            Influence strength
        """
        return float(self.Influence[i, j])
    
    def compute_domain_influence(
        self, 
        domains: np.ndarray
    ) -> np.ndarray:
        """
        Compute average influence per domain pair.
        
        Args:
            domains: Array of domain labels for each sample
            
        Returns:
            D × D matrix of average influence between domains
        """
        unique_domains = np.unique(domains)
        D = len(unique_domains)
        domain_inf = np.zeros((D, D))
        
        for i, d_i in enumerate(unique_domains):
            idx_i = np.where(domains == d_i)[0]
            
            for j, d_j in enumerate(unique_domains):
                idx_j = np.where(domains == d_j)[0]
                
                # Extract submatrix
                sub_matrix = self.Influence[np.ix_(idx_i, idx_j)]
                
                # Average absolute influence
                domain_inf[i, j] = np.mean(np.abs(sub_matrix))
        
        logger.info(f"Domain influence matrix computed: shape {domain_inf.shape}")
        
        return domain_inf
    
    def remove_client_contribution(
        self,
        client_id: int,
        S_c: np.ndarray,
        K_c: np.ndarray,
        n_c: int,
        woodbury_jitter: float = 1e-5
    ) -> 'InfluenceMatrix':
        """
        Remove client contribution from Influence Matrix.
        
        Implements client removal via resolvent update.
        
        Args:
            client_id: Client identifier
            S_c: Client selection matrix
            K_c: Client kernel matrix
            n_c: Number of samples for client
            woodbury_jitter: Numerical stability jitter
            
        Returns:
            New InfluenceMatrix with client removed
        """
        # Update resolvent
        G_new_resolvent = self.resolvent.update_for_client_removal(
            client_id, S_c, K_c, n_c, self.N, woodbury_jitter
        )
        
        # Create new Influence Matrix
        new_inf_mat = InfluenceMatrix.__new__(InfluenceMatrix)
        new_inf_mat.K_global = self.K_global
        new_inf_mat.lambda_reg = self.lambda_reg
        new_inf_mat.N = self.N
        new_inf_mat.I_N = self.I_N
        new_inf_mat.resolvent = G_new_resolvent
        new_inf_mat.G_lambda = G_new_resolvent.G_lambda
        new_inf_mat.Influence = self.I_N - self.lambda_reg * G_new_resolvent.G_lambda
        new_inf_mat._eigenvalues = None
        
        logger.info(f"Client {client_id} contribution removed from Influence Matrix")
        
        return new_inf_mat
    
    def verify_influence_properties(self) -> Dict[str, bool]:
        """
        Verify mathematical properties of Influence Matrix.
        
        Checks:
        - Eigenvalues in [0, 1)
        - Symmetry
        - Positive semi-definiteness
        
        Returns:
            Dictionary with verification results
        """
        eigenvalues = self.compute_eigenvalues()
        
        # Check eigenvalue bounds
        eigenvalues_in_range = np.all((eigenvalues >= 0) & (eigenvalues < 1))
        
        # Check symmetry
        symmetry_error = np.linalg.norm(self.Influence - self.Influence.T, 'fro')
        is_symmetric = symmetry_error < 1e-6
        
        # Check positive semi-definiteness
        is_psd = np.all(eigenvalues >= -1e-6)
        
        properties = {
            'eigenvalues_in_range': bool(eigenvalues_in_range),
            'symmetric': bool(is_symmetric),
            'positive_semi_definite': bool(is_psd),
            'symmetry_error': float(symmetry_error)
        }
        
        logger.info(f"Influence Matrix properties: {properties}")
        
        return properties
    
    def compute_sensitivity_to_perturbation(
        self,
        perturbation_magnitude: float = 0.01
    ) -> float:
        """
        Compute sensitivity of predictions to label perturbations.
        
        Measures stability of Influence Matrix.
        
        Args:
            perturbation_magnitude: Magnitude of label perturbation
            
        Returns:
            Sensitivity measure (change in predictions)
        """
        Y = np.random.randn(self.N)
        Y_perturbed = Y + np.random.randn(self.N) * perturbation_magnitude
        
        pred_original = self.get_predictions(Y)
        pred_perturbed = self.get_predictions(Y_perturbed)
        
        sensitivity = np.linalg.norm(pred_perturbed - pred_original) / (
            np.linalg.norm(pred_original) + 1e-8
        )
        
        return float(sensitivity)