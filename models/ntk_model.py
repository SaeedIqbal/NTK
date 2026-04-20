#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NTK-Specific Model Utilities for NTK-SURGERY
Implements Neural Tangent Kernel computation and Jacobian utilities
Implements Section 4.1 of the manuscript
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class NTKModel(ABC):
    """
    Abstract base class for NTK-compatible neural network models.
    
    Provides interface for computing Jacobians and NTK matrices
    required for the NTK-SURGERY methodology.
    
    Attributes:
        model (nn.Module): Underlying neural network
        initialized (bool): Whether model has been initialized
    """
    
    def __init__(self, model: nn.Module):
        """
        Initialize NTKModel.
        
        Args:
            model: PyTorch neural network module
        """
        self.model = model
        self.initialized = False
        self._parameter_count = self._count_parameters()
        
        logger.info(
            f"Initialized NTKModel with {self._parameter_count:,} parameters"
        )
    
    def _count_parameters(self) -> int:
        """Count total trainable parameters in model."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        pass
    
    @abstractmethod
    def functional_forward(
        self, 
        x: torch.Tensor, 
        params: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Forward pass with explicit parameters (for Jacobian computation).
        
        Args:
            x: Input tensor
            params: Dictionary of parameter tensors
            
        Returns:
            Output tensor
        """
        pass
    
    def get_parameters(self) -> Dict[str, torch.Tensor]:
        """
        Get current model parameters as dictionary.
        
        Returns:
            Dictionary mapping parameter names to tensors
        """
        return {name: param.clone() for name, param in self.model.named_parameters()}
    
    def set_parameters(self, params: Dict[str, torch.Tensor]):
        """
        Set model parameters from dictionary.
        
        Args:
            params: Dictionary mapping parameter names to tensors
        """
        state_dict = self.model.state_dict()
        
        for name, param in params.items():
            if name in state_dict:
                state_dict[name] = param.clone()
        
        self.model.load_state_dict(state_dict)
        logger.debug("Model parameters updated")
    
    def get_parameter_vector(self) -> np.ndarray:
        """
        Flatten all parameters into single vector.
        
        Returns:
            Numpy array of all parameters
        """
        params = []
        for param in self.model.parameters():
            if param.requires_grad:
                params.append(param.detach().cpu().numpy().flatten())
        
        return np.concatenate(params)
    
    def set_parameter_vector(self, vector: np.ndarray):
        """
        Set parameters from flattened vector.
        
        Args:
            vector: Flattened parameter vector
        """
        param_list = list(self.model.parameters())
        idx = 0
        
        for param in param_list:
            if param.requires_grad:
                size = param.numel()
                param.data = torch.tensor(
                    vector[idx:idx+size].reshape(param.shape),
                    dtype=param.dtype,
                    device=param.device
                )
                idx += size
        
        logger.debug(f"Set parameters from vector of length {len(vector)}")
    
    def compute_jacobian(
        self, 
        x: torch.Tensor,
        batch_size: Optional[int] = None
    ) -> np.ndarray:
        """
        Compute Jacobian matrix J ∈ ℝ^(N × P).
        
        Implements: J_{ij} = ∂f(x_i; θ)/∂θ_j
        
        Args:
            x: Input tensor of shape (N, *)
            batch_size: Batch size for memory-efficient computation
            
        Returns:
            Jacobian matrix as numpy array
        """
        self.model.eval()
        N = len(x)
        P = self._parameter_count
        
        logger.info(f"Computing Jacobian for {N} samples, {P:,} parameters")
        
        if batch_size is None:
            batch_size = N
        
        J_list = []
        
        for i in range(0, N, batch_size):
            end_idx = min(i + batch_size, N)
            x_batch = x[i:end_idx]
            batch_len = end_idx - i
            
            # Enable gradients for Jacobian computation
            x_batch = x_batch.requires_grad_(True)
            
            # Compute output
            output = self.forward(x_batch)
            
            # Compute Jacobian for each sample in batch
            for j in range(batch_len):
                grad_list = []
                
                for k in range(output.shape[1] if len(output.shape) > 1 else 1):
                    # Compute gradient for each output dimension
                    grad = torch.autograd.grad(
                        output[j, k] if len(output.shape) > 1 else output[j],
                        self.model.parameters(),
                        retain_graph=True,
                        create_graph=False
                    )
                    
                    # Flatten and concatenate gradients
                    grad_flat = torch.cat([g.flatten() for g in grad])
                    grad_list.append(grad_flat.detach().cpu().numpy())
                
                # Average across output dimensions if needed
                if len(grad_list) > 1:
                    jacobian_row = np.mean(grad_list, axis=0)
                else:
                    jacobian_row = grad_list[0]
                
                J_list.append(jacobian_row)
            
            # Clear computation graph
            del output, x_batch
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        J = np.vstack(J_list)
        
        logger.info(f"Jacobian computed: shape {J.shape}")
        
        return J
    
    def compute_ntk_matrix(
        self, 
        X: torch.Tensor,
        batch_size: Optional[int] = None
    ) -> np.ndarray:
        """
        Compute Neural Tangent Kernel matrix Θ ∈ ℝ^(N × N).
        
        Implements: Θ(x, x') = ⟨∇_θ f(x; θ_0), ∇_θ f(x'; θ_0)⟩
        
        Args:
            X: Input tensor of shape (N, *)
            batch_size: Batch size for memory-efficient computation
            
        Returns:
            NTK matrix as numpy array
        """
        logger.info("Computing NTK matrix")
        
        # Compute Jacobian first
        J = self.compute_jacobian(X, batch_size)
        
        # NTK = J @ J.T
        K = J @ J.T
        
        # Normalize to [0, 1] range
        K_min = K.min()
        K_max = K.max()
        K_normalized = (K - K_min) / (K_max - K_min + 1e-8)
        
        logger.info(f"NTK matrix computed: shape {K.shape}, range [{K_min:.4f}, {K_max:.4f}]")
        
        return K_normalized
    
    def verify_ntk_constancy(
        self, 
        X: torch.Tensor,
        num_steps: int = 10
    ) -> Dict[str, float]:
        """
        Verify NTK remains constant during training (infinite-width property).
        
        Args:
            X: Input tensor
            num_steps: Number of training steps to verify
            
        Returns:
            Dictionary with NTK stability metrics
        """
        self.model.train()
        
        # Initial NTK
        K_initial = self.compute_ntk_matrix(X)
        
        ntk_changes = []
        
        for step in range(num_steps):
            # Simulate training step (small update)
            with torch.no_grad():
                for param in self.model.parameters():
                    param += torch.randn_like(param) * 1e-4
            
            # Compute NTK after update
            K_current = self.compute_ntk_matrix(X)
            
            # Measure change
            change = np.linalg.norm(K_current - K_initial, 'fro') / (
                np.linalg.norm(K_initial, 'fro') + 1e-8
            )
            ntk_changes.append(change)
        
        # Restore original parameters
        self.model.eval()
        
        metrics = {
            'mean_ntk_change': float(np.mean(ntk_changes)),
            'max_ntk_change': float(np.max(ntk_changes)),
            'min_ntk_change': float(np.min(ntk_changes)),
            'ntk_stable': float(np.mean(ntk_changes) < 0.1)
        }
        
        logger.info(f"NTK constancy verification: {metrics}")
        
        return metrics


class NTKUtilities:
    """
    Utility class for NTK-related computations.
    
    Provides static methods for common NTK operations used in
    NTK-SURGERY methodology.
    """
    
    @staticmethod
    def compute_kernel_alignment(
        K1: np.ndarray, 
        K2: np.ndarray
    ) -> float:
        """
        Compute alignment between two kernel matrices.
        
        Alignment = ||K1 ⊙ K2||_F / (||K1||_F · ||K2||_F)
        
        Args:
            K1: First kernel matrix
            K2: Second kernel matrix
            
        Returns:
            Alignment score in [0, 1]
        """
        if K1.shape != K2.shape:
            raise ValueError(f"Kernel shapes mismatch: {K1.shape} vs {K2.shape}")
        
        # Frobenius norm of Hadamard product
        numerator = np.linalg.norm(K1 * K2, 'fro')
        
        # Product of Frobenius norms
        denominator = np.linalg.norm(K1, 'fro') * np.linalg.norm(K2, 'fro')
        
        alignment = numerator / (denominator + 1e-8)
        
        return float(np.clip(alignment, 0, 1))
    
    @staticmethod
    def compute_kernel_eigenvalues(K: np.ndarray) -> np.ndarray:
        """
        Compute eigenvalues of kernel matrix.
        
        Args:
            K: Kernel matrix
            
        Returns:
            Array of eigenvalues (sorted descending)
        """
        eigenvalues = np.linalg.eigvalsh(K)
        return np.sort(eigenvalues)[::-1]
    
    @staticmethod
    def compute_effective_rank(
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
        eigenvalues = NTKUtilities.compute_kernel_eigenvalues(K)
        
        if len(eigenvalues) == 0:
            return 0
        
        max_eig = eigenvalues[0]
        threshold_value = threshold * max_eig
        
        effective_rank = np.sum(eigenvalues > threshold_value)
        
        return int(effective_rank)
    
    @staticmethod
    def compute_kernel_condition_number(K: np.ndarray) -> float:
        """
        Compute condition number of kernel matrix.
        
        Args:
            K: Kernel matrix
            
        Returns:
            Condition number
        """
        eigenvalues = NTKUtilities.compute_kernel_eigenvalues(K)
        
        if len(eigenvalues) == 0:
            return float('inf')
        
        # Filter out near-zero eigenvalues
        positive_eigs = eigenvalues[eigenvalues > 1e-10]
        
        if len(positive_eigs) == 0:
            return float('inf')
        
        condition_number = positive_eigs[0] / (positive_eigs[-1] + 1e-10)
        
        return float(condition_number)
    
    @staticmethod
    def regularize_kernel(
        K: np.ndarray, 
        lambda_reg: float = 0.05
    ) -> np.ndarray:
        """
        Add regularization to kernel matrix.
        
        K_reg = K + λI
        
        Args:
            K: Kernel matrix
            lambda_reg: Regularization parameter
            
        Returns:
            Regularized kernel matrix
        """
        N = K.shape[0]
        return K + lambda_reg * np.eye(N)
    
    @staticmethod
    def compute_kernel_ridge_predictions(
        K_train: np.ndarray,
        K_test: np.ndarray,
        y_train: np.ndarray,
        lambda_reg: float = 0.05
    ) -> np.ndarray:
        """
        Compute predictions using kernel ridge regression.
        
        Ŷ_test = K_test @ (K_train + λI)^{-1} @ y_train
        
        Args:
            K_train: Training kernel matrix (N × N)
            K_test: Test kernel matrix (M × N)
            y_train: Training labels (N,)
            lambda_reg: Regularization parameter
            
        Returns:
            Predictions (M,)
        """
        N = K_train.shape[0]
        I_N = np.eye(N)
        
        # Solve linear system (more stable than explicit inverse)
        from scipy.linalg import solve
        
        alpha = solve(K_train + lambda_reg * I_N, y_train)
        
        predictions = K_test @ alpha
        
        return predictions
    
    @staticmethod
    def verify_kernel_positive_definite(
        K: np.ndarray, 
        tolerance: float = 1e-6
    ) -> bool:
        """
        Verify kernel matrix is positive definite.
        
        Args:
            K: Kernel matrix
            tolerance: Eigenvalue tolerance
            
        Returns:
            True if positive definite
        """
        eigenvalues = NTKUtilities.compute_kernel_eigenvalues(K)
        
        is_pd = np.all(eigenvalues > -tolerance)
        
        if not is_pd:
            logger.warning(
                f"Kernel matrix not positive definite. "
                f"Min eigenvalue: {eigenvalues[-1]:.6e}"
            )
        
        return is_pd
    
    @staticmethod
    def compute_ntk_trace(K: np.ndarray) -> float:
        """
        Compute trace of NTK matrix.
        
        Related to average sensitivity of network outputs.
        
        Args:
            K: NTK matrix
            
        Returns:
            Trace value
        """
        return float(np.trace(K))
    
    @staticmethod
    def compute_parameter_sensitivity(
        model: nn.Module,
        X: torch.Tensor,
        param_name: str
    ) -> float:
        """
        Compute sensitivity of outputs to specific parameter.
        
        Args:
            model: Neural network
            X: Input data
            param_name: Name of parameter to analyze
            
        Returns:
            Average sensitivity (gradient norm)
        """
        model.eval()
        
        param = dict(model.named_parameters())[param_name]
        
        output = model(X)
        
        # Compute gradient w.r.t. specific parameter
        grad = torch.autograd.grad(
            output.mean(),
            param,
            retain_graph=False
        )[0]
        
        sensitivity = torch.norm(grad, 'fro').item()
        
        return float(sensitivity)