#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Section 4.1: The Federated NTK Representation
Implements Equation (1) and (2) from the manuscript

This module handles:
- Neural Tangent Kernel computation at initialization
- Federated kernel aggregation across clients
- Client selection matrix construction
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union
import logging
from scipy.linalg import inv
import warnings

logger = logging.getLogger(__name__)


class NTKKernelComputer:
    """
    Computes Neural Tangent Kernel matrices for neural networks.
    
    Implements Equation (1): Θ(x, x'; θ_0) = ⟨∇_θ f(x; θ_0), ∇_θ f(x'; θ_0)⟩
    
    Attributes:
        model (nn.Module): Neural network model
        theta_0 (dict): Initial parameters at which NTK is computed
        device (str): Computing device ('cpu' or 'cuda')
    """
    
    def __init__(self, model: nn.Module, device: str = 'cpu'):
        """
        Initialize NTKKernelComputer.
        
        Args:
            model: PyTorch neural network model
            device: Computing device
        """
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
        
        # Store initial parameters
        self.theta_0 = {
            name: param.clone().detach() 
            for name, param in self.model.named_parameters()
        }
        
        self._parameter_count = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        
        logger.info(
            f"Initialized NTKKernelComputer with {self._parameter_count:,} parameters"
        )
    
    def compute_jacobian(
        self, 
        X: torch.Tensor,
        batch_size: Optional[int] = None
    ) -> np.ndarray:
        """
        Compute Jacobian matrix J ∈ ℝ^(N × P).
        
        Implements: J_{ij} = ∂f(x_i; θ_0)/∂θ_j
        
        Args:
            X: Input tensor of shape (N, *)
            batch_size: Batch size for memory-efficient computation
            
        Returns:
            Jacobian matrix as numpy array of shape (N, P)
        """
        self.model.load_state_dict(self.theta_0)
        self.model.eval()
        
        N = len(X)
        P = self._parameter_count
        
        if batch_size is None:
            batch_size = min(N, 32)  # Default batch size
        
        logger.info(f"Computing Jacobian: {N} samples, {P:,} parameters, batch_size={batch_size}")
        
        J_list = []
        
        with torch.no_grad():
            for i in range(0, N, batch_size):
                end_idx = min(i + batch_size, N)
                X_batch = X[i:end_idx].to(self.device)
                batch_len = end_idx - i
                
                # Enable gradients for this batch
                X_batch.requires_grad_(True)
                
                # Compute output
                output = self.model(X_batch)
                
                # Compute Jacobian for each sample in batch
                for j in range(batch_len):
                    grad_list = []
                    
                    # For classification, compute gradient for each output dimension
                    num_outputs = output.shape[1] if len(output.shape) > 1 else 1
                    
                    for k in range(num_outputs):
                        # Compute gradient
                        grad = torch.autograd.grad(
                            output[j, k] if num_outputs > 1 else output[j],
                            self.model.parameters(),
                            retain_graph=(k < num_outputs - 1),
                            create_graph=False,
                            allow_unused=True
                        )
                        
                        # Flatten and concatenate gradients
                        grad_flat = torch.cat([
                            g.flatten() if g is not None else torch.zeros(1, device=self.device)
                            for g in grad
                        ])
                        grad_list.append(grad_flat.cpu().numpy())
                    
                    # Average across output dimensions
                    jacobian_row = np.mean(grad_list, axis=0)
                    J_list.append(jacobian_row)
                
                # Clear memory
                del output, X_batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        J = np.vstack(J_list)
        
        logger.info(f"Jacobian computed: shape {J.shape}")
        
        return J
    
    def compute_ntk_matrix(
        self, 
        X: torch.Tensor,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Compute Neural Tangent Kernel matrix Θ ∈ ℝ^(N × N).
        
        Implements: K = J @ J.T where K_{ij} = Θ(x_i, x_j; θ_0)
        
        Args:
            X: Input tensor of shape (N, *)
            normalize: Whether to normalize kernel to [0, 1] range
            
        Returns:
            NTK matrix as numpy array of shape (N, N)
        """
        logger.info("Computing NTK matrix")
        
        # Compute Jacobian first
        J = self.compute_jacobian(X)
        
        # NTK = J @ J.T
        K = J @ J.T
        
        if normalize:
            # Normalize to [0, 1] range for numerical stability
            K_min = K.min()
            K_max = K.max()
            K = (K - K_min) / (K_max - K_min + 1e-8)
            
            logger.info(f"NTK normalized: range [{K.min():.4f}, {K.max():.4f}]")
        
        return K
    
    def verify_kernel_positive_definite(
        self, 
        K: np.ndarray,
        tolerance: float = 1e-6
    ) -> bool:
        """
        Verify kernel matrix is positive semi-definite.
        
        Args:
            K: Kernel matrix
            tolerance: Eigenvalue tolerance
            
        Returns:
            True if positive semi-definite
        """
        eigenvalues = np.linalg.eigvalsh(K)
        min_eig = eigenvalues.min()
        
        is_psd = min_eig > -tolerance
        
        if not is_psd:
            logger.warning(
                f"Kernel matrix not positive semi-definite. "
                f"Min eigenvalue: {min_eig:.6e}"
            )
        
        return is_psd


class FederatedNTK:
    """
    Federated Neural Tangent Kernel Representation.
    
    Implements Section 4.1 of the manuscript including:
    - Client selection matrix construction
    - Client-specific kernel computation
    - Global kernel aggregation via Equation (2)
    
    Attributes:
        X (np.ndarray): Global input data
        y (np.ndarray): Global labels
        clients (dict): Client data partitions
        lambda_reg (float): Regularization parameter
    """
    
    def __init__(
        self,
        model: nn.Module,
        X: np.ndarray,
        y: np.ndarray,
        clients: Dict[int, np.ndarray],
        lambda_reg: float = 0.05,
        device: str = 'cpu'
    ):
        """
        Initialize FederatedNTK.
        
        Args:
            model: Neural network model
            X: Global input data (N × d)
            y: Global labels (N,)
            clients: Dict mapping client_id to sample indices
            lambda_reg: Regularization parameter λ
            device: Computing device
        """
        self.model = model
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.clients = clients
        self.lambda_reg = lambda_reg
        self.device = device
        
        self.N = X.shape[0]
        self.M = len(clients)
        
        # Compute client selection matrices S_c
        self.S_matrices = self._compute_selection_matrices()
        
        # Initialize kernel matrices
        self.K_global = None
        self.K_client = {}
        self.J_global = None
        
        # Initialize kernel computer
        self.kernel_computer = NTKKernelComputer(model, device)
        
        logger.info(
            f"Initialized FederatedNTK: N={self.N}, M={self.M}, λ={lambda_reg}"
        )
    
    def _compute_selection_matrices(self) -> Dict[int, np.ndarray]:
        """
        Compute client selection matrices S_c ∈ {0, 1}^(N × n_c).
        
        S_c maps local client indices to global indices.
        Satisfies: Σ S_c S_c^T = I_N and S_c^T S_c' = δ_cc' I_{n_c}
        
        Returns:
            Dictionary mapping client_id to selection matrix
        """
        S_matrices = {}
        
        for client_id, indices in self.clients.items():
            n_c = len(indices)
            S_c = np.zeros((self.N, n_c))
            
            for i, global_idx in enumerate(indices):
                S_c[global_idx, i] = 1
            
            S_matrices[client_id] = S_c
            
            logger.debug(f"Client {client_id}: S_c shape {S_c.shape}")
        
        return S_matrices
    
    def compute_jacobian(self) -> np.ndarray:
        """
        Compute global Jacobian at initialization θ_0.
        
        Returns:
            Jacobian matrix J ∈ ℝ^(N × P)
        """
        self.J_global = self.kernel_computer.compute_jacobian(self.X)
        return self.J_global
    
    def compute_global_kernel(self, J: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute global kernel matrix K_global ∈ ℝ^(N × N).
        
        Implements: K_global = J @ J.T
        
        Alternatively verifies Equation (2):
        K_global = Σ (n_c/N) S_c K_c S_c^T
        
        Args:
            J: Pre-computed Jacobian (optional)
            
        Returns:
            Global kernel matrix
        """
        if J is None:
            if self.J_global is None:
                self.J_global = self.compute_jacobian()
            J = self.J_global
        
        # Direct computation: K = J @ J.T
        self.K_global = J @ J.T
        
        # Normalize
        K_min = self.K_global.min()
        K_max = self.K_global.max()
        self.K_global = (self.K_global - K_min) / (K_max - K_min + 1e-8)
        
        logger.info(f"Global kernel computed: shape {self.K_global.shape}")
        
        return self.K_global
    
    def compute_client_kernels(self) -> Dict[int, np.ndarray]:
        """
        Compute client-specific kernel matrices K_c ∈ ℝ^(n_c × n_c).
        
        Implements: [K_c]_{ij} = Θ(x_{c,i}, x_{c,j}; θ_0)
        
        Returns:
            Dictionary mapping client_id to client kernel
        """
        if self.K_global is None:
            self.compute_global_kernel()
        
        for client_id, S_c in self.S_matrices.items():
            # K_c = S_c^T @ K_global @ S_c
            self.K_client[client_id] = S_c.T @ self.K_global @ S_c
            
            logger.debug(
                f"Client {client_id}: K_c shape {self.K_client[client_id].shape}"
            )
        
        return self.K_client
    
    def aggregate_kernels(self) -> np.ndarray:
        """
        Aggregate client kernels to reconstruct K_global.
        
        Verifies Equation (2): K_global = Σ (n_c/N) S_c K_c S_c^T
        
        Returns:
            Reconstructed global kernel
        """
        K_reconstructed = np.zeros((self.N, self.N))
        
        for client_id, K_c in self.K_client.items():
            S_c = self.S_matrices[client_id]
            n_c = K_c.shape[0]
            
            # Weighted contribution
            K_reconstructed += (n_c / self.N) * (S_c @ K_c @ S_c.T)
        
        # Verify reconstruction
        if self.K_global is not None:
            reconstruction_error = np.linalg.norm(
                K_reconstructed - self.K_global, 'fro'
            ) / (np.linalg.norm(self.K_global, 'fro') + 1e-8)
            
            logger.info(f"Kernel reconstruction error: {reconstruction_error:.6e}")
        
        return K_reconstructed
    
    def get_kernel_ridge_predictions(
        self, 
        Y: np.ndarray
    ) -> np.ndarray:
        """
        Compute predictions using kernel ridge regression.
        
        Implements: Ŷ = K_global (K_global + λI)^{-1} Y
                   = (I_N - λ(K_global + λI)^{-1}) Y
        
        Args:
            Y: Label vector
            
        Returns:
            Predicted function values
        """
        if self.K_global is None:
            raise ValueError("Kernel matrix not computed. Call compute_global_kernel first.")
        
        I_N = np.eye(self.N)
        
        # Resolvent: G_λ = (K_global + λI)^{-1}
        G_lambda = inv(self.K_global + self.lambda_reg * I_N + 1e-6 * I_N)
        
        # Predictions: Ŷ = K_global @ G_λ @ Y
        Y_hat = self.K_global @ G_lambda @ Y
        
        logger.info(f"Kernel ridge predictions computed: shape {Y_hat.shape}")
        
        return Y_hat
    
    def verify_orthogonality(self) -> Dict[str, float]:
        """
        Verify orthogonality conditions of selection matrices.
        
        Checks: Σ S_c S_c^T = I_N and S_c^T S_c' = δ_cc' I_{n_c}
        
        Returns:
            Dictionary with orthogonality metrics
        """
        # Check Σ S_c S_c^T = I_N
        sum_SST = np.zeros((self.N, self.N))
        for S_c in self.S_matrices.values():
            sum_SST += S_c @ S_c.T
        
        orthogonality_error_1 = np.linalg.norm(sum_SST - np.eye(self.N), 'fro')
        
        # Check S_c^T S_c' = δ_cc' I_{n_c}
        orthogonality_error_2 = 0.0
        client_ids = list(self.S_matrices.keys())
        
        for i, c1 in enumerate(client_ids):
            for c2 in client_ids[i:]:
                S_c1 = self.S_matrices[c1]
                S_c2 = self.S_matrices[c2]
                
                product = S_c1.T @ S_c2
                
                if c1 == c2:
                    # Should be identity
                    error = np.linalg.norm(product - np.eye(S_c1.shape[1]), 'fro')
                else:
                    # Should be zero
                    error = np.linalg.norm(product, 'fro')
                
                orthogonality_error_2 = max(orthogonality_error_2, error)
        
        metrics = {
            'sum_SST_error': float(orthogonality_error_1),
            'orthogonality_error': float(orthogonality_error_2),
            'orthogonality_valid': orthogonality_error_1 < 1e-6 and orthogonality_error_2 < 1e-6
        }
        
        logger.info(f"Orthogonality verification: {metrics}")
        
        return metrics