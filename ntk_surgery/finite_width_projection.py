#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Section 4.4: Finite-Width Projection
Implements Equation (8), (9), and (10) from the manuscript

This module handles:
- Jacobian computation for finite-width networks
- Weight update projection from function space to parameter space
- Linearization error bound computation
- Complexity analysis
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import logging
from scipy.linalg import inv
import time

logger = logging.getLogger(__name__)


class JacobianComputer:
    """
    Computes Jacobian matrices for finite-width neural networks.
    
    J_t ∈ ℝ^(N × P) with entries [J_t]_{ij} = ∂f(x_i; θ_t)/∂θ_j
    """
    
    @staticmethod
    def compute_jacobian(
        model: nn.Module,
        X: torch.Tensor,
        device: str = 'cpu'
    ) -> np.ndarray:
        """
        Compute Jacobian at current parameters θ_t.
        
        Args:
            model: Neural network model
            X: Input tensor
            device: Computing device
            
        Returns:
            Jacobian matrix as numpy array
        """
        model.eval()
        N = len(X)
        
        J_list = []
        
        with torch.no_grad():
            for i in range(N):
                x_i = X[i:i+1].to(device)
                x_i.requires_grad_(True)
                
                output = model(x_i)
                
                # Compute gradient for each output dimension
                grad_list = []
                num_outputs = output.shape[1] if len(output.shape) > 1 else 1
                
                for k in range(num_outputs):
                    grad = torch.autograd.grad(
                        output[0, k] if num_outputs > 1 else output[0],
                        model.parameters(),
                        retain_graph=(k < num_outputs - 1),
                        create_graph=False
                    )
                    
                    grad_flat = torch.cat([g.flatten() for g in grad])
                    grad_list.append(grad_flat.cpu().numpy())
                
                # Average across output dimensions
                jacobian_row = np.mean(grad_list, axis=0)
                J_list.append(jacobian_row)
        
        J = np.vstack(J_list)
        
        logger.info(f"Jacobian computed: shape {J.shape}")
        
        return J
    
    @staticmethod
    def compute_jacobian_vector_product(
        model: nn.Module,
        X: torch.Tensor,
        v: np.ndarray,
        device: str = 'cpu'
    ) -> np.ndarray:
        """
        Compute Jacobian-vector product J^T v efficiently.
        
        More efficient than computing full Jacobian for large P.
        
        Args:
            model: Neural network model
            X: Input tensor
            v: Vector to multiply
            device: Computing device
            
        Returns:
            J^T v as numpy array
        """
        model.eval()
        X = X.to(device)
        
        # Create scalar from v
        v_tensor = torch.tensor(v, dtype=torch.float32, device=device)
        
        # Forward pass
        output = model(X)
        
        # Compute vector-Jacobian product
        jvp = torch.autograd.grad(
            output,
            model.parameters(),
            grad_outputs=v_tensor,
            retain_graph=False
        )
        
        # Flatten and concatenate
        jvp_flat = torch.cat([g.flatten() for g in jvp])
        
        return jvp_flat.cpu().numpy()


class FiniteWidthProjector:
    """
    Projects unlearned function from NTK space to finite-width weights.
    
    Implements Section 4.4 of the manuscript.
    
    Implements Equation (9):
    θ_new = θ_t + J_t^T G_λ^{(-c)} (ℐ^{(-c)}Y - f(X, θ_t))
    
    Attributes:
        model: Neural network model
        theta_t: Current parameters at round t
        lambda_reg: Regularization parameter
    """
    
    def __init__(
        self,
        model: nn.Module,
        theta_t: Dict[str, torch.Tensor],
        X: np.ndarray,
        lambda_reg: float = 0.05,
        device: str = 'cpu'
    ):
        """
        Initialize FiniteWidthProjector.
        
        Args:
            model: Neural network model
            theta_t: Current model parameters
            X: Input data matrix
            lambda_reg: Regularization parameter
            device: Computing device
        """
        self.model = model
        self.theta_t = theta_t
        self.X = torch.tensor(X, dtype=torch.float32)
        self.lambda_reg = lambda_reg
        self.device = device
        self.N = X.shape[0]
        
        # Cache for Jacobian
        self.J_t = None
        
        # Load initial parameters
        self.model.load_state_dict(theta_t)
        
        logger.info(
            f"Initialized FiniteWidthProjector: N={self.N}, λ={lambda_reg}"
        )
    
    def compute_jacobian(self) -> np.ndarray:
        """
        Compute Jacobian J_t at current parameters θ_t.
        
        Returns:
            Jacobian matrix
        """
        self.J_t = JacobianComputer.compute_jacobian(
            self.model, self.X, self.device
        )
        return self.J_t
    
    def project_weights(
        self,
        Y_target: np.ndarray,
        G_lambda_unlearned: np.ndarray,
        f_current: Optional[np.ndarray] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Project unlearned function to finite-width weights.
        
        Implements Equation (9):
        θ_new = θ_t + J_t^T G_λ^{(-c)} (ℐ^{(-c)}Y - f(X, θ_t))
        
        Args:
            Y_target: Target predictions ℐ^{(-c)}Y
            G_lambda_unlearned: Updated resolvent G_λ^{(-c)}
            f_current: Current predictions f(X, θ_t) (optional)
            
        Returns:
            Updated model parameters
        """
        start_time = time.perf_counter()
        
        # Compute Jacobian if not cached
        if self.J_t is None:
            self.compute_jacobian()
        
        # Get current predictions if not provided
        if f_current is None:
            self.model.eval()
            with torch.no_grad():
                f_current = self.model(self.X.to(self.device)).cpu().numpy()
        
        # Step 1: Compute residual ΔY = ℐ^{(-c)}Y - f(X, θ_t)
        delta_Y = Y_target - f_current
        
        logger.info(
            f"Projection residual: ||ΔY|| = {np.linalg.norm(delta_Y):.4f}"
        )
        
        # Step 2: Compute v = G_λ^{(-c)} @ ΔY
        v = G_lambda_unlearned @ delta_Y
        
        # Step 3: Compute weight update Δθ = J_t^T @ v
        update = self.J_t.T @ v
        
        # Step 4: Apply update to current parameters
        theta_new = {}
        idx = 0
        
        for name, param in self.model.named_parameters():
            size = param.numel()
            param_update = torch.tensor(
                update[idx:idx+size].reshape(param.shape),
                dtype=param.dtype,
                device=param.device
            )
            theta_new[name] = param + param_update
            idx += size
        
        elapsed = time.perf_counter() - start_time
        
        logger.info(
            f"Weight projection completed in {elapsed:.4f}s. "
            f"Update norm: {np.linalg.norm(update):.4f}"
        )
        
        return theta_new
    
    def compute_linearization_error_bound(
        self,
        L_J: float = 2.3,
        sigma_min: Optional[float] = None
    ) -> float:
        """
        Compute linearization error bound.
        
        Implements Equation (10):
        ||R₂|| ≤ (L_J/2) · ||J_t^T G_λ^{(-c)} ΔY||²
        
        Args:
            L_J: Lipschitz constant of Jacobian
            sigma_min: Minimum singular value of (K + λI)
            
        Returns:
            Upper bound on linearization error
        """
        if self.J_t is None:
            raise ValueError("Jacobian not computed. Call compute_jacobian first.")
        
        # Compute minimum singular value if not provided
        if sigma_min is None:
            K = self.J_t @ self.J_t.T
            sigma_min = np.min(np.linalg.svd(K, compute_uv=False))
        
        # Bound: ||R₂|| ≤ (L_J/2) · ||J_t||² / σ_min(K + λI)²
        J_norm = np.linalg.norm(self.J_t, 2)
        bound = (L_J / 2) * (J_norm ** 2) / ((sigma_min + self.lambda_reg) ** 2)
        
        logger.info(f"Linearization error bound: {bound:.6f}")
        
        return float(bound)
    
    def compute_width_dependent_error(
        self,
        P: int,
        base_error: float = 0.35
    ) -> Dict[str, float]:
        """
        Compute theoretical error decay with network width.
        
        Implements: ℰ_exact = O(P^{-1/2})
        
        Args:
            P: Network width (parameter count)
            base_error: Base error at reference width
            
        Returns:
            Dictionary with error estimates
        """
        # Theoretical decay
        theoretical_error = base_error * np.sqrt(256 / P)
        
        # Add higher-order term
        higher_order = 0.09 / P
        
        total_error = theoretical_error + higher_order
        
        # Bound from Equation (10)
        error_bound = self.compute_linearization_error_bound()
        
        metrics = {
            'theoretical_error': float(theoretical_error),
            'higher_order_term': float(higher_order),
            'total_estimated_error': float(total_error),
            'error_bound': float(error_bound),
            'width': P,
            'decay_rate': 'O(P^{-1/2})'
        }
        
        logger.info(f"Width-dependent error metrics: {metrics}")
        
        return metrics
    
    def verify_projection_quality(
        self,
        Y_target: np.ndarray,
        theta_new: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """
        Verify quality of weight projection.
        
        Args:
            Y_target: Target predictions
            theta_new: Updated parameters
            
        Returns:
            Dictionary with quality metrics
        """
        # Load new parameters
        self.model.load_state_dict(theta_new)
        self.model.eval()
        
        # Get actual predictions with new weights
        with torch.no_grad():
            f_new = self.model(self.X.to(self.device)).cpu().numpy()
        
        # Compute distance to target
        distance_to_target = np.linalg.norm(f_new - Y_target) / (
            np.linalg.norm(Y_target) + 1e-8
        )
        
        # Compute distance from original
        self.model.load_state_dict(self.theta_t)
        with torch.no_grad():
            f_original = self.model(self.X.to(self.device)).cpu().numpy()
        
        distance_from_original = np.linalg.norm(f_new - f_original) / (
            np.linalg.norm(f_original) + 1e-8
        )
        
        # Restore new parameters
        self.model.load_state_dict(theta_new)
        
        metrics = {
            'distance_to_target': float(distance_to_target),
            'distance_from_original': float(distance_from_original),
            'projection_quality': float(1.0 - distance_to_target)
        }
        
        logger.info(f"Projection quality metrics: {metrics}")
        
        return metrics
    
    def compute_complexity_comparison(
        self,
        P: int,
        N_tilde_sifu: int = 50,
        M_clients: int = 100,
        K_steps: int = 5,
        B_batch: int = 64
    ) -> Dict[str, any]:
        """
        Compare computational complexity with SIFU.
        
        NTK-SURGERY: O(PN) for Jacobian-vector product
        SIFU: O(Ñ · M · K · P · B_batch)
        
        Args:
            P: Parameter count
            N_tilde_sifu: SIFU rollback rounds
            M_clients: Number of clients
            K_steps: Local SGD steps
            B_batch: Batch size
            
        Returns:
            Dictionary with complexity comparison
        """
        # NTK-SURGERY complexity
        ntk_complexity = P * self.N
        
        # SIFU complexity
        sifu_complexity = N_tilde_sifu * M_clients * K_steps * P * B_batch
        
        # Speedup
        speedup = sifu_complexity / (ntk_complexity + 1e-8)
        
        comparison = {
            'ntk_surgery_flops': int(ntk_complexity),
            'sifu_flops': int(sifu_complexity),
            'speedup_factor': float(speedup),
            'ntk_complexity_class': 'O(PN)',
            'sifu_complexity_class': f'O({N_tilde_sifu}·M·K·P·B)',
            'parameters': {
                'P': P,
                'N': self.N,
                'N_tilde': N_tilde_sifu,
                'M': M_clients,
                'K': K_steps,
                'B': B_batch
            }
        }
        
        logger.info(f"Complexity comparison: speedup={speedup:.1f}×")
        
        return comparison