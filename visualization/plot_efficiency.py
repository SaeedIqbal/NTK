#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Efficiency Visualization for NTK-SURGERY
Implements efficiency metrics from Section 5.2

This module creates publication-quality plots for:
- Communication rounds comparison
- Server compute time analysis
- FLOPs and computational complexity
- Speedup factors vs baseline methods
- Scalability with client count

All visualizations follow ACCV formatting guidelines with serif fonts.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass, field
from pathlib import Path
import json

logger = logging.getLogger(__name__)


@dataclass
class EfficiencyConfig:
    """
    Configuration for efficiency visualization.
    
    Attributes:
        figsize: Figure size (width, height)
        font_size: Base font size
        title_font_size: Title font size
        label_font_size: Label font size
        tick_font_size: Tick font size
        legend_font_size: Legend font size
        line_width: Line width for plots
        bar_width: Bar width for bar charts
        color_palette: Color palette for methods
        save_dpi: DPI for saved figures
        format: Save format ('pdf', 'png', 'svg')
        results_dir: Directory for saving results
        log_scale: Whether to use log scale for time plots
    """
    figsize: Tuple[float, float] = (16, 10)
    font_size: int = 12
    title_font_size: int = 14
    label_font_size: int = 12
    tick_font_size: int = 10
    legend_font_size: int = 10
    line_width: float = 1.8
    bar_width: float = 0.6
    color_palette: Dict[str, str] = field(default_factory=lambda: {
        'NTK-SURGERY': '#e74c3c',
        'SIFU': '#9b59b6',
        'FedEraser': '#3498db',
        'Fine-Tuning': '#e67e22',
        'Scratch': '#2c3e50',
        'FedSGD': '#f39c12'
    })
    save_dpi: int = 600
    format: str = 'pdf'
    results_dir: str = 'results/visualizations/efficiency'
    log_scale: bool = True


class EfficiencyPlotter:
    """
    Creates publication-quality efficiency visualizations for NTK-SURGERY.
    
    Implements efficiency metrics visualization with:
    - Bar charts of communication rounds
    - Log-scale server time comparison
    - FLOPs analysis across methods
    - Speedup factor visualization
    - Client scalability plots
    
    All plots use serif fonts and white backgrounds per ACCV guidelines.
    """
    
    def __init__(self, config: Optional[EfficiencyConfig] = None):
        """
        Initialize EfficiencyPlotter.
        
        Args:
            config: Visualization configuration
        """
        self.config = config if config is not None else EfficiencyConfig()
        self._setup_plotting_style()
        
        # Create results directory
        Path(self.config.results_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info("Initialized EfficiencyPlotter")
    
    def _setup_plotting_style(self):
        """Setup matplotlib style for ACCV publication."""
        plt.rcParams.update({
            'font.size': self.config.font_size,
            'font.family': 'serif',
            'font.serif': ['DejaVu Serif'],
            'axes.labelsize': self.config.label_font_size,
            'axes.titlesize': self.config.title_font_size,
            'xtick.labelsize': self.config.tick_font_size,
            'ytick.labelsize': self.config.tick_font_size,
            'legend.fontsize': self.config.legend_font_size,
            'figure.dpi': self.config.save_dpi,
            'savefig.dpi': self.config.save_dpi,
            'savefig.bbox': 'tight',
            'savefig.format': self.config.format,
            'lines.linewidth': self.config.line_width,
            'axes.linewidth': 0.8,
            'grid.linewidth': 0.5,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'xtick.major.width': 0.8,
            'ytick.major.width': 0.8,
        })
        sns.set_style("whitegrid")
    
    def plot_communication_rounds(
        self,
        results: Dict[str, Dict],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create bar chart of communication rounds.
        
        Args:
            results: Dictionary of method results
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.config.figsize)
        
        methods = []
        rounds = []
        colors = []
        
        for method_name, method_results in results.items():
            if method_name not in self.config.color_palette:
                continue
                
            comm_rounds = method_results.get('communication_rounds', 0)
            if isinstance(comm_rounds, list):
                comm_rounds = np.mean(comm_rounds)
            
            methods.append(method_name)
            rounds.append(comm_rounds)
            colors.append(self.config.color_palette[method_name])
        
        bars = ax.bar(methods, rounds, color=colors, width=self.config.bar_width)
        
        # Add value labels on bars
        for bar, rounds_val in zip(bars, rounds):
            ax.text(
                bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{rounds_val:.0f}', ha='center', va='bottom',
                fontsize=self.config.tick_font_size
            )
        
        ax.set_ylabel('Communication Rounds ↓', fontsize=self.config.label_font_size)
        ax.set_title('Communication Efficiency', fontsize=self.config.title_font_size, pad=12)
        ax.set_yscale('log' if self.config.log_scale else 'linear')
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        plt.xticks(rotation=45, ha='right')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=self.config.save_dpi)
            logger.info(f"Saved communication rounds plot to {save_path}")
        
        return fig
    
    def plot_server_time(
        self,
        results: Dict[str, Dict],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create bar chart of server compute time.
        
        Args:
            results: Dictionary of method results
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.config.figsize)
        
        methods = []
        times = []
        colors = []
        
        for method_name, method_results in results.items():
            if method_name not in self.config.color_palette:
                continue
                
            server_time = method_results.get('unlearning_time', 0.0)
            if isinstance(server_time, list):
                server_time = np.mean(server_time)
            
            methods.append(method_name)
            times.append(server_time)
            colors.append(self.config.color_palette[method_name])
        
        bars = ax.bar(methods, times, color=colors, width=self.config.bar_width)
        
        # Add value labels on bars
        for bar, time_val in zip(bars, times):
            ax.text(
                bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{time_val:.1f}s', ha='center', va='bottom',
                fontsize=self.config.tick_font_size
            )
        
        ax.set_ylabel('Server Time (seconds) ↓', fontsize=self.config.label_font_size)
        ax.set_title('Server Compute Time', fontsize=self.config.title_font_size, pad=12)
        ax.set_yscale('log' if self.config.log_scale else 'linear')
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        plt.xticks(rotation=45, ha='right')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=self.config.save_dpi)
            logger.info(f"Saved server time plot to {save_path}")
        
        return fig
    
    def plot_speedup_factors(
        self,
        results: Dict[str, Dict],
        baseline: str = 'Scratch',
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create bar chart of speedup factors.
        
        Args:
            results: Dictionary of method results
            baseline: Baseline method for comparison
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.config.figsize)
        
        methods = []
        speedups = []
        colors = []
        
        baseline_time = results.get(baseline, {}).get('unlearning_time', 1.0)
        
        for method_name, method_results in results.items():
            if method_name not in self.config.color_palette or method_name == baseline:
                continue
                
            method_time = method_results.get('unlearning_time', 1.0)
            if isinstance(method_time, list):
                method_time = np.mean(method_time)
            
            speedup = baseline_time / (method_time + 1e-8)
            
            methods.append(method_name)
            speedups.append(speedup)
            colors.append(self.config.color_palette[method_name])
        
        bars = ax.bar(methods, speedups, color=colors, width=self.config.bar_width)
        
        # Add value labels on bars
        for bar, speedup_val in zip(bars, speedups):
            ax.text(
                bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{speedup_val:.0f}×', ha='center', va='bottom',
                fontsize=self.config.tick_font_size
            )
        
        ax.axhline(y=1, color='crimson', linestyle='--', linewidth=1.2,
                  label=f'{baseline} Baseline')
        ax.set_ylabel('Speedup Factor', fontsize=self.config.label_font_size)
        ax.set_title(f'Computational Efficiency Gain\nvs. {baseline}',
                    fontsize=self.config.title_font_size, pad=12)
        ax.set_yscale('log' if self.config.log_scale else 'linear')
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        ax.legend(fontsize=self.config.legend_font_size, frameon=False)
        plt.xticks(rotation=45, ha='right')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=self.config.save_dpi)
            logger.info(f"Saved speedup factors plot to {save_path}")
        
        return fig
    
    def plot_client_scalability(
        self,
        results: Dict[str, Dict],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create line plot of client scalability.
        
        Args:
            results: Dictionary of method results with client counts
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.config.figsize)
        
        # Group results by method and client count
        method_data = {}
        
        for method_name, method_results in results.items():
            if method_name not in self.config.color_palette:
                continue
                
            client_counts = method_results.get('client_counts', [])
            times = method_results.get('times_by_clients', [])
            
            if client_counts and times:
                method_data[method_name] = {
                    'clients': client_counts,
                    'times': times,
                    'color': self.config.color_palette[method_name]
                }
        
        # Plot each method
        for method_name, data in method_data.items():
            ax.loglog(data['clients'], data['times'], 'o-',
                     color=data['color'], linewidth=self.config.line_width,
                     markersize=6, label=method_name)
        
        ax.set_xlabel('Number of Clients', fontsize=self.config.label_font_size)
        ax.set_ylabel('Server Time (seconds)', fontsize=self.config.label_font_size)
        ax.set_title('Scalability with Client Count', fontsize=self.config.title_font_size, pad=12)
        ax.grid(True, which='both', linestyle='--', alpha=0.4)
        ax.legend(loc='upper left', frameon=False)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=self.config.save_dpi)
            logger.info(f"Saved client scalability plot to {save_path}")
        
        return fig
    
    def plot_flops_comparison(
        self,
        results: Dict[str, Dict],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create bar chart of FLOPs comparison.
        
        Args:
            results: Dictionary of method results
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.config.figsize)
        
        methods = []
        flops = []
        colors = []
        
        for method_name, method_results in results.items():
            if method_name not in self.config.color_palette:
                continue
                
            flop_count = method_results.get('flops', 0)
            if isinstance(flop_count, list):
                flop_count = np.mean(flop_count)
            
            methods.append(method_name)
            flops.append(flop_count)
            colors.append(self.config.color_palette[method_name])
        
        bars = ax.bar(methods, flops, color=colors, width=self.config.bar_width)
        
        # Add value labels on bars (in scientific notation)
        for bar, flop_val in zip(bars, flops):
            if flop_val >= 1e9:
                label = f'{flop_val/1e9:.1f}G'
            elif flop_val >= 1e6:
                label = f'{flop_val/1e6:.1f}M'
            else:
                label = f'{flop_val:.0f}'
            
            ax.text(
                bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05*max(flops),
                label, ha='center', va='bottom',
                fontsize=self.config.tick_font_size
            )
        
        ax.set_ylabel('FLOPs', fontsize=self.config.label_font_size)
        ax.set_title('Computational Complexity', fontsize=self.config.title_font_size, pad=12)
        ax.set_yscale('log')
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        plt.xticks(rotation=45, ha='right')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=self.config.save_dpi)
            logger.info(f"Saved FLOPs comparison plot to {save_path}")
        
        return fig


def plot_efficiency_metrics(
    results_file: str,
    output_dir: str = 'results/visualizations/efficiency',
    config: Optional[EfficiencyConfig] = None
) -> Dict[str, str]:
    """
    Plot efficiency metrics from results file.
    
    Args:
        results_file: Path to results JSON file
        output_dir: Output directory for plots
        config: Visualization configuration
        
    Returns:
        Dictionary of saved plot paths
    """
    # Load results
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Create plotter
    plotter = EfficiencyPlotter(config)
    plotter.config.results_dir = output_dir
    
    # Create plots
    plot_paths = {}
    
    plot_paths['comm_rounds'] = Path(output_dir) / 'communication_rounds.pdf'
    plotter.plot_communication_rounds(results, str(plot_paths['comm_rounds']))
    
    plot_paths['server_time'] = Path(output_dir) / 'server_time.pdf'
    plotter.plot_server_time(results, str(plot_paths['server_time']))
    
    plot_paths['speedup'] = Path(output_dir) / 'speedup_factors.pdf'
    plotter.plot_speedup_factors(results, save_path=str(plot_paths['speedup']))
    
    plot_paths['scalability'] = Path(output_dir) / 'client_scalability.pdf'
    plotter.plot_client_scalability(results, str(plot_paths['scalability']))
    
    plot_paths['flops'] = Path(output_dir) / 'flops_comparison.pdf'
    plotter.plot_flops_comparison(results, str(plot_paths['flops']))
    
    return plot_paths


def create_efficiency_comparison(
    results: Dict[str, Dict],
    output_file: str
) -> Dict[str, Any]:
    """
    Create efficiency comparison summary.
    
    Args:
        results: Dictionary of method results
        output_file: Path to save summary
        
    Returns:
        Summary dictionary
    """
    summary = {
        'methods': list(results.keys()),
        'efficiency_metrics': {
            'communication_rounds': {},
            'server_time': {},
            'flops': {},
            'speedup_vs_scratch': {}
        },
        'best_methods': {}
    }
    
    # Extract metrics
    for metric in ['communication_rounds', 'server_time', 'flops']:
        values = {}
        for method, result in results.items():
            val = result.get(metric, 0.0)
            if isinstance(val, list):
                val = np.mean(val)
            values[method] = val
        
        summary['efficiency_metrics'][metric] = values
        
        # Best method (minimum for all efficiency metrics)
        best_method = min(values, key=values.get)
        summary['best_methods'][metric] = best_method
    
    # Speedup vs Scratch
    scratch_time = results.get('Scratch', {}).get('server_time', 1.0)
    speedups = {}
    for method, result in results.items():
        if method == 'Scratch':
            continue
        method_time = result.get('server_time', 1.0)
        if isinstance(method_time, list):
            method_time = np.mean(method_time)
        speedups[method] = scratch_time / (method_time + 1e-8)
    
    summary['efficiency_metrics']['speedup_vs_scratch'] = speedups
    summary['best_methods']['speedup_vs_scratch'] = max(speedups, key=speedups.get)
    
    # Save summary
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    return summary