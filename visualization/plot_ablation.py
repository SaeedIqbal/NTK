#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ablation Study Visualization for NTK-SURGERY
Implements component-wise analysis from Section 5.3

This module creates publication-quality plots for:
- Component contribution analysis
- Lambda sensitivity curves
- Width scaling relationships
- Ablation variant comparison
- Performance degradation analysis

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
class AblationConfig:
    """
    Configuration for ablation visualization.
    
    Attributes:
        figsize: Figure size (width, height)
        font_size: Base font size
        title_font_size: Title font size
        label_font_size: Label font size
        tick_font_size: Tick font size
        legend_font_size: Legend font size
        line_width: Line width for plots
        marker_size: Marker size for scatter plots
        color_palette: Color palette for variants
        save_dpi: DPI for saved figures
        format: Save format ('pdf', 'png', 'svg')
        results_dir: Directory for saving results
    """
    figsize: Tuple[float, float] = (24, 5)
    font_size: int = 12
    title_font_size: int = 12
    label_font_size: int = 12
    tick_font_size: int = 12
    legend_font_size: int = 12
    line_width: float = 1.2
    marker_size: int = 6
    color_palette: Dict[str, str] = field(default_factory=lambda: {
        'Full NTK-SURGERY': '#e74c3c',
        'w/o NTK Rep': '#e67e22',
        'w/o Influence Matrix': '#f39c12',
        'w/o Surgery Operator': '#9b59b6',
        'w/o Finite-Width Proj': '#3498db',
        'Weight-Space Baseline': '#7f8c8d'
    })
    save_dpi: int = 600
    format: str = 'pdf'
    results_dir: str = 'results/visualizations/ablation'


class AblationPlotter:
    """
    Creates publication-quality ablation visualizations for NTK-SURGERY.
    
    Implements ablation study visualization with:
    - Lambda sensitivity analysis
    - Width scaling relationships
    - Radar plots for component contributions
    - Utility trade-off scatter plots
    - Scalability analysis
    - NTK alignment bar charts
    - Exactness score distributions
    
    All plots use serif fonts and white backgrounds per ACCV guidelines.
    """
    
    def __init__(self, config: Optional[AblationConfig] = None):
        """
        Initialize AblationPlotter.
        
        Args:
            config: Visualization configuration
        """
        self.config = config if config is not None else AblationConfig()
        self._setup_plotting_style()
        
        # Create results directory
        Path(self.config.results_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info("Initialized AblationPlotter")
    
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
            'axes.linewidth': 1.0,
            'grid.linewidth': 0.5,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'xtick.major.width': 0.8,
            'ytick.major.width': 0.8,
        })
        sns.set_style("whitegrid")
    
    def plot_lambda_sensitivity(
        self,
        results: Dict[str, Dict],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create lambda sensitivity plot.
        
        Args:
            results: Dictionary of ablation results
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.config.figsize)
        
        for variant_name, variant_results in results.items():
            if variant_name not in self.config.color_palette:
                continue
                
            color = self.config.color_palette[variant_name]
            
            # Extract lambda and exactness data
            if 'lambda_values' in variant_results and 'exactness_by_lambda' in variant_results:
                lambdas = variant_results['lambda_values']
                exactness = variant_results['exactness_by_lambda']
                
                if isinstance(exactness, list) and isinstance(lambdas, list):
                    # Plot with error bands if available
                    if 'exactness_std_by_lambda' in variant_results:
                        std = variant_results['exactness_std_by_lambda']
                        ax.fill_between(lambdas, 
                                      np.array(exactness) - np.array(std),
                                      np.array(exactness) + np.array(std),
                                      alpha=0.15, color=color)
                    
                    ax.plot(lambdas, exactness, 'o-', color=color,
                           linewidth=self.config.line_width)
        
        ax.set_xscale('log')
        ax.set_xlabel('Lambda', fontsize=self.config.label_font_size)
        ax.set_ylabel('Exactness', fontsize=self.config.label_font_size)
        ax.set_title('Lambda Sensitivity', fontsize=self.config.title_font_size)
        ax.grid(True, linestyle='--', alpha=0.4)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=self.config.save_dpi)
            logger.info(f"Saved lambda sensitivity plot to {save_path}")
        
        return fig
    
    def plot_width_scaling(
        self,
        results: Dict[str, Dict],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create width scaling plot.
        
        Args:
            results: Dictionary of ablation results
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.config.figsize)
        
        for variant_name, variant_results in results.items():
            if variant_name not in self.config.color_palette:
                continue
                
            color = self.config.color_palette[variant_name]
            
            # Extract width and exactness data
            if 'width_values' in variant_results and 'exactness_by_width' in variant_results:
                widths = variant_results['width_values']
                exactness = variant_results['exactness_by_width']
                
                if isinstance(exactness, list) and isinstance(widths, list):
                    # Plot with error bands if available
                    if 'exactness_std_by_width' in variant_results:
                        std = variant_results['exactness_std_by_width']
                        ax.fill_between(widths,
                                      np.array(exactness) - np.array(std),
                                      np.array(exactness) + np.array(std),
                                      alpha=0.15, color=color)
                    
                    ax.loglog(widths, exactness, 's-', color=color,
                             linewidth=self.config.line_width)
        
        # Add reference O(P^{-1/2}) line
        if widths:
            ref_x = np.array([widths[0], widths[-1]])
            ref_y = 0.95 * (ref_x / widths[0]) ** (-0.5)
            ax.loglog(ref_x, ref_y, 'k--', linewidth=1, label='Theoretical O(P^{-1/2})')
        
        ax.set_xlabel('Width', fontsize=self.config.label_font_size)
        ax.set_ylabel('Exactness', fontsize=self.config.label_font_size)
        ax.set_title('Width Scaling', fontsize=self.config.title_font_size)
        ax.grid(True, which='both', linestyle='--', alpha=0.4)
        ax.legend(loc='upper right', fontsize=self.config.legend_font_size, frameon=False)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=self.config.save_dpi)
            logger.info(f"Saved width scaling plot to {save_path}")
        
        return fig
    
    def plot_radar_chart(
        self,
        results: Dict[str, Dict],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create radar chart for component contributions.
        
        Args:
            results: Dictionary of ablation results
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        from math import pi
        
        # Categories for radar chart
        categories = ['Exactness', 'Forget', 'Retain', 'Align', 'Efficiency']
        N = len(categories)
        angles = [n / N * 2 * pi for n in range(N)]
        angles += angles[:1]  # Close the loop
        
        fig, ax = plt.subplots(figsize=(self.config.figsize[0]/3, self.config.figsize[1]), 
                              subplot_kw=dict(projection='polar'))
        
        for variant_name, variant_results in results.items():
            if variant_name not in self.config.color_palette:
                continue
                
            # Extract normalized metrics
            values = []
            
            # Exactness Score
            exactness = variant_results.get('exactness_score', 0.0)
            if isinstance(exactness, list):
                exactness = np.mean(exactness)
            values.append(exactness)
            
            # Forget Accuracy (convert to score where lower is better)
            forget = variant_results.get('forget_accuracy', 50.0)
            if isinstance(forget, list):
                forget = np.mean(forget)
            forget_score = 1.0 - min(forget / 50.0, 1.0)
            values.append(forget_score)
            
            # Retain Accuracy
            retain = variant_results.get('retain_accuracy', 0.0)
            if isinstance(retain, list):
                retain = np.mean(retain)
            values.append(retain / 100.0)
            
            # NTK Alignment
            alignment = variant_results.get('ntk_alignment', 0.0)
            if isinstance(alignment, list):
                alignment = np.mean(alignment)
            values.append(alignment)
            
            # Efficiency (speedup vs scratch)
            speedup = variant_results.get('speedup_vs_scratch', 1.0)
            if isinstance(speedup, list):
                speedup = np.mean(speedup)
            efficiency = min(speedup / 100.0, 1.0)
            values.append(efficiency)
            
            # Close the loop
            values += values[:1]
            
            # Plot
            color = self.config.color_palette[variant_name]
            ax.plot(angles, values, 'o-', linewidth=2, markersize=3,
                   color=color, label=variant_name)
            ax.fill(angles, values, alpha=0.2, color=color)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([cat[:8] for cat in categories], fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_title('Component\nContribution', pad=20, fontsize=11)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=self.config.save_dpi)
            logger.info(f"Saved radar chart to {save_path}")
        
        return fig
    
    def plot_utility_tradeoff(
        self,
        results: Dict[str, Dict],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create utility trade-off scatter plot.
        
        Args:
            results: Dictionary of ablation results
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.config.figsize)
        
        for variant_name, variant_results in results.items():
            if variant_name not in self.config.color_palette:
                continue
                
            color = self.config.color_palette[variant_name]
            
            forget = variant_results.get('forget_accuracy', [])
            retain = variant_results.get('retain_accuracy', [])
            
            if isinstance(forget, list) and isinstance(retain, list):
                ax.scatter(forget, retain, c=color, s=6, alpha=0.7)
            elif isinstance(forget, (int, float)) and isinstance(retain, (int, float)):
                ax.scatter([forget], [retain], c=color, s=6, alpha=0.7)
        
        ax.set_xlabel('Forget', fontsize=self.config.label_font_size)
        ax.set_ylabel('Retain', fontsize=self.config.label_font_size)
        ax.set_title('Utility Trade-off', fontsize=self.config.title_font_size)
        ax.grid(True, linestyle='--', alpha=0.4)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=self.config.save_dpi)
            logger.info(f"Saved utility trade-off plot to {save_path}")
        
        return fig
    
    def plot_scalability(
        self,
        results: Dict[str, Dict],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create scalability plot.
        
        Args:
            results: Dictionary of ablation results
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.config.figsize)
        
        for variant_name, variant_results in results.items():
            if variant_name not in self.config.color_palette:
                continue
                
            color = self.config.color_palette[variant_name]
            
            # Extract client count and time data
            if 'client_counts' in variant_results and 'time_by_clients' in variant_results:
                clients = variant_results['client_counts']
                times = variant_results['time_by_clients']
                
                if isinstance(times, list) and isinstance(clients, list):
                    ax.loglog(clients, times, '^-', color=color,
                             linewidth=self.config.line_width)
        
        # Add reference scaling lines
        if clients:
            x_ref = np.array([clients[0], clients[-1]])
            ax.loglog(x_ref, 0.5*(x_ref/clients[0])**2, 'g:', linewidth=1, label='O(M²)')
            ax.loglog(x_ref, 2*(x_ref/clients[0])**3, 'r:', linewidth=1, label='O(M³)')
        
        ax.set_xlabel('Clients', fontsize=self.config.label_font_size)
        ax.set_ylabel('Time (s)', fontsize=self.config.label_font_size)
        ax.set_title('Scalability', fontsize=self.config.title_font_size)
        ax.grid(True, which='both', linestyle='--', alpha=0.4)
        ax.legend(loc='upper left', fontsize=self.config.legend_font_size, frameon=False)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=self.config.save_dpi)
            logger.info(f"Saved scalability plot to {save_path}")
        
        return fig
    
    def plot_ntk_alignment_bar(
        self,
        results: Dict[str, Dict],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create NTK alignment bar chart.
        
        Args:
            results: Dictionary of ablation results
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.config.figsize)
        
        variants = []
        alignments = []
        colors = []
        
        for variant_name, variant_results in results.items():
            if variant_name not in self.config.color_palette:
                continue
                
            alignment = variant_results.get('ntk_alignment', 0.0)
            if isinstance(alignment, list):
                alignment = np.mean(alignment)
            
            variants.append(variant_name)
            alignments.append(alignment)
            colors.append(self.config.color_palette[variant_name])
        
        ax.bar(np.arange(len(variants)), alignments, color=colors, width=0.5)
        ax.set_ylabel('Alignment', fontsize=self.config.label_font_size)
        ax.set_title('NTK Alignment', fontsize=self.config.title_font_size)
        ax.set_ylim(0, 1)
        ax.set_xticks(np.arange(len(variants)))
        ax.set_xticklabels([v[:10] for v in variants], rotation=45, ha='right', fontsize=8)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=self.config.save_dpi)
            logger.info(f"Saved NTK alignment bar chart to {save_path}")
        
        return fig
    
    def plot_exactness_boxplot(
        self,
        results: Dict[str, Dict],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create exactness score box plot.
        
        Args:
            results: Dictionary of ablation results
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.config.figsize)
        
        data = []
        variants = []
        colors = []
        
        for variant_name, variant_results in results.items():
            if variant_name not in self.config.color_palette:
                continue
                
            exactness = variant_results.get('exactness_score', [])
            if not isinstance(exactness, list):
                exactness = [exactness]
            
            data.append(exactness)
            variants.append(variant_name)
            colors.append(self.config.color_palette[variant_name])
        
        bp = ax.boxplot(data, patch_artist=True)
        
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        ax.set_ylabel('Exactness', fontsize=self.config.label_font_size)
        ax.set_title('Exactness Distribution', fontsize=self.config.title_font_size)
        ax.set_ylim(0, 1)
        ax.set_xticks(np.arange(1, len(variants)+1))
        ax.set_xticklabels([v[:10] for v in variants], rotation=45, ha='right', fontsize=8)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=self.config.save_dpi)
            logger.info(f"Saved exactness box plot to {save_path}")
        
        return fig


def plot_ablation_study(
    results_file: str,
    output_dir: str = 'results/visualizations/ablation',
    config: Optional[AblationConfig] = None
) -> Dict[str, str]:
    """
    Plot ablation study from results file.
    
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
    plotter = AblationPlotter(config)
    plotter.config.results_dir = output_dir
    
    # Create plots
    plot_paths = {}
    
    plot_paths['lambda_sensitivity'] = Path(output_dir) / 'lambda_sensitivity.pdf'
    plotter.plot_lambda_sensitivity(results, str(plot_paths['lambda_sensitivity']))
    
    plot_paths['width_scaling'] = Path(output_dir) / 'width_scaling.pdf'
    plotter.plot_width_scaling(results, str(plot_paths['width_scaling']))
    
    plot_paths['radar_chart'] = Path(output_dir) / 'radar_chart.pdf'
    plotter.plot_radar_chart(results, str(plot_paths['radar_chart']))
    
    plot_paths['utility_tradeoff'] = Path(output_dir) / 'utility_tradeoff.pdf'
    plotter.plot_utility_tradeoff(results, str(plot_paths['utility_tradeoff']))
    
    plot_paths['scalability'] = Path(output_dir) / 'scalability.pdf'
    plotter.plot_scalability(results, str(plot_paths['scalability']))
    
    plot_paths['ntk_alignment'] = Path(output_dir) / 'ntk_alignment.pdf'
    plotter.plot_ntk_alignment_bar(results, str(plot_paths['ntk_alignment']))
    
    plot_paths['exactness_boxplot'] = Path(output_dir) / 'exactness_boxplot.pdf'
    plotter.plot_exactness_boxplot(results, str(plot_paths['exactness_boxplot']))
    
    return plot_paths


def create_ablation_summary(
    results: Dict[str, Dict],
    output_file: str
) -> Dict[str, Any]:
    """
    Create ablation study summary.
    
    Args:
        results: Dictionary of ablation results
        output_file: Path to save summary
        
    Returns:
        Summary dictionary
    """
    summary = {
        'variants': list(results.keys()),
        'metrics': {
            'exactness_score': {},
            'forget_accuracy': {},
            'retain_accuracy': {},
            'ntk_alignment': {},
            'speedup_vs_scratch': {}
        },
        'best_variant': {},
        'component_contributions': {}
    }
    
    # Extract metrics
    for metric in ['exactness_score', 'forget_accuracy', 'retain_accuracy', 'ntk_alignment']:
        values = {}
        for variant, result in results.items():
            val = result.get(metric, 0.0)
            if isinstance(val, list):
                val = np.mean(val)
            values[variant] = val
        
        summary['metrics'][metric] = values
        
        # Best variant (min for forget, max for others)
        if metric == 'forget_accuracy':
            best_variant = min(values, key=values.get)
        else:
            best_variant = max(values, key=values.get)
        
        summary['best_variant'][metric] = best_variant
    
    # Component contribution analysis
    full_result = results.get('Full NTK-SURGERY', {})
    full_exactness = full_result.get('exactness_score', 0.0)
    if isinstance(full_exactness, list):
        full_exactness = np.mean(full_exactness)
    
    for variant, result in results.items():
        if variant == 'Full NTK-SURGERY':
            continue
            
        variant_exactness = result.get('exactness_score', 0.0)
        if isinstance(variant_exactness, list):
            variant_exactness = np.mean(variant_exactness)
        
        contribution = full_exactness - variant_exactness
        summary['component_contributions'][variant] = contribution
    
    # Save summary
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    return summary