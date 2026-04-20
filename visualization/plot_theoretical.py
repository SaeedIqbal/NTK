#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Theoretical Metrics Visualization for NTK-SURGERY
Implements theoretical compliance metrics from Section 5.2

This module creates publication-quality plots for:
- NTK Alignment Score across datasets
- Sensitivity Bound Ratio comparison
- Condition number analysis
- Spectral properties of influence matrices
- Theoretical vs empirical correlation

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
class TheoreticalConfig:
    """
    Configuration for theoretical visualization.
    
    Attributes:
        figsize: Figure size (width, height)
        font_size: Base font size
        title_font_size: Title font size
        label_font_size: Label font size
        tick_font_size: Tick font size
        legend_font_size: Legend font size
        line_width: Line width for plots
        marker_size: Marker size for scatter plots
        color_palette: Color palette for methods
        save_dpi: DPI for saved figures
        format: Save format ('pdf', 'png', 'svg')
        results_dir: Directory for saving results
        heatmap_cmap: Colormap for heatmaps
    """
    figsize: Tuple[float, float] = (16, 10)
    font_size: int = 12
    title_font_size: int = 14
    label_font_size: int = 12
    tick_font_size: int = 10
    legend_font_size: int = 10
    line_width: float = 1.8
    marker_size: int = 6
    color_palette: Dict[str, str] = field(default_factory=lambda: {
        'NTK-SURGERY': '#e74c3c',
        'SIFU': '#9b59b6',
        'FedEraser': '#3498db',
        'Fine-Tuning': '#e67e22',
        'Scratch': '#2c3e50'
    })
    save_dpi: int = 600
    format: str = 'pdf'
    results_dir: str = 'results/visualizations/theoretical'
    heatmap_cmap: str = 'RdYlBu_r'


class TheoreticalPlotter:
    """
    Creates publication-quality theoretical visualizations for NTK-SURGERY.
    
    Implements theoretical compliance visualization with:
    - NTK Alignment Score bar charts
    - Sensitivity Bound Ratio comparison
    - Heatmap of theoretical metrics
    - Correlation between theoretical and empirical metrics
    - Spectral analysis plots
    
    All plots use serif fonts and white backgrounds per ACCV guidelines.
    """
    
    def __init__(self, config: Optional[TheoreticalConfig] = None):
        """
        Initialize TheoreticalPlotter.
        
        Args:
            config: Visualization configuration
        """
        self.config = config if config is not None else TheoreticalConfig()
        self._setup_plotting_style()
        
        # Create results directory
        Path(self.config.results_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info("Initialized TheoreticalPlotter")
    
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
    
    def plot_ntk_alignment(
        self,
        results: Dict[str, Dict],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create bar chart of NTK Alignment Scores.
        
        Args:
            results: Dictionary of method results
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.config.figsize)
        
        methods = []
        alignments = []
        colors = []
        
        for method_name, method_results in results.items():
            if method_name not in self.config.color_palette:
                continue
                
            alignment = method_results.get('ntk_alignment', 0.0)
            if isinstance(alignment, list):
                alignment = np.mean(alignment)
            
            methods.append(method_name)
            alignments.append(alignment)
            colors.append(self.config.color_palette[method_name])
        
        bars = ax.bar(methods, alignments, color=colors, width=0.6)
        
        # Add value labels on bars
        for bar, align_val in zip(bars, alignments):
            ax.text(
                bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{align_val:.3f}', ha='center', va='bottom',
                fontsize=self.config.tick_font_size
            )
        
        ax.set_ylabel('NTK Alignment Score ↑', fontsize=self.config.label_font_size)
        ax.set_title('NTK Alignment Across Methods', fontsize=self.config.title_font_size, pad=12)
        ax.set_ylim(0, 1.05)
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        plt.xticks(rotation=45, ha='right')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=self.config.save_dpi)
            logger.info(f"Saved NTK alignment plot to {save_path}")
        
        return fig
    
    def plot_sensitivity_ratio(
        self,
        results: Dict[str, Dict],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create bar chart of Sensitivity Bound Ratios.
        
        Args:
            results: Dictionary of method results
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.config.figsize)
        
        methods = []
        ratios = []
        colors = []
        
        for method_name, method_results in results.items():
            if method_name not in self.config.color_palette:
                continue
                
            ratio = method_results.get('sensitivity_bound_ratio', 0.0)
            if isinstance(ratio, list):
                ratio = np.mean(ratio)
            
            methods.append(method_name)
            ratios.append(ratio)
            colors.append(self.config.color_palette[method_name])
        
        bars = ax.bar(methods, ratios, color=colors, width=0.6)
        
        # Add value labels on bars
        for bar, ratio_val in zip(bars, ratios):
            ax.text(
                bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'{ratio_val:.0f}×', ha='center', va='bottom',
                fontsize=self.config.tick_font_size
            )
        
        ax.set_ylabel('Sensitivity Bound Ratio ↑', fontsize=self.config.label_font_size)
        ax.set_title('Sensitivity Bound Ratio (ζ_SIFU/ζ_NTK)', 
                    fontsize=self.config.title_font_size, pad=12)
        ax.set_yscale('log')
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        plt.xticks(rotation=45, ha='right')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=self.config.save_dpi)
            logger.info(f"Saved sensitivity ratio plot to {save_path}")
        
        return fig
    
    def plot_theoretical_heatmap(
        self,
        results: Dict[str, Dict],
        metrics: List[str] = ['ntk_alignment', 'sensitivity_bound_ratio', 'condition_number'],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create heatmap of theoretical metrics.
        
        Args:
            results: Dictionary of method results
            metrics: List of metrics to include
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        # Prepare data matrix
        methods = []
        data_matrix = []
        
        for method_name, method_results in results.items():
            if method_name not in self.config.color_palette:
                continue
                
            methods.append(method_name)
            row = []
            
            for metric in metrics:
                val = method_results.get(metric, 0.0)
                if isinstance(val, list):
                    val = np.mean(val)
                row.append(val)
            
            data_matrix.append(row)
        
        if not data_matrix:
            return plt.figure()
        
        data_array = np.array(data_matrix)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=self.config.figsize)
        
        im = ax.imshow(data_array, cmap=self.config.heatmap_cmap, aspect='auto')
        
        # Add annotations
        for i in range(len(methods)):
            for j in range(len(metrics)):
                val = data_array[i, j]
                color = 'black' if val < np.max(data_array)/2 else 'white'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                       fontsize=self.config.tick_font_size, color=color)
        
        ax.set_xticks(range(len(metrics)))
        ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics],
                          fontsize=self.config.tick_font_size)
        ax.set_yticks(range(len(methods)))
        ax.set_yticklabels(methods, fontsize=self.config.tick_font_size)
        ax.set_title('Theoretical Metrics Heatmap', fontsize=self.config.title_font_size, pad=12)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Metric Value', rotation=270, labelpad=15, fontsize=self.config.label_font_size)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=self.config.save_dpi)
            logger.info(f"Saved theoretical heatmap to {save_path}")
        
        return fig
    
    def plot_theoretical_vs_empirical(
        self,
        results: Dict[str, Dict],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create scatter plot of theoretical vs empirical metrics.
        
        Args:
            results: Dictionary of method results
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.config.figsize)
        
        for method_name, method_results in results.items():
            if method_name not in self.config.color_palette:
                continue
                
            theoretical = method_results.get('ntk_alignment', 0.0)
            empirical = method_results.get('exactness_score', 0.0)
            
            if isinstance(theoretical, list):
                theoretical = np.mean(theoretical)
            if isinstance(empirical, list):
                empirical = np.mean(empirical)
            
            ax.scatter(
                theoretical, empirical,
                c=self.config.color_palette[method_name],
                s=self.config.marker_size*10,
                alpha=0.8, label=method_name
            )
        
        # Add diagonal line
        min_val = min(ax.get_xlim()[0], ax.get_ylim()[0])
        max_val = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, linewidth=1)
        
        ax.set_xlabel('NTK Alignment (Theoretical) ↑', fontsize=self.config.label_font_size)
        ax.set_ylabel('Exactness Score (Empirical) ↑', fontsize=self.config.label_font_size)
        ax.set_title('Theoretical vs Empirical Correlation', 
                    fontsize=self.config.title_font_size, pad=12)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.legend(loc='lower right', frameon=False)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=self.config.save_dpi)
            logger.info(f"Saved theoretical vs empirical plot to {save_path}")
        
        return fig
    
    def plot_condition_number_analysis(
        self,
        results: Dict[str, Dict],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create bar chart of condition numbers.
        
        Args:
            results: Dictionary of method results
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.config.figsize)
        
        methods = []
        cond_numbers = []
        colors = []
        
        for method_name, method_results in results.items():
            if method_name not in self.config.color_palette:
                continue
                
            cond_num = method_results.get('condition_number', 0.0)
            if isinstance(cond_num, list):
                cond_num = np.mean(cond_num)
            
            methods.append(method_name)
            cond_numbers.append(cond_num)
            colors.append(self.config.color_palette[method_name])
        
        bars = ax.bar(methods, cond_numbers, color=colors, width=0.6)
        
        # Add value labels on bars (scientific notation)
        for bar, cond_val in zip(bars, cond_numbers):
            if cond_val >= 1e6:
                label = f'{cond_val/1e6:.1f}M'
            elif cond_val >= 1e3:
                label = f'{cond_val/1e3:.1f}K'
            else:
                label = f'{cond_val:.0f}'
            
            ax.text(
                bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05*max(cond_numbers),
                label, ha='center', va='bottom',
                fontsize=self.config.tick_font_size
            )
        
        ax.set_ylabel('Condition Number', fontsize=self.config.label_font_size)
        ax.set_title('Numerical Stability Analysis', fontsize=self.config.title_font_size, pad=12)
        ax.set_yscale('log')
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        plt.xticks(rotation=45, ha='right')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=self.config.save_dpi)
            logger.info(f"Saved condition number plot to {save_path}")
        
        return fig


def plot_theoretical_metrics(
    results_file: str,
    output_dir: str = 'results/visualizations/theoretical',
    config: Optional[TheoreticalConfig] = None
) -> Dict[str, str]:
    """
    Plot theoretical metrics from results file.
    
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
    plotter = TheoreticalPlotter(config)
    plotter.config.results_dir = output_dir
    
    # Create plots
    plot_paths = {}
    
    plot_paths['ntk_alignment'] = Path(output_dir) / 'ntk_alignment.pdf'
    plotter.plot_ntk_alignment(results, str(plot_paths['ntk_alignment']))
    
    plot_paths['sensitivity_ratio'] = Path(output_dir) / 'sensitivity_ratio.pdf'
    plotter.plot_sensitivity_ratio(results, str(plot_paths['sensitivity_ratio']))
    
    plot_paths['theoretical_heatmap'] = Path(output_dir) / 'theoretical_heatmap.pdf'
    plotter.plot_theoretical_heatmap(results, save_path=str(plot_paths['theoretical_heatmap']))
    
    plot_paths['theoretical_vs_empirical'] = Path(output_dir) / 'theoretical_vs_empirical.pdf'
    plotter.plot_theoretical_vs_empirical(results, str(plot_paths['theoretical_vs_empirical']))
    
    plot_paths['condition_number'] = Path(output_dir) / 'condition_number.pdf'
    plotter.plot_condition_number_analysis(results, str(plot_paths['condition_number']))
    
    return plot_paths


def create_theoretical_analysis(
    results: Dict[str, Dict],
    output_file: str
) -> Dict[str, Any]:
    """
    Create theoretical analysis summary.
    
    Args:
        results: Dictionary of method results
        output_file: Path to save summary
        
    Returns:
        Summary dictionary
    """
    summary = {
        'methods': list(results.keys()),
        'theoretical_metrics': {
            'ntk_alignment': {},
            'sensitivity_bound_ratio': {},
            'condition_number': {}
        },
        'best_methods': {},
        'correlations': {}
    }
    
    # Extract metrics
    for metric in ['ntk_alignment', 'sensitivity_bound_ratio', 'condition_number']:
        values = {}
        for method, result in results.items():
            val = result.get(metric, 0.0)
            if isinstance(val, list):
                val = np.mean(val)
            values[method] = val
        
        summary['theoretical_metrics'][metric] = values
        
        # Best method (max for alignment and ratio, min for condition number)
        if metric == 'condition_number':
            best_method = min(values, key=values.get)
        else:
            best_method = max(values, key=values.get)
        
        summary['best_methods'][metric] = best_method
    
    # Calculate correlation between NTK alignment and exactness score
    alignments = []
    exactness_scores = []
    methods_list = []
    
    for method, result in results.items():
        align = result.get('ntk_alignment', 0.0)
        exact = result.get('exactness_score', 0.0)
        
        if isinstance(align, list):
            align = np.mean(align)
        if isinstance(exact, list):
            exact = np.mean(exact)
        
        alignments.append(align)
        exactness_scores.append(exact)
        methods_list.append(method)
    
    if len(alignments) > 1:
        # Pearson correlation
        mean_align = np.mean(alignments)
        mean_exact = np.mean(exactness_scores)
        
        numerator = np.sum((alignments - mean_align) * (exactness_scores - mean_exact))
        denom_align = np.sqrt(np.sum((alignments - mean_align) ** 2))
        denom_exact = np.sqrt(np.sum((exactness_scores - mean_exact) ** 2))
        
        if denom_align * denom_exact > 1e-8:
            correlation = numerator / (denom_align * denom_exact)
            summary['correlations']['ntk_alignment_vs_exactness'] = float(correlation)
    
    # Save summary
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    return summary