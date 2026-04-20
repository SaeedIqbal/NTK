#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unlearning Efficacy Visualization for NTK-SURGERY
Implements Figure 2 from the manuscript

This module creates publication-quality plots for:
- Forget Accuracy vs Retain Accuracy scatter plots
- Exactness Score bar charts and box plots
- Multi-dataset efficacy comparison
- Method ranking by exactness score

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
class EfficacyConfig:
    """
    Configuration for efficacy visualization.
    
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
        'Full NTK-SURGERY': '#e74c3c',
        'w/o NTK Rep': '#e67e22',
        'w/o Influence Matrix': '#f39c12',
        'w/o Surgery Operator': '#9b59b6',
        'w/o Finite-Width Proj': '#3498db',
        'Weight-Space Baseline': '#7f8c8d',
        'Scratch (Gold)': '#2c3e50',
        'SIFU': '#9b59b6',
        'FedEraser': '#3498db',
        'Fine-Tuning': '#e74c3c'
    })
    save_dpi: int = 600
    format: str = 'pdf'
    results_dir: str = 'results/visualizations/efficacy'


class EfficacyPlotter:
    """
    Creates publication-quality efficacy visualizations for NTK-SURGERY.
    
    Implements Figure 2 from the manuscript with:
    - Scatter plots of Forget vs Retain Accuracy
    - Bar charts of Exactness Scores
    - Box plots showing distribution across datasets
    - Radar plots for multi-metric comparison
    
    All plots use serif fonts and white backgrounds per ACCV guidelines.
    """
    
    def __init__(self, config: Optional[EfficacyConfig] = None):
        """
        Initialize EfficacyPlotter.
        
        Args:
            config: Visualization configuration
        """
        self.config = config if config is not None else EfficacyConfig()
        self._setup_plotting_style()
        
        # Create results directory
        Path(self.config.results_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info("Initialized EfficacyPlotter")
    
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
    
    def plot_forget_vs_retain(
        self,
        results: Dict[str, Dict],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create scatter plot of Forget Accuracy vs Retain Accuracy.
        
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
                
            color = self.config.color_palette[method_name]
            
            # Extract metrics
            forget_acc = method_results.get('forget_accuracy', [])
            retain_acc = method_results.get('retain_accuracy', [])
            
            if isinstance(forget_acc, list) and isinstance(retain_acc, list):
                ax.scatter(
                    forget_acc, retain_acc,
                    c=color, s=self.config.marker_size,
                    alpha=0.7, label=method_name
                )
            elif isinstance(forget_acc, (int, float)) and isinstance(retain_acc, (int, float)):
                ax.scatter(
                    [forget_acc], [retain_acc],
                    c=color, s=self.config.marker_size,
                    alpha=0.7, label=method_name
                )
        
        ax.set_xlabel('Forget Accuracy (%) ↓', fontsize=self.config.label_font_size)
        ax.set_ylabel('Retain Accuracy (%) ↑', fontsize=self.config.label_font_size)
        ax.set_title('Utility Trade-off', fontsize=self.config.title_font_size, pad=12)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.legend(loc='upper left', frameon=False, ncol=2)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=self.config.save_dpi)
            logger.info(f"Saved forget vs retain plot to {save_path}")
        
        return fig
    
    def plot_exactness_scores(
        self,
        results: Dict[str, Dict],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create bar chart of Exactness Scores.
        
        Args:
            results: Dictionary of method results
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.config.figsize)
        
        methods = []
        scores = []
        colors = []
        
        for method_name, method_results in results.items():
            if method_name not in self.config.color_palette:
                continue
                
            exactness = method_results.get('exactness_score', 0.0)
            if isinstance(exactness, list):
                exactness = np.mean(exactness)
            
            methods.append(method_name)
            scores.append(exactness)
            colors.append(self.config.color_palette[method_name])
        
        bars = ax.bar(methods, scores, color=colors, width=0.6)
        
        # Add value labels on bars
        for bar, score in zip(bars, scores):
            ax.text(
                bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{score:.3f}', ha='center', va='bottom',
                fontsize=self.config.tick_font_size
            )
        
        ax.set_ylabel('Exactness Score ↑', fontsize=self.config.label_font_size)
        ax.set_title('Exactness Score Comparison', fontsize=self.config.title_font_size, pad=12)
        ax.set_ylim(0, 1.05)
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        plt.xticks(rotation=45, ha='right')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=self.config.save_dpi)
            logger.info(f"Saved exactness scores plot to {save_path}")
        
        return fig
    
    def plot_exactness_boxplot(
        self,
        results: Dict[str, Dict],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create box plot of Exactness Score distributions.
        
        Args:
            results: Dictionary of method results
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.config.figsize)
        
        data = []
        labels = []
        colors = []
        
        for method_name, method_results in results.items():
            if method_name not in self.config.color_palette:
                continue
                
            exactness = method_results.get('exactness_score', [])
            if not isinstance(exactness, list):
                exactness = [exactness]
            
            data.append(exactness)
            labels.append(method_name)
            colors.append(self.config.color_palette[method_name])
        
        bp = ax.boxplot(data, patch_artist=True, widths=0.6)
        
        # Color boxes
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_ylabel('Exactness Score', fontsize=self.config.label_font_size)
        ax.set_title('Exactness Distribution', fontsize=self.config.title_font_size, pad=12)
        ax.set_ylim(0, 1.05)
        plt.xticks(range(1, len(labels)+1), labels, rotation=45, ha='right')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=self.config.save_dpi)
            logger.info(f"Saved exactness boxplot to {save_path}")
        
        return fig
    
    def plot_radar_chart(
        self,
        results: Dict[str, Dict],
        metrics: List[str] = ['Exactness', 'Forget', 'Retain', 'Alignment', 'Efficiency'],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create radar chart for multi-metric comparison.
        
        Args:
            results: Dictionary of method results
            metrics: List of metrics to include
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        from math import pi
        
        # Normalize metrics to [0, 1]
        normalized_data = {}
        
        for method_name, method_results in results.items():
            if method_name not in self.config.color_palette:
                continue
                
            normalized_values = []
            
            for metric in metrics:
                if metric == 'Exactness':
                    value = method_results.get('exactness_score', 0.0)
                    normalized = value
                elif metric == 'Forget':
                    value = method_results.get('forget_accuracy', 100.0)
                    # Lower is better, normalize to [0, 1] where 0 = best
                    normalized = 1.0 - min(value / 50.0, 1.0)
                elif metric == 'Retain':
                    value = method_results.get('retain_accuracy', 0.0)
                    normalized = value / 100.0
                elif metric == 'Alignment':
                    value = method_results.get('ntk_alignment', 0.0)
                    normalized = value
                elif metric == 'Efficiency':
                    value = method_results.get('speedup_vs_scratch', 1.0)
                    normalized = min(value / 100.0, 1.0)
                else:
                    normalized = 0.0
                
                normalized_values.append(max(0.0, min(1.0, normalized)))
            
            normalized_data[method_name] = normalized_values
        
        # Create radar plot
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        
        angles = [n / len(metrics) * 2 * pi for n in range(len(metrics))]
        angles += angles[:1]  # Close the loop
        
        for method_name, values in normalized_data.items():
            values_closed = values + [values[0]]
            color = self.config.color_palette[method_name]
            
            ax.plot(angles, values_closed, 'o-', linewidth=2, markersize=6,
                   color=color, label=method_name)
            ax.fill(angles, values_closed, alpha=0.25, color=color)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m[:8] for m in metrics], fontsize=self.config.tick_font_size)
        ax.set_ylim(0, 1)
        ax.set_title('Component Contribution Analysis', fontsize=self.config.title_font_size, 
                    y=1.05, pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=self.config.legend_font_size)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=self.config.save_dpi)
            logger.info(f"Saved radar chart to {save_path}")
        
        return fig
    
    def create_comprehensive_efficacy_plot(
        self,
        results: Dict[str, Dict],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create comprehensive 1x4 efficacy visualization.
        
        Args:
            results: Dictionary of method results
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 4, figsize=(24, 6))
        fig.suptitle('Unlearning Efficacy Metrics', fontsize=16, y=0.98, fontweight='bold')
        
        # Plot 1: Forget vs Retain
        self.plot_forget_vs_retain(results, save_path=None)
        fig1 = plt.gcf()
        axes[0].clear()
        for artist in fig1.axes[0].artists:
            axes[0].add_artist(artist)
        for collection in fig1.axes[0].collections:
            axes[0].add_collection(collection)
        for line in fig1.axes[0].lines:
            axes[0].add_line(line)
        axes[0].set_xlabel(fig1.axes[0].get_xlabel())
        axes[0].set_ylabel(fig1.axes[0].get_ylabel())
        axes[0].set_title(fig1.axes[0].get_title())
        axes[0].grid(fig1.axes[0].get_gridspec())
        axes[0].legend().remove()
        
        # Plot 2: Exactness Scores
        self.plot_exactness_scores(results, save_path=None)
        fig2 = plt.gcf()
        axes[1].clear()
        for patch in fig2.axes[0].patches:
            axes[1].add_patch(patch)
        for text in fig2.axes[0].texts:
            axes[1].text(text.get_position()[0], text.get_position()[1], text.get_text(),
                        fontsize=text.get_fontsize())
        axes[1].set_ylabel(fig2.axes[0].get_ylabel())
        axes[1].set_title(fig2.axes[0].get_title())
        axes[1].set_ylim(fig2.axes[0].get_ylim())
        axes[1].grid(fig2.axes[0].get_gridspec())
        
        # Plot 3: Box Plot
        self.plot_exactness_boxplot(results, save_path=None)
        fig3 = plt.gcf()
        axes[2].clear()
        for artist in fig3.axes[0].artists:
            axes[2].add_artist(artist)
        for collection in fig3.axes[0].collections:
            axes[2].add_collection(collection)
        axes[2].set_ylabel(fig3.axes[0].get_ylabel())
        axes[2].set_title(fig3.axes[0].get_title())
        axes[2].set_ylim(fig3.axes[0].get_ylim())
        
        # Plot 4: Radar Chart
        self.plot_radar_chart(results, save_path=None)
        fig4 = plt.gcf()
        axes[3] = fig4.axes[0]
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=self.config.save_dpi)
            logger.info(f"Saved comprehensive efficacy plot to {save_path}")
        
        return fig


def plot_unlearning_efficacy(
    results_file: str,
    output_dir: str = 'results/visualizations/efficacy',
    config: Optional[EfficacyConfig] = None
) -> Dict[str, str]:
    """
    Plot unlearning efficacy from results file.
    
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
    plotter = EfficacyPlotter(config)
    plotter.config.results_dir = output_dir
    
    # Create plots
    plot_paths = {}
    
    plot_paths['forget_vs_retain'] = Path(output_dir) / 'forget_vs_retain.pdf'
    plotter.plot_forget_vs_retain(results, str(plot_paths['forget_vs_retain']))
    
    plot_paths['exactness_scores'] = Path(output_dir) / 'exactness_scores.pdf'
    plotter.plot_exactness_scores(results, str(plot_paths['exactness_scores']))
    
    plot_paths['exactness_boxplot'] = Path(output_dir) / 'exactness_boxplot.pdf'
    plotter.plot_exactness_boxplot(results, str(plot_paths['exactness_boxplot']))
    
    plot_paths['radar_chart'] = Path(output_dir) / 'radar_chart.pdf'
    plotter.plot_radar_chart(results, save_path=str(plot_paths['radar_chart']))
    
    plot_paths['comprehensive'] = Path(output_dir) / 'comprehensive_efficacy.pdf'
    plotter.create_comprehensive_efficacy_plot(results, str(plot_paths['comprehensive']))
    
    return plot_paths


def create_efficacy_summary(
    results: Dict[str, Dict],
    output_file: str
) -> Dict[str, Any]:
    """
    Create efficacy summary statistics.
    
    Args:
        results: Dictionary of method results
        output_file: Path to save summary
        
    Returns:
        Summary dictionary
    """
    summary = {
        'methods': list(results.keys()),
        'metrics': {
            'forget_accuracy': {},
            'retain_accuracy': {},
            'exactness_score': {}
        },
        'best_methods': {}
    }
    
    for metric in ['forget_accuracy', 'retain_accuracy', 'exactness_score']:
        values = {}
        for method, result in results.items():
            val = result.get(metric, 0.0)
            if isinstance(val, list):
                val = np.mean(val)
            values[method] = val
        
        summary['metrics'][metric] = values
        
        # Find best method (min for forget, max for others)
        if metric == 'forget_accuracy':
            best_method = min(values, key=values.get)
        else:
            best_method = max(values, key=values.get)
        
        summary['best_methods'][metric] = best_method
    
    # Save summary
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    return summary