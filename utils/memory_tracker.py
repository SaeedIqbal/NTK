#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Memory Tracking for NTK-SURGERY
Implements Memory Usage Monitoring and Profiling

This module provides:
- GPU and CPU memory usage tracking
- Memory allocation monitoring
- Memory leak detection
- Memory usage reporting
- Automatic memory cleanup

All memory tracking is designed for scientific computing workloads.
"""

import torch
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path
import logging
import gc
import os
import psutil

logger = logging.getLogger(__name__)


@dataclass
class MemoryConfig:
    """
    Configuration for memory tracking.
    
    Attributes:
        track_gpu: Whether to track GPU memory
        track_cpu: Whether to track CPU memory
        track_interval: Tracking interval in seconds
        max_memory_gb: Maximum allowed memory in GB
        enable_gc: Whether to enable garbage collection
        gc_threshold: Garbage collection threshold
        log_memory_usage: Whether to log memory usage
        memory_log_file: Path to memory log file
        alert_threshold: Memory usage threshold for alerts (0-1)
    """
    track_gpu: bool = True
    track_cpu: bool = True
    track_interval: float = 1.0
    max_memory_gb: float = 32.0
    enable_gc: bool = True
    gc_threshold: int = 1000
    log_memory_usage: bool = True
    memory_log_file: str = 'memory_usage.log'
    alert_threshold: float = 0.9
    
    def __post_init__(self):
        """Validate configuration."""
        if self.max_memory_gb <= 0:
            raise ValueError("max_memory_gb must be positive")
        if not 0 <= self.alert_threshold <= 1:
            raise ValueError("alert_threshold must be in [0, 1]")


@dataclass
class MemoryReport:
    """
    Data class for memory usage report.
    
    Attributes:
        timestamp: Report timestamp
        gpu_allocated: GPU allocated memory in MB
        gpu_reserved: GPU reserved memory in MB
        gpu_max_allocated: GPU max allocated memory in MB
        cpu_used: CPU used memory in MB
        cpu_available: CPU available memory in MB
        cpu_percent: CPU memory usage percentage
        process_memory: Process memory usage in MB
        num_objects: Number of Python objects
        gc_count: Garbage collector counts
    """
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    gpu_allocated: float = 0.0
    gpu_reserved: float = 0.0
    gpu_max_allocated: float = 0.0
    cpu_used: float = 0.0
    cpu_available: float = 0.0
    cpu_percent: float = 0.0
    process_memory: float = 0.0
    num_objects: int = 0
    gc_count: Tuple[int, int, int] = field(default_factory=lambda: (0, 0, 0))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'timestamp': self.timestamp,
            'gpu_allocated_mb': self.gpu_allocated,
            'gpu_reserved_mb': self.gpu_reserved,
            'gpu_max_allocated_mb': self.gpu_max_allocated,
            'cpu_used_mb': self.cpu_used,
            'cpu_available_mb': self.cpu_available,
            'cpu_percent': self.cpu_percent,
            'process_memory_mb': self.process_memory,
            'num_objects': self.num_objects,
            'gc_count': list(self.gc_count)
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


class MemoryTracker:
    """
    Tracks memory usage for NTK-SURGERY experiments.
    
    Provides:
    - GPU and CPU memory monitoring
    - Memory usage logging
    - Memory leak detection
    - Automatic garbage collection
    - Memory usage alerts
    - Historical tracking and reporting
    """
    
    _instance: Optional['MemoryTracker'] = None
    
    def __new__(cls, config: Optional[MemoryConfig] = None) -> 'MemoryTracker':
        """
        Create or retrieve MemoryTracker instance (singleton pattern).
        
        Args:
            config: Memory configuration
            
        Returns:
            MemoryTracker instance
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        
        return cls._instance
    
    def __init__(self, config: Optional[MemoryConfig] = None):
        """
        Initialize MemoryTracker.
        
        Args:
            config: Memory configuration
        """
        if self._initialized:
            return
        
        self.config = config if config is not None else MemoryConfig()
        self.memory_history: List[MemoryReport] = []
        self.start_time = datetime.now()
        self.peak_memory = MemoryReport()
        
        # Initialize tracking
        if self.config.enable_gc:
            gc.set_threshold(self.config.gc_threshold)
        
        # Create log file directory
        if self.config.log_memory_usage:
            Path(self.config.memory_log_file).parent.mkdir(parents=True, exist_ok=True)
        
        self._initialized = True
        
        logger.info("Initialized MemoryTracker")
    
    def get_gpu_memory(self) -> Dict[str, float]:
        """
        Get GPU memory usage.
        
        Returns:
            Dictionary with GPU memory metrics
        """
        if not self.config.track_gpu or not torch.cuda.is_available():
            return {
                'allocated': 0.0,
                'reserved': 0.0,
                'max_allocated': 0.0
            }
        
        allocated = torch.cuda.memory_allocated() / 1024 / 1024  # MB
        reserved = torch.cuda.memory_reserved() / 1024 / 1024  # MB
        max_allocated = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
        
        return {
            'allocated': allocated,
            'reserved': reserved,
            'max_allocated': max_allocated
        }
    
    def get_cpu_memory(self) -> Dict[str, float]:
        """
        Get CPU memory usage.
        
        Returns:
            Dictionary with CPU memory metrics
        """
        if not self.config.track_cpu:
            return {
                'used': 0.0,
                'available': 0.0,
                'percent': 0.0
            }
        
        memory = psutil.virtual_memory()
        
        return {
            'used': memory.used / 1024 / 1024,  # MB
            'available': memory.available / 1024 / 1024,  # MB
            'percent': memory.percent
        }
    
    def get_process_memory(self) -> float:
        """
        Get current process memory usage.
        
        Returns:
            Process memory in MB
        """
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024  # MB
    
    def get_object_count(self) -> int:
        """
        Get number of Python objects.
        
        Returns:
            Number of objects
        """
        return len(gc.get_objects())
    
    def get_gc_count(self) -> Tuple[int, int, int]:
        """
        Get garbage collector counts.
        
        Returns:
            Tuple of (gen0, gen1, gen2) counts
        """
        return gc.get_count()
    
    def track_memory(self, label: str = '') -> MemoryReport:
        """
        Track current memory usage.
        
        Args:
            label: Optional label for this tracking point
            
        Returns:
            MemoryReport object
        """
        gpu_mem = self.get_gpu_memory()
        cpu_mem = self.get_cpu_memory()
        
        report = MemoryReport(
            gpu_allocated=gpu_mem['allocated'],
            gpu_reserved=gpu_mem['reserved'],
            gpu_max_allocated=gpu_mem['max_allocated'],
            cpu_used=cpu_mem['used'],
            cpu_available=cpu_mem['available'],
            cpu_percent=cpu_mem['percent'],
            process_memory=self.get_process_memory(),
            num_objects=self.get_object_count(),
            gc_count=self.get_gc_count()
        )
        
        # Add label to timestamp
        if label:
            report.timestamp = f"{report.timestamp}_{label}"
        
        # Store in history
        self.memory_history.append(report)
        
        # Update peak memory
        self._update_peak_memory(report)
        
        # Check for alerts
        self._check_memory_alerts(report)
        
        # Log if enabled
        if self.config.log_memory_usage:
            self._log_memory_usage(report)
        
        return report
    
    def _update_peak_memory(self, report: MemoryReport):
        """
        Update peak memory tracking.
        
        Args:
            report: Current memory report
        """
        if report.gpu_allocated > self.peak_memory.gpu_allocated:
            self.peak_memory.gpu_allocated = report.gpu_allocated
        
        if report.cpu_used > self.peak_memory.cpu_used:
            self.peak_memory.cpu_used = report.cpu_used
        
        if report.process_memory > self.peak_memory.process_memory:
            self.peak_memory.process_memory = report.process_memory
    
    def _check_memory_alerts(self, report: MemoryReport):
        """
        Check for memory usage alerts.
        
        Args:
            report: Current memory report
        """
        # Check GPU memory
        if report.gpu_allocated > 0:
            max_gpu = self.config.max_memory_gb * 1024  # Convert to MB
            if report.gpu_allocated / max_gpu > self.config.alert_threshold:
                logger.warning(
                    f"High GPU memory usage: {report.gpu_allocated:.1f}MB "
                    f"({report.gpu_allocated/max_gpu*100:.1f}%)"
                )
        
        # Check CPU memory
        if report.cpu_percent > self.config.alert_threshold * 100:
            logger.warning(
                f"High CPU memory usage: {report.cpu_percent:.1f}%"
            )
    
    def _log_memory_usage(self, report: MemoryReport):
        """
        Log memory usage to file.
        
        Args:
            report: Memory report to log
        """
        log_file = Path(self.config.memory_log_file)
        
        with open(log_file, 'a') as f:
            f.write(report.to_json() + '\n')
    
    def cleanup_memory(self):
        """
        Clean up memory by running garbage collection.
        """
        if not self.config.enable_gc:
            return
        
        # Run garbage collection
        gc.collect()
        
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("Memory cleanup completed")
    
    def reset_peak_memory(self):
        """Reset peak memory tracking."""
        self.peak_memory = MemoryReport()
        logger.info("Peak memory tracking reset")
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """
        Get memory usage summary.
        
        Returns:
            Dictionary with memory summary
        """
        if not self.memory_history:
            return {}
        
        gpu_allocated_values = [r.gpu_allocated for r in self.memory_history]
        cpu_used_values = [r.cpu_used for r in self.memory_history]
        process_memory_values = [r.process_memory for r in self.memory_history]
        
        summary = {
            'tracking_duration_seconds': (datetime.now() - self.start_time).total_seconds(),
            'num_tracking_points': len(self.memory_history),
            'gpu_memory': {
                'current': self.memory_history[-1].gpu_allocated if self.memory_history else 0,
                'peak': self.peak_memory.gpu_allocated,
                'average': float(np.mean(gpu_allocated_values)),
                'min': float(np.min(gpu_allocated_values)),
                'max': float(np.max(gpu_allocated_values))
            },
            'cpu_memory': {
                'current': self.memory_history[-1].cpu_used if self.memory_history else 0,
                'peak': self.peak_memory.cpu_used,
                'average': float(np.mean(cpu_used_values)),
                'min': float(np.min(cpu_used_values)),
                'max': float(np.max(cpu_used_values))
            },
            'process_memory': {
                'current': self.memory_history[-1].process_memory if self.memory_history else 0,
                'peak': self.peak_memory.process_memory,
                'average': float(np.mean(process_memory_values))
            },
            'gc_counts': {
                'current': self.memory_history[-1].gc_count if self.memory_history else (0, 0, 0)
            }
        }
        
        return summary
    
    def save_memory_report(self, filepath: Optional[str] = None):
        """
        Save memory report to file.
        
        Args:
            filepath: Path to save report
        """
        if filepath is None:
            filepath = Path(self.config.memory_log_file).with_suffix('.json')
        else:
            filepath = Path(filepath)
        
        report_data = {
            'config': self.config.__dict__,
            'summary': self.get_memory_summary(),
            'peak_memory': self.peak_memory.to_dict(),
            'history': [r.to_dict() for r in self.memory_history]
        }
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        logger.info(f"Saved memory report to {filepath}")
    
    def print_memory_summary(self):
        """Print memory summary to console."""
        summary = self.get_memory_summary()
        
        if not summary:
            print("No memory tracking data available")
            return
        
        print("\n" + "=" * 60)
        print("MEMORY USAGE SUMMARY")
        print("=" * 60)
        print(f"Tracking Duration: {summary.get('tracking_duration_seconds', 0):.1f}s")
        print(f"Tracking Points: {summary.get('num_tracking_points', 0)}")
        print(f"\nGPU Memory:")
        print(f"  Current: {summary['gpu_memory']['current']:.1f} MB")
        print(f"  Peak: {summary['gpu_memory']['peak']:.1f} MB")
        print(f"  Average: {summary['gpu_memory']['average']:.1f} MB")
        print(f"\nCPU Memory:")
        print(f"  Current: {summary['cpu_memory']['current']:.1f} MB")
        print(f"  Peak: {summary['cpu_memory']['peak']:.1f} MB")
        print(f"  Average: {summary['cpu_memory']['average']:.1f} MB")
        print(f"\nProcess Memory:")
        print(f"  Current: {summary['process_memory']['current']:.1f} MB")
        print(f"  Peak: {summary['process_memory']['peak']:.1f} MB")
        print("=" * 60 + "\n")


def track_memory(label: str = '', config: Optional[MemoryConfig] = None) -> MemoryReport:
    """
    Convenience function to track memory usage.
    
    Args:
        label: Optional label for tracking point
        config: Memory configuration
        
    Returns:
        MemoryReport object
    """
    tracker = MemoryTracker(config)
    return tracker.track_memory(label)


def get_memory_usage() -> Dict[str, float]:
    """
    Get current memory usage.
    
    Returns:
        Dictionary with memory usage metrics
    """
    tracker = MemoryTracker()
    
    return {
        'gpu_allocated': tracker.get_gpu_memory()['allocated'],
        'gpu_reserved': tracker.get_gpu_memory()['reserved'],
        'cpu_used': tracker.get_cpu_memory()['used'],
        'cpu_percent': tracker.get_cpu_memory()['percent'],
        'process_memory': tracker.get_process_memory()
    }