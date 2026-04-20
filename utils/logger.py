#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Logging Utilities for NTK-SURGERY
Implements Comprehensive Logging Infrastructure

This module provides:
- Centralized logging configuration
- Multiple log handlers (console, file)
- Custom log formatters for scientific computing
- Log level management
- Log rotation and archival

All logging follows best practices for scientific reproducibility.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime
import json
import os


@dataclass
class LoggerConfig:
    """
    Configuration for logging system.
    
    Attributes:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files
        console_output: Whether to output to console
        file_output: Whether to output to file
        log_format: Log message format string
        date_format: Date format string
        max_bytes: Maximum log file size in bytes
        backup_count: Number of backup log files to keep
        enable_json: Whether to enable JSON logging
    """
    name: str = 'NTK-SURGERY'
    level: str = 'INFO'
    log_dir: str = 'logs'
    console_output: bool = True
    file_output: bool = True
    log_format: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    date_format: str = '%Y-%m-%d %H:%M:%S'
    max_bytes: int = 10 * 1024 * 1024  # 10 MB
    backup_count: int = 5
    enable_json: bool = False
    
    def __post_init__(self):
        """Validate configuration."""
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if self.level.upper() not in valid_levels:
            raise ValueError(f"Invalid log level: {self.level}")


class LogFormatter(logging.Formatter):
    """
    Custom log formatter for scientific computing.
    
    Provides enhanced formatting with:
    - Color-coded log levels for console output
    - Detailed timestamp information
    - Module and function name inclusion
    - Support for JSON formatting
    """
    
    # ANSI color codes for console output
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m'       # Reset
    }
    
    def __init__(self, fmt: Optional[str] = None, 
                 datefmt: Optional[str] = None,
                 use_color: bool = True):
        """
        Initialize LogFormatter.
        
        Args:
            fmt: Log format string
            datefmt: Date format string
            use_color: Whether to use color for console output
        """
        super().__init__(fmt, datefmt)
        self.use_color = use_color
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record.
        
        Args:
            record: Log record to format
            
        Returns:
            Formatted log message
        """
        # Add color for console output
        if self.use_color and record.levelname in self.COLORS:
            record.levelname = (
                f"{self.COLORS[record.levelname]}{record.levelname}{self.COLORS['RESET']}"
            )
        
        # Add module and function name
        record.module_short = record.module[:20]
        record.function_short = record.funcName[:20] if record.funcName else 'unknown'
        
        return super().format(record)


class JsonFormatter(logging.Formatter):
    """
    JSON log formatter for structured logging.
    
    Outputs logs in JSON format for easier parsing and analysis.
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record as JSON.
        
        Args:
            record: Log record to format
            
        Returns:
            JSON formatted log message
        """
        log_data = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'message': record.getMessage(),
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ['args', 'asctime', 'created', 'exc_info', 'exc_text',
                          'filename', 'funcName', 'id', 'levelname', 'levelno',
                          'lineno', 'module', 'msecs', 'message', 'msg', 'name',
                          'pathname', 'process', 'processName', 'relativeCreated',
                          'stack_info', 'thread', 'threadName']:
                log_data[key] = value
        
        return json.dumps(log_data, default=str)


class Logger:
    """
    Centralized logging manager for NTK-SURGERY.
    
    Provides:
    - Multiple log handlers (console, file, JSON)
    - Log level management
    - Log rotation and archival
    - Structured logging support
    - Performance tracking
    """
    
    _instances: Dict[str, 'Logger'] = {}
    
    def __new__(cls, config: Optional[LoggerConfig] = None) -> 'Logger':
        """
        Create or retrieve logger instance (singleton pattern).
        
        Args:
            config: Logger configuration
            
        Returns:
            Logger instance
        """
        if config is None:
            config = LoggerConfig()
        
        if config.name not in cls._instances:
            instance = super().__new__(cls)
            instance._initialized = False
            cls._instances[config.name] = instance
        
        return cls._instances[config.name]
    
    def __init__(self, config: Optional[LoggerConfig] = None):
        """
        Initialize Logger.
        
        Args:
            config: Logger configuration
        """
        if self._initialized:
            return
        
        self.config = config if config is not None else LoggerConfig()
        self.logger = logging.getLogger(self.config.name)
        self.logger.setLevel(getattr(logging, self.config.level.upper()))
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Setup handlers
        self._setup_console_handler()
        
        if self.config.file_output:
            self._setup_file_handler()
        
        if self.config.enable_json:
            self._setup_json_handler()
        
        self._initialized = True
        
        # Create log directory
        if self.config.file_output:
            Path(self.config.log_dir).mkdir(parents=True, exist_ok=True)
    
    def _setup_console_handler(self):
        """Setup console log handler."""
        if not self.config.console_output:
            return
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, self.config.level.upper()))
        
        formatter = LogFormatter(
            fmt=self.config.log_format,
            datefmt=self.config.date_format,
            use_color=True
        )
        
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
    
    def _setup_file_handler(self):
        """Setup file log handler with rotation."""
        from logging.handlers import RotatingFileHandler
        
        log_file = Path(self.config.log_dir) / f"{self.config.name}.log"
        
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=self.config.max_bytes,
            backupCount=self.config.backup_count
        )
        file_handler.setLevel(getattr(logging, self.config.level.upper()))
        
        formatter = LogFormatter(
            fmt=self.config.log_format,
            datefmt=self.config.date_format,
            use_color=False
        )
        
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
    
    def _setup_json_handler(self):
        """Setup JSON log handler for structured logging."""
        from logging.handlers import RotatingFileHandler
        
        json_file = Path(self.config.log_dir) / f"{self.config.name}.json.log"
        
        json_handler = RotatingFileHandler(
            json_file,
            maxBytes=self.config.max_bytes,
            backupCount=self.config.backup_count
        )
        json_handler.setLevel(getattr(logging, self.config.level.upper()))
        
        formatter = JsonFormatter()
        json_handler.setFormatter(formatter)
        self.logger.addHandler(json_handler)
    
    def debug(self, msg: str, **kwargs):
        """Log debug message."""
        self.logger.debug(msg, extra=kwargs)
    
    def info(self, msg: str, **kwargs):
        """Log info message."""
        self.logger.info(msg, extra=kwargs)
    
    def warning(self, msg: str, **kwargs):
        """Log warning message."""
        self.logger.warning(msg, extra=kwargs)
    
    def error(self, msg: str, **kwargs):
        """Log error message."""
        self.logger.error(msg, extra=kwargs)
    
    def critical(self, msg: str, **kwargs):
        """Log critical message."""
        self.logger.critical(msg, extra=kwargs)
    
    def exception(self, msg: str, **kwargs):
        """Log exception with traceback."""
        self.logger.exception(msg, extra=kwargs)
    
    def log_experiment_start(self, experiment_name: str, config: Dict[str, Any]):
        """
        Log experiment start with configuration.
        
        Args:
            experiment_name: Name of experiment
            config: Experiment configuration
        """
        self.info(
            f"Experiment started: {experiment_name}",
            experiment_name=experiment_name,
            config=config
        )
    
    def log_experiment_end(self, experiment_name: str, results: Dict[str, Any], 
                          duration: float):
        """
        Log experiment end with results.
        
        Args:
            experiment_name: Name of experiment
            results: Experiment results
            duration: Experiment duration in seconds
        """
        self.info(
            f"Experiment completed: {experiment_name}",
            experiment_name=experiment_name,
            results=results,
            duration=duration
        )
    
    def log_metric(self, metric_name: str, value: float, step: Optional[int] = None):
        """
        Log metric value.
        
        Args:
            metric_name: Name of metric
            value: Metric value
            step: Optional step number
        """
        self.info(
            f"Metric: {metric_name} = {value}",
            metric_name=metric_name,
            metric_value=value,
            step=step
        )
    
    def get_logger(self) -> logging.Logger:
        """
        Get underlying logging.Logger instance.
        
        Returns:
            Logger instance
        """
        return self.logger
    
    def set_level(self, level: str):
        """
        Set logging level.
        
        Args:
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        self.logger.setLevel(getattr(logging, level.upper()))
        
        for handler in self.logger.handlers:
            handler.setLevel(getattr(logging, level.upper()))
    
    def close(self):
        """Close all handlers and cleanup."""
        for handler in self.logger.handlers:
            handler.close()
            self.logger.removeHandler(handler)


def setup_logger(name: str = 'NTK-SURGERY', 
                 level: str = 'INFO',
                 log_dir: str = 'logs',
                 console_output: bool = True,
                 file_output: bool = True) -> Logger:
    """
    Setup and return logger instance.
    
    Args:
        name: Logger name
        level: Logging level
        log_dir: Directory for log files
        console_output: Whether to output to console
        file_output: Whether to output to file
        
    Returns:
        Logger instance
    """
    config = LoggerConfig(
        name=name,
        level=level,
        log_dir=log_dir,
        console_output=console_output,
        file_output=file_output
    )
    
    return Logger(config)


def get_logger(name: str = 'NTK-SURGERY') -> Logger:
    """
    Get existing logger instance.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    if name in Logger._instances:
        return Logger._instances[name]
    else:
        return setup_logger(name)