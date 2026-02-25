"""
Logging utilities for the Quantum Circuit Dataset Pipeline.

This module provides centralized logging configuration to ensure
consistent logging across all modules and reproducible debugging.

"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from config import CONFIG


def setup_logger(
    name: str,
    log_file: Optional[Path] = None,
    level: str = None
) -> logging.Logger:
    """
    Set up a logger with consistent formatting.
    
    Parameters
    ----------
    name : str
        Name of the logger (typically __name__ of the calling module).
    log_file : Optional[Path]
        Path to the log file. If None, uses default from CONFIG.
    level : str
        Logging level. If None, uses CONFIG.log_level.
    
    Returns
    -------
    logging.Logger
        Configured logger instance.
    
    Examples
    --------
    >>> logger = setup_logger(__name__)
    >>> logger.info("Pipeline started")
    """
    if level is None:
        level = CONFIG.log_level
    
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = CONFIG.paths.logs_dir / f"pipeline_{timestamp}.log"
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_formatter = logging.Formatter(
        '%(levelname)s: %(message)s'
    )
    
    # File handler
    CONFIG.paths.logs_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    console_handler.setFormatter(console_formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


class PipelineLogger:
    """
    Context manager for pipeline stage logging.
    
    This class provides structured logging for pipeline stages,
    including timing information and error handling.
    
    Attributes
    ----------
    logger : logging.Logger
        The underlying logger instance.
    stage_name : str
        Name of the current pipeline stage.
    start_time : datetime
        Start time of the stage.
    
    Examples
    --------
    >>> with PipelineLogger("PDF Extraction") as pl:
    ...     pl.log("Processing paper 2410.08073")
    ...     # ... processing code ...
    """
    
    def __init__(self, stage_name: str, logger: Optional[logging.Logger] = None):
        """
        Initialize the pipeline logger.
        
        Parameters
        ----------
        stage_name : str
            Name of the pipeline stage.
        logger : Optional[logging.Logger]
            Logger instance. If None, creates a new one.
        """
        self.stage_name = stage_name
        self.logger = logger or setup_logger(f"pipeline.{stage_name}")
        self.start_time = None
    
    def __enter__(self):
        """Start timing the stage."""
        self.start_time = datetime.now()
        self.logger.info(f"=== Starting: {self.stage_name} ===")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Log completion or error."""
        duration = datetime.now() - self.start_time
        if exc_type is None:
            self.logger.info(
                f"=== Completed: {self.stage_name} in {duration.total_seconds():.2f}s ==="
            )
        else:
            self.logger.error(
                f"=== Failed: {self.stage_name} after {duration.total_seconds():.2f}s ===",
                exc_info=True
            )
        return False  # Don't suppress exceptions
    
    def log(self, message: str, level: str = "info"):
        """
        Log a message at the specified level.
        
        Parameters
        ----------
        message : str
            Message to log.
        level : str
            Log level ('debug', 'info', 'warning', 'error').
        """
        getattr(self.logger, level.lower())(f"[{self.stage_name}] {message}")
    
    def progress(self, current: int, total: int, message: str = ""):
        """
        Log progress information.
        
        Parameters
        ----------
        current : int
            Current progress count.
        total : int
            Total expected count.
        message : str
            Additional message.
        """
        percentage = (current / total * 100) if total > 0 else 0
        self.logger.info(
            f"[{self.stage_name}] Progress: {current}/{total} ({percentage:.1f}%) {message}"
        )
