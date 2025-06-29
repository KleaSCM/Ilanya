#!/usr/bin/env python3
"""
Ilanya Logging Utilities

Standardized logging utilities for tests and demos with organized directory structure.
Provides consistent logging format: [date][test_type][name_of_test][thing_its_testing]

Author: KleaSCM
Email: KleaSCM@gmail.com
License: MIT
Version: 0.1.0
"""

import os
import logging
from datetime import datetime
from typing import Optional


def setup_logger(
    engine_type: str,
    test_type: str,
    test_name: str,
    test_target: str,
    log_level: str = "INFO",
    log_dir: str = "Logs"
) -> logging.Logger:
    """
    Set up a logger with standardized naming and directory structure.
    
    Args:
        engine_type: Either 'trait' or 'desire'
        test_type: Either 'test' or 'demo'
        test_name: Name of the test/demo
        test_target: What the test/demo is testing
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_dir: Base directory for logs
    
    Returns:
        Configured logger instance
    """
    # Create filename without timestamp for persistent logging
    filename = f"{test_type}_{test_name}_{test_target}.log"
    
    # Create directory path
    log_path = os.path.join(log_dir, engine_type, f"{test_type}s")
    os.makedirs(log_path, exist_ok=True)
    
    # Full file path
    file_path = os.path.join(log_path, filename)
    
    # Create logger
    logger = logging.getLogger(f"{engine_type}_{test_type}_{test_name}")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Create file handler with append mode
    file_handler = logging.FileHandler(file_path, mode='a')
    file_handler.setLevel(getattr(logging, log_level.upper()))
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    
    # Create formatter with timestamp
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Set formatter for handlers
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Log initial setup message with run separator
    logger.info("=" * 80)
    logger.info(f"NEW TEST RUN - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Logger initialized for {engine_type} {test_type}: {test_name} - {test_target}")
    logger.info(f"Log file: {file_path}")
    logger.info("=" * 80)
    
    return logger


def get_log_file_path(
    engine_type: str,
    test_type: str,
    test_name: str,
    test_target: str,
    log_dir: str = "Logs"
) -> str:
    """
    Get the log file path without creating a logger.
    
    Args:
        engine_type: Either 'trait' or 'desire'
        test_type: Either 'test' or 'demo'
        test_name: Name of the test/demo
        test_target: What the test/demo is testing
        log_dir: Base directory for logs
    
    Returns:
        Full path to the log file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{test_type}_{test_name}_{test_target}.log"
    log_path = os.path.join(log_dir, engine_type, f"{test_type}s")
    return os.path.join(log_path, filename)


def log_test_start(logger: logging.Logger, test_name: str, test_description: str):
    """Log the start of a test."""
    logger.info("=" * 80)
    logger.info(f"STARTING TEST: {test_name}")
    logger.info(f"DESCRIPTION: {test_description}")
    logger.info(f"START TIME: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)


def log_test_end(logger: logging.Logger, test_name: str, success: bool, duration: Optional[float] = None):
    """Log the end of a test."""
    status = "PASSED" if success else "FAILED"
    logger.info("=" * 80)
    logger.info(f"TEST {status}: {test_name}")
    if duration is not None:
        logger.info(f"DURATION: {duration:.2f} seconds")
    logger.info(f"END TIME: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)


def log_demo_start(logger: logging.Logger, demo_name: str, demo_description: str):
    """Log the start of a demo."""
    logger.info("=" * 80)
    logger.info(f"STARTING DEMO: {demo_name}")
    logger.info(f"DESCRIPTION: {demo_description}")
    logger.info(f"START TIME: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)


def log_demo_end(logger: logging.Logger, demo_name: str, duration: Optional[float] = None):
    """Log the end of a demo."""
    logger.info("=" * 80)
    logger.info(f"DEMO COMPLETED: {demo_name}")
    if duration is not None:
        logger.info(f"DURATION: {duration:.2f} seconds")
    logger.info(f"END TIME: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80) 