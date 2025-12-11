"""
Logging Configuration Module
This module provides centralized logging functionality for the Multi-Disease Detection project.
"""

import logging
import os
from datetime import datetime
from pathlib import Path


# Create logs directory if it doesn't exist
LOGS_DIR = Path(__file__).parent.parent.parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)

# Generate log filename with timestamp
LOG_FILE = f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log"
LOG_FILE_PATH = LOGS_DIR / LOG_FILE


def get_logger(name: str = __name__) -> logging.Logger:
    """
    Get a configured logger instance
    
    Args:
        name: Name of the logger (usually __name__ from calling module)
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Only configure if not already configured
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        
        simple_formatter = logging.Formatter(
            "%(levelname)s - %(message)s"
        )
        
        # File handler with detailed formatting
        file_handler = logging.FileHandler(LOG_FILE_PATH, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        
        # Console handler with simple formatting
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(simple_formatter)
        
        # Add handlers
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        # Prevent propagation to root logger
        logger.propagate = False
    
    return logger


# Create default logger for the module
logger = get_logger(__name__)


def log_function_call(func):
    """
    Decorator to log function calls with arguments and return values
    
    Args:
        func: Function to decorate
        
    Returns:
        Wrapped function with logging
    """
    from functools import wraps
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        func_logger = get_logger(func.__module__)
        func_logger.debug(f"Calling {func.__name__} with args={args[:2]}... kwargs={list(kwargs.keys())}")
        
        try:
            result = func(*args, **kwargs)
            func_logger.debug(f"{func.__name__} completed successfully")
            return result
        except Exception as e:
            func_logger.error(f"{func.__name__} failed with error: {str(e)}")
            raise
    
    return wrapper


if __name__ == "__main__":
    # Test logging
    test_logger = get_logger("test")
    test_logger.debug("This is a debug message")
    test_logger.info("This is an info message")
    test_logger.warning("This is a warning message")
    test_logger.error("This is an error message")
    test_logger.critical("This is a critical message")
    
    print(f"\nLog file created at: {LOG_FILE_PATH}")
