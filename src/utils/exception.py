"""
Custom Exception Handler Module
This module defines custom exceptions for the Multi-Disease Detection project.
"""

import sys
from typing import Optional


class CustomException(Exception):
    """Base custom exception class for the project"""
    
    def __init__(self, error_message: str, error_detail: Optional[sys.exc_info] = None):
        """
        Initialize custom exception with detailed error information
        
        Args:
            error_message: The error message
            error_detail: System exception info tuple
        """
        super().__init__(error_message)
        self.error_message = error_message
        
        if error_detail is not None:
            self.error_message = self._get_detailed_error_message(error_message, error_detail)
    
    def _get_detailed_error_message(self, error_message: str, error_detail) -> str:
        """
        Generate detailed error message with file name and line number
        
        Args:
            error_message: Basic error message
            error_detail: System exception info
            
        Returns:
            Formatted error message with context
        """
        _, _, exc_tb = error_detail
        
        if exc_tb is not None:
            file_name = exc_tb.tb_frame.f_code.co_filename
            line_number = exc_tb.tb_lineno
            
            return f"Error occurred in script: [{file_name}] at line [{line_number}]: {error_message}"
        
        return error_message
    
    def __str__(self):
        return self.error_message


class DataLoadError(CustomException):
    """Exception raised for data loading errors"""
    pass


class PreprocessingError(CustomException):
    """Exception raised for preprocessing errors"""
    pass


class ModelLoadError(CustomException):
    """Exception raised for model loading errors"""
    pass


class PredictionError(CustomException):
    """Exception raised for prediction errors"""
    pass


class ConfigurationError(CustomException):
    """Exception raised for configuration errors"""
    pass


class ValidationError(CustomException):
    """Exception raised for validation errors"""
    pass
