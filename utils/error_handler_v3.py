import logging
from typing import Any, Dict, Optional
import traceback
import sys
from pathlib import Path

class ErrorHandler:
    def __init__(self, log_dir: str = "logs/errors"):
        """Initialize error handler with logging configuration."""
        self.logger = self._setup_logger(log_dir)
        self.error_counts: Dict[str, int] = {
            'parsing': 0,
            'validation': 0,
            'critical': 0,
            'file': 0
        }

    def _setup_logger(self, log_dir: str) -> logging.Logger:
        """Set up logging configuration."""
        logger = logging.getLogger('ErrorHandler')
        logger.setLevel(logging.ERROR)

        # Create log directory if it doesn't exist
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        # File handler for error logging
        fh = logging.FileHandler(log_path / 'parser_errors.log')
        fh.setLevel(logging.ERROR)

        # Console handler for immediate feedback
        ch = logging.StreamHandler(sys.stderr)
        ch.setLevel(logging.ERROR)

        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        # Add handlers to logger
        logger.addHandler(fh)
        logger.addHandler(ch)

        return logger

    def handle_parsing_error(self, error: Exception, line_number: int, content: str):
        """Handle errors during log line parsing."""
        self.error_counts['parsing'] += 1
        error_msg = (
            f"Parsing error at line {line_number}: {str(error)}\n"
            f"Content: {content[:200]}..."
        )
        self.logger.error(error_msg)
        self._log_detailed_error(error)

    def handle_validation_error(self, error: Exception, field: str, value: Any):
        """Handle validation errors for specific fields."""
        self.error_counts['validation'] += 1
        error_msg = (
            f"Validation error for field '{field}': {str(error)}\n"
            f"Value: {str(value)}"
        )
        self.logger.error(error_msg)
        self._log_detailed_error(error)

    def handle_file_error(self, error: Exception, file_path: str):
        """Handle errors related to file operations."""
        self.error_counts['file'] += 1
        error_msg = f"File operation error for {file_path}: {str(error)}"
        self.logger.error(error_msg)
        self._log_detailed_error(error)

    def handle_critical_error(self, error: Exception):
        """Handle critical errors that require immediate attention."""
        self.error_counts['critical'] += 1
        error_msg = f"Critical error: {str(error)}"
        self.logger.critical(error_msg)
        self._log_detailed_error(error)

    def handle_entry_error(self, error: Exception, line_number: int, content: str):
        """Handle errors during log entry processing."""
        error_msg = (
            f"Entry processing error at line {line_number}: {str(error)}\n"
            f"Content: {content[:200]}..."
        )
        self.logger.error(error_msg)
        self._log_detailed_error(error)

    def handle_save_error(self, error: Exception):
        """Handle errors during save operations."""
        error_msg = f"Save operation error: {str(error)}"
        self.logger.error(error_msg)
        self._log_detailed_error(error)

    def _log_detailed_error(self, error: Exception):
        """Log detailed error information including stack trace."""
        self.logger.debug(
            "Detailed error information:\n" +
            "".join(traceback.format_exception(
                type(error), error, error.__traceback__
            ))
        )

    def get_error_summary(self) -> Dict[str, int]:
        """Get a summary of all errors encountered."""
        return {
            'total_errors': sum(self.error_counts.values()),
            **self.error_counts
        }

    def reset_error_counts(self):
        """Reset all error counters."""
        for key in self.error_counts:
            self.error_counts[key] = 0