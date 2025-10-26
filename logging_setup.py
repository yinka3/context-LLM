import logging
import sys
from logging.handlers import RotatingFileHandler

def setup_logging(log_level=logging.INFO, log_file="tier1_app.log"):
    """
    Configures a centralized logger for the application.

    Sets up a root logger that outputs to both a rotating file and the console.
    This function should be called once at the very beginning of the application's
    entry point.
    """
    # Get the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Prevent multiple handlers being added if this function is called more than once
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Rotating File Handler
    # Creates a new log file when the current one reaches 2MB, keeps 10 old logs.
    file_handler = RotatingFileHandler(
        log_file, maxBytes=2*1024*1024, backupCount=10
    )
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    # Console Handler
    # Outputs logs to the standard console (stdout).
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    logging.info("Logging configured successfully.")