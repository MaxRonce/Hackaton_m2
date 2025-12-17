import logging
import sys
from logging import Logger

def setup_logger(name: str = "ml_pipeline", level: int = logging.INFO) -> Logger:
    """
    Sets up a logger with a standard format for the pipeline.
    
    Args:
        name (str): Name of the logger.
        level (int): Logging level.
        
    Returns:
        Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Check if handlers already exist to avoid duplicate logs
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
    return logger
