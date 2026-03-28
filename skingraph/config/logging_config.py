import logging
import sys
import os

def setup_logging():
    """Configures structured logging for the SkinGraph system."""
    
    # Define format: timestamp, level, module name, message
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Get level from env, default to INFO (os.getenv is acceptable here)
    log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Suppress noisy third-party logs
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("groq").setLevel(logging.WARNING)
    
    logger = logging.getLogger("skingraph")
    logger.info(f"Logging initialized at {log_level_str} level")
    
    return logger
