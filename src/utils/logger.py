import logging
import os
import multiprocessing

LOGGER_NAME = "plexzahf"


def configure_logging():
    """Configure the root logger once, at application startup"""
    logger = logging.getLogger(LOGGER_NAME)

    # Only configure if not already configured
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)

        # Create a formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(processName)s - %(levelname)s - %(message)s"
        )

        # Only create a file handler in the main process
        if multiprocessing.current_process().name == "MainProcess":
            # Create logs directory if it doesn't exist
            if not os.path.exists("logs"):
                os.makedirs("logs")

            # Use a timestamp that won't change during execution
            # Format timestamp as readable datetime (e.g., 2023-05-25_14-30-45)
            from datetime import datetime

            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            # Create a handler that logs to a file
            file_handler = logging.FileHandler(f"logs/{timestamp}.log")
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        # Always add the stream handler for console output
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger
