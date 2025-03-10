import os
import logging
import sys
from datetime import datetime


def setup_logging(log_level=logging.INFO, log_file=None):
    """
    Set up logging for the application.
    """
    # Create logs directory if it doesn't exist
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
            print(f"Created log directory: {log_dir}")

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_format)
    root_logger.addHandler(console_handler)

    # Create file handler if log_file is specified
    if log_file:
        try:
            file_handler = logging.FileHandler(log_file, mode='a')
            file_handler.setLevel(log_level)
            file_format = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(file_format)
            root_logger.addHandler(file_handler)
            print(f"Added file handler for: {log_file}")

            # Test write to the log file
            root_logger.info(f"Logging initialized to file: {log_file}")
        except Exception as e:
            print(f"Error setting up file logging: {str(e)}")
            # Continue with console logging only

    return root_logger


def main():
    # Create a timestamped log file in the /logs directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "..", "logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    log_file_path = os.path.join(log_dir, f"test_logging_{timestamp}.log")
    print(f"Setting up logging to file: {log_file_path}")

    # Set up logging
    logger = setup_logging(log_level=logging.INFO, log_file=log_file_path)

    # Log some messages
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")

    # Check if the log file was created
    if os.path.exists(log_file_path):
        print(f"Log file created successfully: {log_file_path}")
        print(f"Log file size: {os.path.getsize(log_file_path)} bytes")

        # Print the contents of the log file
        print("\nLog file contents:")
        with open(log_file_path, 'r') as f:
            print(f.read())
    else:
        print(f"Failed to create log file: {log_file_path}")


if __name__ == "__main__":
    main()
