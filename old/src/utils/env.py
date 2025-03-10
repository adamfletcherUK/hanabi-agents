import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# Set up logging
logger = logging.getLogger(__name__)


def load_environment_variables():
    """Load environment variables from .env file."""
    # Try to find .env file in different locations
    possible_locations = [
        Path.cwd() / ".env",                    # Current directory
        Path.cwd().parent / ".env",             # Parent directory
        Path(__file__).parent.parent.parent / ".env"  # Project root
    ]

    env_loaded = False
    for env_path in possible_locations:
        if env_path.exists():
            logger.info(f"Loading environment variables from {env_path}")
            env_loaded = load_dotenv(dotenv_path=env_path)
            if env_loaded:
                break

    if not env_loaded:
        logger.warning(
            "No .env file found. Checking environment variables directly.")

    # Check for required environment variables
    required_vars = ["OPENAI_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]

    if missing_vars:
        logger.error(
            f"Missing required environment variables: {', '.join(missing_vars)}")
        logger.error(
            "Please set these variables in your .env file or environment")
        return False

    logger.info("All required environment variables are set")
    return True
