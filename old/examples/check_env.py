from src.utils.env import load_environment_variables
import os
import logging
import sys
from pathlib import Path
import datetime

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# Set up logging
project_root = Path(__file__).parent.parent
logs_dir = project_root / "logs"
logs_dir.mkdir(exist_ok=True)

# Generate a timestamped log filename
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = logs_dir / f"env_check_{timestamp}.log"

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def check_environment():
    """Check if the environment is set up correctly."""
    logger.info("Checking environment setup...")

    # Check Python version
    python_version = sys.version
    logger.info(f"Python version: {python_version}")

    # Check current directory
    current_dir = Path.cwd()
    logger.info(f"Current directory: {current_dir}")

    # Check if .env file exists
    env_file = current_dir / ".env"
    if env_file.exists():
        logger.info(f".env file found at {env_file}")
    else:
        logger.warning(f".env file not found at {env_file}")

        # Check other possible locations
        other_locations = [
            current_dir.parent / ".env",
            project_root / ".env"
        ]

        for loc in other_locations:
            if loc.exists():
                logger.info(f".env file found at {loc}")
                break
        else:
            logger.warning("No .env file found in any expected location")

    # Try to load environment variables
    if load_environment_variables():
        logger.info("Environment variables loaded successfully")

        # Check OpenAI API key
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            logger.info(f"OPENAI_API_KEY found: {api_key[:5]}...")
        else:
            logger.error("OPENAI_API_KEY not found")

        # Check model name
        model_name = os.getenv("MODEL_NAME")
        if model_name:
            logger.info(f"MODEL_NAME found: {model_name}")
        else:
            logger.warning("MODEL_NAME not found, will use default")
    else:
        logger.error("Failed to load environment variables")

    # Check required packages
    try:
        import langchain_openai
        logger.info("langchain_openai package found")
    except ImportError:
        logger.error("langchain_openai package not found")

    try:
        import langgraph
        logger.info("langgraph package found")
    except ImportError:
        logger.error("langgraph package not found")

    try:
        import pydantic
        logger.info("pydantic package found")
    except ImportError:
        logger.error("pydantic package not found")

    try:
        import dotenv
        logger.info("python-dotenv package found")
    except ImportError:
        logger.error("python-dotenv package not found")

    logger.info("Environment check complete")


if __name__ == "__main__":
    check_environment()
