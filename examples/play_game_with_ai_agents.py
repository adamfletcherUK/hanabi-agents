from hanabi_agents.agents.discussion.discussion_manager import DiscussionManager
from hanabi_agents.agents.ai_agent import AIAgent
from hanabi_agents.game.engine import GameEngine
import os
import sys
import logging
from dotenv import load_dotenv

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
load_dotenv()

# Set up logging based on environment variables
log_level_str = os.getenv("LOG_LEVEL", "INFO")
log_level = getattr(logging, log_level_str.upper(), logging.INFO)

# Configure basic console logging
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Add file logging if enabled
if os.getenv("ENABLE_FILE_LOGGING", "false").lower() == "true":
    log_file_path = os.getenv("LOG_FILE_PATH", "logs/hanabi.log")

    # Create directory for log file if it doesn't exist
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    # Add file handler
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(file_handler)

logger = logging.getLogger(__name__)

# Get game configuration from environment
max_turns = int(os.getenv("MAX_TURNS", "20"))
num_players = int(os.getenv("NUM_PLAYERS", "3"))
model_name = os.getenv("MODEL_NAME", "gpt-4-turbo")


def main():
    """Run a game of Hanabi with AI agents."""
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY not found in environment variables")
        print("Please set your OPENAI_API_KEY in the .env file or environment")
        return

    # Create the game engine
    logger.info(f"Creating game engine with {num_players} players")
    engine = GameEngine(num_players=num_players)

    # Create the AI agents
    logger.info(f"Creating AI agents using model: {model_name}")
    agents = [
        AIAgent(agent_id=i, name=f"Player {i}", model_name=model_name)
        for i in range(num_players)
    ]

    # Create the discussion manager
    discussion_manager = DiscussionManager()

    # Initialize the game
    logger.info("Initializing game")
    engine._initialize_game()

    # Play the game
    game_over = False
    turn_count = 0

    while not game_over and turn_count < max_turns:
        # Get the current game state
        game_state = engine.get_game_summary()

        # Get the current player
        current_player_id = game_state.current_player
        current_agent = next(
            agent for agent in agents if agent.agent_id == current_player_id)

        # Log the current state
        logger.info(f"=== Turn {turn_count + 1} ===")
        logger.info(
            f"Current player: {current_agent.name} (Player {current_player_id})")
        logger.info(
            f"Clue tokens: {game_state.clue_tokens}/{game_state.max_clue_tokens}")
        logger.info(f"Fuse tokens: {game_state.fuse_tokens}/3")
        logger.info(f"Score: {game_state.score}/25")

        # Print the current state for user
        print(f"\n=== Turn {turn_count + 1} ===")
        print(
            f"Current player: {current_agent.name} (Player {current_player_id})")
        print(
            f"Clue tokens: {game_state.clue_tokens}/{game_state.max_clue_tokens}")
        print(f"Fuse tokens: {game_state.fuse_tokens}/3")
        print(f"Score: {game_state.score}/25")

        # Conduct a discussion
        logger.info("Starting discussion phase")
        print("\n--- Discussion Phase ---")
        discussion_summary = discussion_manager.conduct_discussion(
            game_state, agents)
        logger.info(f"Discussion summary: {discussion_summary}")
        print(f"\nDiscussion Summary:\n{discussion_summary}")

        # Decide on an action
        logger.info("Starting action phase")
        print("\n--- Action Phase ---")
        action = current_agent.decide_action(game_state, discussion_summary)
        logger.info(f"Action decided: {action}")
        print(f"{current_agent.name} decides to: {action}")

        # Execute the action
        result = engine.execute_action(current_player_id, action)
        logger.info(f"Action result: {result}")

        # Notify the agent of the result
        current_agent.notify_action_result(action, result)

        # Check if the game is over
        game_over = engine.is_game_over()

        # Increment the turn count
        turn_count += 1

    # Print the final state
    game_state = engine.get_game_summary()
    logger.info("=== Game Over ===")
    logger.info(f"Final score: {game_state.score}/25")
    logger.info(f"Reason: {engine.get_game_over_reason()}")

    print("\n=== Game Over ===")
    print(f"Final score: {game_state.score}/25")
    print(f"Reason: {engine.get_game_over_reason()}")


if __name__ == "__main__":
    main()
