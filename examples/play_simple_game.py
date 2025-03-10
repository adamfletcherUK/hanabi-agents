from hanabi_agents.game.engine import GameEngine
from hanabi_agents.utils import logging as hanabi_logging
from hanabi_agents.utils import game_logger
import os
import sys
import logging
import random
from dotenv import load_dotenv

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
load_dotenv()

# Set up logging
log_level_str = os.getenv("LOG_LEVEL", "INFO")
log_level = getattr(logging, log_level_str.upper(), logging.INFO)
hanabi_logging.setup_logging(log_level=log_level)

logger = logging.getLogger(__name__)

# Game configuration
NUM_PLAYERS = 2
MAX_TURNS = 10


def random_action(game_state, player_id):
    """Generate a random valid action for the given player."""
    # Get available actions
    actions = []

    # Add play card actions
    for i in range(len(game_state.hands[player_id])):
        actions.append({"type": "play_card", "card_index": i})

    # Add discard actions if we don't have max clue tokens
    if game_state.clue_tokens < game_state.max_clue_tokens:
        for i in range(len(game_state.hands[player_id])):
            actions.append({"type": "discard", "card_index": i})

    # Add clue actions if we have clue tokens
    if game_state.clue_tokens > 0:
        for target_id in game_state.hands.keys():
            if target_id != player_id:  # Can't clue yourself
                # Color clues
                for color in ["red", "yellow", "green", "blue", "white"]:
                    actions.append({
                        "type": "give_clue",
                        "target_id": target_id,
                        "clue": {"type": "color", "value": color}
                    })

                # Number clues
                for number in range(1, 6):
                    actions.append({
                        "type": "give_clue",
                        "target_id": target_id,
                        "clue": {"type": "number", "value": number}
                    })

    # Choose a random action
    return random.choice(actions)


def main():
    """Run a simple game of Hanabi with random actions."""
    # Create the game engine
    logger.info(f"Creating game engine with {NUM_PLAYERS} players")
    engine = GameEngine(num_players=NUM_PLAYERS)

    # Initialize the game
    logger.info("Initializing game")
    print("\n🎮 STARTING NEW GAME OF HANABI 🎮")
    print("=" * 50)

    # Log initial game state
    game_logger.log_game_state(engine, print_to_console=True)

    # Play the game
    game_over = False
    turn_count = 0

    while not game_over and turn_count < MAX_TURNS:
        # Get the current game state
        game_state = engine.get_game_state()

        # Get the current player
        current_player_id = game_state.current_player

        # Log turn information
        game_logger.log_turn_info(
            turn_count + 1,
            f"Player {current_player_id}",
            current_player_id,
            game_state.clue_tokens,
            game_state.max_clue_tokens,
            game_state.fuse_tokens,
            game_state.score,
            print_to_console=True
        )

        # Log detailed game state
        game_logger.log_game_state(engine, print_to_console=True)

        # Generate a random action
        print("\n--- 🎲 Random Action Phase ---")
        action = random_action(game_state, current_player_id)
        logger.info(f"Random action: {action}")

        # Display the formatted action
        action_display = game_logger.format_action_for_display(
            action, f"Player {current_player_id}")
        print(action_display)

        # Execute the action
        result = engine.execute_action(current_player_id, action)
        logger.info(f"Action result: {result}")

        # Log the action result
        game_logger.log_action_result(
            action, result, f"Player {current_player_id}", print_to_console=True)

        # Log the updated game state after the action
        game_logger.log_game_state(engine, print_to_console=True)

        # Check if the game is over
        game_over = engine.is_game_over()

        # Increment the turn count
        turn_count += 1

    # Log game over information
    game_state = engine.get_game_state()
    game_logger.log_game_over(
        game_state.score,
        engine.get_game_over_reason(),
        print_to_console=True
    )


if __name__ == "__main__":
    main()
