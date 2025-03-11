import os
import sys
from typing import Dict, Any
import logging
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from src.game.state import GameState, Card, Color, ClueAction
from src.agents.ai_agent import AIAgent
from src.agents.tools.play_card import _play_card_impl
from src.agents.tools.give_clue import _give_clue_impl
from src.agents.tools.discard import _discard_impl

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_test_game_state() -> GameState:
    """Create a test game state with some predefined cards and settings."""
    # Initialize a simple game state with all required fields
    game_state = GameState(
        deck=[],  # Empty deck for testing
        hands={
            0: [  # AI agent's hand
                Card(color=Color.RED, number=1),
                Card(color=Color.BLUE, number=2),
                Card(color=Color.GREEN, number=3),
                Card(color=Color.WHITE, number=4),
                Card(color=Color.YELLOW, number=5)
            ],
            1: [  # Other player's hand
                Card(color=Color.RED, number=1),
                Card(color=Color.RED, number=2),
                Card(color=Color.RED, number=3),
                Card(color=Color.RED, number=4),
                Card(color=Color.RED, number=5)
            ]
        },
        firework_piles={color: [] for color in Color},  # Empty firework piles
        discard_pile=[],  # Empty discard pile
        clue_tokens=4,  # Start with fewer clue tokens to allow discarding
        max_clue_tokens=8,
        fuse_tokens=3,
        max_fuse_tokens=3,
        current_player=0,  # AI agent's turn
        turn_count=0,
        game_over=False,
        score=0
    )

    return game_state


def test_tools(agent_id: int, game_state: GameState):
    """Test all available tools with the given game state."""
    logger.info("Testing all tools...")

    # Test play card tool
    logger.info("\nTesting play_card_tool...")
    if game_state.is_valid_move(agent_id, "play_card", card_index=0):
        result = _play_card_impl(agent_id, 0, game_state)
        logger.info(f"Play card result: {result}")
    else:
        logger.warning("Play card action is not valid")

    # Test give clue tool
    logger.info("\nTesting give_clue_tool...")
    clue = ClueAction(type="color", value="red")
    if game_state.is_valid_move(agent_id, "give_clue", target_id=1, clue=clue.model_dump()):
        result = _give_clue_impl(agent_id, 1, "color", "red", game_state)
        logger.info(f"Give clue result: {result}")
    else:
        logger.warning("Give clue action is not valid")

    # Test discard tool
    logger.info("\nTesting discard_tool...")
    if game_state.is_valid_move(agent_id, "discard", card_index=1):
        result = _discard_impl(agent_id, 1, game_state)
        logger.info(f"Discard result: {result}")
    else:
        logger.warning("Discard action is not valid")


def main():
    # Load environment variables
    load_dotenv()

    # Get the model name from environment
    model_name = os.getenv("MODEL_NAME", "o3-mini")
    logger.info(f"Using model: {model_name}")

    # Create a test game state
    game_state = create_test_game_state()

    # Initialize the AI agent with model_name
    agent = AIAgent(
        agent_id=0,
        model_name=model_name
    )

    # Test the tools
    test_tools(agent.agent_id, game_state)

    logger.info("\nAll tool tests completed!")


if __name__ == "__main__":
    main()
