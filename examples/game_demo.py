#!/usr/bin/env python3
"""
Demo script for the Hanabi game engine.

This script demonstrates how to use the Hanabi game engine without agents.
It manually executes actions to simulate a simple game.
"""

from hanabi_agents.utils.logging import setup_logging
from hanabi_agents.game.state import Color
from hanabi_agents.game.engine import GameEngine
import logging
import os
import sys

# Add the parent directory to the path so we can import the hanabi_agents package
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))


def main():
    """Run a simple demonstration of the Hanabi game engine."""
    # Set up logging
    logger = setup_logging(log_level=logging.INFO)

    # Note: When a player plays an invalid card, the game engine logs a warning about losing a fuse token.
    # This is expected game behavior, not a software issue.

    # Initialize game engine with 2 players and seed 1
    logger.info("Initializing game engine with 2 players and seed 1")
    engine = GameEngine(num_players=2, seed=1)

    # Print initial game state
    logger.info("Initial game state:")
    logger.info(f"  Deck size: {len(engine.state.deck)}")
    logger.info(
        f"  Player 0 hand: {[str(card) for card in engine.state.hands[0]]}")
    logger.info(
        f"  Player 1 hand: {[str(card) for card in engine.state.hands[1]]}")
    logger.info(f"  Clue tokens: {engine.state.clue_tokens}")
    logger.info(f"  Fuse tokens: {engine.state.fuse_tokens}")
    logger.info(f"  Score: {engine.state.score}")

    # Manually execute some actions
    logger.info("\nExecuting actions:")

    # Player 0 gives a clue to Player 1
    logger.info("\nPlayer 0 gives a color clue to Player 1")
    clue_action = {
        "type": "give_clue",
        "target_id": 1,
        "clue": {"type": "color", "value": engine.state.hands[1][0].color.value}
    }
    result = engine.execute_action(0, clue_action)
    logger.info(f"  Result: {result}")
    logger.info(f"  Clue tokens: {engine.state.clue_tokens}")

    # Player 1 plays a card (potentially invalid)
    logger.info(
        "\nPlayer 1 plays a card (potentially invalid - this may trigger a fuse token loss, which is normal game behavior)")
    play_action = {
        "type": "play_card",
        "card_index": 0
    }
    result = engine.execute_action(1, play_action)
    logger.info(f"  Result: {result}")
    logger.info(f"  Score: {engine.state.score}")
    logger.info(f"  Fuse tokens: {engine.state.fuse_tokens}")
    logger.info(
        f"  Firework piles: {[f'{color.value}: {[str(c) for c in pile]}' for color, pile in engine.state.firework_piles.items() if pile]}")

    # Player 0 discards a card
    logger.info("\nPlayer 0 discards a card")
    discard_action = {
        "type": "discard",
        "card_index": 0
    }
    result = engine.execute_action(0, discard_action)
    logger.info(f"  Result: {result}")
    logger.info(f"  Clue tokens: {engine.state.clue_tokens}")
    logger.info(
        f"  Discard pile: {[str(card) for card in engine.state.discard_pile]}")

    # Player 1 plays a valid card (additional turn)
    logger.info("\nPlayer 1 attempts to play a valid card")
    # Find a valid card in player 1's hand
    valid_card_found = False
    for card_index, card in enumerate(engine.state.hands[1]):
        color = card.color
        number = card.number
        pile = engine.state.firework_piles.get(color, [])

        # Check if this card is valid to play
        if (not pile and number == 1) or (pile and pile[-1].number == number - 1):
            valid_card_found = True
            logger.info(f"  Found valid card to play: {card}")
            play_action = {
                "type": "play_card",
                "card_index": card_index
            }
            result = engine.execute_action(1, play_action)
            logger.info(f"  Result: {result}")
            logger.info(f"  Score: {engine.state.score}")
            logger.info(f"  Firework piles: {[f'{color.value}: {[str(c) for c in pile]}' for color,
                        pile in engine.state.firework_piles.items() if pile]}")
            break

    if not valid_card_found:
        logger.info("  No valid cards found in Player 1's hand")

    # Print final game state
    logger.info("\nFinal game state:")
    logger.info(f"  Deck size: {len(engine.state.deck)}")
    logger.info(
        f"  Player 0 hand: {[str(card) for card in engine.state.hands[0]]}")
    logger.info(
        f"  Player 1 hand: {[str(card) for card in engine.state.hands[1]]}")
    logger.info(f"  Clue tokens: {engine.state.clue_tokens}")
    logger.info(f"  Fuse tokens: {engine.state.fuse_tokens}")
    logger.info(f"  Score: {engine.state.score}")
    logger.info(f"  Turn count: {engine.state.turn_count}")
    logger.info(f"  Firework piles: {[f'{color.value}: {[str(c) for c in pile]}' for color,
                pile in engine.state.firework_piles.items() if pile]}")

    # Print game summary
    logger.info("\nGame summary:")
    summary = engine.get_game_summary()
    for key, value in summary.items():
        logger.info(f"  {key}: {value}")


if __name__ == "__main__":
    main()
