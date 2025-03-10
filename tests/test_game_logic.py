#!/usr/bin/env python3
"""
Test script for the Hanabi game logic.

This script tests the core game logic components to ensure they work correctly.
"""

from hanabi_agents.game.engine import GameEngine
from hanabi_agents.game.state import GameState, Card, Color, ClueAction
import os
import sys
import unittest

# Add the parent directory to the path so we can import the hanabi_agents package
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))


class TestGameState(unittest.TestCase):
    """Test cases for the GameState class."""

    def test_initialization(self):
        """Test that a GameState can be initialized correctly."""
        state = GameState()
        self.assertEqual(state.clue_tokens, 8)
        self.assertEqual(state.fuse_tokens, 3)
        self.assertEqual(state.score, 0)
        self.assertEqual(state.turn_count, 0)
        self.assertFalse(state.game_over)

    def test_get_view_for(self):
        """Test that get_view_for returns a filtered view of the game state."""
        state = GameState()

        # Add some cards to hands
        state.hands[0] = [
            Card(color=Color.RED, number=1),
            Card(color=Color.BLUE, number=2)
        ]
        state.hands[1] = [
            Card(color=Color.GREEN, number=3),
            Card(color=Color.YELLOW, number=4)
        ]

        # Get view for player 0
        view = state.get_view_for(0)

        # Player 0 should not see their own cards
        self.assertFalse(view.hands[0][0].is_visible)
        self.assertFalse(view.hands[0][1].is_visible)

        # Player 0 should see player 1's cards
        self.assertEqual(view.hands[1][0].color, Color.GREEN)
        self.assertEqual(view.hands[1][0].number, 3)
        self.assertEqual(view.hands[1][1].color, Color.YELLOW)
        self.assertEqual(view.hands[1][1].number, 4)

    def test_clue_validation(self):
        """Test that clue validation works correctly."""
        # Valid color clue
        clue = ClueAction(type="color", value="red")
        self.assertEqual(clue.type, "color")
        self.assertEqual(clue.value, "red")

        # Valid number clue
        clue = ClueAction(type="number", value=3)
        self.assertEqual(clue.type, "number")
        self.assertEqual(clue.value, 3)

        # Invalid color
        with self.assertRaises(ValueError):
            ClueAction(type="color", value="purple")

        # Invalid number
        with self.assertRaises(ValueError):
            ClueAction(type="number", value=6)

        # Invalid type
        with self.assertRaises(ValueError):
            ClueAction(type="invalid", value="red")


class TestGameEngine(unittest.TestCase):
    """Test cases for the GameEngine class."""

    def test_initialization(self):
        """Test that a GameEngine can be initialized correctly."""
        engine = GameEngine(num_players=2)
        self.assertEqual(engine.num_players, 2)
        self.assertIsNotNone(engine.state)

        # Check that the deck was created correctly
        # 50 cards in deck, 5 cards per player
        self.assertEqual(len(engine.state.deck), 50 - 10)

        # Check that hands were dealt correctly
        self.assertEqual(len(engine.state.hands), 2)
        self.assertEqual(len(engine.state.hands[0]), 5)
        self.assertEqual(len(engine.state.hands[1]), 5)

    def test_execute_play_card(self):
        """Test that playing a card works correctly."""
        engine = GameEngine(num_players=2)

        # Set up a known game state
        engine.state.hands[0] = [
            Card(color=Color.RED, number=1),  # Playable
            Card(color=Color.BLUE, number=2)  # Not playable yet
        ]

        # Play the first card (should succeed)
        result = engine._execute_play_card(0, 0)
        self.assertTrue(result["success"])
        self.assertEqual(engine.state.score, 1)
        self.assertEqual(len(engine.state.firework_piles[Color.RED]), 1)

        # Check that the hand was updated
        self.assertEqual(len(engine.state.hands[0]), 2)  # Drew a new card

    def test_execute_give_clue(self):
        """Test that giving a clue works correctly."""
        engine = GameEngine(num_players=2)

        # Set up a known game state
        engine.state.hands[1] = [
            Card(color=Color.RED, number=1),
            Card(color=Color.RED, number=2)
        ]

        # Give a color clue
        clue = {"type": "color", "value": "red"}
        result = engine._execute_give_clue(0, 1, clue)

        self.assertTrue(result["success"])
        self.assertEqual(result["affected_indices"], [0, 1])
        self.assertEqual(engine.state.clue_tokens, 7)

        # Check that the cards were marked as clued
        self.assertTrue(engine.state.hands[1][0].color_clued)
        self.assertTrue(engine.state.hands[1][1].color_clued)

    def test_execute_discard(self):
        """Test that discarding a card works correctly."""
        engine = GameEngine(num_players=2)

        # Set up a known game state
        engine.state.hands[0] = [
            Card(color=Color.RED, number=1),
            Card(color=Color.BLUE, number=2)
        ]
        engine.state.clue_tokens = 7  # Use one token

        # Discard the first card
        result = engine._execute_discard(0, 0)

        self.assertTrue(result["success"])
        self.assertEqual(engine.state.clue_tokens, 8)
        self.assertEqual(len(engine.state.discard_pile), 1)
        self.assertEqual(engine.state.discard_pile[0].color, Color.RED)
        self.assertEqual(engine.state.discard_pile[0].number, 1)

        # Check that the hand was updated
        self.assertEqual(len(engine.state.hands[0]), 2)  # Drew a new card


if __name__ == "__main__":
    unittest.main()
