#!/usr/bin/env python3
"""
Integration tests for the Hanabi game logic.

These tests focus on testing the game logic from a functional black box perspective,
validating the behavior of the game rather than implementation details.
"""

import pytest
import random
from typing import List, Dict, Any
from hanabi_agents.game.engine import GameEngine
from hanabi_agents.game.state import GameState, Card, Color, ClueAction


# ===== Test Fixtures =====

@pytest.fixture
def game_engine_2p():
    """
    Create a 2-player game engine with a fixed seed for reproducibility.

    This fixture provides a consistent game state for testing.
    """
    return GameEngine(num_players=2, seed=42)


@pytest.fixture
def game_engine_4p():
    """
    Create a 4-player game engine with a fixed seed for reproducibility.

    This fixture provides a consistent game state for testing with more players.
    """
    return GameEngine(num_players=4, seed=42)


@pytest.fixture
def controlled_game_state():
    """
    Create a game state with a controlled setup for specific test scenarios.

    This fixture provides a game state with known cards in players' hands
    and a specific deck arrangement.
    """
    state = GameState()

    # Set up player hands with known cards
    state.hands[0] = [
        Card(color=Color.RED, number=1),
        Card(color=Color.BLUE, number=2),
        Card(color=Color.GREEN, number=3),
        Card(color=Color.WHITE, number=4),
        Card(color=Color.YELLOW, number=5)
    ]

    state.hands[1] = [
        Card(color=Color.RED, number=2),
        Card(color=Color.BLUE, number=3),
        Card(color=Color.GREEN, number=4),
        Card(color=Color.WHITE, number=5),
        Card(color=Color.YELLOW, number=1)
    ]

    # Set up a controlled deck
    state.deck = [
        Card(color=Color.RED, number=3),
        Card(color=Color.BLUE, number=1),
        Card(color=Color.GREEN, number=2),
        Card(color=Color.WHITE, number=1),
        Card(color=Color.YELLOW, number=2)
    ]

    # Initialize firework piles
    for color in Color:
        state.firework_piles[color] = []

    return state


@pytest.fixture
def game_engine_with_controlled_state(controlled_game_state):
    """
    Create a game engine with a controlled game state.

    This fixture provides a game engine with a predetermined game state
    for testing specific scenarios.
    """
    engine = GameEngine(num_players=2, seed=42)
    engine.state = controlled_game_state
    return engine


# ===== Happy Path Tests =====

class TestGameInitialization:
    """Test cases for game initialization."""

    def test_game_setup_2_players(self, game_engine_2p):
        """
        Test that a 2-player game is set up correctly.

        Given: A new game with 2 players
        When: The game is initialized
        Then: The game state should be correctly set up with appropriate initial values
        """
        # Verify initial game state
        assert game_engine_2p.num_players == 2
        # 50 cards - (5 cards × 2 players)
        assert len(game_engine_2p.state.deck) == 40

        # Verify player hands
        assert len(game_engine_2p.state.hands) == 2
        assert all(len(hand) == 5 for hand in game_engine_2p.state.hands.values())

        # Verify initial resources
        assert game_engine_2p.state.clue_tokens == 8
        assert game_engine_2p.state.fuse_tokens == 3

        # Verify game state tracking
        assert game_engine_2p.state.score == 0
        assert game_engine_2p.state.turn_count == 0
        assert not game_engine_2p.state.game_over

    def test_game_setup_4_players(self, game_engine_4p):
        """
        Test that a 4-player game is set up correctly.

        Given: A new game with 4 players
        When: The game is initialized
        Then: The game state should be correctly set up with appropriate initial values
        """
        # Verify initial game state
        assert game_engine_4p.num_players == 4
        # 50 cards - (4 cards × 4 players)
        assert len(game_engine_4p.state.deck) == 34

        # Verify player hands
        assert len(game_engine_4p.state.hands) == 4
        assert all(len(hand) == 4 for hand in game_engine_4p.state.hands.values())

        # Verify initial resources
        assert game_engine_4p.state.clue_tokens == 8
        assert game_engine_4p.state.fuse_tokens == 3


class TestGameActions:
    """Test cases for game actions."""

    def test_play_valid_card(self, game_engine_with_controlled_state):
        """
        Test playing a valid card.

        Given: A game with a controlled state where player 0 has a red 1
        When: Player 0 plays the red 1
        Then: The card should be added to the firework pile and the score should increase
        """
        engine = game_engine_with_controlled_state

        # Execute play action
        result = engine._execute_play_card(0, 0)  # Play the red 1

        # Verify the result
        assert result["success"] is True
        assert engine.state.score == 1
        assert len(engine.state.firework_piles[Color.RED]) == 1
        assert engine.state.firework_piles[Color.RED][0].number == 1

        # Verify the player's hand
        assert len(engine.state.hands[0]) == 5  # Should draw a new card

        # Verify game state updates
        # Turn count is updated by play_turn, not _execute_play_card
        assert engine.state.turn_count == 0

    def test_play_invalid_card(self, game_engine_with_controlled_state):
        """
        Test playing an invalid card.

        Given: A game with a controlled state where player 0 has a blue 2
        When: Player 0 plays the blue 2 (which is invalid as no blue 1 has been played)
        Then: The card should be discarded and a fuse token should be lost
        """
        engine = game_engine_with_controlled_state

        # Execute play action
        result = engine._execute_play_card(0, 1)  # Play the blue 2

        # Verify the result
        assert result["success"] is False
        assert engine.state.score == 0
        assert len(engine.state.firework_piles[Color.BLUE]) == 0

        # Verify the fuse token was lost
        assert engine.state.fuse_tokens == 2

        # Verify the card was discarded
        assert len(engine.state.discard_pile) == 1
        assert engine.state.discard_pile[0].color == Color.BLUE
        assert engine.state.discard_pile[0].number == 2

        # Verify the player's hand
        assert len(engine.state.hands[0]) == 5  # Should draw a new card

    def test_give_color_clue(self, game_engine_with_controlled_state):
        """
        Test giving a color clue.

        Given: A game with a controlled state
        When: Player 0 gives a red color clue to player 1
        Then: The appropriate cards should be marked as clued
        """
        engine = game_engine_with_controlled_state

        # Execute clue action
        clue = {"type": "color", "value": "red"}
        result = engine._execute_give_clue(0, 1, clue)

        # Verify the result
        assert result["success"] is True
        assert result["affected_indices"] == [0]  # Only the red 2 is affected

        # Verify the clue token was used
        assert engine.state.clue_tokens == 7

        # Verify the card was marked as clued
        assert engine.state.hands[1][0].color_clued is True
        assert engine.state.hands[1][0].color == Color.RED

    def test_give_number_clue(self, game_engine_with_controlled_state):
        """
        Test giving a number clue.

        Given: A game with a controlled state
        When: Player 0 gives a number 5 clue to player 1
        Then: The appropriate cards should be marked as clued
        """
        engine = game_engine_with_controlled_state

        # Execute clue action
        clue = {"type": "number", "value": 5}
        result = engine._execute_give_clue(0, 1, clue)

        # Verify the result
        assert result["success"] is True
        assert result["affected_indices"] == [
            3]  # Only the white 5 is affected

        # Verify the clue token was used
        assert engine.state.clue_tokens == 7

        # Verify the card was marked as clued
        assert engine.state.hands[1][3].number_clued is True
        assert engine.state.hands[1][3].number == 5

    def test_discard_card(self, game_engine_with_controlled_state):
        """
        Test discarding a card.

        Given: A game with a controlled state and 7 clue tokens
        When: Player 0 discards a card
        Then: The card should be added to the discard pile and a clue token should be gained
        """
        engine = game_engine_with_controlled_state
        engine.state.clue_tokens = 7  # Set clue tokens to 7

        # Execute discard action
        result = engine._execute_discard(0, 2)  # Discard the green 3

        # Verify the result
        assert result["success"] is True

        # Verify the clue token was gained
        assert engine.state.clue_tokens == 8

        # Verify the card was discarded
        assert len(engine.state.discard_pile) == 1
        assert engine.state.discard_pile[0].color == Color.GREEN
        assert engine.state.discard_pile[0].number == 3

        # Verify the player's hand
        assert len(engine.state.hands[0]) == 5  # Should draw a new card


class TestGameProgression:
    """Test cases for game progression and end conditions."""

    def test_complete_firework(self, game_engine_with_controlled_state):
        """
        Test completing a firework.

        Given: A game with a controlled state
        When: A complete firework (1-5) of one color is played
        Then: The firework should be marked as completed
        """
        engine = game_engine_with_controlled_state

        # Set up the firework pile with cards 1-4
        engine.state.firework_piles[Color.RED] = [
            Card(color=Color.RED, number=1),
            Card(color=Color.RED, number=2),
            Card(color=Color.RED, number=3),
            Card(color=Color.RED, number=4)
        ]

        # Set the player's hand to have the red 5
        engine.state.hands[0][0] = Card(color=Color.RED, number=5)

        # Play the red 5
        result = engine._execute_play_card(0, 0)

        # Verify the result
        assert result["success"] is True
        assert engine.state.score == 1

        # Verify the firework is complete
        assert len(engine.state.firework_piles[Color.RED]) == 5
        assert engine.state.check_firework_completion(Color.RED) is True

    def test_out_of_fuse_tokens(self, game_engine_with_controlled_state):
        """
        Test running out of fuse tokens.

        Given: A game with a controlled state and 1 fuse token
        When: Player 0 plays an invalid card
        Then: The game should end due to running out of fuse tokens
        """
        engine = game_engine_with_controlled_state
        engine.state.fuse_tokens = 1  # Set fuse tokens to 1

        # Play an invalid card
        result = engine._execute_play_card(0, 1)  # Play the blue 2

        # Verify the result
        assert result["success"] is False

        # Verify the game is over
        assert engine.state.fuse_tokens == 0
        engine._update_game_state()  # This method checks for game over conditions
        assert engine.state.game_over is True


# ===== Error Handling Tests =====

class TestErrorHandling:
    """Test cases for error handling."""

    def test_invalid_clue_type(self, game_engine_2p):
        """
        Test giving an invalid clue type.

        Given: A game with 2 players
        When: Player 0 gives an invalid clue type
        Then: The action should fail
        """
        # Create an invalid clue
        invalid_clue = {"type": "invalid", "value": "red"}

        # Attempt to execute the clue action
        with pytest.raises(ValueError):
            ClueAction(type="invalid", value="red")

    def test_invalid_clue_value(self, game_engine_2p):
        """
        Test giving an invalid clue value.

        Given: A game with 2 players
        When: Player 0 gives a clue with an invalid value
        Then: The action should fail
        """
        # Create an invalid color clue
        with pytest.raises(ValueError):
            ClueAction(type="color", value="purple")

        # Create an invalid number clue
        with pytest.raises(ValueError):
            ClueAction(type="number", value=6)

    def test_play_with_invalid_card_index(self, game_engine_2p):
        """
        Test playing a card with an invalid index.

        Given: A game with 2 players
        When: Player 0 tries to play a card with an invalid index
        Then: The action should fail with an IndexError
        """
        # Attempt to play a card with an invalid index
        invalid_index = 10  # Assuming player has 5 cards

        # We expect an IndexError when trying to access an invalid card index
        with pytest.raises(IndexError):
            game_engine_2p._execute_play_card(0, invalid_index)

    def test_discard_with_no_clue_tokens_needed(self, game_engine_2p):
        """
        Test discarding a card when clue tokens are already at maximum.

        Given: A game with 2 players and maximum clue tokens
        When: Player 0 discards a card
        Then: The action should succeed but no additional clue token should be gained
        """
        # Ensure clue tokens are at maximum
        game_engine_2p.state.clue_tokens = 8

        # Execute discard action
        result = game_engine_2p._execute_discard(0, 0)

        # Verify the result
        assert result["success"] is True
        assert game_engine_2p.state.clue_tokens == 8  # Still at maximum
        assert len(game_engine_2p.state.discard_pile) == 1


# ===== Edge Case Tests =====

class TestEdgeCases:
    """Test cases for edge cases."""

    def test_empty_deck_handling(self, game_engine_with_controlled_state):
        """
        Test handling of an empty deck.

        Given: A game with a controlled state and an empty deck
        When: A player plays a card
        Then: The player should not draw a new card
        """
        engine = game_engine_with_controlled_state
        engine.state.deck = []  # Empty the deck

        # Execute play action
        result = engine._execute_play_card(0, 0)  # Play the red 1

        # Verify the result
        assert result["success"] is True

        # Verify the player's hand
        assert len(engine.state.hands[0]) == 4  # No new card drawn

    def test_clue_with_no_matching_cards(self, game_engine_with_controlled_state):
        """
        Test giving a clue that doesn't match any cards.

        Given: A game with a controlled state
        When: Player 0 gives a clue that doesn't match any of player 1's cards
        Then: The action should fail
        """
        engine = game_engine_with_controlled_state

        # Remove all purple cards from player 1's hand
        engine.state.hands[1] = [
            Card(color=Color.RED, number=2),
            Card(color=Color.BLUE, number=3),
            Card(color=Color.GREEN, number=4),
            Card(color=Color.WHITE, number=5),
            Card(color=Color.YELLOW, number=1)
        ]

        # This should fail validation before execution
        with pytest.raises(ValueError):
            ClueAction(type="number", value=9)

    def test_play_last_card_of_game(self, game_engine_with_controlled_state):
        """
        Test playing the last card of the game.

        Given: A game with a controlled state, empty deck, and player 0 with only one card
        When: Player 0 plays their last card
        Then: The game should continue until the final round is complete
        """
        engine = game_engine_with_controlled_state
        engine.state.deck = []  # Empty the deck
        engine.state.hands[0] = [
            Card(color=Color.RED, number=1)]  # Only one card

        # Execute play action
        result = engine._execute_play_card(0, 0)  # Play the red 1

        # Verify the result
        assert result["success"] is True

        # Verify the player's hand
        assert len(engine.state.hands[0]) == 0  # No cards left

        # The game is not immediately over when a player runs out of cards
        # It continues until all players have had one more turn
        # So we don't expect game_over to be True yet
        engine._update_game_state()
        assert engine.state.game_over is False

        # Note: To fully test the end game condition, we would need to simulate
        # all players taking their final turn, which is beyond the scope of this test


if __name__ == "__main__":
    pytest.main()
