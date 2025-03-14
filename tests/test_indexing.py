import pytest
from hanabi_agents.agents.ai_agent import AIAgent
import logging

# Configure logging for tests
logging.basicConfig(level=logging.INFO)


class TestIndexing:
    """Tests for verifying correct handling of 1-indexed to 0-indexed conversion."""

    def test_to_zero_indexed_conversion(self):
        """Test the to_zero_indexed method."""
        agent = AIAgent(agent_id=0, model_name="test")

        # Test conversion of valid 1-indexed values
        assert agent.to_zero_indexed(1) == 0
        assert agent.to_zero_indexed(2) == 1
        assert agent.to_zero_indexed(5) == 4

        # Test handling of invalid values
        assert agent.to_zero_indexed(0) == 0  # Should prevent negative result
        assert agent.to_zero_indexed("1") == 0  # Should handle string input
        assert agent.to_zero_indexed(
            "not a number") == "not a number"  # Should return unchanged

    def test_verify_tool_call_indexing(self):
        """Test that verify_tool_call properly converts card indices."""
        agent = AIAgent(agent_id=0, model_name="test")

        # Test play_card tool with 1-indexed position
        play_tool_call = {
            "name": "play_card_tool",
            "args": {"card_index": 3}  # 1-indexed position (3rd card)
        }

        # Should convert to 0-indexed (2)
        verified = agent.verify_tool_call(play_tool_call)
        assert verified["name"] == "play_card_tool"
        assert verified["args"]["card_index"] == 2  # 0-indexed position

        # Test discard tool with 1-indexed position
        discard_tool_call = {
            "name": "discard_tool",
            "args": {"card_index": 5}  # 1-indexed position (5th card)
        }

        # Should convert to 0-indexed (4)
        verified = agent.verify_tool_call(discard_tool_call)
        assert verified["name"] == "discard_tool"
        assert verified["args"]["card_index"] == 4  # 0-indexed position

        # Test handling of out-of-range values (with hand size constraint)
        agent.current_game_state = type('obj', (object,), {
            'hands': {0: [1, 2, 3]}  # Hand with 3 cards
        })

        out_of_range_call = {
            "name": "play_card_tool",
            # 1-indexed position (10th card, too large)
            "args": {"card_index": 10}
        }

        # Should clamp to hand size and convert to 0-indexed (2)
        verified = agent.verify_tool_call(out_of_range_call)
        # 0-indexed position (3rd card)
        assert verified["args"]["card_index"] == 2

    def test_validation_after_conversion(self):
        """Test that validation properly handles 0-indexed values after conversion."""
        from hanabi_agents.game.state import GameState
        from hanabi_agents.game.state import Card, Color

        # Create a simple game state with 3 cards in player 0's hand
        game_state = GameState(
            hands={
                0: [
                    Card(color=Color.RED, number=1),
                    Card(color=Color.BLUE, number=2),
                    Card(color=Color.GREEN, number=3)
                ]
            }
        )

        # Verify that the game state accepts 0-indexed positions
        assert game_state.is_valid_move(
            0, "play_card", card_index=0)  # Valid 0-indexed position
        assert game_state.is_valid_move(
            0, "play_card", card_index=2)  # Valid 0-indexed position
        assert not game_state.is_valid_move(
            0, "play_card", card_index=3)  # Invalid 0-indexed position
