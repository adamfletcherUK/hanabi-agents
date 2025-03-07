import pytest
from src.game.state import GameState, Color, Card
from src.game.engine import GameEngine
from src.agents.ai_agent import AIAgent
from src.communication.discussion import DiscussionManager


@pytest.fixture
def game_state():
    """Create a basic game state for testing."""
    state = GameState()
    # Add some cards to hands
    state.hands[0] = [
        Card(color=Color.RED, number=1),
        Card(color=Color.BLUE, number=2),
        Card(color=Color.GREEN, number=3)
    ]
    state.hands[1] = [
        Card(color=Color.YELLOW, number=1),
        Card(color=Color.WHITE, number=2),
        Card(color=Color.RED, number=2)
    ]
    return state


def test_game_state_validation(game_state):
    """Test basic game state validation."""
    # Test valid play
    assert game_state.is_valid_move(0, "play_card", card_index=0)

    # Test invalid play (wrong player)
    assert not game_state.is_valid_move(1, "play_card", card_index=0)

    # Test valid clue
    assert game_state.is_valid_move(0, "give_clue",
                                    target_id=1,
                                    clue={"type": "color", "value": "red"})

    # Test invalid clue (no tokens)
    game_state.clue_tokens = 0
    assert not game_state.is_valid_move(0, "give_clue",
                                        target_id=1,
                                        clue={"type": "color", "value": "red"})


def test_discussion_manager():
    """Test basic discussion manager functionality."""
    manager = DiscussionManager(max_rounds=2)

    # Test adding contributions
    manager.add_contribution(0, "I think we should play the red 1")
    manager.add_contribution(1, "Agreed, that's a safe play")

    # Test discussion summary
    summary = manager.get_discussion_summary()
    assert "Agent 0" in summary
    assert "Agent 1" in summary

    # Test consensus
    assert not manager.has_reached_consensus()

    # Add more contributions to reach consensus
    manager.add_contribution(2, "Yes, let's play red 1")
    manager.add_contribution(3, "I agree with playing red 1")
    manager.add_contribution(4, "Red 1 is the best play")

    assert manager.has_reached_consensus()


def test_game_engine_initialization():
    """Test game engine initialization."""
    engine = GameEngine()
    assert len(engine.state.hands) == 5  # 5 players
    assert engine.state.clue_tokens == 8
    assert engine.state.fuse_tokens == 3
    assert not engine.state.game_over


def test_game_engine_action_execution():
    """Test game engine action execution."""
    engine = GameEngine()

    # Test playing a card
    action = {
        "type": "play_card",
        "card_index": 0
    }
    success = engine.execute_action(0, action)
    assert success

    # Test giving a clue
    action = {
        "type": "give_clue",
        "target_id": 1,
        "clue": {"type": "color", "value": "red"}
    }
    success = engine.execute_action(1, action)
    assert success

    # Test discarding
    action = {
        "type": "discard",
        "card_index": 0
    }
    success = engine.execute_action(2, action)
    assert success
