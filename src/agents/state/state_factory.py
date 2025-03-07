from typing import Dict, Any, List
from ...game.state import GameState
from langchain_core.messages import HumanMessage


def create_initial_state(
    game_state: GameState,
    discussion_strings: List[str],
    game_history_strings: List[str]
) -> Dict[str, Any]:
    """
    Create the initial state for the reasoning graph (discussion phase).

    Args:
        game_state: Current game state
        discussion_strings: Processed discussion history
        game_history_strings: Processed game history

    Returns:
        Initial state dictionary for the reasoning graph
    """
    return {
        "game_state": game_state,
        "discussion_history": discussion_strings,
        "game_history": game_history_strings,
        "current_thoughts": []
    }


def create_action_state(
    game_state: GameState,
    discussion_strings: List[str],
    game_history_strings: List[str],
    memory: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Create the initial state for the reasoning graph (action phase).

    Args:
        game_state: Current game state
        discussion_strings: Processed discussion history
        game_history_strings: Processed game history
        memory: Optional memory dictionary containing agent's memory

    Returns:
        Initial state dictionary for the reasoning graph with messages
    """
    # Create the basic state
    state = create_initial_state(
        game_state, discussion_strings, game_history_strings)

    # Add messages for action phase
    state["messages"] = [HumanMessage(
        content="It's your turn to take an action in the Hanabi game.")]

    # If we have memory with proposed tool calls from the discussion phase, include them
    if memory and "proposed_tool_calls" in memory:
        state["proposed_tool_calls"] = memory["proposed_tool_calls"]

    return state
