from typing import Dict, Any, List
from ...game.state import GameState
from langchain_core.messages import HumanMessage
from ..utils.discussion import process_discussion_history


def create_initial_state(
    game_state: GameState,
    discussion_history: List[str] or str
) -> Dict[str, Any]:
    """
    Create the initial state for the reasoning graph (discussion phase).

    Args:
        game_state: Current game state
        discussion_history: Discussion history (either as a list or a string)

    Returns:
        Initial state dictionary for the reasoning graph
    """
    # Process discussion history if it's a list
    if isinstance(discussion_history, list):
        discussion_strings, game_history_strings = process_discussion_history(
            discussion_history, {})
    else:
        # If it's already a string, use it directly
        discussion_strings = [discussion_history] if discussion_history else []
        game_history_strings = []

    # Create the initial state with all required fields
    state = {
        "game_state": game_state,
        "discussion_history": discussion_strings,
        "game_history": game_history_strings,
        "current_thoughts": [],
        "execution_path": [],
        "is_action_phase": False  # Default to discussion phase
    }

    return state


def create_action_state(
    game_state: GameState,
    discussion_summary: str or List[str],
    memory: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Create the initial state for the reasoning graph (action phase).

    Args:
        game_state: Current game state
        discussion_summary: Summary of the discussion (either as a string or a list)
        memory: Optional memory dictionary containing agent's memory

    Returns:
        Initial state dictionary for the reasoning graph with messages
    """
    # Create the basic state
    state = create_initial_state(game_state, discussion_summary)

    # Format the discussion summary for the message
    formatted_summary = discussion_summary
    if isinstance(discussion_summary, list):
        formatted_summary = "\n".join(discussion_summary)

    # Add messages for action phase - ensure we always have this key
    state["messages"] = [HumanMessage(
        content=f"It's your turn to take an action in the Hanabi game. Here's the discussion summary:\n{formatted_summary}"
    )]

    # If we have memory with proposed tool calls from the discussion phase, include them
    if memory and "proposed_tool_calls" in memory:
        state["proposed_tool_calls"] = memory["proposed_tool_calls"]

    # Add any game history from memory if available
    if memory and "game_history" in memory:
        state["game_history"].extend(memory["game_history"])

    # Add any previous actions from memory if available
    if memory and "previous_actions" in memory:
        state["previous_actions"] = memory["previous_actions"]

    # Add a flag to indicate this is the action phase
    state["is_action_phase"] = True

    # Add a flag to indicate the node execution path
    state["execution_path"] = []

    return state
