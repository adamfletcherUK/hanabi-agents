from typing import Dict, Any, List, Tuple


def process_discussion_history(discussion_history: list, memory: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """
    Process discussion history into strings and extract game history.

    Args:
        discussion_history: Raw discussion history (may be objects or strings)
        memory: Agent's memory dictionary

    Returns:
        Tuple of (discussion_strings, game_history_strings)
    """
    # Check if discussion_history contains objects or strings
    has_discussion_objects = False
    has_game_history = False

    if discussion_history and hasattr(discussion_history[0], 'content'):
        has_discussion_objects = True

    # Check for game history (entries with turn_number attribute)
    if discussion_history and hasattr(discussion_history[0], 'turn_number'):
        has_game_history = True

    # Convert discussion history to strings
    discussion_strings = []
    game_history_strings = []

    # Process based on the type of history objects
    if has_game_history:
        # Separate current discussion from previous turns
        current_turn = max(
            entry.turn_number for entry in discussion_history)
        current_discussion = [
            entry for entry in discussion_history if entry.turn_number == current_turn]
        previous_discussions = [
            entry for entry in discussion_history if entry.turn_number < current_turn]

        # Convert to strings
        discussion_strings = [
            entry.content for entry in current_discussion]
        game_history_strings = [f"Turn {entry.turn_number+1}, Agent {entry.agent_id}: {entry.content}"
                                for entry in previous_discussions]

        # Update memory with latest game history
        if memory is not None:
            memory["game_history"] = game_history_strings
    elif has_discussion_objects:
        discussion_strings = [
            entry.content for entry in discussion_history]
    else:
        # Already strings
        discussion_strings = discussion_history

    # Get game history from memory if not in discussion
    if not has_game_history and memory is not None and "game_history" in memory:
        game_history_strings = memory["game_history"]

    return discussion_strings, game_history_strings
