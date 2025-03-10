from typing import Dict, Any, List
from ...game.state import GameState, Color


def give_clue_tool(agent_id: int, target_id: int, clue_type: str, clue_value: Any, game_state: GameState) -> Dict[str, Any]:
    """
    Tool for giving a clue to another player.

    Args:
        agent_id: ID of the agent giving the clue
        target_id: ID of the player receiving the clue
        clue_type: Type of clue ("color" or "number")
        clue_value: Value of the clue (color name or number)
        game_state: Current state of the game

    Returns:
        Dictionary with the result of the action
    """
    # Validate the clue type
    if clue_type not in ["color", "number"]:
        return {
            "success": False,
            "error": f"Invalid clue type: {clue_type}. Must be 'color' or 'number'.",
            "action_type": "clue"
        }

    # Validate the clue value
    if clue_type == "color":
        # Convert string to Color enum if needed
        if isinstance(clue_value, str):
            try:
                clue_value = clue_value.lower()
                # Check if the value is a valid color
                if clue_value not in [c.value for c in Color]:
                    return {
                        "success": False,
                        "error": f"Invalid color value: {clue_value}",
                        "action_type": "clue"
                    }
            except (ValueError, AttributeError):
                return {
                    "success": False,
                    "error": f"Invalid color value: {clue_value}",
                    "action_type": "clue"
                }
    elif clue_type == "number":
        # Convert string to int if needed
        if isinstance(clue_value, str):
            try:
                clue_value = int(clue_value)
            except ValueError:
                return {
                    "success": False,
                    "error": f"Invalid number value: {clue_value}",
                    "action_type": "clue"
                }

        # Check if the number is valid
        if not (1 <= clue_value <= 5):
            return {
                "success": False,
                "error": f"Invalid number value: {clue_value}. Must be between 1 and 5.",
                "action_type": "clue"
            }

    # Validate the action
    if not game_state.is_valid_move(agent_id, "clue", target_id=target_id, clue_type=clue_type, clue_value=clue_value):
        return {
            "success": False,
            "error": f"Invalid clue action: target_id={target_id}, clue_type={clue_type}, clue_value={clue_value}",
            "action_type": "clue"
        }

    # Get the affected cards
    affected_indices = _get_affected_indices(
        game_state, target_id, clue_type, clue_value)

    return {
        "success": True,
        "action_type": "clue",
        "target_id": target_id,
        "clue_type": clue_type,
        "clue_value": clue_value,
        "affected_indices": affected_indices,
        "message": f"Giving {clue_type} clue {clue_value} to player {target_id}, affecting cards at indices {affected_indices}"
    }


def _get_affected_indices(game_state: GameState, target_id: int, clue_type: str, clue_value: Any) -> List[int]:
    """
    Get the indices of cards affected by a clue.

    Args:
        game_state: Current state of the game
        target_id: ID of the player receiving the clue
        clue_type: Type of clue ("color" or "number")
        clue_value: Value of the clue (color name or number)

    Returns:
        List of indices of affected cards
    """
    affected_indices = []

    # Get the target player's hand
    hand = game_state.hands.get(target_id, [])

    # Find cards that match the clue
    for i, card in enumerate(hand):
        if clue_type == "color" and card.color.value == clue_value:
            affected_indices.append(i)
        elif clue_type == "number" and card.number == clue_value:
            affected_indices.append(i)

    return affected_indices
