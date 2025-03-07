from typing import Dict, Any
from ...game.state import Color


def give_clue_tool(agent_id: int, target_id: int, clue_type: str, clue_value: Any, game_state: Any) -> Dict[str, Any]:
    """Give a clue to another player."""
    # Validate the action
    if game_state.clue_tokens <= 0:
        raise ValueError("No clue tokens available")

    if target_id == agent_id:
        raise ValueError("Cannot give clue to yourself")

    if target_id not in game_state.hands:
        raise ValueError(f"Invalid target player: {target_id}")

    if clue_type not in ["color", "number"]:
        raise ValueError(f"Invalid clue type: {clue_type}")

    # Validate clue value based on type
    if clue_type == "color":
        if not isinstance(clue_value, str):
            raise ValueError(
                f"Color value must be a string, got {type(clue_value)}")

        if clue_value not in [c.value for c in Color]:
            raise ValueError(
                f"Invalid color value: {clue_value}. Must be one of {[c.value for c in Color]}")

    elif clue_type == "number":
        # Convert to int if it's a string
        if isinstance(clue_value, str):
            try:
                clue_value = int(clue_value)
            except ValueError:
                raise ValueError(
                    f"Invalid number value: {clue_value}. Must be convertible to an integer.")

        if not isinstance(clue_value, int):
            raise ValueError(
                f"Number value must be an integer, got {type(clue_value)}")

        if not (1 <= clue_value <= 5):
            raise ValueError(
                f"Invalid number value: {clue_value}. Must be between 1 and 5.")

    # Check if the clue matches any cards in the target's hand
    target_hand = game_state.hands[target_id]
    matches = False
    for card in target_hand:
        if (clue_type == "color" and card.color.value == clue_value) or \
           (clue_type == "number" and card.number == clue_value):
            matches = True
            break

    if not matches:
        raise ValueError(
            f"No {clue_type} {clue_value} cards in player {target_id}'s hand")

    # Format the action for the game engine
    return {
        "type": "give_clue",
        "target_id": target_id,
        "clue": {
            "type": clue_type,
            "value": clue_value
        }
    }
